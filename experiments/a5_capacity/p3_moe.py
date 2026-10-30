"""Problem 3: four full-size experts, top-1 routing and lagged quantile balance.

Balanced batches use three batched matrix products. Skewed batches use only
nonempty experts, avoiding 4x padding at router collapse. No token is dropped.
"""
import torch
from torch import nn
from torch.nn import functional as F
from modeling import AutoregressiveLM, LlamaRMSNorm


def swiglu(x, gate, up, down):
    return (F.silu(x @ gate)*(x @ up)) @ down


_compiled_swiglu = None


class Experts(nn.Module):
    def __init__(self, config):
        super().__init__()
        d, i = config.hidden_size, config.intermediate_size
        self.gate = nn.Parameter(torch.empty(4, d, i))
        self.up = nn.Parameter(torch.empty(4, d, i))
        self.down = nn.Parameter(torch.empty(4, i, d))
        self.reset_parameters()

    def reset_parameters(self, clone=False):
        for p in (self.gate, self.up, self.down):
            std = p.shape[-2]**-.5
            nn.init.trunc_normal_(p, std=std, a=-3*std, b=3*std)
            if clone:
                with torch.no_grad():p[1:].copy_(p[:1].expand_as(p[1:]))

    def compute(self, x, expert=None):
        global _compiled_swiglu
        fn = swiglu
        if x.is_cuda:
            if _compiled_swiglu is None:
                _compiled_swiglu = torch.compile(swiglu, dynamic=True)
            fn = _compiled_swiglu
        weights = (self.gate, self.up, self.down)
        return fn(x, *(w if expert is None else w[expert] for w in weights))


class RouterMLP(nn.Module):
    def __init__(self, config, *, routing='learned', norm=False, noise=0., clone=False,
                 unit=False, balance=0., quantile=False, top_k=1):
        super().__init__()
        self.experts = Experts(config)
        self.router = nn.Linear(config.hidden_size, 4, bias=False)
        self.norm = LlamaRMSNorm(config.hidden_size, 1e-5) if norm else nn.Identity()
        self.routing, self.noise, self.clone = routing, noise, clone
        self.unit, self.balance, self.quantile, self.top_k = unit, balance, quantile, top_k
        if routing in ('random', 'frozen'):self.router.requires_grad_(False)
        self.register_buffer('bias', torch.zeros(4))
        self.pending = []
        self.aux_loss = None
        self.record, self.stats = False, {}

    def forward(self, hidden):
        x = hidden.flatten(0, 1)
        logits = self.router(self.norm(x)).float()
        if self.training and self.noise:logits = logits + self.noise*torch.randn_like(logits)
        p = logits.softmax(-1)
        k = self.top_k
        selected = torch.randint(4, (len(x), k), device=x.device) if self.routing == 'random' else (logits+self.bias).topk(k, -1).indices
        weights = torch.ones_like(selected, dtype=x.dtype) if self.unit or self.routing == 'random' else (4/k)*p.gather(-1, selected).to(x.dtype)
        counts = torch.bincount(selected.flatten(), minlength=4)
        shares = counts.float()/selected.numel()
        self.aux_loss = self.balance*4*(shares.detach()*p.mean(0)).sum()
        if self.training and torch.is_grad_enabled() and self.quantile:
            alpha = (logits.detach()+self.bias).topk(k+1, -1).values[:, -1:]
            self.pending.append((logits.detach()-alpha).clone())
        # Sort assignments once, then pack one contiguous batch for each expert.
        order = selected.flatten().argsort(stable=True)
        tokens = torch.arange(len(x), device=x.device).repeat_interleave(k)[order]
        counts_cpu = counts.tolist()  # One routing synchronization per layer.
        capacity = max(counts_cpu)
        if capacity <= 1.5*selected.numel()/4:
            expert = selected.flatten()[order]
            offset = torch.repeat_interleave(counts.cumsum(0)-counts, counts)
            slot = expert*capacity + torch.arange(len(tokens), device=x.device)-offset
            packed = x.new_zeros((4*capacity, x.shape[-1])).index_copy(0, slot, x[tokens])
            values = self.experts.compute(packed.view(4, capacity, -1)).flatten(0, 1)[slot]
        else:
            chunks = x[tokens].split(counts_cpu)
            values = torch.cat([self.experts.compute(chunk, e) for e, chunk in enumerate(chunks) if len(chunk)])
        out = torch.zeros_like(x).index_add(0, tokens, values*weights.flatten()[order, None])
        if self.record:
            from .diagnostics import router_metrics
            self.stats = router_metrics(p, selected)
        return out.reshape_as(hidden)

    @torch.no_grad()
    def finish_step(self):
        if self.pending:
            # Quantile threshold over all accumulation microbatches, for NEXT step.
            residual = torch.cat(self.pending)
            target = max(1, len(residual)*self.top_k//4)
            beta = residual.kthvalue(len(residual)-target+1, dim=0).values
            self.bias.copy_(-beta+beta.mean())
            self.pending.clear()


class MoELM(AutoregressiveLM):
    def __init__(self, config, *, router=None, **kwargs):
        super().__init__(config, **kwargs)
        for block in self.model.layers:block.mlp = RouterMLP(config, **(router or {}))

    def post_initialize(self):
        for block in self.model.layers:block.mlp.experts.reset_parameters(block.mlp.clone)

    def auxiliary_loss(self):
        return sum(block.mlp.aux_loss for block in self.model.layers if block.mlp.aux_loss is not None)

    def finish_step(self):
        for block in self.model.layers:block.mlp.finish_step()


if __name__ == '__main__':
    from .sweeps import launch_one
    launch_one('moe-learned', .003)  # Edit the variant and LR.
