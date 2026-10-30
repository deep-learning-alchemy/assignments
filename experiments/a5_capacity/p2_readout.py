"""Problem 2: readout interventions and the SpamLang toy task."""
import math
import torch
from torch import nn
from torch.nn import functional as F
from modeling import AutoregressiveLM


class Readout(nn.Module):
    def __init__(self, width, vocab, rank, method='softmax'):
        super().__init__()
        self.method = method
        count = 4 if method == 'mos' else 1
        self.down = nn.ModuleList([nn.Linear(width, rank, bias=False) for _ in range(count)])
        self.up = nn.ModuleList([nn.Linear(rank, vocab, bias=False) for _ in range(count)])
        self.gate = nn.Linear(width, count, bias=False) if count > 1 else None

    def reset_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                std = m.in_features**-.5
                nn.init.trunc_normal_(m.weight, std=std, a=-3*std, b=3*std)

    def forward(self, h):
        z = [up(down(h)).float() for down, up in zip(self.down, self.up)]
        if self.method == 'mos':
            return torch.logsumexp(torch.stack([x.log_softmax(-1) for x in z], -2)
                                   + self.gate(h).float().log_softmax(-1)[..., None], -2)
        return z[0] + F.logsigmoid(z[0]) if self.method == 'sigsoftmax' else z[0]


class HeadLM(AutoregressiveLM):
    def __init__(self, config, *, rank=None, method='softmax', **kwargs):
        super().__init__(config, **kwargs)
        self.method = method
        if rank is not None or method == 'mos':
            self.lm_head = Readout(config.hidden_size, config.vocab_size,
                                   rank or config.hidden_size, method)

    def get_output_embeddings(self):
        return self.lm_head.up[0] if isinstance(self.lm_head, Readout) else self.lm_head

    def post_initialize(self):
        if isinstance(self.lm_head, Readout):
            self.lm_head.reset_parameters()

    def forward(self, input_ids=None, **kwargs):
        z = super().forward(input_ids, **kwargs)
        if self.method == 'sigsoftmax' and not isinstance(self.lm_head, Readout):
            z = z.float() + F.logsigmoid(z.float())
        return z


class CPLM(HeadLM):
    """Causal prefix pointer with a learned copy gate and LM-return sink."""
    def __init__(self, config, *, rank=None, **kwargs):
        super().__init__(config, rank=rank, **kwargs)
        d = config.hidden_size
        self.copy_q = nn.Linear(d, d, bias=False)
        self.copy_k = nn.Linear(d, d, bias=False)
        self.copy_gate = nn.Linear(d, 1, bias=False)
        self.sink_key = nn.Parameter(torch.empty(d))
        nn.init.normal_(self.sink_key, std=d**-.5)

    def forward(self, input_ids=None, attention_mask=None, position_ids=None):
        h = self.model(input_ids, attention_mask=attention_mask, position_ids=position_ids)
        logits = self.lm_head(h).float()
        q, k = self.copy_q(h), self.copy_k(h)
        scores = (q @ k.transpose(-1, -2)).float()/math.sqrt(q.shape[-1])
        t = input_ids.shape[1]
        mask = torch.ones(t, t, device=h.device, dtype=torch.bool).tril()
        if attention_mask is not None:
            mask = mask & attention_mask[:, None, :].bool()
        scores = scores.masked_fill(~mask, -torch.inf)
        sink = (q.float()*self.sink_key.float()).sum(-1)/math.sqrt(q.shape[-1])
        attention = torch.cat((sink[..., None], scores), -1).log_softmax(-1)
        mass = torch.zeros_like(logits).scatter_add(
            -1, input_ids[:, None, :].expand_as(scores), attention[..., 1:].exp())
        # Paper gate: c = sigmoid(g(h) - logsumexp(vocabulary logits)).
        z = self.copy_gate(h).squeeze(-1).float() - logits.logsumexp(-1)
        log_c, log_not_c = F.logsigmoid(z), F.logsigmoid(-z)
        log_lm_weight = torch.logaddexp(log_not_c, log_c + attention[..., 0])
        log_mass = torch.where(mass > 0, mass.clamp_min(torch.finfo(mass.dtype).tiny).log(), -torch.inf)
        result = torch.logaddexp(log_lm_weight[..., None] + logits.log_softmax(-1),
                                log_c[..., None] + log_mass)
        # Normalized log probabilities are valid inputs to the native CE loss.
        return result


def training_config(vocab=4096, width=64, lr=.001, root='cache/a5_capacity/spam'):
    import json
    from dataclasses import replace
    from pathlib import Path
    import numpy as np
    from data import TokenDatasetConfig
    from .p1_depth_width import architecture
    from .sweeps import config
    root = Path(root)/f'v{vocab}-w{width}'
    for split, size, seed in [('train', 102400, 17), ('val', 2048, 18)]:
        folder = root/split
        folder.mkdir(parents=True, exist_ok=True)
        ids = np.random.default_rng(seed).integers(vocab, size=size, dtype=np.uint16)
        np.repeat(ids[:, None], 64, 1).tofile(folder/'tokens.bin')
        (folder/'metadata.json').write_text(json.dumps(dict(dtype='uint16', num_sequences=size,
                                                         seq_len=64, vocab_size=vocab)))
    shape = replace(architecture(1, width, vocab, 64), num_attention_heads=4,
                    num_key_value_heads=4, head_dim=width//4)
    return config('dense', lr, model_config=shape,
                  train_dataset=TokenDatasetConfig('train', 'spam', 64, path=str(root/'train')),
                  val_dataset=TokenDatasetConfig('val', 'spam', 64, path=str(root/'val')),
                  num_train_sequences=102400, batch_size=128, num_micro_batches=1,
                  data_seed=None, run_name=f'a5-spam-v{vocab}-w{width}-lr{lr:g}')


if __name__ == '__main__':
    from .sweeps import launch_one
    launch_one('softmax-r64', .003)  # Edit the variant and LR.
