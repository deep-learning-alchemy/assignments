"""Assignment measurements. Run separately from training on fixed held-out tokens."""
import math
from contextlib import contextmanager
import torch
from torch.nn import functional as F


@contextmanager
def fp32_math(device):
    previous = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    try:
        with torch.autocast(device_type=device, enabled=False):yield
    finally:
        torch.backends.cuda.matmul.allow_tf32 = previous


@torch.no_grad()
def router_metrics(probabilities, selected):
    p = probabilities.float()
    loads = torch.bincount(selected.flatten(), minlength=p.shape[-1]).float()/selected.numel()
    return dict(load_cv=float(loads.std(unbiased=False)/loads.mean()),
                entropy=float(-(p*p.clamp_min(1e-30).log()).sum(-1).mean()/math.log(p.shape[-1])),
                mean_max_probability=float(p.max(-1).values.mean()),
                loads=loads.cpu().tolist())


def summarize_routers(layers):
    return dict(load_cv=max(s['load_cv'] for s in layers),
                entropy=sum(s['entropy'] for s in layers)/len(layers),
                mean_max_probability=sum(s['mean_max_probability'] for s in layers)/len(layers))


@torch.no_grad()
def gradient_energy(logits, targets, weight):
    """FP32 energy outside col(weight); no V×V projector is materialized."""
    with fp32_math(logits.device.type):
        g = logits.float().softmax(-1).flatten(0, -2)
        g[torch.arange(len(g), device=g.device), targets.flatten()] -= 1
        u, s, _ = torch.linalg.svd(weight.float(), full_matrices=False)
        basis = u[:, s > s.max()*1e-5]
        total = g.square().sum()
        # Orthogonality gives ||g-UU'g||² = ||g||²-||gU||².
        energy = None if total == 0 else float((1-(g@basis).square().sum()/total).clamp(0, 1))
        return dict(suppressed_fraction=energy, rank=basis.shape[1],gradient_energy=float(total),
                    random_subspace_reference=1-basis.shape[1]/weight.shape[0])


def signal_profile(model, tokens):
    """FP32 signals per core pass (or dense block), preserving parameter grads."""
    states, updates, hooks = [], [], []
    layers = model.model.layers
    core_size = getattr(model, 'core_size', 1)
    pass_start = [None]
    def remember_start(module, args):
        pass_start[0] = args[0]
    def capture(module, args, output):
        states.append(output)
        before = pass_start[0] if core_size > 1 and module is layers[core_size] else args[0]
        updates.append(float((output.detach()-before.detach()).float().square().mean().sqrt()
                             /before.detach().float().square().mean().sqrt().clamp_min(1e-12)))
    blocks = layers
    if core_size > 1:
        hooks.append(layers[1].register_forward_pre_hook(remember_start))
        blocks = [layers[0], layers[core_size], layers[-1]]
    # Register once per MODULE; a shared block's one hook fires at every pass.
    for block in dict.fromkeys(blocks):
        hooks.append(block.register_forward_hook(capture))
    previous_mode = model.training
    model.eval()
    try:
        with torch.enable_grad(), fp32_math(tokens.device.type):
            logits = model(tokens)
            loss = F.cross_entropy(logits[:, :-1].float().flatten(0, 1), tokens[:, 1:].flatten())
            gradients = torch.autograd.grad(loss, states)
        return [dict(application=i + 1, activation_rms=float(h.detach().float().square().mean().sqrt()),
                     gradient_rms=float(g.float().square().mean().sqrt()), relative_update=update)
                for i, (h, g, update) in enumerate(zip(states, gradients, updates))]
    finally:
        for hook in hooks:hook.remove()
        model.train(previous_mode)


def head_weight(model):
    from .p2_readout import Readout
    head = model.lm_head
    if not isinstance(head,Readout):return head.weight
    if head.method=='mos':raise ValueError('MoS has no single linear output matrix; inspect its components separately.')
    with fp32_math(head.up[0].weight.device.type):
        return head.up[0].weight.float()@head.down[0].weight.float()


@torch.no_grad()
def measure_model(model,tokens):
    from .p3_moe import RouterMLP
    from .p2_readout import CPLM
    from .p4_looping import LoopLM
    previous_mode = model.training
    model.eval()
    routers=[m for m in model.modules() if isinstance(m,RouterMLP)]
    flags=[m.record for m in routers]
    try:
        for m in routers:m.record=True
        logits=model(tokens)
        from train import causal_lm_loss
        result=dict(nll=float(causal_lm_loss(logits,tokens)),sequences=tokens.shape[0],
                    context=tokens.shape[1], parameters=sum(p.numel() for p in model.parameters()))
        if routers:
            layers=[m.stats for m in routers]
            result['routers']=summarize_routers(layers)
            result['router_layers']=layers
            if all(m.routing=='random' for m in routers):
                result['routers'].update(entropy=None,mean_max_probability=None)
            for m,stats in zip(routers,layers):
                if m.routing=='random':stats.update(entropy=None,mean_max_probability=None)
        elif isinstance(model,LoopLM):
            result.update(executed_depth=model.executed_depth, signals=signal_profile(model,tokens))
        elif not isinstance(model,CPLM) and getattr(model,'method','softmax')=='softmax':
            result['gradient_energy']=gradient_energy(logits[:,:-1],tokens[:,1:],head_weight(model))
        else:
            result['gradient_energy_note']='The single linear-head projection diagnostic applies to ordinary softmax, not the mixture or nonlinear outputs.'
        return result
    finally:
        for m,flag in zip(routers,flags):m.record=flag
        model.train(previous_mode)
