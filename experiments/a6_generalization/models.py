"""DNA vocabulary, attention-only dropout, and the A5 architecture carryovers."""

import torch
from modeling import AutoregressiveLM, initialize_model
from experiments.a5_capacity.p1_depth_width import architecture
from .common import VOCAB, reference


def build(config, root=None):
    c = config
    shape = architecture(c.depth, c.width, vocab=len(VOCAB), context=c.context)
    torch.manual_seed(c.seed)
    if c.architecture == 'moe':
        from experiments.a5_capacity.p3_moe import MoELM

        model = MoELM(shape, router=dict(quantile=True, balance=0.0))
    elif c.architecture == 'loop':
        from experiments.a5_capacity.p4_looping import LoopLM

        model = LoopLM(shape, core_size=1, boundary_norm=True, recall=1.0)
    elif c.architecture == 'dense':
        model = AutoregressiveLM(shape)
    else:
        raise ValueError('Use dense/moe/loop, or add your own architecture here.')
    torch.manual_seed(c.seed)
    initialize_model(model)
    if hasattr(model, 'post_initialize'):
        model.post_initialize()
    # Match the DNA experiments' Llama3 frequency scaling, locally to A6.
    inv = model.model.rotary_emb.inv_freq
    wavelength = 2 * torch.pi / inv
    smooth = (8192 / wavelength - 1) / 3
    model.model.rotary_emb.inv_freq = torch.where(
        wavelength > 8192,
        inv / 32,
        torch.where(wavelength >= 2048, (1 - smooth) * inv / 32 + smooth * inv, inv),
    )
    for block in model.model.layers:
        block.self_attn.dropout = c.attention_dropout  # Residual dropout stays zero.
    if c.initialization != 'random':
        state = torch.load(
            reference(f'checkpoints/{c.initialization}.pt', root),
            map_location='cpu',
            weights_only=False,
        )['model']
        if c.initialization == 'dclm':
            state = {
                n: p
                for n, p in state.items()
                if n not in ('model.embed_tokens.weight', 'lm_head.weight')
            }
            result = model.load_state_dict(state, strict=False)
            assert (
                set(result.missing_keys) == {'model.embed_tokens.weight', 'lm_head.weight'}
                and not result.unexpected_keys
            )
        else:
            model.load_state_dict(state)
    for name, p in model.named_parameters():
        if c.adaptation == 'frozen_head':
            p.requires_grad_(name != 'lm_head.weight')
        elif c.adaptation == 'frozen_body':
            p.requires_grad_(name in ('model.embed_tokens.weight', 'lm_head.weight'))
        elif c.adaptation not in ('full', 'toward_initialization'):
            raise ValueError('Unknown adaptation mode.')
    return model
