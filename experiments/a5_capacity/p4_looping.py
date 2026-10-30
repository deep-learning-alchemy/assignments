"""Problem 4: one prelude, a shared core, and one coda."""
import torch
from torch import nn
from modeling import AutoregressiveLM


class LoopLM(AutoregressiveLM):
    def __init__(self, config, *, passes=6, core_size=None, boundary_norm=False,
                 recall=0., final_boundary=True, **kwargs):
        super().__init__(config, **kwargs)
        if not isinstance(passes, int) or passes < 1:
            raise ValueError('passes must be a positive integer.')
        middle = config.num_hidden_layers - 2
        if core_size is None:
            if middle < 1 or middle % passes:
                raise ValueError('Passes must divide the middle depth; use core_size=1 for a loop-count ladder.')
            core_size = middle // passes
        if not 1 <= core_size <= middle:
            raise ValueError('core_size must fit between the prelude and coda.')
        self.passes, self.core_size = passes, core_size
        self.eval_passes = None
        self.recall, self.final_boundary = recall, final_boundary
        self.boundary = (nn.RMSNorm(config.hidden_size, eps=1e-5, elementwise_affine=False)
                         if boundary_norm else nn.Identity())
        blocks = list(self.model.layers)
        core = blocks[1:1 + core_size]
        # Repeat references to the same modules: matrices AND norm gains are shared.
        self.model.layers = nn.ModuleList([blocks[0], *(core * passes), blocks[-1]])

    @property
    def executed_depth(self):
        passes = self.passes if self.eval_passes is None else self.eval_passes
        return 2 + self.core_size * passes

    def forward(self, input_ids=None, attention_mask=None, position_ids=None):
        passes = self.passes if self.eval_passes is None else self.eval_passes
        if not isinstance(passes, int) or passes < 1:
            raise ValueError('Evaluation passes must be a positive integer.')
        if self.training and passes != self.passes:
            raise ValueError('Change eval_passes only in evaluation mode.')
        h = self.model.embed_tokens(input_ids)
        if position_ids is None:
            position_ids = torch.arange(h.shape[1], device=h.device)[None]
        rope = self.model.rotary_emb(h, position_ids)
        h = self.model.layers[0](h, rope, attention_mask)
        anchor = h
        for r in range(passes):
            for j in range(self.core_size):
                h = self.model.layers[1 + j](h, rope, attention_mask)
            if r < passes-1 or self.final_boundary:
                h = self.boundary(h) + self.recall*anchor
        h = self.model.layers[-1](h, rope, attention_mask)
        return self.lm_head(self.model.norm(h))


if __name__ == '__main__':
    from .sweeps import launch_one
    launch_one('loop-r6', .003)  # Edit the variant and LR.
