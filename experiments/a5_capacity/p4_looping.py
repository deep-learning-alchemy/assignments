"""Problem 4: one prelude, a shared core, and one coda."""
import torch
from torch import nn
from copy import deepcopy
from modeling import AutoregressiveLM


class LoopLM(AutoregressiveLM):
    def __init__(self, config, *, passes=6, core_size=None, boundary_norm=False,
                 recall=0., final_boundary=True, tied=True, switch_step=None,
                 switch_to='unique', merge='mean', **kwargs):
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
        if switch_step is not None and (not isinstance(switch_step, int) or switch_step < 1):
            raise ValueError('switch_step counts completed optimizer updates and must be positive.')
        if switch_to not in ('unique', 'loop', 'reset') or merge not in ('mean', 'first'):
            raise ValueError('Use switch_to=unique/loop/reset and merge=mean/first.')
        self.switch_step, self.switch_to, self.merge = switch_step, switch_to, merge
        self.tied, self.completed_steps = tied, 0
        self.register_buffer('_tied', torch.tensor(tied))
        self.register_buffer('_steps', torch.tensor(0, dtype=torch.long))
        self.eval_passes = None
        self.recall, self.final_boundary = recall, final_boundary
        self.boundary = (nn.RMSNorm(config.hidden_size, eps=1e-5, elementwise_affine=False)
                         if boundary_norm else nn.Identity())
        blocks = list(self.model.layers)
        core = blocks[1:1 + core_size]
        # Repeat references to the same modules: matrices AND norm gains are shared.
        if not tied and passes * core_size > middle:
            raise ValueError('An initially unique model needs enough configured middle blocks.')
        middle_blocks = core * passes if tied else blocks[1:1 + passes * core_size]
        self.model.layers = nn.ModuleList([blocks[0], *middle_blocks, blocks[-1]])

    @torch.no_grad()
    def set_sharing(self, tied, merge='mean'):
        """Clone each occurrence to untie; average corresponding blocks to tie."""
        if merge not in ('mean', 'first'):
            raise ValueError('merge must be mean or first.')
        if tied == self.tied:
            return
        blocks = list(self.model.layers)
        if tied:
            core = []
            for j in range(self.core_size):
                group = blocks[1+j:-1:self.core_size]
                block = deepcopy(group[0])
                if merge == 'mean':
                    states = [b.state_dict() for b in group]
                    block.load_state_dict({k: torch.stack([s[k].float() for s in states]).mean(0).to(v)
                                           for k, v in states[0].items()})
                core.append(block)
            middle = core * self.passes
        else:
            # deepcopy the occurrences separately; copying the list preserves aliases.
            middle = [deepcopy(b) for b in blocks[1:-1]]
        self.model.layers = nn.ModuleList([blocks[0], *middle, blocks[-1]])
        self._tied.fill_(tied)
        self.tied = tied

    def _load_from_state_dict(self, state_dict, prefix, *args):
        # Restore parameter aliases before loading either model or Adam state.
        self.set_sharing(bool(state_dict[prefix + '_tied']), merge='first')
        self.completed_steps = int(state_dict[prefix + '_steps'])
        super()._load_from_state_dict(state_dict, prefix, *args)

    def attach_optimizer(self, optimizer):
        """Reset Adam once at the switch; retain the scheduler's optimizer object."""
        from optimizers import build_masked_weight_decay_parameter_groups
        def refresh_groups():
            decay = max(g['weight_decay'] for g in optimizer.param_groups)
            groups = build_masked_weight_decay_parameter_groups(self, decay)
            if len(groups) != len(optimizer.param_groups):
                raise ValueError('Switching must preserve the optimizer group layout.')
            for old, new in zip(optimizer.param_groups, groups):
                old['params'] = new['params']
            optimizer.state.clear()
            optimizer.zero_grad(set_to_none=True)
        load_hook = self.register_load_state_dict_post_hook(lambda *unused: refresh_groups())
        def finish(*unused):
            self.completed_steps += 1
            self._steps.fill_(self.completed_steps)
            if self.switch_step is None or self.completed_steps != self.switch_step:
                return
            observer = getattr(self, 'switch_observer', lambda model, phase: None)
            observer(self, 'before')
            if self.switch_to != 'reset':
                self.set_sharing(self.switch_to == 'loop', self.merge)
            refresh_groups()
            observer(self, 'after')
            print(f'Completed update {self.completed_steps}: {self.switch_to}; optimizer state reset.', flush=True)
        return load_hook, optimizer.register_step_post_hook(finish)

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
        if not self.tied and passes > self.passes:
            raise ValueError('A unique model has no extra blocks; evaluate at most its trained passes.')
        h = self.model.embed_tokens(input_ids)
        if position_ids is None:
            position_ids = torch.arange(h.shape[1], device=h.device)[None]
        rope = self.model.rotary_emb(h, position_ids)
        h = self.model.layers[0](h, rope, attention_mask)
        anchor = h
        for r in range(passes):
            for j in range(self.core_size):
                index = j if self.tied else r * self.core_size + j
                h = self.model.layers[1 + index](h, rope, attention_mask)
            if r < passes-1 or self.final_boundary:
                h = self.boundary(h) + self.recall*anchor
        h = self.model.layers[-1](h, rope, attention_mask)
        return self.lm_head(self.model.norm(h))


if __name__ == '__main__':
    from .sweeps import launch_one
    launch_one('loop-r6', .003)  # Edit the variant and LR.
