"""Short editable experiment definitions. Use the native recipe for local GPU or Modal runs."""
from dataclasses import replace
from copy import deepcopy
from train import TrainConfig
from .p1_depth_width import architecture

VARIANTS = {
    'dense': (None, {}),
    'mos': ('experiments.a5_capacity.p2_readout:HeadLM', dict(method='mos')),
    'sigsoftmax': ('experiments.a5_capacity.p2_readout:HeadLM', dict(method='sigsoftmax')),
    'cplm': ('experiments.a5_capacity.p2_readout:CPLM', {}),
    'moe-random': ('experiments.a5_capacity.p3_moe:MoELM', dict(router=dict(routing='random', balance=0.))),
    'moe-frozen': ('experiments.a5_capacity.p3_moe:MoELM', dict(router=dict(routing='frozen', balance=0.))),
    'moe-learned': ('experiments.a5_capacity.p3_moe:MoELM', dict(router=dict(balance=0.))),
    'moe-balanced': ('experiments.a5_capacity.p3_moe:MoELM', dict(router=dict(balance=.01))),
    'moe-norm': ('experiments.a5_capacity.p3_moe:MoELM', dict(router=dict(norm=True, balance=0.))),
    'moe-noisy': ('experiments.a5_capacity.p3_moe:MoELM', dict(router=dict(noise=.05, balance=0.))),
    'moe-cloned': ('experiments.a5_capacity.p3_moe:MoELM', dict(router=dict(clone=True, balance=0.))),
    'moe-unit': ('experiments.a5_capacity.p3_moe:MoELM', dict(router=dict(unit=True, balance=0.))),
    'moe-quantile': ('experiments.a5_capacity.p3_moe:MoELM', dict(router=dict(quantile=True, balance=0.))),
    'moe-full': ('experiments.a5_capacity.p3_moe:MoELM', dict(router=dict(top_k=4, balance=0.))),
}
for rank in (32, 64, 128, 256):
    for method in ('softmax', 'mos', 'sigsoftmax'):
        VARIANTS[f'{method}-r{rank}'] = ('experiments.a5_capacity.p2_readout:HeadLM', dict(rank=rank, method=method))
    VARIANTS[f'cplm-r{rank}'] = ('experiments.a5_capacity.p2_readout:CPLM', dict(rank=rank))

# P4(a): eight applications, with different numbers of distinct core blocks.
for passes in (1, 2, 3, 6):
    VARIANTS[f'loop-r{passes}'] = ('experiments.a5_capacity.p4_looping:LoopLM', dict(passes=passes))
for key, extra in [('boundary-norm', dict(boundary_norm=True)),
                   ('recall', dict(recall=1.)),
                   ('norm-recall', dict(boundary_norm=True, recall=1.))]:
    VARIANTS['loop-'+key] = ('experiments.a5_capacity.p4_looping:LoopLM', dict(passes=6, **extra))

# P4(d): three distinct blocks; R core applications gives total depth R+2.
for passes in (1, 2, 3, 4, 6, 18, 20):
    VARIANTS[f'loop-1block-r{passes}'] = ('experiments.a5_capacity.p4_looping:LoopLM', dict(passes=passes, core_size=1))

# P4(e): the full run has 9,375 updates; switch after update 4,688.
for name, tied, target in [('loop-to-unique', True, 'unique'),
                            ('unique-to-loop', False, 'loop'),
                            ('loop-reset', True, 'reset'),
                            ('unique-reset', False, 'reset')]:
    VARIANTS[name] = ('experiments.a5_capacity.p4_looping:LoopLM',
                      dict(passes=6, tied=tied, switch_step=4688, switch_to=target))


def config(name='dense', lr=.003, *, depth=8, width=None, **overrides):
    builder, kwargs = VARIANTS[name]
    cfg = TrainConfig(model_config=architecture(depth, width or 64*depth),
                      model_builder=builder, model_builder_kwargs=deepcopy(kwargs),
                      learning_rate=lr, num_micro_batches=4,
                      torch_compile_mode='default',
                      run_name=f'a5-{name}-d{depth}-w{width or 64*depth}-lr{lr:g}',
                      wandb_tags=('a5', 'student', name))
    return replace(cfg, **overrides)


def train(config):
    """Install the model's update hook without changing the shared trainer."""
    import train as native
    switching = config.model_builder_kwargs.get('switch_step') is not None
    if switching and (config.ema_decays or config.num_micro_batches == 1):
        raise ValueError('Switching uses the default four microbatches and no EMA; avoid compiled parameter captures.')
    if switching:
        import torch
        if torch.cuda.device_count() > 1:
            raise ValueError('Architecture switching currently supports one visible GPU.')
    original = native.build_optimizer
    hooks = []
    def build(model, *args, **kwargs):
        optimizer = original(model, *args, **kwargs)
        if hasattr(model, 'attach_optimizer') and switching:
            hooks.extend(model.attach_optimizer(optimizer))
            if switching:
                # Measure both sides of the same update, on the native held-out set.
                import json
                from pathlib import Path
                val = native.load_token_dataset(config.val_dataset, role='val', data_seed=config.data_seed)
                batches = native.create_batches(val, config.batch_size // config.num_micro_batches)
                path = Path(config.model_dir) / native.training_run_name(config) / 'switch_metrics.json'
                def observe(model, phase):
                    loss = float(native.evaluate(model, batches, config, next(model.parameters()).device))
                    record = dict(phase=phase, completed_steps=model.completed_steps, loss=loss,
                                  tied=model.tied, parameters=sum(p.numel() for p in model.parameters()))
                    events = json.loads(path.read_text()) if path.exists() else []
                    events = [e for e in events if e['phase'] != phase] + [record]
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_text(json.dumps(events, indent=2))
                    print('Switch measurement:', record, flush=True)
                    if config.wandb_online:
                        import wandb
                        wandb.log({f'switch/{phase}_val_loss':loss, 'optimizer_step':model.completed_steps-1})
                model.switch_observer = observe
        elif hasattr(model, 'finish_step'):
            optimizer.register_step_post_hook(lambda *unused: model.finish_step())
        return optimizer
    native.build_optimizer = build
    try:
        return native.train(config)
    finally:
        native.build_optimizer = original
        for hook in hooks:
            hook.remove()


def run(config):
    """Modal worker; regenerate SpamLang because host caches are not uploaded."""
    from modal_utils import MODAL_MODEL_DIR, user_volume
    if config.train_dataset.tag == 'spam':
        from .p2_readout import training_config
        data = training_config(config.model_config.vocab_size, config.model_config.hidden_size,
                               root='/tmp/a5_capacity/spam')
        config = replace(config, train_dataset=data.train_dataset, val_dataset=data.val_dataset)
    try:
        train(replace(config, model_dir=str(MODAL_MODEL_DIR)))
        return dict(run_name=config.run_name, checkpoint=str(MODAL_MODEL_DIR/config.run_name))
    finally:
        user_volume.commit()


def launch(config, gpu='H100'):
    import modal
    from modal_utils import app, build_image, VOLUME_MOUNTS, MODAL_ENVIRONMENT, secrets
    remote = app.function(image=build_image(), volumes=VOLUME_MOUNTS,
                          secrets=secrets(include_wandb=config.wandb_online),
                          gpu=gpu, cpu=8, memory=32768, timeout=24*3600)(run)
    with modal.enable_output(), app.run(environment_name=MODAL_ENVIRONMENT, detach=True):
        call = remote.spawn(config)
        print('Modal call:', call.object_id, 'run:', config.run_name)
        return call.object_id


def launch_one(name='dense', lr=.003, *, local=False, dry_run=False, **overrides):
    cfg = config(name, lr, **overrides)
    if dry_run:
        print(cfg)
        return cfg
    return train(cfg) if local else launch(cfg)


if __name__ == '__main__':
    launch_one('moe-learned', dry_run=True)
