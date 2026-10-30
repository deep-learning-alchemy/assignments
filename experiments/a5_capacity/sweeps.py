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
    """One post-optimizer hook updates lagged router biases per effective batch."""
    import train as native
    original = native.build_optimizer
    def build(model, *args, **kwargs):
        optimizer = original(model, *args, **kwargs)
        if hasattr(model, 'finish_step'):
            optimizer.register_step_post_hook(lambda *unused: model.finish_step())
        return optimizer
    native.build_optimizer = build
    try:
        return native.train(config)
    finally:
        native.build_optimizer = original


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
