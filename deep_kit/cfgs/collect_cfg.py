import os
from omegaconf import OmegaConf as Ocfg
import importlib.resources
from .. import cfgs
from .. import utils

Ocfg.register_new_resolver('len', lambda x: len(x))

cfg_base = Ocfg.create(importlib.resources.read_text(cfgs, 'base.yml'))
cfg_default_exp = Ocfg.load('cfgs/default/experiment.yml')
cfg_cli = Ocfg.from_cli()

assert hasattr(cfg_cli.exp, 'name')
if cfg_cli.exp.name.startswith('train'):
    cfg_cli.exp.mode = 'train'
    if os.path.exists(f'cfgs/train/{cfg_cli.exp.name}.yml'):
        cfg_exp = Ocfg.load(f'cfgs/train/{cfg_cli.exp.name}.yml')
    else:
        cfg_exp = Ocfg.load(f'cfgs/{cfg_cli.exp.name}.yml')
elif cfg_cli.exp.name.startswith('test'):
    cfg_cli.exp.mode = 'test'
    if os.path.exists(f'cfgs/test/{cfg_cli.exp.name}.yml'):
        cfg_exp = Ocfg.load(f'cfgs/test/{cfg_cli.exp.name}.yml')
    else:
        cfg_exp = Ocfg.load(f'cfgs/{cfg_cli.exp.name}.yml')
else:
    raise ValueError
cfg_exp.exp.name = cfg_cli.exp.name
cfg_default_model = Ocfg.load(f'cfgs/default/models/{cfg_exp.model.name}.yml')
cfg_default_dataset = Ocfg.load(f'cfgs/default/datasets/{cfg_exp.dataset.name}.yml')

cfg = Ocfg.unsafe_merge(cfg_base, cfg_default_exp, cfg_default_model, cfg_default_dataset, cfg_exp, cfg_cli)
cfg.var = Ocfg.create(flags={"allow_objects": True})
Ocfg.resolve(cfg)

cfg.exp.train.optimizer = Ocfg.masked_copy(cfg.exp.train.optimizer, ['name', 'lr', cfg.exp.train.optimizer['name']])
keys_masked_sch = ['name']
if cfg.exp.train.scheduler.name is not None:
    keys_masked_sch.append(cfg.exp.train.scheduler.name)
cfg.exp.train.scheduler = Ocfg.masked_copy(cfg.exp.train.scheduler, keys_masked_sch)

assert cfg.exp.mode in ['train', 'test']

cfg.exp.name = cfg.exp.name.replace('/', '-')
Ocfg.set_readonly(cfg, True)
Ocfg.set_readonly(cfg.var, False)
