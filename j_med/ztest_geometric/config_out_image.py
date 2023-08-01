import ml_collections
import jax
import numpy as np
from ml_collections import config_dict

def get_cfg():
    cfg = config_dict.ConfigDict()
    # cfg.total_steps=8000
    cfg.total_steps=20000
    cfg.learning_rate=0.000009
    # cfg.learning_rate=0.000002
    # cfg.learning_rate=0.0000009
    cfg.convolution_channels=32
    cfg.batch_size=240
    cfg.batch_size_pmapped=np.max([cfg.batch_size//jax.local_device_count(),1])
    cfg.img_size = (cfg.batch_size,256,256,1)
    cfg.img_size_pmapped = (cfg.batch_size_pmapped,256,256,1)
    cfg.r_x_total=3
    cfg.r_y_total=3
    cfg.num_dim=2
    cfg.orig_grid_shape= (cfg.img_size[1]//2**cfg.r_x_total,cfg.img_size[2]//2**cfg.r_y_total,cfg.num_dim)
    cfg.r=8
    #controls how many additional points per primary triangle will be added
    cfg.num_additional_points=2
    #control what is the index of first additional point
    cfg.primary_control_points_offset=9



    cfg.weights_channels=8
    cfg.epsilon=0.0000000000001
    cfg.optax_name = 'big_vision.scale_by_adafactor'
    cfg.optax = dict(beta2_cap=0.95)

    cfg.lr = cfg.learning_rate
    cfg.wd = 0.00001 # default is 0.0001; paper used 0.3, effective wd=0.3*lr
    cfg.schedule = dict(
        warmup_steps=20,
        decay_type='linear',
        linear_end=cfg.lr/100,
    )

    # GSAM settings.
    # Note: when rho_max=rho_min and alpha=0, GSAM reduces to SAM.
    cfg.gsam = dict(
        rho_max=0.6,
        rho_min=0.1,
        alpha=0.6,
        lr_max=cfg.get_ref('lr'),
        lr_min=cfg.schedule.get_ref('linear_end') * cfg.get_ref('lr'),
    )

    #setting how frequent the checkpoints should be performed
    cfg.divisor_checkpoint=15
    cfg.divisor_logging=5
    cfg.to_save_check_point=False

    cfg.is_gsam=False


    cfg = ml_collections.FrozenConfigDict(cfg)

    return cfg


def get_dynamic_cfgs():
  dynamic_cfg_a = config_dict.ConfigDict()
  dynamic_cfg_a.is_beg=True
  dynamic_cfg_a = ml_collections.config_dict.FrozenConfigDict(dynamic_cfg_a)

  dynamic_cfg_b = config_dict.ConfigDict()
  dynamic_cfg_b.is_beg=False
  dynamic_cfg_b = ml_collections.config_dict.FrozenConfigDict(dynamic_cfg_b)
  return [dynamic_cfg_a, dynamic_cfg_b]