import ml_collections
import jax
import numpy as np
from ml_collections import config_dict

def get_cfg():
    cfg = config_dict.ConfigDict()
    # cfg.total_steps=8000
    cfg.total_steps=8000
    # cfg.learning_rate=0.00002 #used for warmup with average coverage loss
    # cfg.learning_rate=0.0000001
    # cfg.learning_rate=0.00000001
    # cfg.learning_rate=0.00000000001
    cfg.learning_rate=0.00000009
    cfg.convolution_channels=32

    #true_num_dim - number of the dimensions num_dim is number of the masks needed to describe it
    cfg.true_num_dim=2
    cfg.num_dim=4
    cfg.batch_size=120

    cfg.batch_size_pmapped=np.max([cfg.batch_size//jax.local_device_count(),1])
    cfg.img_size = (cfg.batch_size,256,256,1)
    cfg.img_size_pmapped = (cfg.batch_size_pmapped,256,256,1)
    cfg.masks_size = (cfg.batch_size_pmapped,256,256,cfg.num_dim)
    cfg.bi_channel_size = (cfg.batch_size_pmapped,256,256,2)
    cfg.label_size = (cfg.batch_size_pmapped,256,256,1)
    cfg.deconv_multi_zero_shape = (cfg.batch_size_pmapped,256,256,cfg.convolution_channels)
    cfg.r_x_total= 3
    cfg.r_y_total= 3
    cfg.orig_grid_shape= (cfg.img_size[1]//2**cfg.r_x_total,cfg.img_size[2]//2**cfg.r_y_total,cfg.num_dim)
    cfg.masks_num= 4# number of mask (4 in 2D and 8 in 3D)
    cfg.volume_corr= 10000# for standardizing the volume - we want to penalize the very big and very small supervoxels 
                        # the bigger the number here the smaller the penalty
    ##getting the importance of the losses associated with deconvolutions
    ## generally last one is most similar to the actual image - hence should be most important
    cfg.deconves_importances=(0.1,0.5,1.0)

    #just for numerical stability
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
    cfg.divisor_checkpoint=10
    cfg.divisor_logging=8
    cfg.to_save_check_point=False

    cfg.is_gsam=False
    config_dict.ConfigDict()
    # convSpecs=(
    #    {'in_channels':1,'out_channels':cfg.convolution_channels, 'kernel_size':(5,5),'stride':(1,1) },
    #    {'in_channels':1,'out_channels':cfg.convolution_channels, 'kernel_size':(5,5),'stride':(1,1) },
    #    {'in_channels':1,'out_channels':cfg.convolution_channels, 'kernel_size':(5,5),'stride':(1,1) },
    #    {'in_channels':1,'out_channels':cfg.convolution_channels, 'kernel_size':(5,5),'stride':(1,1) },
    #    {'in_channels':1,'out_channels':cfg.convolution_channels, 'kernel_size':(5,5),'stride':(1,1) },
    #    {'in_channels':1,'out_channels':cfg.convolution_channels, 'kernel_size':(5,5),'stride':(1,1) },
    #    {'in_channels':1,'out_channels':cfg.convolution_channels, 'kernel_size':(5,5),'stride':(1,1) },
    #    {'in_channels':1,'out_channels':cfg.convolution_channels, 'kernel_size':(5,5),'stride':(1,1) }
    #     )
    # cfg.convSpecs= list(map(FrozenDict,convSpecs ))
    # self.convSpecs=[{'in_channels':3,'out_channels':4, 'kernel_size':(3,3,3),'stride':(2,2,2) }
    # ,{'in_channels':4,'out_channels':4, 'kernel_size':(3,3,3),'stride':(2,2,2) }
    # ,{'in_channels':4,'out_channels':8, 'kernel_size':(3,3,3),'stride':(1,1,1) }
    # ,{'in_channels':8,'out_channels':16, 'kernel_size':(3,3,3),'stride':(2,2,2) }]

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