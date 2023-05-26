import ml_collections
import jax
import numpy as np
from ml_collections import config_dict

def get_cfg():
    cfg = config_dict.ConfigDict()
    cfg.total_steps=7000
    # cfg.learning_rate=0.00002 #used for warmup with average coverage loss
    # cfg.learning_rate=0.0000001
    cfg.learning_rate=0.00000001

    cfg.num_dim=4
    cfg.batch_size=50

    cfg.batch_size_pmapped=np.max([cfg.batch_size//jax.local_device_count(),1])
    cfg.img_size = (cfg.batch_size,256,256,1)
    cfg.label_size = (cfg.batch_size,256,256,1)
    cfg.r_x_total= 3
    cfg.r_y_total= 3
    cfg.orig_grid_shape= (cfg.img_size[1]//2**cfg.r_x_total,cfg.img_size[2]//2**cfg.r_y_total,cfg.num_dim)
    cfg.masks_num= 4# number of mask (4 in 2D and 8 in 3D)
    cfg.volume_corr= 10000# for standardizing the volume - we want to penalize the very big and very small supervoxels 
                        # the bigger the number here the smaller the penalty

    ##getting the importance of the losses associated with deconvolutions
    ## generally last one is most similar to the actual image - hence should be most important
    cfg.deconves_importances=(0.1,0.5,1.0)
    #some constant multipliers related to the fact that those losses are disproportionally smaller than the other ones
    cfg.edge_loss_multiplier=10.0
    cfg.feature_loss_multiplier=10.0
    cfg.percent_weak_edges=0.45

    ### how important we consider diffrent losses at diffrent stages of the training loop
    #0)consistency_loss,1)rounding_loss,2)feature_variance_loss,3)edgeloss,4)average_coverage_loss,5)consistency_between_masks_loss,6)
    cfg.initial_weights_epochs_len=0 #number of epochs when initial_loss_weights would be used
    cfg.initial_loss_weights=(
        1.0 #rounding_loss
        ,0000.1 #feature_variance_loss
        ,0000.1 #edgeloss
        ,1.0 #consistency_between_masks_loss
        )

    cfg.actual_segmentation_loss_weights=(
        0.1 #rounding_loss
        ,1.0 #feature_variance_loss
        ,1.0 #edgeloss
        ,0.00001 #consistency_between_masks_loss
        )

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
    cfg.divisor_logging=1
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