import ml_collections
import jax
import numpy as np
from ml_collections import config_dict
import more_itertools
import toolz
import einops
from jax import lax, random, numpy as jnp
import orbax.checkpoint
from datetime import datetime
from flax.training import orbax_utils
import os
import h5py

num_images_concatenated = 2


def apply_on_single(dat):    
    batch_images,batch_labels,slic_image=dat
    batch_images=batch_images[0,0,64:-64,64:-64,14:-14]
    batch_labels=batch_labels[0,0,64:-64,64:-64,14:-14]
    
    batch_images= einops.rearrange(batch_images, 'x y z-> z x y' )
    batch_labels= einops.rearrange(batch_labels, 'x y z->z x y' )

    return batch_images,batch_labels


def add_batches(cached_subj,cfg,batch_size=None):
  
  if(batch_size==None):
     batch_size=cfg.batch_size
  batch_size_pmapped=np.max([batch_size//jax.local_device_count(),1])


  cached_subj=list(map(apply_on_single,cached_subj ))
  batch_images,batch_labels=list(toolz.sandbox.core.unzip(cached_subj))
  batch_images= list(batch_images)
  batch_labels= list(batch_labels)
  batch_images= jnp.concatenate(batch_images,axis=0 )
  batch_labels= jnp.concatenate(batch_labels,axis=0 ) #b y z
  #padding to be able to use batch size efficiently
  target_size= int(np.ceil(batch_images.shape[0]/batch_size))*batch_size
  to_pad=target_size- batch_images.shape[0]
  if(to_pad>0):
     batch_images= jnp.pad(batch_images,((0,to_pad),(0,0),(0,0)))
     batch_labels= jnp.pad(batch_labels,((0,to_pad),(0,0),(0,0)))
  batch_images= einops.rearrange(batch_images,'(d pm b) x y->d pm b x y 1',b=batch_size_pmapped,pm=jax.local_device_count())  
  batch_labels= einops.rearrange(batch_labels,'(d pm b) x y->d pm b x y 1',b=batch_size_pmapped,pm=jax.local_device_count())  
  print(f"add_batches batch_images {batch_images.shape} batch_labels {batch_labels.shape}")
  batch_images=batch_images[:-1,:,:,:,:,:]
  batch_labels=batch_labels[:-1,:,:,:,:,:]
  return batch_images,batch_labels



def get_check_point_folder():
  now = datetime.now()
  checkPoint_folder=f"/workspaces/Jax_cuda_med/data/checkpoints/{now}"
  checkPoint_folder=checkPoint_folder.replace(' ','_')
  checkPoint_folder=checkPoint_folder.replace(':','_')
  checkPoint_folder=checkPoint_folder.replace('.','_')
  os.makedirs(checkPoint_folder)
  return checkPoint_folder  
   

def save_checkpoint(index,epoch,cfg,checkPoint_folder,state,loss):
    if(index==0 and epoch%cfg.divisor_checkpoint==0 and cfg.to_save_check_point):
        chechpoint_epoch_folder=f"{checkPoint_folder}/{epoch}"
        # os.makedirs(chechpoint_epoch_folder)

        orbax_checkpointer = orbax.checkpoint.PyTreeCheckpointer()
        ckpt = {'model': state, 'config': cfg,'loss':loss}
        save_args = orbax_utils.save_args_from_target(ckpt)
        orbax_checkpointer.save(chechpoint_epoch_folder, ckpt, save_args=save_args)

def save_examples_to_hdf5(masks,batch_images_prim,curr_label  ):
    f = h5py.File('/workspaces/Jax_cuda_med/data/hdf5_loc/example_mask.hdf5', 'w')
    f.create_dataset(f"masks",data= masks)
    f.create_dataset(f"image",data= batch_images_prim)
    f.create_dataset(f"label",data= curr_label)
    f.close()   



# def add_batches(cached_subj,cfg):
#   cached_subj=list(more_itertools.chunked(cached_subj, num_images_concatenated))
#   cached_subj=list(map(toolz.sandbox.core.unzip,cached_subj ))
#   cached_subj=list(map(lambda inn: list(map(list,inn)),cached_subj ))
#   cached_subj=list(map(lambda inn: list(map(np.concatenate,inn)),cached_subj ))
#   cached_subj=list(map(apply_on_single,cached_subj ))
#   batch_images,batch_labels=toolz.sandbox.core.unzip,cached_subj()
#   return cached_subj