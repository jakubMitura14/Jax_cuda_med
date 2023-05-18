from flax import linen as nn
import numpy as np
from typing import Any, Callable, Optional, Tuple, Type, List
from jax import lax, random, numpy as jnp
import einops
import torchio
import torch
import matplotlib.pyplot as plt
import tempfile
import shutil
import os
import glob
import jax
# import monai_swin_nD
import tensorflow as tf
# import monai_einops
import torch 
import einops
import torchio as tio
import optax
from flax.training import train_state  # Useful dataclass to keep train state
from torch.utils.data import DataLoader
import my_jax_3d_regr as my_jax_3d_regr
from my_jax_3d_regr import SwinTransformer
import augmentations.simpleTransforms as simpleTransforms
from flax.linen.linear import Dense

data_dir='/root/data'
train_images = sorted(
    glob.glob(os.path.join(data_dir, "imagesTr", "*.nii.gz")))
train_labels = sorted(
    glob.glob(os.path.join(data_dir, "labelsTr", "*.nii.gz")))
data_dicts = [
    {"image": image_name, "label": label_name}
    for image_name, label_name in zip(train_images, train_labels)
]

rang=list(range(0,len(train_images)))

subjects_list=list(map(lambda index:tio.Subject(image=tio.ScalarImage(train_images[index],),label=tio.LabelMap(train_labels[index])),rang ))
subjects_list_train=subjects_list[:-9]
subjects_list_val=subjects_list[-9:]

transforms = [
    tio.RescaleIntensity(out_min_max=(0, 1)),
    tio.Resample((5.5,5.5,5.5)),
    tio.transforms.CropOrPad((64,64,32)),
]
transform = tio.Compose(transforms)
subjects_dataset = tio.SubjectsDataset(subjects_list_train, transform=transform)



prng = jax.random.PRNGKey(42)

feature_size  = 12 #by how long vector each image patch will be represented
in_chans=2
depths= (2, 2, 2, 2)
num_heads = (3, 3, 3, 3)
#how much the window should be rolled
shift_sizes= ((2,2,2),(0,0,0),(2,2,2),(0,0,0)  )
#how much relative to original resolution (after embedding) should be reduced
downsamples=(False,True,True,True)



patch_size = (4,4,4)
window_size = (4,4,4) # in my definition it is number of patches it holds
img_size = (1,2,64,64,32)

def focal_loss(inputs, targets):
    """
    based on https://www.kaggle.com/code/bigironsphere/loss-function-library-keras-pytorch/notebook
    """
    alpha = 0.8
    gamma = 2        
    #comment out if your model contains a sigmoid or equivalent activation layer
    # inputs = jax.nn.sigmoid(inputs)       
    inputs=jnp.ravel(inputs)
    targets=jnp.ravel(targets)
    #first compute binary cross-entropy 
    BCE = optax.softmax_cross_entropy(inputs, targets)
    BCE_EXP = jnp.exp(-BCE)
    focal_loss = alpha * (1-BCE_EXP)**gamma * BCE                    
    return focal_loss

def dice_metr(y_pred,y_true):
    """
    Computes the Dice coefficient, a measure of set similarity.
    Parameters
    ----------
    im1 : array-like, bool
        Any array of arbitrary size. If not boolean, will be converted.
    im2 : array-like, bool
        Any other array of identical size. If not boolean, will be converted.
    Returns
    -------
    dice : float
        Dice coefficient as a float on range [0,1].
        Maximum similarity = 1
        No similarity = 0
        Both are empty (sum eq to zero) = empty_score
        
    Notes
    -----
    The order of inputs for `dice` is irrelevant. The result will be
    identical if `im1` and `im2` are switched.
    """
    empty_score=1.0
    inputs = jax.nn.sigmoid(y_pred)
    inputs =inputs >= 0.5   
    im1 = inputs.astype(np.bool)
    im2 = y_true.astype(np.bool)
    im_sum = im1.sum() + im2.sum()
    if im_sum == 0:
        return empty_score

    # Compute Dice coefficient
    intersection = jnp.logical_and(im1, im2)

    return 2. * intersection.sum() / im_sum


jax_swin= my_jax_3d_regr.SwinTransformer(img_size=img_size
                ,in_chans=in_chans
                ,embed_dim=feature_size
                ,window_size=window_size
                ,patch_size=patch_size
                ,depths=depths
                ,num_heads=num_heads
                ,shift_sizes=shift_sizes
                ,downsamples=downsamples                           
                )

total_steps=100

def create_train_state(learning_rate):
    """Creates initial `TrainState`."""
    input=jnp.ones(img_size)
    params = jax_swin.init(prng, input)['params'] # initialize parameters by passing a template image
    # bb= jax_swin.apply({'params': params},input,train=False)
    # print(f"jax shapee 0 {bb.shape} ")
    warmup_exponential_decay_scheduler = optax.warmup_exponential_decay_schedule(init_value=0.001, peak_value=0.0003,
                                                                                warmup_steps=int(total_steps*0.2),
                                                                                transition_steps=total_steps,
                                                                                decay_rate=0.8,
                                                                                transition_begin=int(total_steps*0.2),
                                                                                end_value=0.0001)    
    
    tx = optax.chain(
        optax.clip_by_global_norm(1.5),  # Clip gradients at norm 1.5
        optax.adamw(learning_rate=warmup_exponential_decay_scheduler)
    )


    return train_state.TrainState.create(
        apply_fn=jax_swin.apply, params=params, tx=tx)

state = create_train_state(0.0001)

# @nn.jit
def train_step(state, label,rot_lab,train):
  """Train for a single step."""
  def loss_fn(params):
    label_conc=jnp.concatenate((label,rot_lab),axis=1)
    logits = state.apply_fn({'params': params}, label_conc)
    #print(f"logits {logits.shape} ::: rot shape {jnp.array([rot]).shape}")
    loss = focal_loss(logits, label)
    # print(f"loss {loss} ")
    #loss= optax.l2_loss(logits, jnp.array([rot]))[0]
    return loss, logits

  grad_fn = jax.grad(loss_fn, has_aux=True)
  grads, logits = grad_fn(state.params)
  state = state.apply_gradients(grads=grads)
  f_l=focal_loss(logits, label)

  return state,f_l,logits

train=True
cached_subj=[]
training_loader = DataLoader(subjects_dataset, batch_size=1, num_workers=8)
rots = np.random.random(len(training_loader))
print(f"rots {np.min(rots)} {np.max(rots)} ")
rots_labs= []
for subject, rot in zip(training_loader,rots) :
    cached_subj.append(subject)
    label=subject['label'][tio.DATA].numpy()
    #print(f"label.shape {label.shape}  ")
    _,_,Nx,Ny,Nz= label.shape
    trans_mat_inv = jnp.linalg.inv(simpleTransforms.rotate_3d(rot,0.00,0.0)[0:3,0:3])
    rot_lab=simpleTransforms.apply_affine_rotation(label[0,0,:,:,:],trans_mat_inv,Nx,Ny,Nz)
    rots_labs.append(jnp.reshape(rot_lab,label.shape))

for epoch in range(1, total_steps):
    dicee=0
    f_ll=0
    for subject, rot_lab,rot in zip(cached_subj,rots_labs,rots ) :
        # image=subject['image'][tio.DATA].numpy()
        label=subject['label'][tio.DATA].numpy()
        # label=jnp.concatenate((label,rot_lab),axis=1)
        # print(f"label shape {label.shape}")
        # print(f"#### {jnp.sum(label)} ")
        state,f_l,logits=train_step(state, label,rot_lab,train)
        # dice=dice_metr(logits,label)
        # dicee=dicee+dice
        f_ll=f_ll+f_l
    print(f"epoch {epoch} f_l {f_ll/len(subjects_dataset)} ")
    # print(image.shape)


