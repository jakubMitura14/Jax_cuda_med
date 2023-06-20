import jax
import io
import base64
import time
from functools import partial
from typing import NamedTuple
import subprocess

import PIL
import numpy as np
import matplotlib.pylab as pl
"""
idea in general is to have a function that takes a point coordinates and output the value
so we have stored for each supervoxel a center, a voxel characteristic (for now just simple gaussian)
and set of vectors in diffrent directions basically in polar coordinates
next given a point coordinates in the image we can look at a location value for this point of each supervoxel
so we will take the voctor from the center of the analyzed supervoxel and look for the dot products of this vector with the stored vectors in the supervoxel publish_display_data
we than take either sum or max dot product as the score we than take into account wheather the query vecotr is shorter or longer than stored one
so if it is shorter we give high value if it is not small  
we will have a score for each supervoxel that will mark basically are the coordinates in the shape or not
next we will softmax or sth like that to exaggerate the influence of best fit 
lastly we will multiply by the output values of the haussians - hence all the supervoxels becouse of the low scores should not contribute significantly 
and the single one should

later when wavelet or some other we should also have the way to get the value in a particular spot without instantiating whole image 

note 
in order to avoid collapse of the shape to the point there should be additionall loss function that will maximize the distance between the points within supervoxel
"""
import jax
import jax.numpy as jnp
import io
import base64
import time
from functools import partial
from typing import NamedTuple
import subprocess
import optax
import PIL
import numpy as np
import matplotlib.pylab as pl
import einops 
import SimpleITK as sitk
import itertools
from flax import linen as nn
import functools
from typing import Any, Callable, Sequence, Optional
import jax
from jax import lax, random, numpy as jnp
from flax.core import freeze, unfreeze
from flax import linen as nn
import chex
import ml_collections
from jax_smi import initialise_tracking
from flax.training import train_state  
from flax import jax_utils  

cpus = jax.devices("cpu")
gpus = jax.devices("gpu")

def norm(v, axis=-1, keepdims=False, eps=0.0):
  return jnp.sqrt((v*v).sum(axis, keepdims=keepdims).clip(eps))

def normalize(v, axis=-1, eps=1e-20):
  return v/norm(v, axis, keepdims=True, eps=eps)
def standardize_to_sum_1(v, axis=-1, eps=1e-20):
  a=v-jnp.min(v)
  # print(f"a {a}")
  return a/jnp.sum(jnp.ravel(a))

def get_initial_points_around(r):
    """
    given center point will give set of points that will create a verticies of shape around this point
    the first try of this function is a cube
    center_point- jax array with 3 float entries
    r - a float that is controlling the initial shape size if it would be a sphere it would be a radius
    """
    res= list(set(itertools.combinations([-r,0.0,r,-r,0.0,r,-r,0.0,r],3)))
    return jnp.array( list(filter(lambda points: not (points[0]==0.0 and points[1]==0.0 and points[0]==0.0)  ,res))).astype(float)
     
def get_corrected_dist(query_point_normalized,around,eps,pow=3):
    """
    our goal is to get the information how distant in a query point direction it can be to be still within shape
    so first we need to asses the angle in respect to all of its verticies to know in which direction to look
    next we asses how far in this direction the shape goes from the center (which in this case is the center of coordinate system)
    query_point_normalized: the location of the query points given the shape center is the center of coordianate system
    around - set of points that are the verticies of the shape in the same coordinate system as above
    eps - added for numerical stability
    pow - need to be odd number  the bigger the more exact the calculation the smaller the more stable numerically and simpler to differentiate
    """
    #vmap cosine similarity
    cosine_similarity_multi=jax.vmap(partial(optax.cosine_similarity, epsilon=eps),(0,None))
    angle_similarity= jax.nn.softmax(jnp.power(((cosine_similarity_multi(around,query_point_normalized))+1),pow))
    dists = jnp.sqrt(jnp.sum(jnp.power(around,2),axis=1))
    print(f"query_point_normalized {query_point_normalized.shape} dists {dists.shape} angle_similarity {angle_similarity.shape} around {around.shape}")

    corrected_dist= jnp.sum(dists*angle_similarity)#try also max
    return corrected_dist

def soft_less(max_cube,curr,to_pow=90):
    """
    soft version of less operator if max cube is bigger than curr we will get number close to 1
    if curr is bigger than max_cube we will gt close to 0
    to_pow - the bigger the more exact but less numerically stable solution we get
    """
    diff_a =  jnp.power(to_pow,(max_cube-curr))# if it is positive we are inside if negative outside
    diff_b =  jnp.power(to_pow,(curr-max_cube))# if it is positive we are outside if negative inside
    return (diff_a/(diff_a+diff_b))

def get_value_for_point(param_s_vox, value_param,query_point,cfg ):
    """
    given a point coordinates will return the value on the basis of current supervoxel data that is stored in param_s_vox
    param_s_vox - contain supervoxel parameters
        0-3) supervoxel center
        4-end verticies points (3 float each)
    value_param - parameters required to compute value for points inside the shape
    query_point - point that we are analizing wheather is in the given supervoxel or not
    """
    super_voxel_center=param_s_vox[0,:]
    #coordinates of verticies
    around= param_s_vox[1:,:]
    # around= einops.rearrange(around,'(v c)->v c',c=3)
    #normalizing so we are acting as if the center of supervoxel is the center of coordinate system
    query_point_normalized=jnp.subtract(query_point,super_voxel_center)
    #getting the distance from the center of the shape to ith border in the direction of query point
    corrected_dist=get_corrected_dist(query_point_normalized,around,cfg.eps,pow=cfg.pow)
    # distance from center of the shepe to query point
    # print(f"query_point_normalizedaaa {query_point_normalized} ")
    dist_query= jnp.sqrt(jnp.sum(jnp.power(query_point_normalized,2)))
    #closer to 1 if query point is in shape closer to 0 if outside
    soft_is_in_shape=soft_less(corrected_dist,dist_query,to_pow=cfg.to_pow)
    res= value_param*soft_is_in_shape
    return res

def get_in_shape_to_compute(indicies,cfg):
    """
    We need to calculate the computational plan - we get the devices number 
            - which need to define our first axis by jax.local_device_count() 
    we supply the max per gpu num - so we will have the xmap correctly established
    so we have 
    pmap - fisrs axis = number of devices
    scan - just to match the data and other constraints
    vmap - with number equal or smaller to supplied max per device number     
    """
    max_per_device=cfg.max_per_device
    start_index=cfg.start_index
    index_increment=cfg.index_increment

    padding_value=-5
    padding=((padding_value, padding_value),(padding_value,padding_value))

    n_gpu=jax.local_device_count() 
    n_points= indicies.shape[0]
    gpu_times_max_per_gpu=n_gpu*max_per_device
    remainderr= np.remainder(n_points,n_gpu)
    remainder_big= np.remainder(n_points,gpu_times_max_per_gpu)
    # in case we do not need to scan just pmap and vmap
    if(gpu_times_max_per_gpu>=n_points):
        #now we need to check is it divisible by n_gpu if yes nothing need to be done if not 
        #we need to pad it
        if(remainderr!=0):
            indicies= jnp.pad(indicies, ((0, remainderr), (0, 0)), 'constant', constant_values=padding)
        return einops.rearrange(indicies,'(g p) c->g 1 p c',g=n_gpu)
    #now for case we need scanning we need to calculate the optimal padding plan 
    else:
        #first maybe we are lucky and all is divisible
        if(remainder_big==0):
            return einops.rearrange(indicies,'(g s p) c->g s p c',g=n_gpu,p=max_per_device)
        #if padding is needed we need to make an effort and calculate it
        else:
            #now we will first go to the closest number divisible by n_gpu up from n_points from max_per_device
            n_points=n_points+remainderr
            max_per_device=max_per_device+np.remainder(max_per_device,n_gpu)
            max_per_device=max_per_device*n_gpu
            #now we will go with stepping up in n_point and stepping down from max_per_device and check for
            #for a way to get the divisible outcome with as small steps as possible
            index=start_index
            old_index=0
            while(True):
                # for the case that we can not find the solution we leave possibility to increase search space
                index=index+index_increment
                arranges=jnp.arange(old_index,index,n_gpu)
                old_index=index
                lenn=len(arranges)
                n_points_arr=arranges+n_points
                max_per_device_arr=max_per_device-arranges
                ind_matrix=jnp.repeat(jnp.reshape(n_points_arr,(1,lenn)),lenn,axis=0 )
                scan_matrix=jnp.transpose(jnp.repeat(jnp.reshape(max_per_device_arr,(1,lenn)),lenn ,axis=0))

                # we get it into a matrix get a remainders 
                remainder_matrix= jnp.remainder(ind_matrix,scan_matrix)
                # #we look for entry with zeros and return its indicies
                remm=jnp.asarray(remainder_matrix==0).nonzero()
                zero_ind_array=jnp.stack(remm,axis=1)
                if(zero_ind_array.shape[0]>0):
                    #now we look for the entry with smallest number of padding steps that meets our criteria
                    min_index=jnp.argmin(jnp.sum(zero_ind_array,axis=1))
                    #on the basis of calculated values we add padding to indicies and set scanning axis correctly
                    to_pad=n_points_arr[zero_ind_array[min_index,1]]
                    scan_axis_num=max_per_device_arr[zero_ind_array[min_index,0]]
                    # to_pad=n_points_arr[zero_ind_array[min_index,1]]
                    # scan_axis_num=max_per_device_arr[zero_ind_array[min_index,0]]
                    divv=to_pad/scan_axis_num
                    # print(f"divv {divv} zero_ind_array {zero_ind_array[min_index,:]}  to_pad {to_pad}  scan_axis_num {scan_axis_num} multi {scan_axis_num*n_gpu}")
                    indicies= jnp.pad(indicies, ((0,to_pad-n_points), (0, 0)), 'constant', constant_values=padding)
                    # print(f"scan_axis_num {scan_axis_num} indicies {indicies.shape} n_gpu {n_gpu}")
                    # print(f"indicies {indicies.shape[0]}  calculated {indicies.shape[0]/(scan_axis_num)}  ")
                    # return einops.rearrange(indicies,'(g p) c->g s p c',g=n_gpu,s=scan_axis_num)
                    return einops.rearrange(indicies,'(g s p) c->g s p c',g=n_gpu,p=int(scan_axis_num/n_gpu))

def get_initial_supervoxel_shape_param(shapee,r):
    """
    return the initial centers and relative verticies of supervoxels 
    verticies has location relative to its center
    shape is (v p c) - v - super voxel p - point c - coordinate of the point
    """
    verticies=get_initial_points_around(r)
    xx,yy,zz=list(map(lambda index :jnp.arange(r,shapee[index]-r,r),[0,1,2]))
    centers=list(map(lambda x:list(map( lambda y: list(map(lambda z: jnp.array([x,y,z])    
                        ,zz))
                    ,yy))            
                ,xx ))
    centers= list(itertools.chain(*centers))
    centers= list(itertools.chain(*centers))
    centers=jnp.array(centers)
    centers=einops.rearrange(centers,'v c-> v 1 c')
    # return centers
    verticies_multi=einops.repeat(verticies,'p c->v p c',v=len(centers) )
    shape_param=jnp.concatenate([centers,verticies_multi], axis=1)
    return shape_param

class super_voxels_analyze(nn.Module):
    cfg: ml_collections.config_dict.config_dict.ConfigDict

    def get_shape_params(self,rng,dummy):
        return get_initial_supervoxel_shape_param(self.cfg.input_shape,self.cfg.r)
    def get_value_params(self,rng,super_voxel_num):
        return jnp.arange(super_voxel_num).astype(float)
    
    def for_scan(self,query_point, param_pair):
       shape_param_singe,value_param_single = param_pair
       #we need to check wheather we have positive values negative values indicate that it is just a padding
       res=get_value_for_point(shape_param_singe, value_param_single,query_point,self.cfg)
       return query_point,res#res

    def to_call(self, query_point):
        shape_param = self.param('shape_param_single_s_vox',
                            self.get_shape_params,())
        value_param = self.param('value_param_single_s_vox',
                            self.get_value_params,(shape_param.shape[0]))
        
        scanned_b = lax.scan(self.for_scan, query_point, (shape_param, value_param))[1]
        return jnp.sum(jnp.ravel(scanned_b))
    # def to_call_when_none(self, query_point):
    #     return jnp.array([0.0])


    @nn.compact
    def __call__(self,query_point):
        # print(f"query_pointtttt {query_point}")
        #we need to verify that we do not deal with a padding points
        # return lax.select(query_point[0]>-1, self.to_call(query_point), self.to_call_when_none(query_point))
        res= lax.select(query_point[0]>-1, self.to_call(query_point), 0.0)

        return res

"""
first function transformation we vmap over all points given to this point 
the maximum number of points is controlled via hyperparameter and is controlling
gpu memory usage
"""
vmapped_super_vox_model = nn.vmap(
    super_voxels_analyze,
    in_axes=0, out_axes=0,
    variable_axes={'params': None},
    split_rngs={'params': True})


"""
second transformation needed if we will not be able to squeeze all of the points computations
on a GPU memory we will scan over batches and each batch will be vmapped as shown above
"""
def get_vmapped_scanned_super_vox_model(cfg):

    return nn.vmap(
    vmapped_super_vox_model,
    in_axes=0, out_axes=0,
    variable_axes={'params': None},
    split_rngs={'params': True})

    # return nn.remat_scan(vmapped_super_vox_model
    #                      ,lengths=(1,cfg.final_shape[0])
    #                      , variable_broadcast=False
    #                     #  ,variable_broadcast={'params': None}
    #                      )


    # return nn.scan(
    #             vmapped_super_vox_model,
    #             in_axes=0, out_axes=0
    #             ,length=cfg.final_shape[0]
    #             # ,variable_broadcast={'params': 0}
    #             ,variable_axes={'params': 0}
    #             ,split_rngs={'params': True}
    #             )



@functools.partial(jax.pmap,static_broadcasted_argnums=1, axis_name='ensemble')#,static_broadcasted_argnums=(2)
def create_train_state(rng,cfg:ml_collections.config_dict.FrozenConfigDict):
    """
    create a training state
    """
    model = get_vmapped_scanned_super_vox_model(cfg)(cfg)
    # dummy_image=jnp.ones((cfg.final_shape[1],cfg.final_shape[2],cfg.final_shape[3]))
    dummy_image=jnp.ones((cfg.final_shape))
    params = model.init(rng, dummy_image)['params']
    param_count = sum(x.size for x in jax.tree_leaves(params))
    print(f"parameter num {param_count}")
    # dummy_image=jnp.ones((cfg.final_shape[1],cfg.final_shape[2],cfg.final_shape[3]))
    # print(f"dummy_image {dummy_image.shape}")
    # model.init(rng,dummy_image )#['params']
    # print(f" got paramse in train state")
    tx = optax.chain(
    optax.clip_by_global_norm(1.5),  # Clip gradients at norm 1.5
    optax.adamw(learning_rate=0.00000001))
    # tx = optax.adamw(learning_rate=0.00001)
    return train_state.TrainState.create(
        apply_fn=model.apply, params=params, tx=tx)


@functools.partial(jax.pmap, axis_name='ensemble',static_broadcasted_argnums=0)#,static_broadcasted_argnums=(2)
def apply_model(cfg,state, indicies,orig_image):
    """
    get a loss and return both output and gradients required for model update
    """
    def loss_fn(params):
        values = get_vmapped_scanned_super_vox_model(cfg)(cfg).apply({'params': params}, indicies)
        orig_flat=jnp.ravel(orig_image)

        loss = optax.l2_loss(jax.nn.sigmoid(jnp.ravel(values)[0:orig_flat.shape[0]]),orig_flat).mean()
        return loss, values

    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    (loss, values), grads = grad_fn(state.params)

    return grads, loss, values

@jax.pmap
def update_model(state, grads):
  """
  on the basis of calculated gradients and using given optimizer 
  will update parameters
  """
  return state.apply_gradients(grads=grads)

def train_epoch(state, indicies,orig_image, rng):
    #replicate needed for multi gpu training
    indicies_repl = jax_utils.replicate(indicies)
    orig_image_repl = jax_utils.replicate(orig_image)

    grads, loss, values = apply_model(cfg,state, indicies_repl, orig_image_repl)
    state = update_model(state, grads)
    #unreplicating needed for multi gpu training
    epoch_loss=jax_utils.unreplicate(loss)
    values=jax_utils.unreplicate(values)

    return state, epoch_loss, values




from jax.config import config
# config.update("jax_disable_jit", True)
initialise_tracking()

cfg = ml_collections.config_dict.ConfigDict()
cfg.r=4.5# size of super voxel
cfg.eps=1e-20 #for numerical stability
cfg.to_pow=110# rendering hyperparameter controling exactness and differentiability
cfg.pow=5# rendering hyperparameter controling exactness and differentiability
cfg.input_shape=(40,40,40)
cfg.max_per_device= 100000#1000000 # indicates how many points at once can be vmapped to be processed in the GPU to avoid out of memory errors
#two variables controlling the search space of optimal padding strategy the first number 
# is controlling how much gpu memory will be used, second haw fast it will jump (the faster the less optimal the padding)
cfg.start_index=50000
cfg.index_increment=1000


# value_params=jnp.arange(len(supervoxel_centers))
image = jnp.ones(cfg.input_shape)
indicies= einops.rearrange(jnp.indices(image.shape),'c x y z->(x y z) c')
orig_indicies_len=indicies.shape[0]

indicies=get_in_shape_to_compute(indicies,cfg)
indicies= einops.rearrange(indicies,'g s p c->(g s) p c')

print(f"get compute shape {indicies.shape}")
# super_vox_model=super_voxels_analyze(cfg)
rng = jax.random.PRNGKey(42)
rng,rng_new = jax.random.split(rng)
# rng_a,rng_b,rng_c=jax.random.split(rng,split=3)
rng_2=jax.random.split(rng,num=jax.local_device_count() )
cfg.final_shape=indicies.shape


cfg = ml_collections.config_dict.FrozenConfigDict(cfg)

state=create_train_state(rng_2,cfg)

num_epochs=5
for epoch in range(1, num_epochs + 1):
    state, epoch_loss, values=train_epoch(state, indicies,image, rng_new)
    print(f"epoch_loss {epoch_loss}")



# https://flax.readthedocs.io/en/latest/guides/ensembling.html

# aa=vmapped_super_vox_model.init(jax.random.PRNGKey(0),indicies)#query_point
# # aa= jax.jit(aa)
# tic_loop = time.perf_counter()
# # print(f"aaaa {aa}")
# zz= vmapped_super_vox_model.apply(aa,indicies)
# x = random.uniform(random.PRNGKey(0), (1000, 1000))
# jnp.dot(x, x).block_until_ready() 
# toc_loop = time.perf_counter()
# print(f"loop {toc_loop - tic_loop:0.4f} seconds")


# # print(f"zzz {zz[0]}")
# image_res = jnp.reshape(zz[0],cfg.input_shape)
# image_res = sitk.GetImageFromArray(image_res)
# sitk.WriteImage(image_res,"/workspaces/Jax_cuda_med/data/explore/cube.nii.gz")







# layer.init(query_point)

# mono_super_voxel_stack = nn.scan(
#     mono_super_voxel,
#     variable_axes={
#         "params": 0
#         },
#     split_rngs={
#         "params": True,
#     },
#     length=supervoxel_centers.shape[0],
#     # in_axes=(None,0,0)
#     in_axes=(nn.broadcast,)
#     ,out_axes=0
# )(cfg,supervoxel_centers,value_params) #

# aaaa=mono_super_voxel_stack(query_point)

# print(f"aaa {type(mono_super_voxel_stack)}")
# supervoxel_params=get_initial_supervoxel_shape_param(cfg.input_shape,cfg.r)


# layer=render_super_voxels_mono_point(cfg,supervoxel_centers,value_params)
# layer.init(jax.random.PRNGKey(0),query_point)#query_point
# layer.init(query_point)

# render_super_voxels(cfg,supervoxel_centers,value_params)


print("finish")





# class render_super_voxels(nn.Module):
#     cfg: ml_collections.config_dict.config_dict.ConfigDict
#     supervoxel_centers: chex.Array

#     def setup(self):
#         pass
#         #frst we scan over supervoxels - vmapping is too memory costly
#         # self.mono_super_voxels= nn.scan(mono_super_voxel,
#         #     in_axes=(None,0,1), out_axes=0,
#         #     variable_axes={'params': 0},#indicates we do not share parameters
#         #     # variable_broadcast={'params': None}, params None will mean parameter sharing
#         #     split_rngs={'params': True}
#         #     ,length= (len(self.supervoxel_centers))(cfg,self.supervoxel_centers,self.value_params))
#         # #then we vmap a scan module over all of the indicies - so all of the image
#         # self.mono_super_voxels= nn.vmap(
#         #     self.mono_super_voxels,
#         #     in_axes=0, out_axes=0,
#         #     variable_axes={'params': None},# we share hyperparameters - here parameters are set of supervoxel layers with their parameters
#         #     split_rngs={'params': False})
        
#     @nn.compact
#     def __call__(self, indicies):
#       pass
#       # self.mono_super_voxels(indicies)

# class render_super_voxels_mono_point(nn.Module):
#     cfg: ml_collections.config_dict.config_dict.ConfigDict
#     supervoxel_centers: chex.Array
#     value_params:chex.Array
#     def setup(self):
#       self.mono_super_voxels= list(map(lambda index :mono_super_voxel(cfg,self.supervoxel_centers[index,:],jnp.array([self.value_params[index]]))  , range(self.supervoxel_centers.shape[0])))
#       self.mono_super_voxels= jnp.array(self.mono_super_voxels)
#       # self.mono_super_voxels= nn.scan(mono_super_voxel,
#       #     # in_axes=(None,0,1)
#       #     in_axes=0
#       #     ,out_axes=0,
#       #     variable_axes={'params': 0},#indicates we do not share parameters
#       #     # variable_broadcast={'params': None}, params None will mean parameter sharing
#       #     split_rngs={'params': False}
#       #     ,length= (self.supervoxel_centers.shape[0]))(self.cfg,supervoxel_centers,self.value_params)

#     def for_scan(carry_tuple, mono_super_v_module):
#        point,carry=carry_tuple
#        carry_new= mono_super_v_module(point)+carry
#        return carry_new,point
             
#     @nn.compact
#     def __call__(self, query_index):
#       return jax.lax.scan(self.for_scan, (query_index,0) ,self.mono_super_voxels)
#       # return self.mono_super_voxels(query_index)
#       # self.mono_super_voxels(indicies)

# class mono_super_voxel(nn.Module):
#     cfg: ml_collections.config_dict.config_dict.ConfigDict
#     super_voxel_center: chex.Array=jnp.array([1.0,1.0,1.0])
#     value_param_single_s_vox_given: chex.Array=jnp.array([1.0])

#     def get_shape_param_single_s_vox(self,rng,dummy):
#         verticies=get_initial_points_around(self.cfg.r)
#         return jnp.concatenate([self.super_voxel_center,verticies])
#     def get_value_param_single_s_vox(self,rng,dummy):
#         return self.value_param_single_s_vox_given  
  
#     @nn.compact
#     def __call__(self, query_point):
#         shape_param_single_s_vox = self.param('shape_param_single_s_vox',
#                             self.get_shape_param_single_s_vox,())
#         value_param_single_s_vox = self.param('value_param_single_s_vox',
#                             self.get_value_param_single_s_vox,())
#         res= get_value_for_point(shape_param_single_s_vox, value_param_single_s_vox,query_point,self.cfg)        

#         return res,res

