#using xmap in a model https://github.com/venkat4938/flaxformer/blob/87cd1f420e6f686a595facab3a1328a4e16d53ff/flaxformer/architectures/perceiver_ar/slicing.py
# 3d mnist https://www.kaggle.com/datasets/daavoo/3d-mnist
import os
import jax
os.environ["XLA_FLAGS"] = '--xla_force_host_platform_device_count=8' # Use 8 CPU devices
import jax.numpy as jnp
from jax import lax
from jax.nn import one_hot, relu
from jax.scipy.special import logsumexp
from jax.experimental.maps import Mesh
import jax
import numpy as np
from typing import Any, Callable
import flax.linen as nn
from absl import logging
from flax import linen as nn
from flax.metrics import tensorboard
from flax.training import train_state
import jax.numpy as jnp
import ml_collections
import numpy as np
import optax
import tensorflow_datasets as tfds
from jax.experimental.maps import xmap
import optax  # https://github.com/deepmind/optax
import jax
import itertools
from itertools import starmap
import h5py
import functools
import toolz
# jax good for i loop https://github.com/google/jax/discussions/8706
#https://jax.readthedocs.io/en/latest/_autosummary/jax.nn.initializers.glorot_normal.html#jax.nn.initializers.glorot_normal




def get_weight_size(dictt):
  """
  utility function to get the correct size of weights
  """
  return (dictt["out_channels"], dictt["in_channels"]) + dictt["kernel_size"]

def initialize_conv(convSpecs,conv_prngs):
  """
  initialize weights for convolutions
  """
  conv_sizes= list(map(get_weight_size ,convSpecs))
  return list(starmap(initializer,zip(conv_prngs,conv_sizes) ))


def apply_conv(img,index, conv_params,convSpecs  ):
  """
  executes convolutions using given specifications and parameters taken from prepared lists
  """
  return jax.nn.gelu(jax.lax.conv_general_dilated(img, conv_params[index],convSpecs[index]['stride'], 'SAME'))


convSpecs=[{'in_channels':3,'out_channels':4, 'kernel_size':(3,3,3),'stride':(2,2,2) }
,{'in_channels':4,'out_channels':4, 'kernel_size':(3,3,3),'stride':(2,2,2) }
,{'in_channels':4,'out_channels':8, 'kernel_size':(3,3,3),'stride':(1,1,1) }
,{'in_channels':8,'out_channels':16, 'kernel_size':(3,3,3),'stride':(2,2,2) }]





def init_single_dense(dictt):
  return (dictt['in'],dictt['out'])

def init_dense(denseSpecs,dense_prngs):
  """
  on the basis of the specifications in provided dictionary initializes parameters for dense layers
  """
  return list(starmap(initializer,zip(conv_prngs,list(map(init_single_dense ,denseSpecs))) ))

def apply_dense(img,index, dense_params):
  """
  applies the dense layer 
  """
  return jax.nn.gelu(jnp.dot(img,dense_params[index] ))

def predict(img,params):
  """
  combine layers into prediction function
  """
  img= jnp.expand_dims(img,0)
  conv_params,dense_params=params
  c1=apply_conv(img,0, conv_params,convSpecs  )
  c2=apply_conv(c1,1, conv_params,convSpecs  )
  c3=apply_conv(c2,2, conv_params,convSpecs  )
  c4=apply_conv(c3,3, conv_params,convSpecs  )
  d1=apply_dense(jnp.ravel(c4),0, dense_params)
  return apply_dense(d1,1, dense_params)



def loss(params, batchedImages, targets):
  preds = batched_predict(batchedImages,params)
  return jnp.mean(optax.softmax_cross_entropy_with_integer_labels(preds,targets))



#setting up the initializer for convolution weights
initializer = jax.nn.initializers.glorot_normal()#xavier initialization
prng = jax.random.PRNGKey(42)
#defining how the convolutions will be structured

imageDims = (3,16, 16,16)
#calculating the total size after all strided convolutions
strides = list(map(lambda dictt:dictt['stride'] , convSpecs))
stride_total = list(functools.reduce(lambda tuplA, tuplB: (tuplA[0]*tuplB[0],tuplA[1]*tuplB[1],tuplA[2]*tuplB[2] ) , strides))
imageDims_reduced = (toolz.last(convSpecs)['out_channels'],int(imageDims[1]/stride_total[0] ),int(imageDims[2]/stride_total[1] ),int(imageDims[3]/stride_total[2] )  )
denseSpecs=[{'in':np.product(imageDims_reduced),'out':1000},{'in' :1000 , 'out':10}]

#### initialize parameters
conv_prngs= jax.random.split(prng, len(convSpecs))
dense_prngs=jax.random.split(conv_prngs[0], len(denseSpecs))
conv_params= initialize_conv(convSpecs,conv_prngs)
dense_params=init_dense(denseSpecs,dense_prngs)
params=[conv_params,dense_params]

#vmap for batched prediction
batched_predict=jax.vmap(predict,(0,None))
#defining optimizer and its state
optimizer = optax.adam(learning_rate=1e-2)
opt_state = optimizer.init(params)


def load3dMnist():
  ### loading data 
  with h5py.File('/workspaces/forPicaiDocker/jax/3dData/full_dataset_vectors.h5', 'r') as dataset:
      X_train = dataset['X_train'][:]
      X_test = dataset['X_test'][:]
      y_train = dataset['y_train'][:]
      y_test = dataset['y_test'][:]


  config = {
      'min_learning_rate': 1e-6,
      'n_classes': 10,
      'batch_size': 128,
      'width': 16,
      'height': 16,
      'depth': 16,
      'channels': 3,
  }

  ## add the channel dimension to the input dataset
  X_train_shaped = np.ndarray((X_train.shape[0], 4096, config['channels']))
  X_test_shaped = np.ndarray((X_test.shape[0], 4096, config['channels']))

  input_shape = (config['width'], config['height'], config['depth'], config['channels'])
  X_train = np.reshape(X_train_shaped, (X_train_shaped.shape[0], *input_shape))
  X_test = np.reshape(X_test_shaped, (X_test_shaped.shape[0], *input_shape))

  X_train= jnp.moveaxis(X_train,4,1)
  X_test= jnp.moveaxis(X_test,4,1)
  return np.array_split(X_train, 100), np.array_split(X_test, 100), np.array_split(y_train, 100), np.array_split(y_test, 100)

X_train, X_test, y_train, y_test=load3dMnist()


@jax.jit
def step(params, opt_state, batch, labels):
  loss_value, grads = jax.value_and_grad(loss)(params, batch, labels)
  updates, opt_state = optimizer.update(grads, opt_state, params)
  params = optax.apply_updates(params, updates)
  return params, opt_state, loss_value


for epoch in range(1,6):
  for i, (batch, labels) in enumerate(zip(X_train, y_train)):
    params, opt_state, loss_value = step(params, opt_state, batch, labels)
    if i % 100 == 0:
      print(f'epoch {epoch}, loss: {loss_value}')




# X_train[0].shape
# y_train.shape

# batchedImages= jnp.zeros((5,)+imageDims)
# params, opt_state, loss_value=loss(params, batchedImages, jnp.array([0,1,2,3,4]))




# step(params, opt_state, batchedImages, jnp.array([0,1,2,3,4]))

# def loss(params, batch):
#   inputs, targets = batch
#   preds = predict(params, inputs)
#   return -jnp.mean(jnp.sum(preds * targets, axis=1))


# d2.shape
# print(c4.shape)
# imageDims_reduced


# optax.softmax_cross_entropy_with_integer_labels(jnp.zeros((3,10)),jnp.array([0,2,3]))



# optax https://optax.readthedocs.io/en/latest/optax-101.html
# NUM_TRAIN_STEPS = 10



# optax.softmax_cross_entropy(logits,labels)

# c4.shape


# ress.shape



# conv_params[0].shape

# conv_num=6
# copied_channels= jax.lax.fori_loop(1, conv_num, copy_concat, img)


# out_channels=5
# grouped_in_channels=2
# kernel_size=(3,3,3)
# img=jnp.zeros((1,2,64, 96,32))
# ww=(out_channels, grouped_in_channels) + kernel_size
# params=initializer(prng,ww)

# ress= jax.lax.conv_general_dilated(img, params,(2,2,2), 'SAME')
# ress.shape



# img=jnp.zeros((1,3,64, 96,32))













# in_channels=5
# out_channels=10
# conv3d=eqx.nn.Conv3d(in_channels=in_channels, out_channels=out_channels, kernel_size= (3,3,3), padding=1,key=prng)
# xxx=conv3d(jnp.zeros((5,64, 96,32)))

# #idea of multichannel convolutions first map a copy function to have necessary amount and then apply vmap
# # #or lower memory fori loop and immidiately concatenate 

# xxx.shape
# def do_conv_concat(i,val):
#   param,stride=args
#   return jax.lax.conv_general_dilated(img, param,stride, 'SAME')



# jax.lax.fori_loop()

# cc=do_conv(zipped[0])
# cc.shape
# jax.lax.map(do_conv,zipped)



# dstack(tup[, dtype])
# #random states for xavier initializations
# conv_prngs= jax.random.split(prng, len(conv_strides))
# conv_params= list(starmap(initializer,zip(conv_prngs,conv_sizes) ))

# img=jnp.zeros((1,1,64, 96,32))
# index=1
# kernel=conv_params[index]
# stride=conv_strides[index]

# convolved1 = jax.lax.conv_general_dilated(img, conv_params[0],conv_strides[0], 'SAME')
# convolved2= jax.lax.conv_general_dilated(convolved1, conv_params[1],conv_strides[1], 'SAME')
# convolved3 = jax.lax.conv_general_dilated(convolved2, conv_params[2],conv_strides[2], 'SAME')


# convolved = jax.lax.conv_general_dilated(jnp.zeros((1,4,64, 96,32)), jnp.zeros((1,4,3,3,3)),(2,2,2), 'SAME')
# convolved.shape



# convolved.shape#(1, 1, 32, 48, 16)
# deconvolved=jax.lax.conv_transpose(convolved,jnp.zeros((1,1,3,3,3)),(2,2,2), 'SAME')  
# convolved3.shape


# deconvolved2=jax.lax.conv_transpose(deconvolved3,jnp.zeros((1,1,2,2,2)),(2,2,2), 'SAME',transpose_kernel=True)  

# deconvolved3.shape
# convolved3.shape


# jax.lax.conv_transpose()

# w1 = jnp.zeros((784, 512))
# w2 = jnp.zeros((512, 10))
# images = jnp.zeros((128, 784))
# labels = jnp.zeros(128, dtype=jnp.int32)



# img = jnp.zeros((5,3, 66, 66, 66), dtype=jnp.float32)
# kernel = jnp.zeros((1,3,3,3 ,3), dtype=jnp.float32)

# convolved = jax.lax.conv_general_dilated(img, kernel, (2,2, 2), 'SAME')
# convolved.shape


# """
# get list of parametrarized kernels
# """



# """

# """
# def named_predict(w1, w2, image):
#   hidden = relu(lax.pdot(image, w1, 'inputs'))
#   logits = lax.pdot(hidden, w2, 'hidden')
#   return logits - logsumexp(logits, 'classes')

# def named_loss(w12, images, labels,linMod):
#     w1,w2=w12
#     predictions = named_predict(w1, w2, images)
#     num_classes = lax.psum(1, 'classes')
#     bb=lax.psum(jnp.ravel(w1),'inputs')

#     #   print(f"num_classes {num_classes}")
#     targets = one_hot(labels, num_classes, axis='classes')
#     losses = lax.psum(targets * predictions, 'classes')
#     return -lax.pmean(losses,('batch'))#+bb


# in_axes = [[{0:'inputs', 1:'hidden'},
#            {0:'hidden', 1:'classes'}],
#            {0:'batch', 1:'inputs'},
#            {0:'batch'},[...]]

# loss = xmap(named_loss, in_axes=in_axes, out_axes=[...])
# print(loss([w1, w2], images, labels))#should be 2.3


# # bb=lax.concatenate(w1,0)



# # x = jax.numpy.zeros((batch_size, in_size))
# # y = jax.numpy.zeros((batch_size, out_size))
# # grads = loss_fn(model, x, y)


# # # devices = np.array(jax.local_devices())
# # # with Mesh(devices, ('x',)):
# # #   print(loss(w1, w2, images, labels))
# # bb=lax.concatenate(jnp.zeros((5, 5, 5, 5,5), dtype=jnp.float32),'inputs')
# # bb.shape

# # img = jnp.zeros((5,3, 66, 66, 66), dtype=jnp.float32)
# # kernel = jnp.zeros((1,3,3,3 ,3), dtype=jnp.float32)

# # convolved = jax.lax.conv_general_dilated(img, kernel, (2,2, 2), 'SAME')
# # convolved.shape

# # from jax import lax
# # import jax.numpy as jnp
# # img = jnp.zeros((1, 200, 198, 3), dtype=jnp.float32)
# # for k in range(3):
# #   x = 30 + 60*k
# #   y = 20 + 60*k
# #   img = img.at[0, x:x+10, y:y+10, k].set(1.0)
# # kernel = jnp.zeros((3, 3, 3, 3,3), dtype=jnp.float32)
# # kernel += jnp.array([ [[1, 1, 0],
# #                      [1, 0,-1],
# #                      [0,-1,-1]],[[1, 1, 0],
# #                      [1, 0,-1],
# #                      [0,-1,-1]] ,[[1, 1, 0],
# #                      [1, 0,-1],
# #                      [0,-1,-1]]   ] )[:, :, jnp.newaxis, jnp.newaxis]

# # out = lax.conv(img,    # lhs = NCHW image tensor
# #                kernel, # rhs = OIHW conv kernel tensor
# #                (1, 1,1),  # window strides
# #                'SAME') # padding mode
# # print("out shape: ", out.shape)


# # # out = lax.conv_with_general_padding(
# # #   jnp.transpose(img,[0,3,1,2]),    # lhs = NCHW image tensor
# # #   jnp.transpose(kernel,[2,3,0,1]), # rhs = IOHW conv kernel tensor
# # #   (1, 1),  # window strides
# # #   ((2,2),(2,2)), # general padding 2x2
# # #   (1,1),  # lhs/image dilation
# # #   (1,1))  # rhs/kernel dilation



# # # from jax.experimental.maps import xmap
# # # from typing import Any, Callable
# # # import jax.numpy as jnp
# # # from jax import lax
# # # from jax.nn import one_hot, relu
# # # from jax.scipy.special import logsumexp
# # # from jax.experimental.maps import Mesh
# # # import jax
# # # import numpy as np
# # # from typing import Any, Callable
# # # class ArrayType:
# # #   def __getitem__(self, idx):
# # #     return Any
# # # f32 = ArrayType()
# # # i32 = ArrayType()

# # # def my_func(x: f32[{'main':5,'batch': 20}]) -> f32[{'main':5,'batch': 20}]:
# # # #   assert x.shape == (5,)
# # #   # assert x.named_shape == {'batch': 20}  # TODO: Implement named_shape
# # #   return x

# # # x: f32[(20, 5)] = jnp.zeros((20, 5), dtype=np.float32)
# # # f = xmap(my_func,
# # #          in_axes={0: 'batch'},   # Name the first axis of the only argument 'batch'
# # #          out_axes={1: 'batch'})  # Place the 'batch' named axis of the output as the second positional axis
# # # y: f32[(5, 20)] = f(x)
# # # (y == x.T).all()


# # # assert (y == x.T).all()  # The first dimension was removed from x and then re-inserted as the last dim




# # # class CNN(nn.Module):
# # #   """A simple CNN model."""

# # #   @nn.compact
# # #   def __call__(self, x):
# # #     x = nn.Dense(features=10)(x)
# # #     return x
# # # cnn = CNN()
# # # rng = jax.random.PRNGKey(0)
# # # params = cnn.init(rng, jnp.ones(10))['params']

# # # cnn.apply({'params': params},jnp.zeros((10)) )



# # class Linear(eqx.Module):
# #     weight: jax.numpy.ndarray
# #     bias: jax.numpy.ndarray

# #     def __init__(self, in_size, out_size, key):
# #         wkey, bkey = jax.random.split(key)
# #         self.weight = jax.random.normal(wkey, (out_size, in_size))
# #         self.bias = jax.random.normal(bkey, (out_size,))

# #     def __call__(self, x):
# #         return self.weight @ x + self.bias


# # @jax.jit
# # @jax.grad
# # def loss_fn(model, x, y):
# #     pred_y = jax.vmap(model)(x)
# #     return jax.numpy.mean((y - pred_y) ** 2)

# # batch_size, in_size, out_size = 32, 2, 3
# # linMod = Linear(in_size, out_size, key=jax.random.PRNGKey(0))



# # def copy_concat(i,val):
# #   return jnp.concatenate([val, jax.numpy.expand_dims(val[:,0,:,:,:].copy(),1 )],1 )

# # aa=copy_concat(img)
# # aa.shape
# # def test_func2(nmax, mat, mat2, x, sum_res, sum_res2):
# #   def body_fun(n, carry):
# #     sum_res, sum_res2 = carry
# #     sum_res = sum_res + jnp.dot(mat, n * x)
# #     sum_res2 = sum_res2 + jnp.dot(mat2, ((n/2) * x))
# #     return (sum_res, sum_res2)
# #   sum_res, sum_res2 = jax.lax.fori_loop(0, nmax, body_fun, (sum_res, sum_res2))
# #   return sum_res, sum_res2, jnp.sqrt(x * nmax)