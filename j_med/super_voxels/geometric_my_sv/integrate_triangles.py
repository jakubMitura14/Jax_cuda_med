import numpy as np
import matplotlib.pyplot as plt
import toolz
import optax
import jax.numpy as jnp
import jax
from flax import linen as nn
import einops
import seaborn as sns
from itertools import permutations
from itertools import product
import numpy as np
import matplotlib.pyplot as plt
import ipympl
import imageio.v3 as iio
import skimage.color
import skimage.filters
import skimage.measure
import os
from shape_reshape_functions import *
from functools import partial
import math
from control_points_utils import *
from set_points_loc import *
from points_to_areas import *
import itertools

def get_triangles_data():
    """ 
    manually setting data about what triangles are present in each square 
    it is based on set of sv centers and control points - also comments are to the upper left square of the image in 
    /workspaces/jax_cpu_experiments_b/geometric_sv_idea/triangle_geometric_sv.jpg
    we will also use the id of nodes as in the image bottom right
    single triangle data will consist of 4 entries - first 3 will be verticies ids as in image and the last one the id of the sv that is owner of this node
        as in the alpha order 
    we also organize triangles in primary triangles what is very usefull for additional control points managament    
    """
    return jnp.array([
          [[0,8,7,3]#I
         ,[0,8,1,0]]#A

         ,[[0,2,1,0]#B
         ,[0,2,3,1]]#C

         ,[[0,4,3,1]#D
         ,[0,4,5,2]]#L

         ,[[0,6,5,2]#K
         ,[0,6,7,3]]]#J
        )
def iter_zipped(orig,new_ones):
    return list(map(lambda triang: iter_zipped_inner(triang,new_ones) ,orig  ))

def iter_zipped_inner(triang,new_ones):
    on_border=np.append([triang[0]],new_ones)
    # on_border=[triang[0]]+new_ones
    # on_border=np.array(on_border)
    # on_border=np.flip(np.array(on_border),axis=0)
    # print(f"orig {triang} new_ones {new_ones} on_border {on_border}")
    return list(map(lambda i: [on_border[i],on_border[i+1],triang[2],triang[3]]  ,range(len(on_border)-1)))

def get_modified_triangles_data(num_additional_points,primary_control_points_offset):
    """ 
    as we can add variable number of additional control points we need also to include them in the analysis
    """
    triangles_data= get_triangles_data()
    triangles_data= np.array(triangles_data)
    triangles_data_prim= triangles_data
    #num_additional_points tell how many additional points we will insert per primary triangle
    #and we have 4 primary triangles
    triangles_data_new_entries= list(map( lambda i :np.arange(primary_control_points_offset+i*num_additional_points,primary_control_points_offset+i*num_additional_points+num_additional_points),range(4)))
    zipped= list(zip(triangles_data_prim, triangles_data_new_entries))
    triangles_data_new= list(itertools.starmap(iter_zipped,zipped ))

    res=jnp.array(triangles_data_new)
    return  einops.rearrange(res,'a b c d -> (a b c) d' )

# def analyze_single_triangle(curried,verts):
#     """ 
#     given a point it is designed to be scanned over triangles as we can add also additional
#     control points 
#     """
#     vert_b,vert_c=verts
#     x_y,control_points_coords,res,vert_a,sv_id=curried
#     is_in=is_point_in_triangle(x_y,control_points_coords[vert_a,:],control_points_coords[vert_b,:],control_points_coords[vert_c,:])
#     return (x_y,control_points_coords,res.at[sv_id].set(res[sv_id]+is_in ),vert_a,sv_id),None
#     # return curried,None

def analyze_single_triangle(curried,triangle_dat):
    """ 
    given a point it is designed to be scanned over triangles as we can add also additional
    control points 
    """
    x_y,control_points_coords,res=curried
    is_in=is_point_in_triangle(x_y,control_points_coords[triangle_dat[0],:],control_points_coords[triangle_dat[1],:],control_points_coords[triangle_dat[2],:])
    return (x_y,control_points_coords,res.at[triangle_dat[3]].set(res[triangle_dat[3]]+is_in )),None



# def analyze_primary_triangle(curried,triangle_dat):
#     x_y,control_points_coords,res,num_additional_points=curried
#     #we need to iterate over the additional coords also 
#     #so now we scan over triangle_dat supplying info of what verticies to analyze
#     #we also do operation twice for each subtriangles in the primary triangles
#     curried_small=x_y,control_points_coords,res,triangle_dat[0,-2],triangle_dat[0,-1]
#     curr,_=jax.lax.scan(analyze_single_triangle,curried_small,(triangle_dat[0,1:-2], triangle_dat[1,0:-3]))#,length=num_additional_points+1
#     x_y,control_points_coords,res,a,b =curr
#     # x_y,control_points_coords,res=jax.lax.scan(analyze_single_triangle,curried_small,triangle_dat[0,1:-2], triangle_dat[0,0:-3])#,length=num_additional_points+1
    
#     curried_small=x_y,control_points_coords,res,triangle_dat[1,-2],triangle_dat[1,-1]
#     curr,_=jax.lax.scan(analyze_single_triangle,curried_small,(triangle_dat[0,1:-2], triangle_dat[1,0:-3]))#,length=num_additional_points+1
#     x_y,control_points_coords,res,a,b=curr

#     curried=x_y,control_points_coords,res,num_additional_points
#     return curried,None


def analyze_single_point(x_y,triangles_data,control_points_coords,num_additional_points):
    """ 
    analyze thepoints of sv area (apart from edges) by checking it against the triangles
    we will scan over those triangles and return array of length 4 that will indicate to which sv given point is attached
    x_y - array with 2 entries indicating x and y coordinates of currently analyzed point
    triangles_data - constants describing points and to which sv they belong
    control_points_coords - coordinates of the sv centers and control points in order as indicated at image 
        /workspaces/jax_cpu_experiments_b/geometric_sv_idea/triangle_geometric_sv.jpg
    """
    curried=x_y,control_points_coords,jnp.zeros(4)
    res,_= jax.lax.scan(analyze_single_triangle,curried, triangles_data) #,length=4
    return (res[2]+0.000000000000000000000000000000001)/(jnp.sum(res[2])+0.000000000000000000000000000000001) 
    # return res[2]

v_analyze_single_point=jax.vmap(analyze_single_point,in_axes=(0,None,None,None))
v_v_analyze_single_point=jax.vmap(v_analyze_single_point,in_axes=(0,None,None,None))

def analyze_point_linear(curr_point, control_point,channel_up,channel_down):
    """ 
    as we are analyzing bottom and right border as lines we are just intrested weather given point is up or 
        down the axis from control point
    curr_point - float representing the position in axis of intrest of the point currently analyzed
    control_point - float representing the position in axis of intrest of the point currently analyzed
    channel_up - the channel owned by the sv up the axis
    channel_down - the channel owned by the sv down the axis
    """
    #will give close to 1 if test point is maller than control
    is_test_smaller_than_control=nn.sigmoid( (control_point-curr_point)*10 )
    res=jnp.zeros(4)
    res=res.at[channel_up].set(is_test_smaller_than_control)
    res=res.at[channel_down].set(1-is_test_smaller_than_control)
    return res

v_analyze_point_linear= jax.vmap(analyze_point_linear,in_axes=(0,None,None,None))




def reshuffle_channels(res,sv_area_type,debug_index):
    """ 
    sv_area_type - we have 4 sv_area_type in first node 0 is in upper left corner and we go clockwise so bottom left is 3
        other sv types are set in a way to be consistent with first sv_area_type
    we get res as the input that has the channels organised as if it is first type this funtion is to fix it
    """
    # p = permutations([0,1,2,3])
    # p=list(p)
    # prod=list(product(p,p,p))
    
    def alfa():
        return res
        
    def beta():
        #0 1 2 3 -> 1 0 3 2
        return res[:,:, jnp.array([3, 2, 1, 0])]
        # return res[:,:, jnp.array([1,0,3,2])]
        return res
        
    def delta():
        #0 1 2 3 -> 3 2 1 0
        return res[:,:, jnp.array([2, 3, 0, 1])]   
        # return res[:,:, jnp.array([2,1,0,3])]
        # return res

    def gamma():
        #0 1 2 3 -> 2 3 0 1
        # return res[:,:, jnp.array([2,3,0,1])]
        return res[:,:, jnp.array([1, 0, 3, 2])]
        return res


    functions_list=[alfa,beta,delta,gamma]
    return jax.lax.switch(sv_area_type,functions_list)    


def analyze_square(control_points_coords,diameter
                   ,triangles_data,sv_area_type,debug_index,num_additional_points):
    """ 
    analyzing single square where each corner is created by sv center
    triangles_data- constants describing triangles specified in get_triangles_data function
    diameter - diameter of sv area - what is important the right and bottom edges will be treated separately as their ownership calculations do not require triangle analysis
    control_points_coords - data about location of control points where entries are organized as in the main image
    sv_area_type - we have 4 sv_area_type in first node 0 is in upper left corner and we go clockwise so bottom left is 3
        other sv types are set in a way to be consistent with first sv_area_type
    """
    #get grid of points and apply analyze_single_point ignoring left and bottom borders for now
    grid=einops.rearrange(jnp.mgrid[0:diameter+1, 0:diameter+1],'c x y-> x y c')+control_points_coords[1,:]
    grid=grid[1:,1:,:]
    # print(f"\n ggggrid  \n {grid} \n  ")
    # print(f"grid {grid}")

    grid_right= grid[-1,:,1]
    grid_bottom= grid[:,-1,0]
    grid=grid[0:-1,0:-1,:]

    res=v_v_analyze_single_point(grid,triangles_data,control_points_coords,num_additional_points)
    #analyze bottom and right border we assume that we are in square alpha so right border is between sv 1vs2 and bottom 2 vs 3
    right=v_analyze_point_linear(grid_right,control_points_coords[4,1],1,2)
    bottom=v_analyze_point_linear(grid_bottom,control_points_coords[6,0],3,2)

    # right=right.at[0,:].set(0)
    right=right.at[-1,:].set(0)
    # bottom=bottom.at[0,:].set(0)
    bottom=bottom.at[-1,:].set(0)

    # right=right.at[0,1].set(1)
    right=right.at[-1,2].set(1)

    # bottom=bottom.at[0,3].set(1)
    bottom=bottom.at[-1,2].set(1)

    # recreate full grid
    res= jnp.pad(res,((0,1),(0,1),(0,0)))


    res=res.at[-1,:,:].set(right)
    res=res.at[:,-1,:].set(bottom)


    res=reshuffle_channels(res,sv_area_type,debug_index) 

    #return 4 channel array where each channel tell about given sv and weather this point is owned but that sv
    return res

v_analyze_square=jax.vmap(analyze_square,in_axes=(0,None,None,0,None,None) )
v_v_analyze_square=jax.vmap(v_analyze_square,in_axes=(0,None,None,0,None,None) )
v_v_v_analyze_square=jax.vmap(v_v_analyze_square,in_axes=(0,None,None,0,None,None) )

# @partial(jax.jit, static_argnames=['pmapped_batch_size','sv_diameter','r','diam_x','diam_y','half_r'])
def analyze_all_control_points(modified_control_points_coords,triangles_data
                               ,pmapped_batch_size,sv_diameter,half_r,num_additional_points):    
    
    debug_index=0
    #we prepare data for vmapping - sv area type has this checkerboard organization
    shh=modified_control_points_coords.shape
    sv_area_type=jnp.zeros((shh[1],shh[2]),dtype=int)
    sv_area_type=sv_area_type.at[1::2,0::2].set(3)
    sv_area_type=sv_area_type.at[1::2,1::2].set(2)
    sv_area_type=sv_area_type.at[0::2,1::2].set(1)
    sv_area_type=einops.repeat(sv_area_type, 'x y->b x y', b=pmapped_batch_size)
    #vmap over analyze_square   
    res=v_v_v_analyze_square(modified_control_points_coords,sv_diameter,triangles_data,sv_area_type,debug_index,num_additional_points)
    res= einops.rearrange(res,' b w h x y c-> b (w x) (h y) c')   
    half_r=int(half_r)
    res=res[:,half_r:-half_r,half_r:-half_r,:]
    #returns the 4 channel mask that encode the ownership of the pixels for each supervoxel
    return res



    # triangles_data=get_triangles_data()




