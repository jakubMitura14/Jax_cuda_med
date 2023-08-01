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


def get_triangle_area(p_0,p_1,p_2):
    area = 0.5 * (p_0[0] * (p_1[1] - p_2[1]) + p_1[0] * (p_2[1] - p_0[1]) + p_2[0]
                  * (p_0[1] - p_1[1]))
    return differentiable_abs(area)

def is_point_in_triangle(test_point,sv_center,control_point_a,control_point_b):
    """ 
    basic idea is that if a point is inside the triangle and we will create 3 sub triangles inside 
    where the new point is the apex and the bases are the 3 edges of the tested triangles
    if the sum of the areas of the sub triangles is equal the area of tested triangle the point is most probably inside the tested triangle
    if the sum of the subtriangles areas is diffrent then area of tested triangle it is for sure not in the triangle
    tested triangle will be always build from 3 points where sv center is one of them and other 2 points are sv control points
    adapted from https://stackoverflow.com/questions/59597399/area-of-triangle-using-3-sets-of-coordinates
    added power and sigmoid to the end to make sure that if point is in the triangle it will be approximately 0 and otherwise approximately 1
    """
    main_triangle_area= get_triangle_area(sv_center,control_point_a,control_point_b)
    sub_a=get_triangle_area(test_point,control_point_a,control_point_b)
    sub_b=get_triangle_area(sv_center,test_point,control_point_b)
    sub_c=get_triangle_area(sv_center,control_point_a,test_point)

    subtriangles_area= sub_a+sub_b+sub_c
    area_diff=main_triangle_area-subtriangles_area
    area_diff=jnp.power(area_diff,2)
    return 1-(nn.sigmoid(area_diff*15)-0.5)*2

def lineLineIntersection(A, B, C, D):
    """ 
    based on https://www.geeksforgeeks.org/program-for-point-of-intersection-of-two-lines/
    """
    # Line AB represented as a1x + b1y = c1
    a1 = B[1] - A[1]
    b1 = A[0] - B[0]
    c1 = a1*(A[0]) + b1*(A[1])
 
    # Line CD represented as a2x + b2y = c2
    a2 = D[1] - C[1]
    b2 = C[0] - D[0]
    c2 = a2*(C[0]) + b2*(C[1])
    

    determinant = (a1*b2 - a2*b1)+0.000000000001
    
    # if (determinant == 0):
    #     # The lines are parallel. This is simplified
    #     # by returning a pair of FLT_MAX
    #     return Point(10**9, 10**9)
    # else:
    x = (b2*c1 - b1*c2)/determinant
    y = (a1*c2 - a2*c1)/determinant
    return [x, y]

def orthoProjection(vertex_a,vertex_b,vertex_c):
    """ 
    projection of a point on a line 
    adapted from https://stackoverflow.com/questions/55230528/find-point-where-altitude-meets-base-python
    """
    # abx = bx - ax
    abx = vertex_b[0] - vertex_a[0]
    # aby = by - ay
    aby = vertex_b[1] - vertex_a[1]
    # acx = cx - ax
    acx = vertex_c[0] - vertex_a[0]
    # acy = cy - ay
    acy = vertex_c[1] - vertex_a[1]
    t = (abx * acx + aby * acy) / ((abx * abx + aby * aby)+0.000000001)
    # px = ax + t * abx
    px = vertex_a[0] + t * abx
    # py = ay + t * aby
    py = vertex_a[1] + t * aby
    return [px, py]

def get_point_on_a_line_b(vertex_0,vertex_1,weight):
    diff_x=vertex_1[0]-vertex_0[0]
    diff_y=vertex_1[1]-vertex_0[1]
    return [vertex_0[0]+(diff_x*weight),vertex_0[1]+(diff_y*weight)]

def get_point_inside_triange(vertex_a,vertex_b,vertex_c,edge_weights):
    """ 
    we want to put a new point in a triangle - that will be a new control point
    point is as specified constrained by a triangle weights live on two of the primary triangle edges
    so we take  2 edges establish positions of temporary points by moving on those edges by percentege of their length
    then we get a line between those new points and apply 3rd weight to it so we will move along this new line
    """
    p0=get_point_on_a_line_b(vertex_a,vertex_b,edge_weights[0])
    p1=get_point_on_a_line_b(vertex_b,vertex_c,edge_weights[1])

    res=get_point_on_a_line_b(p0,p1,edge_weights[2])
    return jnp.array(res)


def get_point_inside_square(vertex_a,vertex_b,vertex_c,vertex_d,edge_weights):
    """ 
    we want to put a new point in a square - that will be a new control point
    we will need just to get a point on each edge - connect points from opposing edges by the line 
    and find an intersection point of those lines
    """
    p0=get_point_on_a_line_b(vertex_a,vertex_b,edge_weights[0])
    p1=get_point_on_a_line_b(vertex_c,vertex_d,edge_weights[1])
    
    p2=get_point_on_a_line_b(vertex_a,vertex_d,edge_weights[2])
    p3=get_point_on_a_line_b(vertex_b,vertex_c,edge_weights[3])

    return lineLineIntersection(p0,p1,p2,p3)





