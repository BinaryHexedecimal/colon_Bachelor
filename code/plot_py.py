from math_py import *

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d
from PIL import Image

import numpy as np
import copy
import math

from scipy.interpolate import interpn


def plot_arr(arr, affine, fig_size = 4):
    """
    Given label array, plot the voxels with label 1 in anatomical coordinate system.

    Parameters:
    ----------
    arr :     3D label array

    affine :   4 X 4 numpy array 
           the affine array
    fig_size : integer
           the size of figure
    """
    fig = plt.figure(figsize = (fig_size, fig_size))
    ax = fig.add_subplot(projection = '3d')
 
    atm = arr_2_atm(arr, affine)
   
    ax.scatter(atm[:,0],atm[:,1],atm[:,2], marker = '.', s = 1)
    plt.gca().set_box_aspect((1, 1, 1))

    ax.set_xlabel('X ')
    ax.set_ylabel('Y ')
    ax.set_zlabel('Z ')
    
    ax.set_xlim([-150, 150])
    ax.set_ylim([100,330])
    ax.set_zlim([1200,1700])

    plt.show()


def plot_points(points, fig_size = 4):
    fig = plt.figure(figsize = (fig_size, fig_size))
    ax = fig.add_subplot(projection = '3d')
    
    ax.scatter(points[:, 0], points[:, 1], points[:, 2], marker = '.', s = 1)

    plt.gca().set_box_aspect((1, 1, 1))

    ax.set_xlabel('X ')
    ax.set_ylabel('Y ')
    ax.set_zlabel('Z ')
    
    ax.set_xlim([-150, 150])
    ax.set_ylim([100,330])
    ax.set_zlim([1200,1700])

    plt.show()
    
    
    
def plot_skeleton(skeleton_arr, origin, fig_size = 4):
    """
    Given label array, plot the voxels with label 1 in anatomical coordinate system.

    Parameters:
    ----------
    skeleton_arr : 3D label array
              The skeleton array
    origin :     3 X 1 numpy array 
              The vector by which an array is translated into the anatomical coordinate system
    fig_size :    integer
              the size of figure
    """
    fig = plt.figure(figsize = (fig_size, fig_size))
    ax = fig.add_subplot(projection = '3d')
    
    points= np.argwhere(skeleton_arr==1)
    ax.scatter(points[:, 0] + origin[0], points[:, 1] + origin[1], points[:, 2] + origin[2], marker = '.', s = 1)

    plt.gca().set_box_aspect((1, 1, 1))

    ax.set_xlabel('X ')
    ax.set_ylabel('Y ')
    ax.set_zlabel('Z ')

    ax.set_xlim([-150, 150])
    ax.set_ylim([100,330])
    ax.set_zlim([1200,1700])
    plt.show()


def plot_section(L, half_side_a, half_side_b, idx_X_lst, HUs, dir_, data_filename, threshold_HU = None):
    """
    plot a series of slices in grayscale, as in the way of CT 
    Saved also in the given directory, besides shown in the present window

    Parameters:
    ----------
    L :         integer
              the no. of points on the centerline. That is to say, we will take L slices along centerline
    half_side_a:   integer
              The half side a of slice
    half_side_b:   integer
              The half side b of slice
    idx_X_lst:    list of list (dtype is integer)
              The outer list has L items, which is a list  with various length associated with a slice.
              More specifically, each inner list contains index of points which are 
              intersection between colon surface and this slice.
    HUs :       N X 1 numpy array, where N = L*(2*half_side_a+1)*(2*half_side_b+1) 
              The HU values of points which are arranged in an order of slices X side_a X side_b 
    dir_ :       string
              the directory where the images are saved.
    data_filename: string
              the file name of used CT data
    
    threshold_HU : integer 
              a HU value used to the enhance the contrast of images.
    """
    side_a = 2 * half_side_a + 1
    side_b = 2 * half_side_b + 1
    area = side_a * side_b
    assert L * side_a * side_b == HUs.shape[0]
    
    gs = HU_2_grayscale(HUs, threshold_HU)
    
    x, y = np.arange(side_b),np.arange(side_a)
    xx,yy = np.meshgrid(x, y)
    xy = (np.vstack([xx.ravel(), yy.ravel()])).T
    bredth = 5
    fig,ax = plt.subplots(int(L/bredth)+1, bredth) 
    

    for l in range(L):
        idx_lst = idx_X_lst[l]
        g_l = gs[l * area:(l + 1) * area]
        HU_l = HUs[l * area:(l + 1) * area]
        G = g_l.reshape((side_b, side_a)) 
        H = HU_l.reshape((side_b, side_a))
        for idx in idx_lst:
            m = math.floor(idx / side_a)
            n = idx % side_a
            G[m,n] = 254

        ax[int(np.floor(l / bredth)),l % bredth].scatter(xy[:,0],xy[:,1], c = np.flip(H.T, axis = 1), s = 1, cmap = 'gray')
        ax[int(np.floor(l / bredth)),l % bredth].set_title( f'{l}th section')
        ax[int(np.floor(l / bredth)),l % bredth].axis('equal')    
        
        G=np.flip(G)
        im_ = Image.fromarray(G.T)
        im_ = im_.convert("L")
        im_.save(dir_ + f"{data_filename}_section_{l}.jpg")
        #im_.save(dir_ + f"section_{l}.jpg")
        
        

        