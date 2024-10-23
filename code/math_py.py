import numpy as np
from scipy.ndimage import gaussian_filter

def HU_2_grayscale(HU, HU_low_threshold = None):
    """
    Linear transformation from HU to grayscale

    Parameters:
    ----------
    HU :           n X 1 numpy array 
                 The HU values
    HU_low_threshold :  Integer or float
                  The low threshold of HU. If None, use the mininal value in HU array.
                 should be higher than -1024 

    Returns:
    ----------
    grayscale: n X 1 numpy array 
           grayscale
    """
    hu_min = np.min(HU)
    hu_max = np.max(HU)
    #print(f"The max HU is {hu_max}, and the min HU is {hu_min}")
    if HU_low_threshold == None or HU_low_threshold < -1024:
        HU_low_threshold = hu_min
    normalized = np.where(HU < HU_low_threshold, HU_low_threshold, (HU - HU_low_threshold) / (hu_max - HU_low_threshold))
    grayscale = normalized * 255
    return grayscale

def norm(a):
    """
    Calculate the norm of a vector or N vectors

    Parameters:
    ----------
    a :  1 X D  or N X D numpy array, D is the dimension of vector. 
       vector(s)
       
    Returns:
    ----------
    norm : 1 dimension or N X 1 2-dimensional numpy array. 
       norm(s) of vector(s)
    """
    if a.ndim > 1:
        dot = (a * a).sum(axis = 1)
        norm = np.sqrt(dot)
    else:
        dot = (a * a).sum()
        norm = np.sqrt(dot)
    return norm


def cos(a, b):
    """
    Calculate the cosine between two vectors

    Parameters:
    ----------
    a :  1 X D  or N X D numpy array, D is the dimension of vector. 
       vector(s)
    b :  1 X D  or N X D numpy array, D is the dimension of vector. 
       vector(s)     
  
    Returns:
    ----------
    cos : 1 dimension or N X 1 2-dimensional numpy array. 
       norm(s) of vector(s)
    """
    if a.ndim == 1 and b.ndim == 1:
        dot = (a * b).sum()
        cos = dot / norm(a) / norm(b)
    else:
        dot = (a * b).sum(axis = 1)
        cos = dot / norm(a) / norm(b)
    return cos


def distance(a, b):
    """
    Calculate the Euclide distance between two vectors

    Parameters:
    ----------
    a :  1 X D  or N X D numpy array, D is the dimension of vector. 
       vector(s)
    b :  1 X D  or N X D numpy array, D is the dimension of vector. 
       vector(s)     
  
    Returns:
    ----------
    dis : 1 dimension or N X 1 2-dimensional numpy array. 
        The Euclide distance between two vectors
    """
    diff = a - b
    if diff.ndim != 1:
        dis = np.sqrt((diff * diff).sum(axis = 1))
    else:
        dis = np.sqrt((diff * diff).sum())
    return dis


def smooth(points, sigma = 6):
    """
    Smooth a line segment using gaussian_filter, in order to facilitate subsequent calcultion of tangent

    Parameters:
    ----------
    points :  N X 3 numpy array. 
          coordinates of N points on one segment.
    sigma :  a single number 
          Standard deviation for Gaussian kernel, used to adjust how smooth the segment will become
  
    Returns:
    ----------
    XYZ :   N X 3 numpy array. 
          coordinates of N new points on a more smooth segment.
        
    """
    # Ensure points in float.
    points = points.astype(float)
    
    X = points[:, 0]
    Y = points[:, 1]
    Z = points[:, 2]
    X1 = gaussian_filter(X, sigma)
    Y1 = gaussian_filter(Y, sigma)
    Z1 = gaussian_filter(Z, sigma)
    XYZ = np.concatenate((X1.reshape(-1, 1), Y1.reshape(-1, 1), Z1.reshape(-1, 1)), axis = 1)
    return XYZ

def vx_2_atm(vx_points, affine):
    """
    Transform coordinates from image/voxel coordinate system into anatomical/real-world coordinate system 

    Parameters:
    ----------
    vx_points :  N X 3 numpy array. 
            coordinates of N points in the image/voxel coordinate system
    affine :   4 X 4 numpy array 
            the affine array
  
    Returns:
    ----------
    atm_points :  N X 3 numpy array. 
            coordinates of N points in the anatomical coordinate system
    """
    A = affine[:3,:3]
    t = affine[:3,3]
    atm_points = np.matmul(vx_points , A.T) + t
    return atm_points



def arr_2_atm(arr, affine):
    """
    Given an binary/label array in voxel coordinate system, calculate the coordinates of label "1" in anatomical coordinate system 

    Parameters:
    ----------
    arr :     3D numpy array
            an binary/label array in voxel coordinate system (In this project, 1 for colon)
    affine :   4 X 4 numpy array 
            the affine array
  
    Returns:
    ----------
    atm_points: N X 3 numpy array. 
            coordinates of all label "1" in the anatomical coordinate system
    """    
    vx_points = np.argwhere(arr == 1)
    atm_points = vx_2_atm(vx_points, affine)
    return atm_points



def atm_2_vx(atm_points, affine):
    
    """
    Transform coordinates from anatomical/real-world coordinate system into image/voxel coordinate system 

    Parameters:
    ----------
    atm_points : N X 3 numpy array. 
             coordinates of N points in the anatomical coordinate system

    affine :   4 X 4 numpy array 
            the affine array
  
    Returns:
    ----------
    vx_points :  N X 3 numpy array. 
            coordinates of N points in the image/voxel coordinate system

    """
    A = affine[:3,:3]
    t = affine[:3,3]
    foo = atm_points - t
    vx_points = np.matmul(foo, np.linalg.inv(A.T))
    return vx_points




def approximate_points_atm_to_arr(atm_points):
    """
    Given coordinates of points in anatomical coordinate system, first round them to the nearest integer.
    Then shift these points so that all points have non-negative coordinates.
    Based on them, construct an array, the voxels with points has entry 1 and other voxels 0

    Parameters:
    ----------
    points : N X 3 numpy array. 
          coordinates of N points in the anatomical coordinate system

    Returns:
    ----------
    arr_atm :  3D numpy array
           The input points has label 1 and other 0.
    origin : 1 X 3 numpy array
          The position of new-built array[0,0,0] in the original anatomical coordinate system
    
    """

    # Round elements of the array to the nearest integer, no matter positive or negative
    atm_points= (np.rint(atm_points)).astype(int)
    
    # find the frame
    origin_0, max_0 = np.min(atm_points[:,0]), np.max(atm_points[:,0])
    origin_1, max_1 = np.min(atm_points[:,1]), np.max(atm_points[:,1])
    origin_2, max_2 = np.min(atm_points[:,2]), np.max(atm_points[:,2])
    
    buffer = 2
    arr_atm = np.zeros(( max_0 - origin_0 + buffer, max_1 - origin_1 + buffer, max_2 - origin_2 + buffer))
    arr_atm[atm_points[:,0] - origin_0, atm_points[:,1] - origin_1, atm_points[:,2] - origin_2] = 1
    
    origin = np.array([origin_0, origin_1, origin_2])

    return arr_atm, origin
    
