import nrrd
import nibabel as nib
import numpy as np
from skimage import morphology as sk


# read CT IMAGE
def read_CT(dir_, filename):
    """
    Read the original CT data
    Parameters:
    ----------
    dir_ :    string 
           the directory where the file is saved
    filename : string
           the NIFTI file that will be read
    Returns:
    ----------
    data_CT :  3D numpy array
            the radio density from CT data
    affine :   4 X 4 numpy array
           The affine array storing the relation between image coordinate system and anatomical system 
    
    """
    img = nib.load(dir_ + filename)
    header_CT = img.header
    data_CT = np.array(img.dataobj)

    size = data_CT.shape
    print("CT size is ,", size)

    pixdim = header_CT["pixdim"]
    scale = pixdim[1:4]
    print("scale is ", scale)

    affine = img.affine
    print("Affine array is \n", affine)

    return data_CT, affine


def read_st(dir_,filename, volumn_threshold = 800):
    """
    Read the segmentation data
    Parameters:
    ----------
    dir_ :          string 
                 the directory where the file is saved
    filename :       string
                 the NIFTI file that will be read
    volumn_threshold :  integer
                 the objects with voxels less than the threshold will be treated as misclassification and deleted automatically.
    Returns:
    ----------
    colon :         3D numpy array
                 binary array, where 1 for colon, 0 for the other
    """
    # segment index
    colon, header_colon = nrrd.read(dir_ + filename)
    #remove small clumps
    large = sk.remove_small_objects(colon != 0, volumn_threshold)
    colon = large * 1
    return colon