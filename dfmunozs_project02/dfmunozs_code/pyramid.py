import argparse
import cv2 as cv 
import numpy as np
import sys 
import scipy 
from scipy import ndimage
from scipy  import ndimage
from q1 import conv2
from scipy import signal
import roiSelect
align_ =False
def make_larger(a,b):
    width = int(b.shape[1] )#* scale_percent / 100)
    height = int(b.shape[0] )#* scale_percent / 100)
    dim = (width, height)
    # resize image
    resized = cv.resize(a, dim, interpolation = cv.INTER_NEAREST)  
    return resized
def make_smaller(layer_i_plus_1):
    scale_percent = 50 # percent of original size
    width = int(layer_i_plus_1.shape[1] * scale_percent / 100)
    height = int(layer_i_plus_1.shape[0] * scale_percent / 100)
    dim = (width, height)
    resized = cv.resize(layer_i_plus_1, dim, interpolation = cv.INTER_NEAREST)
    return resized
    
    
def ComputePyr_Gaussinan(in_img, num_layers):
    layer_i= in_img.copy()

    gaussian_pyramid=[np.float32(layer_i)]
    ker=np.zeros((5,5))
    ker[2][2]=1
    ker= ndimage.filters.gaussian_filter(ker,1)
    #x= cv.getGaussianKernel(5,2)
    #ker= x*x.T
    for layer in range(num_layers):
        layer_i_plus_1= conv2(layer_i,ker,2)
        resized = make_smaller(layer_i_plus_1)
        layer_i_plus_1= resized
        gaussian_pyramid.append(np.float32(layer_i_plus_1))
        layer_i=layer_i_plus_1
    return gaussian_pyramid

def ComputePyr_Laplacian(guassian_pyramid):
    laplacian_top_layer= guassian_pyramid[-1]
    num_layers= len(guassian_pyramid)-1
    ker=np.zeros((5,5))
    ker[2][2]=1
    ker= ndimage.filters.gaussian_filter(ker,1)
    ker=ker
    #x= cv.getGaussianKernel(5,2)
    #ker= x*x.T
    laplacian_pyramid=[laplacian_top_layer]
    for layer in range(num_layers,0,-1):
        current_layer=guassian_pyramid[layer]
        gaussian_expanded= make_larger(current_layer, guassian_pyramid[layer-1])
        gaussian_expanded=conv2(gaussian_expanded,ker,2)
        laplacian= (guassian_pyramid[layer-1]-gaussian_expanded)
        laplacian_pyramid.append(laplacian)
    return laplacian_pyramid

def ComputePyr(img, num_layers):
    '''
    Description: This function takes in an image and computes its gaussian and laplacian pyramids.
    The number of layers can be specified but it is checked to make sure that it is valid,
    if the layers are not valid the maximum number of layer possible is used instead.

    Parameters:
    img: input image: Required
    num_layers: Number of layers for the pyramids: required

    Returns:
    gaussian pyramid and laplacian pyramid in a tuple   
    '''
    gaussian_pyramid= ComputePyr_Gaussinan(img,num_layers) 
    laplacian_pyramid= ComputePyr_Laplacian(gaussian_pyramid)
    return (gaussian_pyramid,laplacian_pyramid)

def blend(laplacian_A, laplacian_B, mask_pyramid):
    LS=[]
    for la, lb, mask in zip(laplacian_A,laplacian_B,mask_pyramid):
        ls = la * mask + lb * (1.0 - mask)
        LS.append(ls)
    return LS

def reconstruct(laplacian_pyr):
    laplacian_top = laplacian_pyr[0]
    laplacian_lst = [laplacian_top]
    num_levels = len(laplacian_pyr) - 1
    ker=np.zeros((5,5))
    ker[2][2]=1
    ker= ndimage.filters.gaussian_filter(ker,1)
    for i in range(num_levels):
        size = (laplacian_pyr[i + 1].shape[1], laplacian_pyr[i + 1].shape[0])
        laplacian_expanded = make_larger(laplacian_top, laplacian_pyr[i + 1])
        laplacian_expanded= conv2(laplacian_expanded, ker,2)
        laplacian_top = cv.add(laplacian_pyr[i+1], laplacian_expanded)
        laplacian_lst.append(laplacian_top)
    return laplacian_lst

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument("-i1", "--image1", type=str, default="lena.png", help="path to the input image")
    ap.add_argument("-i2", "--image2",type=str, default="lena.png", help="path to the input image")
    ap.add_argument("-o","--output", type=str, default="out.png", help="path for the output image")
    ap.add_argument("-om","--outputmask", type=str, default="outmask.png", help="path for the mask image")
    args = vars(ap.parse_args())
    im = cv.imread(args["image1"])
    # Load the two images
    img1 = cv.imread(args["image1"]) #Foregorund image
    img2 = cv.imread(args["image2"])   #background image 


def align(img2,img1,r,c):
    a_img = np.zeros(img2.shape)
    a_img[r:img1.shape[0]+r,c:img1.shape[1]+c] = img1
    return a_img.astype('uint8')

if align_ == True:    
# validation for aligning
    if (img2.shape[0] > img1.shape[0]):
    #    global s_flag
    #    s_flag = False
        n_f_img = align(img2,img1,50,35) 
        f_img = np.copy(n_f_img)
        img1 = np.copy(n_f_img)
    else:
        img1 = np.copy(img1)
else:
    img1 = cv.resize(img1, (600, 340))
    img2 = cv.resize(img2, (600,340))  #this resize is just to make sure that the images are the same size if needed if not this can ignored by setting align_ to True

    # Create the mask
    mask, mask_image= roiSelect.roiSelect(img1) #this function returns two mask one is the actual mask used for blending with a
    #magnitude of 1 for the selected region and the other has 255 in the region to use as an illustration to write back to the disk
    cv.imwrite(args["outputmask"], mask_image)    #write the mask to the disk for illustration purposes 
    num_levels = 5
    
    # For image-1, calculate Gaussian and Laplacian
    gaussian_pyr_1 = ComputePyr_Gaussinan(img1, num_levels)
    laplacian_pyr_1 = ComputePyr_Laplacian(gaussian_pyr_1)
    # For image-2, calculate Gaussian and Laplacian
    gaussian_pyr_2 = ComputePyr_Gaussinan(img2, num_levels)
    laplacian_pyr_2 = ComputePyr_Laplacian(gaussian_pyr_2)
    # Calculate the Gaussian pyramid for the mask image and reverse it.
    mask_pyr_final = ComputePyr_Gaussinan(mask, num_levels)
    mask_pyr_final.reverse()
    # Blend the images
    add_laplace = blend(laplacian_pyr_1,laplacian_pyr_2,mask_pyr_final)
    # Reconstruct the images
    final  = reconstruct(add_laplace)
    # Save the final image to the disk    
    cv.imwrite(args["output"], final[num_levels])    

