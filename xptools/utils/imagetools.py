#System imports
import os
import sys

#Image processing imports
import itertools
import cv2
import numpy as np

def open_all_images(dirname):
    """
    Returns a list of all cv2 images found in the directory
    
    Parameters
    ----------
    dirname : string
        Directory to look for the images in
    Returns
    ----------
    list[cv2 images]
        List of cv2 images
    """
    filename_list = []

    for file in os.listdir(dirname):
        if file.endswith(".JPG") or file.endswith(".jpg"):
            filename_list.append(file)

    filename_list.sort()

    print("Found {0} files in the directory".format(len(filename_list)))

    return [cv2.imread(dirname+"/"+file) for file in filename_list]

def compose_matrix(imgs, dirname, lines = 8, bCompress = False):
    """
    Arranges images in a matrix and saves the resulting image to the chosen directory
    
    Parameters
    ----------
    imgs : list (cv2 images)
        List of image to process
    dirname : string
        Path of the directory to save the composition to
    lines : int, optional
        Number of lines in the matrix (default 8)
    bCompress : boolean, optional
        Flag to downscale the resulting composition to 1/10th of the original resultion
    """

    name = dirname.split('/')[-1] + ".jpg" #Name of the exported file
    margin = 20 #Margin between pictures in pixels

    n = len(imgs) # Number of images in the list
    h = lines # Height of the matrix (nb of images)
    if (n % h) == 0:
        w = n // h
    else:
        raise ValueError("The number of images ({0}) is not divisible by the number of lines ({1})".format(n,h))

    #Define the shape of the image to be replicated (all images should have the same shape)
    img_h, img_w, img_c = imgs[0].shape

    #Define the margins in x and y directions
    m_x = margin
    m_y = margin

    #Size of the full size image
    mat_x = img_w * w + m_x * (w - 1)
    mat_y = img_h * h + m_y * (h - 1)

    #Create a matrix of zeros of the right size and fill with 255 (so margins end up white)
    imgmatrix = np.zeros((mat_y, mat_x, img_c),np.uint8)
    imgmatrix.fill(255)

    #Prepare an iterable with the right dimensions
    positions = itertools.product(range(h), range(w))

    for (y_i, x_i), img in zip(positions, imgs):
        x = x_i * (img_w + m_x)
        y = y_i * (img_h + m_y)
        imgmatrix[y:y+img_h, x:x+img_w, :] = img

    print("Writing the composite image to {0}".format(dirname + '/' + name))

    #Write the final image to disc and compress if requested
    if bCompress:
        resized = cv2.resize(imgmatrix, (mat_x//3,mat_y//3), interpolation = cv2.INTER_AREA)
        compression_params = [cv2.IMWRITE_JPEG_QUALITY, 90]
        cv2.imwrite(dirname + '/' + name, resized, compression_params)
    else:
        compression_params = [cv2.IMWRITE_JPEG_QUALITY, 100]
        cv2.imwrite(dirname + '/' + name, imgmatrix, compression_params)