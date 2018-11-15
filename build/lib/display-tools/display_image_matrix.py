#!/usr/bin/env pythonw

"""
Displays the images contained in a directory as a grid (8 lines by default)

Parameters
----------
lines : int
    Number of lines of the image matrix
directory : string
    Path of the directory to be explored
compress : boolean
    Reduces the resolution of the resulting image to reduce filesize
"""

import os
import sys
import argparse

import itertools
import cv2
import numpy as np

def list_files(dirname):
    """Lists the image files in a directory

    Returns a list of openCV image objects corresponding to all the *.JPG files
    found in the directory
    """
    filename_list = []

    for file in os.listdir(dirname):
        if file.endswith(".JPG"):
            filename_list.append(file)

    filename_list.sort()

    print("Found {0} files in the directory".format(len(filename_list)))

    return [cv2.imread(dirname+"/"+file) for file in filename_list]

def compose_matrix(imgs, dirname, lines, compress):
    """Arranges the images from a list in a matrix

    The images are arranged in a grid and the resulting matrix is saved
    """

    expname = dirname.split('/')[-1]

    name = "Crystallization_" + expname + "_per_particle" + ".jpg" #Name of the exported file
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
    if compress:
        resized = cv2.resize(imgmatrix, (mat_x//3,mat_y//3), interpolation = cv2.INTER_AREA)
        compression_params = [cv2.IMWRITE_JPEG_QUALITY, 90]
        cv2.imwrite(dirname + '/' + name, resized, compression_params)
    else:
        compression_params = [cv2.IMWRITE_JPEG_QUALITY, 100]
        cv2.imwrite(dirname + '/' + name, imgmatrix, compression_params)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("-l", "--lines", type=int , required=False, default=8, help="Number of lines of the image matrix")
    ap.add_argument("directory", help="Path of the directory")
    ap.add_argument("-c", "--compress", action='store_true', help="Resizes the finale image to reduce resolution by a factor of 9")

    args = ap.parse_args()

    dirname = args.directory
    lines = args.lines
    compress = args.compress

    compose_matrix(list_files(dirname), dirname, lines, compress)
