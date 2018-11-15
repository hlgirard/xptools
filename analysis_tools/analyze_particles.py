from skimage import io, exposure, img_as_float, img_as_ubyte, morphology, filters, util
from skimage.color import rgb2gray, label2rgb
from skimage.filters import threshold_minimum, sobel
from skimage.segmentation import clear_border
from skimage.measure import label, regionprops
from skimage.morphology import closing, watershed, square

import numpy as np

def analyze_watershed(image):
    """Returns bright on dark particles using a watershed algorithm

    Implements contrast stretching, sobel edge detection and a watershed algortihm to detect particles
    Cleanup is performed by closing and removing the particles that contact the clear_border
    """
    #Convert to grayscale
    i_grey = rgb2gray(image)
    #Convert to uint8
    i_uint = img_as_ubyte(i_grey)
    #Contrast stretching
    p2, p98 = np.percentile(i_uint, (2, 98))
    i_contStretch = exposure.rescale_intensity(i_uint, in_range=(p2, p98))
    #Construct elevation map
    elevation_map = sobel(i_contStretch)
    #Create markers using extremes of the intensity histogram
    markers = np.zeros_like(i_uint)
    threshold = threshold_minimum(i_contStretch)
    markers[i_uint < 0.7*threshold] = 1
    markers[i_uint > 1.3*threshold] = 2
    #Apply the watershed algorithm for segmentation
    i_segmented = watershed(elevation_map, markers)
    #Close open domains
    i_closed = closing(i_segmented, square(3))
    #Remove artifacts connected to image border
    i_cleared = clear_border(i_closed)
    #Label image regions
    label_image = label(i_cleared)
    #Extract region properties
    partProps = regionprops(label_image, intensity_image=i_grey)

    #Return
    return partProps


def analyze_minThreshold(image):
    """Returns bright on dark particles using a thresholding algorithm

    Implements contrast stretching and minimum thresholding to detect particles
    Cleanup is performed by closing and removing the particles that contact the clear_border
    """
    #Convert to grayscale
    i_grey = rgb2gray(image)
    #Convert to uint8
    i_uint = img_as_ubyte(i_grey)
    #Contrast stretching
    p2, p98 = np.percentile(i_uint, (2, 98))
    i_contStretch = exposure.rescale_intensity(i_uint, in_range=(p2, p98))
    #Minimum thresholding
    i_min_thresh = i_uint > threshold_minimum(i_contStretch)
    #Close open domains
    i_closed = closing(i_min_thresh, square(3))
    #Remove artifacts connected to image border
    i_cleared = clear_border(i_closed)
    #Label image regions
    label_image = label(i_cleared)
    #Extract region properties
    partProps = regionprops(label_image, intensity_image=i_grey)

    #Return
    return partProps