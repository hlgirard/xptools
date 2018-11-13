import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.widgets import RectangleSelector
import seaborn as sns
import av
import cv2

import skimage
from skimage import io, img_as_ubyte
from skimage.color import rgb2gray
from skimage.filters import threshold_minimum
from skimage.measure import label, regionprops
from skimage.morphology import closing, square

from joblib import Parallel, delayed

import os

#Local imports
import select_roi

def open_video(file):
    """Opens a video file and returns a stack of numpy arrays corresponding to each frame.
    
    The images in the stack are converted to ubyte and gray scale for further processing.
    """

    v = av.open(file)
    stack = []
    for frame in v.decode(video=0):
        img = frame.to_image()
        #Convert the img to a numpy array, convert to grayscale and ubyte.
        img_gray = img_as_ubyte(rgb2gray(np.asarray(img)))
        stack.append(img_gray)
    return stack


def determine_threshold(stack):
    """Determines the threshold to use for the stack.

    Returns the value obtained by running a minimum threshold algorithm on the median image of the stack.
    TODO: increase robustness by comparing thresholds obtained on different images and choosing the best one.
    """
    img = stack[len(stack)//2]
    return threshold_minimum(img)


def analyze_front(img, thresh, expName, stackIdx):
    """Find the portion of the image with ice and determine the coordinates of the bounding box
    """
    #Threshold the image using the global threshold
    img_min_thresh = img > thresh
    # Closing
    bw = closing(img_min_thresh, square(3))
    # label image regions
    label_image = label(bw)
    #Extract region properties for regions > 100 px^2
    regProp = [i for i in regionprops(label_image)]
    largest = regProp[np.argmax([i.area for i in regProp])]
    #Get largest region properties
    minr, minc, maxr, maxc = largest.bbox
    area = largest.area
    return [expName, stackIdx, area, minr, minc, maxr, maxc, area / img.shape[0]]

def process_movie(file, df_crop):
    """Process a stack of images to determine the progress of the ice front.
    """
    #Extract experiment name from filename
    name = file.split('.')[0].split('/')[2]
    #Open the file and get a stack of grayscale images
    stack = open_video(file)
    #Select the region of interest
    (minRow, minCol, maxRow, maxCol) = df_crop[df_crop['ExpName'] == name]['CroppingBox'].iloc[0]
    #Crop the stack to the right dimensions
    stack_cropped = [img[minRow:maxRow,minCol:maxCol] for img in stack]
    #Determine the threshold
    min_thresh = determine_threshold(stack_cropped)
    #Create a dataframe to hold the data
    col_names = ['ExpName','Frame #', 'Area', 'MinRow', 'MinCol', 'MaxRow', 'MaxCol','HeightPx']
    df = pd.DataFrame(columns=col_names)
    #Analyze pics and drop data in df
    for i in range(len(stack)):
        df.loc[i] = analyze_front(stack[i], min_thresh, name, i)
    #Return the dataframe
    return df

if __name__ == '__main__':

    #TODO: generalize the working directory
    working_dir = 'analysis-tools/test_movies'

    file_list = [working_dir + '/' + file for file in os.listdir(working_dir) if file.endswith('.avi')]
    print("Files to process: " + str(file_list))

    #Process the movies

    #If the movies have been processed already, load from disk, otherwise, process
    savepath = working_dir+"/"+"ProcessedData"+".pkl"

    if os.path.isfile(savepath):
        df = pd.read_pickle(savepath)
        print("Data loaded from disk")
    else:
        print("Did not find data, processing movies")
        #Create a dataframe to hold the cropping boxes
        columns = ['ExpName', 'CroppingBox']
        df_crop = pd.DataFrame(columns=columns)
        #Determine the bounding boxes
        for file in file_list:
            #Extract experiment name from filename
            name = file.split('.')[0].split('/')[2]
            #Open the file and get a stack of grayscale images
            stack = open_video(file)
            #Select the region of interest
            df_crop = df_crop.append({'ExpName':name, 'CroppingBox':select_roi.select_rectangle(stack[len(stack)- 10])}, ignore_index=True)
        #Run the movie analysis in parallel
        df_list = Parallel(n_jobs=-2, verbose=10)(delayed(process_movie)(file, df_crop) for file in file_list)
        #Merge all the dataframes in one and reindex
        df = pd.concat(df_list).reset_index(drop=True)
        #Save dataframe to disk
        df.to_pickle(savepath)

    #Plot the height vs. time and save the graph
    f1 = plt.figure()
    sns.set()
    sns.set_style("ticks")
    sns.set_context("talk")
    sns.set_style({"xtick.direction": "in","ytick.direction": "in"})

    p = []

    for i in range(len(df['ExpName'].unique())):
        p.append(plt.scatter(
            df[df['ExpName'] == df['ExpName'].unique()[i]]['Frame #'],
            df[df['ExpName'] == df['ExpName'].unique()[i]]['HeightPx'],
            label = df['ExpName'].unique()[i]
        ))

    plt.legend(fontsize=16,loc='lower right',frameon=True)
    axes = plt.gca()
    axes.set_xlim([0,1000])
    #axes.set_ylim([0,7])
    plt.xlabel('Time (s)',fontsize=20)
    plt.ylabel('Height of Front (px)',fontsize=20)

    plt.minorticks_on()
    plt.tick_params(axis='both', which='major', labelsize=18)

    f1.set_size_inches(8, 8)
    print('Showing Plot')
    plt.show()


