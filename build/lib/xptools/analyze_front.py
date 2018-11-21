"""Analyze videos of a moving ice front and plot the positions as a function of time

This module takes a directory containing video files. For each file, it asks the user to select a region of interest
and processes the selected area with a minimum threshold to find the largest area. It then plots the height of this area as
a function of time.

Options
-r: force reprocessing of the videos
-p: use plotly instead of matplotlib for plotting
-s: save the output graph
-c: scale factor in px/mm
-f: framerate in frames/s
-a: (experimental) autodetect beginning of the ice progression
"""

# Data analysis
import numpy as np
import pandas as pd

# Image analysis
import skimage
from skimage import io
from skimage.measure import label, regionprops
from skimage.morphology import closing, square

# System
import os
import argparse
from joblib import Parallel, delayed

# Local packages
from xptools.utils import select_roi, videotools

def analyze_front(img, thresh, scale = 1, bAuto = False):
    """
    Process a frame to determine the position of a moving front
    
    Parameters
    ----------
    img : np.array
        image to process (must be gray scale, ubyte)
    thresh : int
        Intensity threshold to use
    scale : float, optional
        Scaling factor in px/mm. Default is 1 px/mm
    bAuto : boolean, optional
        Try to determine the beginning of the front progression automatically (experimental). Default is False
    Returns
    -------
    List (float) [Area, MinRow, MinCol, MaxRow, MaxCol, Height]
        Information and coordinates of the largest detected region. All values in mm, use scale = 1 to obtain pixels
    """
    #Threshold the image using the global threshold
    img_min_thresh = img > thresh
    # Closing
    bw = closing(img_min_thresh, square(3))
    # label image regions
    label_image = label(bw)
    #Extract region properties
    regProp = [i for i in regionprops(label_image, intensity_image=img)]
    if len(regProp) > 0:
        largest = regProp[np.argmax([i.area for i in regProp])]
        #Low mean intensity likely means there is no ice yet
        minIntensity = 240 if bAuto else 0
        if largest.mean_intensity > minIntensity:
            #Get largest region properties
            minr, minc, maxr, maxc = largest.bbox
            area = largest.area
            return [area/scale**2, minr/scale, minc/scale, maxr/scale, maxc/scale, area / (img.shape[0] * scale)]
        else:
            return None
    else:
        return None

def process_movie(file, crop_box = None, scale = 1, framerate = 1, bAuto = False):
    """
    Process a video to determine the progression of a moving front.
    
    Parameters
    ----------
    file : string
        path of the video file to open
    crop_box : tuple (int), optional
        Coordinates of the cropping box in the format (minRow, minCol, maxRow, maxCol) based on numpy coordinates
    scale : float, optional
        Scaling factor in px/mm. Default is 1 px/mm
    framerate : int, optional
        Framerate of the video. Default is 1 frame / s
    bAuto : boolean, optional
        Try to determine the beginning of the front progression automatically (experimental). Default is False
    Returns
    -------
    Dataframe {'ExpName','Frame #', 'Time', 'Area', 'MinRow', 'MinCol', 'MaxRow', 'MaxCol','Height'}
        A dataframe containing the results of the analysis
        ExpName: string - Name of the file without the extension
        Frame #: int - Frame number
        Time: float - Time since start of the video (or icing event if bAuto = True)
        Area: float - Area of the detected region
        MinRow, MinCol, MaxRow, MaxCol: int - Coordinates of the bounding box of the detected region
        Height: float - Height of the region calculated with area / (img.shape[0] * scale
    """
    #Extract experiment name from filename
    name = file.split('.')[0].split('/')[-1]
    #Open the file and get a stack of grayscale images
    stack = videotools.open_video(file)
    if crop_box != None:
        #Select the region of interest
        (minRow, minCol, maxRow, maxCol) = crop_box
        #Crop the stack to the right dimensions
        stack = [img[minRow:maxRow,minCol:maxCol] for img in stack]
    #Determine the threshold
    min_thresh = videotools.determine_threshold(stack)
    #Create a dataframe to hold the data
    col_names = ['ExpName','Frame #', 'Time', 'Area', 'MinRow', 'MinCol', 'MaxRow', 'MaxCol','Height']
    df = pd.DataFrame(columns=col_names)
    #Analyze pics and drop data in dataframe
    for i in range(len(stack)):
        result = analyze_front(stack[i], min_thresh, scale, bAuto)
        if result != None:
            all_fields = [name, i, i/framerate] + result
            df.loc[i] = all_fields
    #Slide the time back to the first frame where ice is detected (if autoprocess is requested)
    if bAuto:
        df['Time'] = df['Time'] - min(df['Time'])
    #Return the dataframe
    return df

def plot_front_position(df, bSave, dirname):
    """Plot the height vs. time of the freezing front using matplotlib
    """

    from matplotlib import pyplot as plt
    
    plt.figure()

    # Use seaborn if available
    try:
        import seaborn as sns
        sns.set()
        sns.set_style("ticks")
        sns.set_context("talk")
        sns.set_style({"xtick.direction": "in","ytick.direction": "in"})
    except ImportError:
        pass
    
    p = []

    for i in range(len(df['ExpName'].unique())):
        p.append(plt.scatter(
            df[df['ExpName'] == df['ExpName'].unique()[i]]['Time'],
            df[df['ExpName'] == df['ExpName'].unique()[i]]['Height'],
            label = df['ExpName'].unique()[i]
        ))

    plt.legend(fontsize=16,loc='lower right',frameon=True)
    plt.xlabel('Time (s)',fontsize=20)
    plt.ylabel('Height of Front (mm)',fontsize=20)

    plt.minorticks_on()
    plt.tick_params(axis='both', which='major', labelsize=18)

    plt.show()

    if bSave:
        plt.savefig(dirname + '/' + 'FrontHeight.pdf', bbox_inches = 'tight')

def plot_front_position_pltly(df, bSave, dirname):
    """Plot the height vs. time of the freezing front using plotly
    """

    import plotly
    import plotly.graph_objs as go
    import plotly.io as pio
    
    data=[]

    for i in range(len(df['ExpName'].unique())):
        data.append(go.Scatter(
            x = df[df['ExpName'] == df['ExpName'].unique()[i]]['Time'],
            y = df[df['ExpName'] == df['ExpName'].unique()[i]]['Height'],
            name = df['ExpName'].unique()[i]
        ))

    fig = go.Figure({
        "data": data,
        "layout": go.Layout(
            width = 800,
            height = 500,
            xaxis=dict(title='Time (s)', linecolor = 'black',linewidth = 2, mirror = True),
            yaxis=dict(title='Height of front (mm)',linecolor = 'black',linewidth = 2, mirror = True),
            showlegend=True
        )}
    )

    plotly.offline.plot(fig, auto_open=True)

    if bSave:
        pio.write_image(fig, dirname + '/' + 'FrontHeight.pdf')


if __name__ == '__main__':

    #Setup parser
    ap = argparse.ArgumentParser()
    ap.add_argument("directory", nargs='?', default=os.getcwd(), help="Path of the directory")
    ap.add_argument("-r", "--reprocess", action='store_true', help="Force the reprocessing of the movies")
    ap.add_argument("-p", "--plotly", action='store_true', help="Use plotly instead of matplotlib for graphing")
    ap.add_argument("-s", "--save", action='store_true', help="Save the resulting plot")
    ap.add_argument("-c", "--scale", type=float, default=1, help="Scale factor in px/mm (default = 1)")
    ap.add_argument("-f", "--framerate", type=int, default=1, help="Frame rate (default = 1)")
    ap.add_argument("-a", "--autoprocess", action='store_true', help="Attempt to detect when the ice first appears")

    #Retrieve arguments
    args = ap.parse_args()

    dirname = args.directory
    bReprocess = args.reprocess
    bPlotly = args.plotly
    bSave = args.save
    scale = args.scale
    framerate = args.framerate
    bAuto = args.autoprocess

    if os.path.isfile(dirname) and dirname.endswith('.avi'):
        file_list=dirname
    elif os.path.isdir(dirname):
        file_list = [dirname + '/' + file for file in os.listdir(dirname) if file.endswith('.avi')]
    else:
        raise ValueError('Invalid file or directory.')

    if len(file_list) == 0:
        raise Exception('No video file in directory.')

    print("Files to process: " + str(file_list))
    
    #Save path for the processed dataframe
    savepath = dirname+"/"+"ProcessedData"+".pkl"

    #If the movies have been processed already, load from disk, otherwise process now
    if os.path.isfile(savepath) and not bReprocess:
        df = pd.read_pickle(savepath)
        print("Data loaded from disk")
    else:
        print("Processing movies")
        df_crop = videotools.obtain_cropping_boxes(file_list)
        #Make a list with file names and cropping boxes
        vid_list = []
        for file in file_list:
            name = file.split('.')[0].split('/')[-1]
            cropping_box = df_crop[df_crop['ExpName'] == name]['CroppingBox'].iloc[0]
            vid_list.append((file, cropping_box))
        #Run the movie analysis in parallel (one thread per movie)
        df_list = Parallel(n_jobs=-2, verbose=10)(delayed(process_movie)(file, box, scale, framerate, bAuto) for (file, box) in vid_list)
        #Merge all the dataframes in one and reindex
        df = pd.concat(df_list).reset_index(drop=True)
        #Save dataframe to disk
        df.to_pickle(savepath)

    if bPlotly:
        plot_front_position_pltly(df, bSave, dirname)
    else:
        plot_front_position(df, bSave, dirname)


