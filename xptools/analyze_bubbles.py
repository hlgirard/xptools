import pandas as pd
import numpy as np

from skimage import io, img_as_ubyte, exposure
from skimage.color import rgb2gray
from skimage.feature import peak_local_max
from skimage.filters import threshold_otsu
from skimage.segmentation import watershed
from skimage.measure import label, regionprops
from skimage.morphology import closing

from scipy import ndimage as ndi

# Package imports
from xptools.utils import videotools

# System imports
import os
import argparse
from joblib import Parallel, delayed
import tqdm

import matplotlib.pyplot as plt


def analyze_bubbles(img, scale = 1, frame = 0):
    """
    Process an image to detect the bubbles (bright on dark assumed)
    
    Parameters
    ----------
    img : np.array
        image to process
    scale : float, optional
        Scaling factor in px/mm. Default is 1 px/mm
    frame : int, optional
        Frame number of the image for inclusion in the dataframe. Default is 0
    Returns
    -------
    DataFrame {'Frame', Label', 'Area', 'Eccentricity', 'Bbox Area'}
        Frame (int): Frame number of the image
        Label (int): Identifier for each bubble
        Area (int): area of the detected bubble (mm2)
        Eccentricity (float): eccentricity of the region
        Bbox Area (int): area of the bounding box (mm2)
    """

    #Convert to grayscale
    img_gray = img_as_ubyte(rgb2gray(img))
    #Threshold the image (otsu) to get the bubbles
    img_bin = img_gray > threshold_otsu(img_gray)
    #Close holes in bubbles
    img_closed = closing(img_bin)
    #Compute distance to background
    dist = ndi.distance_transform_edt(img_closed)
    #Stretch contrast of the distance image
    dist_cont = exposure.rescale_intensity(dist, in_range='image')
    #Find local maximas in distance image and make them markers for the watershed
    local_maxi = peak_local_max(dist_cont, indices=False, footprint=np.ones((10, 10)))
    markers, num_max = ndi.label(local_maxi)
    #Run a watershed algorithm to separate the bubbles
    img_segmented = watershed(-dist_cont, markers, mask = img_closed, watershed_line=True)
    #Label the segmented image
    img_labeled, num_sec = ndi.label(img_segmented)
    #Get the properties of the labeled regions and construct a dataframe
    reg = regionprops(img_labeled, coordinates='rc')
    if len(reg) > 0:
        columns= ['Frame','Label', 'Area', 'Eccentricity', 'Bbox Area']
        df = pd.DataFrame(columns=columns, dtype = np.float64)
        df = df.append([{'Frame':frame, 'Label':i.label, 'Area':i.area/(scale**2), 'Eccentricity':i.eccentricity, 'Bbox Area':i.bbox_area/(scale**2)} for i in reg])
        return df
    else:
        return None

def process_movie(file, crop_box = None, scale = 1, framerate = 1):
    """
    Process a video to determine the bubble distribution.
    
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
    Returns
    -------
    DataFrame {'Frame', Label', 'Area', 'Eccentricity', 'Bbox Area'}
        Frame (int): Frame number of the image
        Label (int): Identifier for each bubble
        Area (int): area of the detected bubble (mm2)
        Eccentricity (float): eccentricity of the region
        Bbox Area (int): area of the bounding box (mm2)
    """
    #Open the file and get a stack of grayscale images
    stack = videotools.open_video(file)
    if crop_box != None:
        #Select the region of interest
        (minRow, minCol, maxRow, maxCol) = crop_box
        #Crop the stack to the right dimensions
        stack = [img[minRow:maxRow,minCol:maxCol] for img in stack]
    #Create a dataframe to hold the data
    df = pd.DataFrame(dtype=np.float64)
    #Analyze pics and drop data in dataframe
    for i in tqdm(range(len(stack))):
        result = analyze_bubbles(stack[i], scale, i)
        if result is not None:
            df = df.append(result, ignore_index = True)
    #Determine time for each frame
    df['Time'] = df['Frame']/framerate
    #Return the dataframe
    return df

def plot_bubble_area_hist(df, bSave = False, dirname = None):
    """Plots a histogram of the area of the bubbles
    """

    import plotly
    import plotly.graph_objs as go
    import plotly.io as pio
    
    data=[]

    for i in range(len(df['Frame'].unique())):
        data.append(go.Histogram(
            x = df[df['Frame'] == df['Frame'].unique()[i]]['Area'],
            name = df['Frame'].unique()[i]
        ))

    fig = go.Figure({
        "data": data,
        "layout": go.Layout(
            width = 800,
            height = 500,
            xaxis=dict(title='Area (mm2)', linecolor = 'black',linewidth = 2, mirror = True),
            showlegend=True
        )}
    )

    plotly.offline.plot(fig, auto_open=True)

    if bSave:
        pio.write_image(fig, dirname + '/' + 'BubbleHist.pdf')

def plot_bubble_area_dist(df, bSave = False, dirname = None):
    """Plots the distribution of bubble size for each frame
    TODO: implement support for simultaneous plotting of different experiments
    """

    import plotly
    from plotly import tools
    import plotly.graph_objs as go
    import plotly.io as pio

    #Everything is plotted for each frame of the video
    gf = df.groupby('Frame')

    # Functions return the first and third quartile
    def q1(x): return x.quantile(0.25)
    def q3(x): return x.quantile(0.75)

    #Plot the distribution of bubble size as a function of time

    trace_q1 = go.Scatter(
        x = gf.Time.agg('median'),
        y = gf.Area.agg(q1),
        mode = 'lines',
        line = dict(width= 0),
        xaxis='x2',
        yaxis='y2',
        name = 'Q1'
    )

    trace_q3 = go.Scatter(
        x = gf.Time.agg('median'),
        y = gf.Area.agg(q3),
        fill = 'tonexty',
        mode = 'none',
        xaxis='x2',
        yaxis='y2',
        name = 'Q3'
    )

    trace_mean = go.Scatter(
        x = gf.Time.agg('median'),
        y = gf.Area.agg('median'),
        xaxis='x2',
        yaxis='y2',
        name = 'mean'
    )
    
    #Plot the number of bubbles detected as a function of time
    trace_num = go.Scatter(
        x = gf.Time.agg('median'),
        y = gf.size(),
        name = 'Number'
    )

    fig = tools.make_subplots(rows=2, cols=1)

    fig.append_trace(trace_q1, 2, 1)
    fig.append_trace(trace_q3, 2, 1)
    fig.append_trace(trace_mean, 2, 1)
    fig.append_trace(trace_num, 1, 1)

    fig['layout'].update(
        width = 800,
        height = 800,
        showlegend=False)

    fig['layout']['yaxis1'].update(title='Number', range=(0,1500), linecolor = 'black',linewidth = 2, mirror = True)
    fig['layout']['yaxis2'].update(title='Area', range=(0,200), linecolor = 'black',linewidth = 2, mirror = True, anchor='x2')
    fig['layout']['xaxis1'].update(title='Time', linecolor = 'black',linewidth = 2, mirror = True)
    fig['layout']['xaxis2'].update(title='Time', linecolor = 'black',linewidth = 2, mirror = True)

    plotly.offline.plot(fig, auto_open=True)

    if bSave:
        pio.write_image(fig, dirname + '/' + 'BubbleDist.pdf')

def main():
    #Setup parser
    ap = argparse.ArgumentParser()
    ap.add_argument("directory", nargs='?', default=os.getcwd(), help="Path of the directory")
    ap.add_argument("-r", "--reprocess", action='store_true', help="Force the reprocessing of the images")
    ap.add_argument("-p", "--plotly", action='store_true', help="Use plotly instead of matplotlib for graphing")
    ap.add_argument("-s", "--save", action='store_true', help="Save the resulting plot")
    ap.add_argument("-f", "--framerate", type=int, default=1, help="Frame rate (default = 1)")
    ap.add_argument("-c", "--scale", type=float, default=1, help="Scale factor in px/mm (default = 1)")

    #Retrieve arguments
    args = ap.parse_args()

    dirname = args.directory
    bReprocess = args.reprocess
    bPlotly = args.plotly
    framerate = args.framerate
    bSave = args.save
    scale = args.scale

    # File or directory?
    isFile = False

    if os.path.isfile(dirname) and (dirname.endswith('.avi') or dirname.endswith('.AVI')):
        file_list=[os.path.abspath(dirname)]
        isFile = True
    elif os.path.isdir(dirname):
        file_list = [dirname + '/' + file for file in os.listdir(dirname) if (dirname.endswith('.avi') or dirname.endswith('.AVI'))]
    else:
        raise ValueError('Invalid file or directory.')

    if len(file_list) == 0:
        raise Exception('No video file in directory.')

    print("Files to process: " + str(file_list))
    
    #Save path for the processed dataframe
    if isFile:
        savedir = os.path.abspath(os.path.dirname(dirname))
        filename = dirname.split('.')[0].split('/')[-1]
        savepath = savedir+"/"+"ProcessedData_"+filename+".pkl"
    else:
        savepath = dirname+"/"+"ProcessedData"+".pkl"

    #If the movies have been processed already, load from disk, otherwise process now
    if os.path.isfile(savepath) and not bReprocess:
        df = pd.read_pickle(savepath)
        print("Data loaded from disk")
    else:
        print("Processing movies")
        dict_crop = videotools.obtain_cropping_boxes(file_list)
        #Make a list with file names and cropping boxes
        vid_list = []
        for file in file_list:
            name = file.split('.')[0].split('/')[-1]
            cropping_box = dict_crop[name]
            vid_list.append((file, cropping_box))
        #Run the movie analysis in parallel (one thread per movie)
        #df_list = Parallel(n_jobs=-2, verbose=10)(delayed(process_movie)(file, box, scale, framerate) for (file, box) in vid_list)
        df_list = []
        for (file, box) in vid_list:
            df_list.append(process_movie(file, box, scale, framerate))
        #Merge all the dataframes in one and reindex
        df = pd.concat(df_list).reset_index(drop=True)
        #Save dataframe to disk
        df.to_pickle(savepath)

    plot_bubble_area_dist(df, bSave, dirname)



if __name__ == '__main__':
    main()