import pandas as pd
import numpy as np

from skimage import img_as_ubyte, exposure
from skimage.color import rgb2gray
from skimage.feature import peak_local_max
from skimage.filters import threshold_otsu
from skimage.segmentation import watershed
from skimage.measure import label, regionprops
from skimage.morphology import closing

from scipy import ndimage as ndi


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
    img_labeled = ndi.label(img_segmented)
    #Get the properties of the labeled regions and construct a dataframe
    reg = regionprops(img_labeled, coordinates='rc')
    columns= ['Frame','Label', 'Area', 'Eccentricity', 'Bbox Area']
    df = pd.DataFrame(columns=columns)
    df = df.append([{'Frame':frame, 'Label':i.label, 'Area':i.area/(scale**2), 'Eccentricity':i.eccentricity, 'Bbox Area':i.bbox_area/(scale**2)} for i in reg])
    return df


def plot_buble_area_hist(df, bSave = False, dirname = None):
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
        pio.write_image(fig, dirname + '/' + 'FrontHeight.pdf')