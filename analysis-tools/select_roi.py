from matplotlib.widgets import RectangleSelector
from matplotlib import pyplot as plt

#Testing imports
from skimage import io, exposure, img_as_float, img_as_ubyte, morphology, filters, util
from skimage.color import rgb2gray, label2rgb


class RectangleSelection(object):
    def __init__(self, img):
        self.rectangle = None
        self.img = img
        self.done = False

        #Setup the figure
        self.fig, self.ax = plt.subplots()
        plt.imshow(self.img, cmap='gray')

        self.RS = RectangleSelector(self.ax, self.onselect,
                                       drawtype='box', useblit=True,
                                       button=[1, 3],  # don't use middle button
                                       minspanx=5, minspany=5,
                                       spancoords='pixels',
                                       interactive=True)

        plt.connect('key_press_event', self.toggle_selector)
        plt.show()

    def onselect(self, e_click, e_release):
        minRow = int(min(e_click.ydata, e_release.ydata))
        minCol = int(min(e_click.xdata, e_release.xdata))
        maxRow = int(max(e_click.ydata, e_release.ydata))
        maxCol = int(max(e_click.xdata, e_release.xdata))
        self.rectangle = (minRow, minCol, maxRow, maxCol)

    def toggle_selector(self, event):
        if event.key in ['Q', 'q'] and self.RS.active:
            self.RS.set_active(False)
        if event.key in ['A', 'a'] and not self.RS.active:
            self.RS.set_active(True)



def select_rectangle(img):
    ''' Create a window requesting a rectangular selection on the passed image

    Support multilple redos, press q to exit
    '''

    selector = RectangleSelection(img)

    print('Select the region of interest then press Q/q to confirm selection and exit.')

    plt.close(selector.fig)

    return selector.rectangle

def select_multi_rectangle(img_list):
    ''' Returns a list containing the coordinates of the selected rectangle for each image

    List of tuple of int, coordinates of the selected rectangle (minRow, minCol, maxRow, maxCol)
    List has the same order as the incoming list
    '''

    #Create list to hold the coordinates
    rectangles = []

    #Determine the bounding boxes
    for img in img_list:
        rectangles.append(select_rectangle(img))
    
    return rectangles


if __name__ == '__main__':
    test_img = img_as_ubyte(rgb2gray(io.imread('analysis-tools/test_movies/test_image.jpg')))
    print('Coordinates of the selected rectangle: ' + str(select_rectangle(test_img)))