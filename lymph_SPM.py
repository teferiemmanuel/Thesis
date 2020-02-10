import glob
import openslide
import cv2 as cv
import csv
import numpy as np
from PIL import Image
from skimage import color
from sklearn.cluster import KMeans
from sklearn.cluster import MiniBatchKMeans
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression


'''
spatial pyramid matching 
'''
def spm(im, res): 
    # flatten the representation of histograms (2 features), for levels 0-2 = 42
    # consider testing levels 0-n, paper says n > 2 isn't significant, but these images are
    # higher res? See equation 3 in SPM intro paper
    # print(dims)
    hist_boxes = 42
    s = (1, hist_boxes)
    histograms = np.zeros(s)


    dims = im.level_dimensions[res]
    SLIDE_HEIGHT = dims[1]
    SLIDE_WIDTH = dims[0]


    # LEVEL 0
    img = im.read_region((0, 0), res, (SLIDE_WIDTH, SLIDE_HEIGHT))
    imgA = np.array(img)
    gray = cv.cvtColor(imgA, cv.COLOR_BGR2GRAY)
    
    sift = cv.xfeatures2d.SIFT_create()
    dense = cv2.FeatureDetector_create("Dense")
    kp_sift = sift.detect(gray,None)
    kp_dense = dense.detect(gray, None)

    histograms[0] = len(kp_sift) // 4
    histograms[1] = len(kp_dense) // 4

    # LEVEL 1
    W_STRIDE = SLIDE_WIDTH // 2
    H_STRIDE = SLIDE_HEIGHT // 2
    # h and w are the top left pixel
    h = 0
    w = 0
    index = 2
    while h < SLIDE_HEIGHT:
        while w < SLIDE_WIDTH:
            img = im.read_region((w, h), 0, (W_STRIDE, H_STRIDE))
            imgA = np.array(img)
            gray = cv.cvtColor(imgA, cv.COLOR_BGR2GRAY)
            kp_sift = sift.detect(gray,None)
            kp_dense = dense.detect(gray, None)
            histograms[index] = len(kp_sift) // 4
            histograms[index] = len(kp_dense) // 4
            index += 1
            # Slide the window right 
            w += W_STRIDE
        # Go back to left end, then slide down one
        w = 0
        h += H_STRIDE

    # LEVEL 2
    W_STRIDE = SLIDE_WIDTH // 4
    H_STRIDE = SLIDE_HEIGHT // 4
    # h and w are the top left pixel
    h = 0
    w = 0
    while h < SLIDE_HEIGHT:
        while w < SLIDE_WIDTH:
            img = im.read_region((w, h), 0, (W_STRIDE, H_STRIDE))
            imgA = np.array(img)
            gray = cv.cvtColor(imgA, cv.COLOR_BGR2GRAY)
            kp_sift = sift.detect(gray,None)
            kp_dense = dense.detect(gray, None)
            histograms[index] = len(kp_sift) // 2
            histograms[index] = len(kp_dense) // 2
            index += 1
            # Slide the window right 
            w += W_STRIDE
        # Go back to left end, then slide down one
        w = 0
        h += H_STRIDE

    return histograms




'''

'''
def train(): 
    # dimensions proportional to lymph slide's (3.45 : 1)
    sift = cv.xfeatures2d.SIFT_create()
    descripTotal = []
    RESOLUTION = 2
    #PATH_LEN = 28
    # Local PATH, 
    # PATH = "/Volumes/Datasets/training/*.svs"
    PATH = "/n/fs/visualai-scr/emmanuel/lymph-130/training/*.svs"
    PATH_LEN = 48

    # saving labels step, won't work on alternate dataset. 
    label_reader = csv.DictReader(open("target.csv"))
    labels = {}
    for row in label_reader:
        labels[row['slide']] = row['target']
    
    # have  dict relating order in filepath iteration in training to slide name fioepath parsed.
    tr_order_to_label = {}
    index = 0
       # get all SIFT features
    for filepath in glob.iglob(PATH):
        im = openslide.OpenSlide(filepath)
        filename = filepath[PATH_LEN:]
        tr_order_to_label[index] = filename

        hist = spm(im, RESOLUTION)

