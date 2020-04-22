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
from sklearn.neighbors import NearestNeighbors
from sklearn.naive_bayes import GaussianNB


#Hyperparams:
RESOLUTION = 2


def sift_n(*args, **kwargs):
    try:
        return cv.xfeatures2d.SIFT_create(*args, **kwargs)
    except:
        return cv.SIFT()

'''pass in black and white  np array '''
def dsift(img, step=5):
    keypoints = [
        cv.KeyPoint(x, y, step)
        for y in range(0, img.shape[0], step) # y, x line 28
        for x in range(0, img.shape[1], step)
    ]
    features = sift_n().compute(img, keypoints)[1]
    features /= features.sum(axis=1).reshape(-1, 1)
    #features = sift_n().compute(img, keypoints)[1]
    # print("features:")
    # print(features)
    # print(len(features))
    # print(len(features[0]))
    # features /= features.sum(axis=1).reshape(-1, 1)
    # print("this is keypoints:")
    # print(keypoints)
    return keypoints #, features


'''
Returns Histogram with applied kernel of image. 
'''
def spm(im, lymph): 
    # flatten the representation of histograms (2 features), for levels 0-2 = 42
    # consider testing levels 0-n, paper says n > 2 isn't significant, but these images are
    # higher res? See equation 3 in SPM intro paper
    # print(dims)
    hist_boxes = 42
    #s = (hist_boxes)
    histograms = np.zeros(hist_boxes)

    SLIDE_HEIGHT = lymph[4] - lymph[3]
    SLIDE_WIDTH = lymph[2] - lymph[1]
    # number of keypoints. 
    DSIFT_TOTAL = ((SLIDE_HEIGHT // 5) + 1) * ((SLIDE_HEIGHT // 5) + 1)

    # LEVEL 0, whole image: 
    img = im.read_region((lymph[1], lymph[3]), RESOLUTION, (SLIDE_WIDTH, SLIDE_HEIGHT))
    imgA = np.array(img)
    gray = cv.cvtColor(imgA, cv.COLOR_BGR2GRAY)
    
    sift = cv.xfeatures2d.SIFT_create()
    #dense = cv.FeatureDetector_create("Dense")
    
    kp_sift = sift.detect(gray,None)
    #kp_dense = dsift(gray) #dense.detect(gray, None)


    histograms[0] = len(kp_sift) // 4
    histograms[1] = DSIFT_TOTAL // 4

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
            print("imgA")
            print(imgA)
            print(np.sum(imgA))
            gray = cv.cvtColor(imgA, cv.COLOR_BGR2GRAY)
            print("greytype")
            print(type(gray))
            print(gray.shape)

            kp_sift = sift.detect(gray,None)
            #kp_dense = dsift(gray, None)
            histograms[index] = len(kp_sift) // 4
            #outermost division is for kernel, inner because one fourth of image
            histograms[index] = (DSIFT_TOTAL // 4) // 4
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
            #kp_dense = dsift(gray, None)
            histograms[index] = len(kp_sift) // 2
            histograms[index] = (DSIFT_TOTAL // 16) // 2
            index += 1
            # Slide the window right 
            w += W_STRIDE
        # Go back to left end, then slide down one
        w = 0
        h += H_STRIDE
    return histograms
'''
return spatial pyramids and and matching class labels. 
'''

def find_bounds_train():
    PATH_LEN = 27
    PATH = "/Volumes/Datasets/training/*.svs"
    sift = cv.xfeatures2d.SIFT_create()
    # saving labels step, won't work on alternate dataset. 
    label_reader = csv.DictReader(open("target.csv"))
    labels = {}
    for row in label_reader:
        labels[row['slide']] = row['target']
    # have dict relating order in filepath iteration in training to slide name fioepath parsed.
    test_order_to_label = {}
    index = 0
    bounded_lymph =[]
    # imcount current image
    im_count = 0
    # print("TESTing entered. ")
    for filepath in glob.iglob(PATH):
        #im current slide
        im = openslide.OpenSlide(filepath)
        #pass if res is unavailable
        if im.level_count < RESOLUTION + 1:
            continue
        # num of descriptors
        desCount = 0
        dims = im.level_dimensions[RESOLUTION]
        SLIDE_HEIGHT = dims[1]
        SLIDE_WIDTH = dims[0]
        filename = filepath[PATH_LEN:]
        slide_label = labels[filename]
        img = im.read_region((0, 0), RESOLUTION, (SLIDE_WIDTH, SLIDE_HEIGHT))
        imgA = np.array(img)
        gray = cv.cvtColor(imgA, cv.COLOR_BGR2GRAY)
        kp   = sift.detect(gray,None)
        BIN_FACTOR = int(200 * 1/RESOLUTION)
        sift_hist_x = np.zeros(SLIDE_WIDTH//BIN_FACTOR + 1, dtype=int)
        # create histogram in x direction showing presence of keypoints in bins
        for k in range(len(kp)):
            x_bin = int(kp[k].pt[0] // BIN_FACTOR)
            sift_hist_x[x_bin] += 1    
        # X histogram is made find X bounds
        #*********************************
        i = 0 
        j = 0
        THRESHOLD = np.sum(sift_hist_x)/(len(sift_hist_x) * 4)
        n = len(sift_hist_x)
        lymph_ends_x = [] 
        while i < n and j < n:
            count = sift_hist_x[i] 
            # scan up to find end of th
            if count >= THRESHOLD:
                start = i
                while sift_hist_x[j] >= THRESHOLD:
                    j += 1
                    if j == n:
                        break
                lymph_ends_x.append(((i+1) * BIN_FACTOR, (j+1) * BIN_FACTOR))
                i = j
            # finished parsing this lymph node. 
            i += 1
            j += 1
        # X bounds is found now find Y bounds, for each file, iterate through bounded lymph
        #******************************
        lymph_ends_y =[]
        for b in range(len(lymph_ends_x)):
            # if x distance is too small, ignore bounds it's noise
            if lymph_ends_x[b][1] - lymph_ends_x[b][0] < 200:
                continue
            # If there are too few y bounds, add more
            if len(lymph_ends_y) <= b:
                lymph_ends_y.append((150, SLIDE_HEIGHT - 150))
            # test_order_to_label relates this traversal order to names of files, which can in classifiy time be converted to diagnoisis
            # index is that traversal order
            test_order_to_label[index] = filename
            index += 1
            # parse out the x bounded region in the image
            img = im.read_region((lymph_ends_x[b][0], 100), RESOLUTION, ((lymph_ends_x[b][1] - lymph_ends_x[b][0]) + 100, SLIDE_HEIGHT - 110))
            # Calculate the x bounded region's lymph SIFT features.   
            imgA = np.asarray(img)
            gray = cv.cvtColor(imgA, cv.COLOR_BGR2GRAY)
            kp = sift.detect(gray,None)

            # Bin the keypoints into y histogram
            BIN_FACTOR = int(200 * 1/RESOLUTION)
            sift_hist_y = np.zeros(SLIDE_HEIGHT//BIN_FACTOR + 1, dtype=int)
            # Bin the following:
            for k in range(len(kp)):
                y_bin = int(kp[k].pt[1] // BIN_FACTOR)
                sift_hist_y[y_bin] += 1   
            # Make threshold proportional to number of sift feats. 
            THRESHOLD = np.sum(sift_hist_y)/(len(sift_hist_y) * 4)
            n = len(sift_hist_y)
            # Find the y-bounds to the lymph nodes. 
            # indices are necessary for the parsing in the y direction
            i = 0 
            j = 0
            champ = None 
            while i < n and j < n:
                count = sift_hist_y[i] 
                #print("A")
                # scan up to find end of th
                if count >= THRESHOLD:
                    start = i
                    while sift_hist_y[j] >= THRESHOLD:
                        j += 1
                        if j == n:
                            break
                    # champion to find largest y bound on multiple lymph x bounds
                    if champ is None:
                        champ = j - i
                        lymph_ends_y.append(((i+1) * BIN_FACTOR, (j+1) * BIN_FACTOR))
                    else:
                        if j - i > champ:
                            lymph_ends_y.pop()
                            champ = j - i
                            lymph_ends_y.append(((i+1) * BIN_FACTOR, (j+1) * BIN_FACTOR))
                    i = j
                # finished parsing this lymph node. 
                i += 1
                j += 1
            if lymph_ends_x[b][1] - lymph_ends_x[b][0] < 200:
                continue
            if len(lymph_ends_y) <= b:
                lymph_ends_y.append((150, SLIDE_HEIGHT - 150))
            bounded_lymph.append((filename, lymph_ends_x[b][0], lymph_ends_x[b][1], lymph_ends_y[b][0], lymph_ends_y[b][1]))
    return bounded_lymph
        #Lymph bounds in all 4 directions has been found and attached to the filename.
        #**************************************


def find_bounds_test():
    PATH_LEN = 29
    PATH = "/Volumes/Datasets/validation/*.svs"
    sift = cv.xfeatures2d.SIFT_create()
    # saving labels step, won't work on alternate dataset. 
    label_reader = csv.DictReader(open("target.csv"))
    labels = {}
    for row in label_reader:
        labels[row['slide']] = row['target']
    # have dict relating order in filepath iteration in training to slide name fioepath parsed.
    test_order_to_label = {}
    index = 0
    bounded_lymph =[]
    # imcount current image
    im_count = 0
    # print("TESTing entered. ")
    for filepath in glob.iglob(PATH):
        #im current slide
        im = openslide.OpenSlide(filepath)
        #pass if res is unavailable
        if im.level_count < RESOLUTION + 1:
            continue
        # num of descriptors
        desCount = 0
        dims = im.level_dimensions[RESOLUTION]
        SLIDE_HEIGHT = dims[1]
        SLIDE_WIDTH = dims[0]
        filename = filepath[PATH_LEN:]
        slide_label = labels[filename]
        img = im.read_region((0, 0), RESOLUTION, (SLIDE_WIDTH, SLIDE_HEIGHT))
        imgA = np.array(img)
        gray = cv.cvtColor(imgA, cv.COLOR_BGR2GRAY)
        kp   = sift.detect(gray,None)
        BIN_FACTOR = int(200 * 1/RESOLUTION)
        sift_hist_x = np.zeros(SLIDE_WIDTH//BIN_FACTOR + 1, dtype=int)
        # create histogram in x direction showing presence of keypoints in bins
        for k in range(len(kp)):
            x_bin = int(kp[k].pt[0] // BIN_FACTOR)
            sift_hist_x[x_bin] += 1    
        # X histogram is made find X bounds
        #*********************************
        i = 0 
        j = 0
        THRESHOLD = np.sum(sift_hist_x)/(len(sift_hist_x) * 4)
        n = len(sift_hist_x)
        lymph_ends_x = [] 
        while i < n and j < n:
            count = sift_hist_x[i] 
            # scan up to find end of th
            if count >= THRESHOLD:
                start = i
                while sift_hist_x[j] >= THRESHOLD:
                    j += 1
                    if j == n:
                        break
                lymph_ends_x.append(((i+1) * BIN_FACTOR, (j+1) * BIN_FACTOR))
                i = j
            # finished parsing this lymph node. 
            i += 1
            j += 1
        # X bounds is found now find Y bounds, for each file, iterate through bounded lymph
        #******************************
        lymph_ends_y =[]
        for b in range(len(lymph_ends_x)):
            # if x distance is too small, ignore bounds it's noise
            if lymph_ends_x[b][1] - lymph_ends_x[b][0] < 200:
                continue
            # If there are too few y bounds, add more
            if len(lymph_ends_y) <= b:
                lymph_ends_y.append((150, SLIDE_HEIGHT - 150))
            # test_order_to_label relates this traversal order to names of files, which can in classifiy time be converted to diagnoisis
            # index is that traversal order
            test_order_to_label[index] = filename
            index += 1
            # parse out the x bounded region in the image
            img = im.read_region((lymph_ends_x[b][0], 100), RESOLUTION, ((lymph_ends_x[b][1] - lymph_ends_x[b][0]) + 100, SLIDE_HEIGHT - 110))
            # Calculate the x bounded region's lymph SIFT features.   
            imgA = np.asarray(img)
            gray = cv.cvtColor(imgA, cv.COLOR_BGR2GRAY)
            kp = sift.detect(gray,None)

            # Bin the keypoints into y histogram
            BIN_FACTOR = int(200 * 1/RESOLUTION)
            sift_hist_y = np.zeros(SLIDE_HEIGHT//BIN_FACTOR + 1, dtype=int)
            # Bin the following:
            for k in range(len(kp)):
                y_bin = int(kp[k].pt[1] // BIN_FACTOR)
                sift_hist_y[y_bin] += 1   
            # Make threshold proportional to number of sift feats. 
            THRESHOLD = np.sum(sift_hist_y)/(len(sift_hist_y) * 4)
            n = len(sift_hist_y)
            # Find the y-bounds to the lymph nodes. 
            # indices are necessary for the parsing in the y direction
            i = 0 
            j = 0
            champ = None 
            while i < n and j < n:
                count = sift_hist_y[i] 
                #print("A")
                # scan up to find end of th
                if count >= THRESHOLD:
                    start = i
                    while sift_hist_y[j] >= THRESHOLD:
                        j += 1
                        if j == n:
                            break
                    # champion to find largest y bound on multiple lymph x bounds
                    if champ is None:
                        champ = j - i
                        lymph_ends_y.append(((i+1) * BIN_FACTOR, (j+1) * BIN_FACTOR))
                    else:
                        if j - i > champ:
                            lymph_ends_y.pop()
                            champ = j - i
                            lymph_ends_y.append(((i+1) * BIN_FACTOR, (j+1) * BIN_FACTOR))
                    i = j
                # finished parsing this lymph node. 
                i += 1
                j += 1
            if lymph_ends_x[b][1] - lymph_ends_x[b][0] < 200:
                continue
            if len(lymph_ends_y) <= b:
                lymph_ends_y.append((150, SLIDE_HEIGHT - 150))
            bounded_lymph.append((filename, lymph_ends_x[b][0], lymph_ends_x[b][1], lymph_ends_y[b][0], lymph_ends_y[b][1]))
    return bounded_lymph
        #Lymph bounds in all 4 directions has been found and attached to the filename.
        #**************************************

def train(): 
    # dimensions proportional to lymph slide's (3.45 : 1)
    sift = cv.xfeatures2d.SIFT_create()
    hist_total = []
    # Local PATH, 
    # PATH_LEN = 27
    # PATH = "/Volumes/Datasets/training/*.svs"
    # PATH = "/n/fs/visualai-scr/emmanuel/lymph-130/training/*.svs"
    # PATH_LEN = 48

    # saving labels step, won't work on alternate dataset. 
    label_reader = csv.DictReader(open("target.csv"))
    labels = {}
    for row in label_reader:
        labels[row['slide']] = row['target']
    # Have dict relating order in filepath iteration in training to slide name fioepath parsed.
    tr_order_to_label = {}
    index = 0
    diagnosis = []
    bounds = find_bounds_train()
    pre_PATH = "/Volumes/Datasets/training/"
    loading = 0

    #iterate through list
    for lymph in bounds:
        whole_path = pre_PATH + lymph[0]
        im = openslide.OpenSlide(whole_path)
        filename = lymph[0] #filepath[PATH_LEN:]
        tr_order_to_label[index] = filename
        diagnosis.append(labels[filename])
        h = spm(im, lymph)
        # normalize h:
        h_sum = np.sum(h)
        for i in range(len(h)):
            h[i] = h[i] / h_sum
        hist_total.append(h)
        index += 1
        print(loading)
        loading += 1
    return hist_total, diagnosis, labels


'''

'''
def test(hist_total_train, diagnosis_train, labels):
    #scale of 0-3 resolution 0 is highest res, 3 is lowest
    PATH_LEN = 24
    #PATH = "/Volumes/Datasets/test/*.svs"
    pre_PATH = "/Volumes/Datasets/validation/"
    #Path for VisualAI cluser:
    # PATH = "/n/fs/visualai-scr/emmanuel/lymph-130/test/*.svs"
    # PATH_LEN = 44
    # have dict relating order in filepath iteration in training to slide name fioepath parsed.
    bounds = find_bounds_test()
    #truths not needed.
    truths = []
    test_hists = []

    # iterate through list
    for lymph in bounds:
        whole_path = pre_PATH + lymph[0]
        im = openslide.OpenSlide(whole_path)
        filename = lymph[0] #filepath[PATH_LEN:]
        #tr_order_to_label[index] = filename
        truths.append(labels[filename])
        h = spm(im, lymph)
        # normalize h:
        h_sum = np.sum(h)
        for i in range(len(h)):
            h[i] = h[i] / h_sum
        test_hists.append(h)

    # Change between NN, SVM, and Logit
    # test_predictions is testSize amount of predictions 
    # hist is test histogram, normalize hist_total
    test_predictions_NN = nearest_neighbor(hist_total_train, diagnosis_train, test_hists)
    test_predictions_SVM = svm(hist_total_train, diagnosis_train, test_hists)
    test_predictions_logit = logistic(hist_total_train, diagnosis_train, test_hists)
    
    # confusion matrix, the key code is actual-predicted
    confusion_NN = {}
    confusion_NN["p-p"] = 0
    confusion_NN["p-n"] = 0
    confusion_NN["n-n"] = 0
    confusion_NN["n-p"] = 0

    confusion_SVM = {}
    confusion_SVM["p-p"] = 0
    confusion_SVM["p-n"] = 0
    confusion_SVM["n-n"] = 0
    confusion_SVM["n-p"] = 0

    confusion_logit = {}
    confusion_logit["p-p"] = 0
    confusion_logit["p-n"] = 0
    confusion_logit["n-n"] = 0
    confusion_logit["n-p"] = 0


    # NN performance on test set. 
    index = 0
    for b in bounds:
        truth = labels[b[0]]
        pred = str(test_predictions_NN[index])
        print("THIS IS PRED::: ")
        print(type(pred))
        index += 1 
        if truth == '1' and pred == '1':
            confusion_NN["p-p"] += 1
        elif truth == '1' and pred == '0':
            confusion_NN["p-n"] += 1
        elif truth == '0' and pred == '1':
            confusion_NN["n-p"] += 1
        else:
            print("double check")
            print(type(pred))
            confusion_NN["n-n"] += 1
    
    # SVM performance on test set. 
    index = 0
    for b in bounds:
        truth = labels[b[0]]
        pred = str(test_predictions_SVM[index])
        print("THIS IS PRED::: ")
        print(type(pred))
        index += 1
        if truth == '1' and pred == '1':
            confusion_SVM["p-p"] += 1
        elif truth == '1' and pred == '0':
            confusion_SVM["p-n"] += 1
        elif truth == '0' and pred == '1':
            confusion_SVM["n-p"] += 1
        else:
            print("double check")
            print(type(pred))
            confusion_SVM["n-n"] += 1

    # logit performance on test set. 
    index = 0
    for b in bounds:
        truth = labels[b[0]]
        pred = str(test_predictions_logit[index])
        print("THIS IS PRED::: ")
        print(type(pred))
        index += 1
        if truth == '1' and pred == '1':
            confusion_logit["p-p"] += 1
        elif truth == '1' and pred == '0':
            confusion_logit["p-n"] += 1
        elif truth == '0' and pred == '1':
            confusion_logit["n-p"] += 1
        else:
            print("double check")
            print(type(pred))
            confusion_logit["n-n"] += 1
        
    

    print("Gaussian Naive Bayes: ")
    print(confusion_NN) 
    print("SVM: (linear kernel)")
    print(confusion_SVM)
    print("Logistic Regression: ")
    print(confusion_logit)


# support vector machine to divide training space by hyperplane
#     test_predictions_SVM = svm(hist_total_train, diagnosis_train, test_hists)

def svm(train_hist, train_y, test_hists):
    # train_len = len(train_hist)
    # y = []
    # for t in range(train_len):
    #     y.append(labels[tr_order_to_label[t]])
    model = SVC(kernel='linear')
    model.fit(train_hist, train_y)
    classified = []
    #for i in range(len(test_hists)):
    prediction = model.predict(test_hists)
    #    classified.append(prediction)
    return prediction #.transpose().flatten() #classified



def logistic(train_hist, train_y, test_hists):
    # train_len = len(train_hist)
    # y = []
    # for t in range(train_len):
    #     y.append(labels[tr_order_to_label[t]])

    model = LogisticRegression(random_state=0).fit(train_hist, train_y)
    classified = []
    #for i in range(len(test_hists)):
    prediction = model.predict(test_hists)
    #    classified.append(prediction)
    return prediction #.transpose().flatten() #classified

#  return each test image's nearest neighbor.
def nearest_neighbor(train_hist, train_y, test_hists):
    #nn = NearestNeighbors().fit(train_hist) #, train_y)
    gnb = GaussianNB()
    y_pred = gnb.fit(train_hist, train_y).predict(test_hists)

    #prediction = nn.kneighbors(test_hists, 1, return_distance=False)
    #prediction = prediction.transpose().flatten()

    return y_pred


train_hist, diagnosis, labels = train()
test(train_hist, diagnosis, labels)



