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
spatial pyramid matching, consider adding 3d feature to histogram, check correctness
as well. 
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
return spatial pyramids and and matching class labels. 
'''
def train(): 
    # dimensions proportional to lymph slide's (3.45 : 1)
    sift = cv.xfeatures2d.SIFT_create()
    hist_total = []
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

    diagnosis = []
    # get all SIFT features
    for filepath in glob.iglob(PATH):
        im = openslide.OpenSlide(filepath)
        filename = filepath[PATH_LEN:]
        tr_order_to_label[index] = filename
        diagnosis.append(labels[filename])
        h = spm(im, RESOLUTION)
        hist_total.append(h)
        index += 1
    
    return hist_total, diagnosis


'''

'''

def test(hist_total, diagnosis):
    #scale of 0-3 resolution 0 is highest res, 3 is lowest
    RESOLUTION = 2
    #PATH_LEN = 24
    #PATH = "/Volumes/Datasets/test/*.svs"
    #Path for VisualAI cluser:
    PATH = "/n/fs/visualai-scr/emmanuel/lymph-130/test/*.svs"
    PATH_LEN = 44
    # have dict relating order in filepath iteration in training to slide name fioepath parsed.
    test_order_to_label = {}
    index = 0

    # get test size (should be 26 unless more images added)
    testSize = 0 
    for filepath in glob.iglob(PATH):
        filename = filepath[PATH_LEN:]
        test_order_to_label[index] = filename
        testSize += 1
        index += 1



    # Change between NN, SVM, and Logit
    # test_predictions is testSize amount of predictions 
    # hist is test histogram, normalize hist_total
    test_predictions_NN = nearest_neighbor(hist_total, diagnosis)
    test_predictions_SVM = svm(hist_total, diagnosis)
    test_predictions_logit = logistic(hist_total, diagnosis)
    
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
    for i in range(testSize):
        truth = label[test_order_to_label[i]]
        pred = test_predictions_NN[i]
        if truth == '1' and pred == '1':
            confusion_NN["p-p"] += 1
        elif truth == '1' and pred == '0':
            confusion_NN["p-n"] += 1
        elif truth == '0' and pred == '1':
            confusion_NN["n-p"] += 1
        else:
            confusion_NN["n-n"] += 1
    
    # SVM performance on test set. 
    for i in range(testSize):
        truth = label[test_order_to_label[i]]
        pred = test_predictions_SVM[i]
        if truth == '1' and pred == '1':
            confusion_SVM["p-p"] += 1
        elif truth == '1' and pred == '0':
            confusion_SVM["p-n"] += 1
        elif truth == '0' and pred == '1':
            confusion_SVM["n-p"] += 1
        else:
            confusion_SVM["n-n"] += 1

    # logit performance on test set. 
    for i in range(testSize):
        truth = label[test_order_to_label[i]]
        pred = test_predictions_logit[i]
        if truth == '1' and pred == '1':
            confusion_logit["p-p"] += 1
        elif truth == '1' and pred == '0':
            confusion_logit["p-n"] += 1
        elif truth == '0' and pred == '1':
            confusion_logit["n-p"] += 1
        else:
            confusion_logit["n-n"] += 1
    

    print("Nearest Neighbor: ")
    print(confusion_NN) 
    print("SVM: (linear kernel)")
    print(confusion_SVM)
    print("Logistic Regression: ")
    print(confusion_logit)


# support vector machine to divide training space by hyperplane
#
def svm(train_hist, test_hist, testSize, tr_order_to_label, labels):
    train_len = len(train_hist)
    y = []
    for t in range(train_len):
        y.append(labels[tr_order_to_label[t]])
    model = SVC(kernel='linear')
    model.fit(train_hist, y)
    classified = []
    for test in range(testSize):
        prediction = model.predict(test_hist[i])
        classified.append(prediction)
    return prediction



def logistic(train_hist, test_hist, testSize, tr_order_to_label, labels):
    train_len = len(train_hist)
    y = []
    for t in range(train_len):
        y.append(labels[tr_order_to_label[t]])

    model = LogisticRegression(random_state=0).fit(train_hist, y)
    classified = []
    for test in range(testSize):
        prediction = model.predict(test_hist[i])
        classified.append(prediction)
    return prediction

#  return each test image's nearest neighbor.
def nearest_neighbor(train_hist, test_hist, testSize, tr_order_to_label, labels):
    train_len = len(train_hist)
    dists = np.zeros((testSize, train_len))

    # for each test histogram row in , find proximity of each train histogram row. Chi-distance metric (changable)
    for test_element in range(testSize):
        for train_element in range(train_len):
            diff = (train_hist[test_element][train_element] - histTest[test_element][train_element]) ** 2
            temp = (train_hist[test_element][train_element] + histTest[test_element][train_element]) * 1.0
            if temp > 0.000001 and temp < -0.000001:
                diff /= temp
            sum += diff
            # distance of particular test element from each training element
            dists[test_element][train_element] = sum

    classified = np.empty(testSize) 
    # champion to find lowest distance in each row. Save that index j, in classified
    for i in range(testSize):
        minValue = dists[i][0]
        minIndex = 0
        for j in range(train_len):
            if classified[i] < minValue:
                minIndex = j
                minValue = dists[i][j]
        # go from index --> slidename --> classification label, of nearest neighbor        
        classified[i] = labels[tr_order_to_label[minIndex]]
    return classified


histograms, kmeans, codebook, labels, tr_order_to_label = train()
test(histograms, kmeans, codebook, labels, tr_order_to_label)



