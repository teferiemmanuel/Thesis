
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
dense sift is an option for feature extraction. 
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

        dims = im.level_dimensions[RESOLUTION]
        # print(dims)
        # dims = dims[RESOLUTION]
        SLIDE_HEIGHT = dims[1]
        SLIDE_WIDTH = dims[0]
        img = im.read_region((0, 0), RESOLUTION, (SLIDE_WIDTH, SLIDE_HEIGHT))
        imgA = np.array(img)
        gray = cv.cvtColor(imgA, cv.COLOR_BGR2GRAY)

        ### Dense SIFT alternative feature to test ###
        # dense=cv2.FeatureDetector_create("Dense")
        # kp=dense.detect(imgGray)
        # kp,des=sift.compute(imgGray,kp)

        #Normal sift 
        kp, des = sift.detectAndCompute(gray,None)
        for d in des: 
            if des is not None:
                descripTotal.extend(des)
        index += 1
                
    # Centroids are codebook, tune hyperparam K
    K = 400
    kmeans = MiniBatchKMeans(n_clusters = K).fit(descripTotal)
    codebook = kmeans.cluster_centers_
    s = (2, len(codebook))
    histograms = np.zeros(s)

    # Get histograms, map all training points to the centroids.  
    for filepath in glob.iglob(PATH):
        im = openslide.OpenSlide(filepath)
        dims = im.level_dimensions[RESOLUTION]
        SLIDE_HEIGHT = dims[1]
        SLIDE_WIDTH = dims[0]
        filename = filepath[PATH_LEN:]
        slide_label = labels[filename]
        img = im.read_region((0, 0), RESOLUTION, (SLIDE_WIDTH, SLIDE_HEIGHT))
        imgA = np.array(img)
        gray = cv.cvtColor(imgA, cv.COLOR_BGR2GRAY)

        kp, des = sift.detectAndCompute(gray,None)
        
        ### Dense SIFT alternative feature to test ###
        # dense=cv2.FeatureDetector_create("Dense")
        # kp=dense.detect(imgGray)
        # kp,des=sift.compute(imgGray,kp)

        if des is not None:
            prediction = kmeans.predict(des)
            histograms[slide_label][prediction] += 1
 
    # Normalize histograms, each class row individually
    NUM_CLASS = 2
    for i in range(NUM_CLASS):
        histograms[i] = np.true_divide(histograms[i], np.sum(histograms[i]))
    return histograms, kmeans, codebook, labels, tr_order_to_label

'''
Can be used for validation step as well. Takes training outputs. 
Labels are 'HobI16-053768896760.svs': '1'


'''
def testNN(tr_histograms, kmeans, codebook, labels, tr_order_to_label):
    #scale of 0-3 resolution 0 is highest res, 3 is lowest
    RESOLUTION = 3
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
        index += 1∏


    #score = np.zeros((testSize, 2))
    dimension = (testSize, len(codebook))
    hist = np.zeros(dimension)
    histTest = np.zeros(dimension)

    #current image
    i = 0
    
    for filepath in glob.iglob(PATH):
        im = openslide.OpenSlide(filepath)
        dims = im.level_dimensions[RESOLUTION]
        SLIDE_HEIGHT = dims[1]
        SLIDE_WIDTH = dims[0]
        filename = filepath[PATH_LEN:]
        desCount = 0
        slide_label = labels[filename]
        img = im.read_region((0, 0), RESOLUTION, (SLIDE_WIDTH, SLIDE_HEIGHT))
        imgA = np.array(img)
        gray = cv.cvtColor(imgA, cv.COLOR_BGR2GRAY)
   
        ### Dense SIFT alternative feature to test ###
        # dense=cv2.FeatureDetector_create("Dense")
        # kp=dense.detect(imgGray)
        # kp,des=sift.compute(imgGray,kp)

        # Normal sift 
        kp, des = sift.detectAndCompute(gray,None)

        # make histogram for the test set. Question: How sparse are the histograms?
        if des is not None:
            for d in des:
                d = np.reshape(d, (1, 128))
                predicted = kmeans.predict(d)
                histogramTest[i][predicted] += 1
                desCount += 1
            hist[i] = np.true_divide(histogramTest[i], desCount)
        i += 1

    # change between NN, SVM, and Logit
    # test_predictions is testSize amount of predictions 
    # hist is test histogram 
    test_predictions_NN = nearest_neighbor(tr_histograms, hist, testSize, tr_order_to_label, labels)
    test_predictions_SVM = svm(tr_histograms, hist, testSize, tr_order_to_label, labels)
    test_predictions_logit = logistic(tr_histograms, hist, testSize, tr_order_to_label, labels)
    
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
