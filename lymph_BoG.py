
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

'''
dense sift is an option for feature extraction. 
'''
def train():
    print("TRAIN entered")
    # dimensions proportional to lymph slide's (3.45 : 1)
    sift = cv.xfeatures2d.SIFT_create()
    #dense= cv.FeatureDetector_create('Dense')

    descripTotal = []
    RESOLUTION = 1
    PATH_LEN = 27
    # Local PATH, 
    PATH = "/Volumes/Datasets/training/*.svs"
    #PATH = "/n/fs/visualai-scr/emmanuel/lymph-130/training/*.svs"
    #PATH_LEN = 48

    # saving labels step, won't work on alternate dataset. 
    label_reader = csv.DictReader(open("target.csv"))
    labels = {}
    for row in label_reader:
        labels[row['slide']] = row['target']
    print("labels made!")
    # have  dict relating order in filepath iteration in training to slide name fioepath parsed.
    tr_order_to_label = {}
    index = 0
    sift_counts = []
    sift_sum = 0

    # get all SIFT features
    for filepath in glob.iglob(PATH):
        im = openslide.OpenSlide(filepath)
        filename = filepath[PATH_LEN:]
        tr_order_to_label[index] = filename
        if im.level_count < RESOLUTION + 1:
            continue
        
        print(index)
        dims = im.level_dimensions[RESOLUTION]
        # print(dims)
        # dims = dims[RESOLUTION]
        SLIDE_HEIGHT = dims[1]
        SLIDE_WIDTH = dims[0]
        img = im.read_region((0, 0), RESOLUTION, (SLIDE_WIDTH, SLIDE_HEIGHT))
        imgA = np.array(img)
        gray = cv.cvtColor(imgA, cv.COLOR_BGR2GRAY)

        ### Dense SIFT alternative feature to test ###
        # kp=dense.detect(imgGray)
        # kp,des=sift.compute(imgGray,kp)

        #Normal sift 
        kp, des = sift.detectAndCompute(gray,None)

        s = len(des)
        #starting from index 1, make sift_counts cumulative sums.
        if index != 0:
            s = sift_counts[index - 1] + s
        sift_counts.append(s)
        descripTotal.extend(des)
        index += 1
    # print("tr_order_to_labels:")
    # print(tr_order_to_label)
    # Centroids are codebook, tune hyperparam K
    K = 50
    x = np.array(descripTotal)
    # print("X value!!!!")
    # print(x)
    kmeans = MiniBatchKMeans(n_clusters = K).fit(x)
    codebook = kmeans.cluster_centers_
    s = (2, len(codebook))
    histograms = np.zeros(s)
    # print("SIFT counts? why only 1?")
    # print(sift_counts)
    # print("x len!!!")
   

    slide_marker = 0
    prediction = kmeans.predict(x)
    # print("x len, sum sift count:")
    # print(len(x))
    # print(np.sum(sift_counts))
    #x len and sum of sift counts are the same size
    for row in range(len(x)):
        diagnosis = int(labels[tr_order_to_label[slide_marker]])
        # print(slide_marker)
        histograms[diagnosis][prediction[row]] += 1
        # It is only a new slide if we pass the number of descriptors of the current slide 
        if sift_counts[slide_marker] == row:
            slide_marker += 1
            print("marker incremented.")
            print(slide_marker)
    #print("Pre-normalized histograms: ")
    # Normalize histograms, each class row individually
    print(histograms)
    NUM_CLASS = 2
    for i in range(NUM_CLASS):
        histograms[i] = np.true_divide(histograms[i], np.sum(histograms[i]))
    return histograms, kmeans, codebook, labels, tr_order_to_label

'''
Can be used for validation step as well. Takes training outputs. 
Labels are 'HobI16-053768896760.svs': '1'
'''
def test(tr_histograms, kmeans, codebook, labels, tr_order_to_label):
    #scale of 0-3 resolution 0 is highest res, 3 is lowest
    RESOLUTION = 1
    #test 23 below
    PATH_LEN = 29
    PATH = "/Volumes/Datasets/validation/*.svs"
    sift = cv.xfeatures2d.SIFT_create()

    #Path for VisualAI cluser:
    #PATH = "/n/fs/visualai-scr/emmanuel/lymph-130/test/*.svs"
    #PATH_LEN = 44
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


    #score = np.zeros((testSize, 2))
    dimension = (testSize, len(codebook))
    hist = np.zeros(dimension)
    histTest = np.zeros(dimension)

    #current image
    i = 0
    print("TESTing entered. ")
    for filepath in glob.iglob(PATH):
        im = openslide.OpenSlide(filepath)
        if im.level_count < RESOLUTION + 1:
            continue
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
        # dense=cv.FeatureDetector_create("Dense")
        # kp=dense.detect(imgGray)
        # kp,des=sift.compute(imgGray,kp)
        # print(histTest)
        # Normal sift 
        
        
        kp, des = sift.detectAndCompute(gray,None)
        
        
        
        # s = len(des)
        # sift_counts.append(s)
        # descripTotal.append(des)
        # index += 1
        # make histogram for the test set. Question: How sparse are the histograms?
        if des is not None:
            for d in des:
                d = np.reshape(d, (1, 128))
                #predicted = which centroid is this feature closest to?
                predicted = kmeans.predict(d)
                # image i 
                histTest[i][predicted] += 1
                desCount += 1
            hist[i] = np.true_divide(histTest[i], desCount)
        i += 1

    # change between NN, SVM, and Logit
    # test_predictions is testSize amount of predictions 
    # hist is test histogram 
    # print("train hist:")
    # print(tr_histograms)
    # print("test hist:")
    print(hist)
    return tr_histograms, hist, testSize, tr_order_to_label, labels, test_order_to_label


def classify(tr_histograms, hist, testSize, tr_order_to_label, labels, test_order_to_label):
    test_predictions_NN = nearest_neighbor(tr_histograms, hist, testSize, tr_order_to_label, labels)
    test_predictions_SVM = svm(tr_histograms, hist, testSize, tr_order_to_label, labels)
    test_predictions_logit = logistic(tr_histograms, hist, testSize, tr_order_to_label, labels)
    print("TESt results NN: ")
    print(test_predictions_NN)
    print("TESt results SVM: ")
    print(test_predictions_SVM)
    print("TESt results logit: ")
    print(test_predictions_logit)
    print(type(test_predictions_logit[0]))



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
        truth = labels[test_order_to_label[i]]
        pred = test_predictions_NN[i]
        if truth == '1' and pred == 1:
            confusion_NN["p-p"] += 1
        elif truth == '1' and pred == 0:
            confusion_NN["p-n"] += 1
            # print("false negative")
            # print(name)
        elif truth == '0' and pred == 1:
            confusion_NN["n-p"] += 1
            # print("false positive:")
            # print(name)
        else:
            confusion_NN["n-n"] += 1
            print(truth)
    
    # SVM performance on test set. 
    print("SVM")
    for i in range(testSize):
        name = test_order_to_label[i]
        truth = labels[name]
        pred = test_predictions_SVM[i]
        if truth == '1' and pred == 1:
            confusion_SVM["p-p"] += 1
        elif truth == '1' and pred == 0:
            confusion_SVM["p-n"] += 1
            print("false negative")
            print(name)
        elif truth == '0' and pred == 1:
            confusion_SVM["n-p"] += 1
            print("false positive:")
            print(name)
        else:
            # print(type(truth))
            # print(type(pred))
            # print(truth)
            # print(pred)
            confusion_SVM["n-n"] += 1

    # logit performance on test set. 
    for i in range(testSize):
        truth = labels[test_order_to_label[i]]
        pred = test_predictions_logit[i]
        if truth == '1' and pred == 1:
            confusion_logit["p-p"] += 1
        elif truth == '1' and pred == 0:
            confusion_logit["p-n"] += 1
        elif truth == '0' and pred == 1:
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
    model = SVC(kernel='linear')
    model.fit(train_hist, [0, 1])
    prediction = model.predict(test_hist)

    return prediction



def logistic(train_hist, test_hist, testSize, tr_order_to_label, labels):
    model = LogisticRegression(random_state=0).fit(train_hist, [0, 1])
    prediction = model.predict(test_hist)
    return prediction

#  return each test image's nearest neighbor.
def nearest_neighbor(train_hist, histTest, testSize, tr_order_to_label, labels):
    nn = NearestNeighbors().fit(train_hist)
    prediction = nn.kneighbors(histTest, 1, return_distance=False)
    prediction = prediction.transpose().flatten()
    print("NN prediction: ")
    print(prediction.shape)
    return prediction


# histograms, kmeans, codebook, labels, tr_order_to_label = train()
# test(histograms, kmeans, codebook, labels, tr_order_to_label)
