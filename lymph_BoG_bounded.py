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
# HYPERPARAMS:
# Scale of 0-3 resolution 0 is highest res, 3 is lowest
RESOLUTION = 3
K = 400



def sift_n(*args, **kwargs):
    try:
        return cv.xfeatures2d.SIFT_create(*args, **kwargs)
    except:
        return cv.SIFT()

'''pass in black and white  np array '''
def dsift(img, step=5):
    keypoints = [
        cv.KeyPoint(x, y, step)
        for y in range(0, img.shape[0], step)
        for x in range(0, img.shape[1], step)
    ]
    features = sift_n().compute(img, keypoints)[1]
    print("features:")
    print(features)
    print(len(features))
    print(len(features[0]))
    features /= features.sum(axis=1).reshape(-1, 1)
    return keypoints, features



def train():
    print("TRAIN entered")
    # dimensions proportional to lymph slide's (3.45 : 1)
    sift = cv.xfeatures2d.SIFT_create()
    #dense= cv.FeatureDetector_create('Dense')

    descripTotal = []
    # Local PATH, 
    PATH_LEN = 27
    PATH = "/Volumes/Datasets/training/*.svs"
    #PATH = "/n/fs/visualai-scr/emmanuel/lymph-130/training/*.svs"
    #PATH_LEN = 48

    # saving labels step, won't work on alternate dataset. 
    label_reader = csv.DictReader(open("target.csv"))
    labels = {}
    for row in label_reader:
        labels[row['slide']] = row['target']
    #print("labels made!")
    # have  dict relating order in filepath iteration in training to slide name (filepath parsed)
    tr_order_to_label = {}
    # dict relating file name with x bounds. 
    #filename_to_bounds ={}
    # index represents a single lymph or slide. 
    index = 0
    sift_counts = []
    sift_sum = 0
    # get al files, get SIFT features
    for filepath in glob.iglob(PATH):
        im = openslide.OpenSlide(filepath)
        filename = filepath[PATH_LEN:]
        #skip images that don't have this resolution.
        if im.level_count < RESOLUTION + 1:
            continue
        dims = im.level_dimensions[RESOLUTION]
        # print(dims)
        # dims = dims[RESOLUTION]
        SLIDE_HEIGHT = dims[1]
        SLIDE_WIDTH = dims[0]
        img = im.read_region((0, 0), RESOLUTION, (SLIDE_WIDTH, SLIDE_HEIGHT))
        imgA = np.asarray(img)
        gray = cv.cvtColor(imgA, cv.COLOR_BGR2GRAY)
        kp = sift.detect(gray,None)

        BIN_FACTOR = int(200 * 1/RESOLUTION)
        sift_hist_x = np.zeros(SLIDE_WIDTH//BIN_FACTOR + 1, dtype=int)
        # bin the following:
        for k in range(len(kp)):
            x_bin = int(kp[k].pt[0] // BIN_FACTOR)
            sift_hist_x[x_bin] += 1    
        #print(sift_hist_x)

        '''
        Find Node bounds in x direction. 
        '''
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
        lymph_ends_y =[]
        for b in range(len(lymph_ends_x)):
            # print(b)
            # if too small, ignore
            if lymph_ends_x[b][1] - lymph_ends_x[b][0] < 200:
                #lymph_ends_x.pop(b)
                continue
            

            img = im.read_region((lymph_ends_x[b][0], 100), RESOLUTION, ((lymph_ends_x[b][1] - lymph_ends_x[b][0]) + 100, SLIDE_HEIGHT - 110))
            #print(lymph_ends[b][0])
            #print(img.width)
            sift_hist_y = np.zeros(SLIDE_HEIGHT//BIN_FACTOR + 1, dtype=int)
            '''
            Calculate the x bounded lymph SIFT features.   
            '''
            imgA = np.asarray(img)
            gray = cv.cvtColor(imgA, cv.COLOR_BGR2GRAY)
            kp = sift.detect(gray,None)
            i = 0 
            j = 0
            '''
            Bin the keypoints into y histogram
            '''
            #x_counter = 0
            i = 0
            BIN_FACTOR = int(200 * 1/RESOLUTION)
            sift_hist_y = np.zeros(SLIDE_HEIGHT//BIN_FACTOR + 1, dtype=int)
            # bin the following:
            for k in range(len(kp)):
                y_bin = int(kp[k].pt[1] // BIN_FACTOR)
                sift_hist_y[y_bin] += 1   
            # Make threshold proportional to number of siftfeats. 
            THRESHOLD = np.sum(sift_hist_y)/(len(sift_hist_y) * 4)
            n = len(sift_hist_y)
            #lymph_ends_y = [] 
            '''
            Find the y-bounds to the lymph nodes. 
            '''
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
        '''
        parse and analyse within those bounds. 
        index is single lymph 
        '''
        for b in range(len(lymph_ends_x)):
            if lymph_ends_x[b][1] - lymph_ends_x[b][0] < 200:
                #lymph_ends_x.pop(b)
                continue
            #print("YLymph:")
            #print(lymph_ends_y)
            #print("XLymph")
            #print(lymph_ends_x)
            if len(lymph_ends_y) <= b:
                lymph_ends_y.append((150, SLIDE_HEIGHT - 150))
                # print("YLymph: ")
                # print(lymph_ends_y)
            
            tr_order_to_label[index] = filename

            #filename_to_bounds[filename] = (lymph_ends_x[b][0], lymph_ends_x[b][1], lymph_ends_y[b][0], lymph_ends_y[b][1])
            try:
                top_point = (lymph_ends_x[b][0], lymph_ends_y[b][0])
                x_delta   = lymph_ends_x[b][1] - lymph_ends_x[b][0]
                y_delta   = lymph_ends_y[b][1] - lymph_ends_y[b][0]
                img = im.read_region(top_point, RESOLUTION, (x_delta, y_delta))
            except:
                print("b: ")
                print(b)
                continue
            #img.show()
            imgA = np.array(img)
            gray = cv.cvtColor(imgA, cv.COLOR_BGR2GRAY)
            #Normal sift 
            #DENSE! DENSE
            kp, des = dsift(gray)
            #kp, des = sift.detectAndCompute(gray,None)
            if not None:
                s = len(des)
                #starting from index 1, make sift_counts cumulative sums.
                if index != 0:
                    s = sift_counts[index - 1] + s
                sift_counts.append(s)
                descripTotal.extend(des)
                index += 1
    x = np.nan_to_num(np.array(descripTotal))
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
    '''
    look into this bottom part, does tr_order_to_label need to be changed because of switch to lymph level.
    '''
    # print("labels")
    # print(labels)
    # print("-----------")
    # print("tr_order_to_label")
    # print(tr_order_to_label)

    for row in range(len(x)):
        #print labels, tr_order, slide_marker.
        #print(row)
        diagnosis = int(labels[tr_order_to_label[slide_marker]])
        # diagnosis is the class, predicion is the codebook of centroid's prediction, row is the particulr descriptor. 
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
    PATH_LEN = 29
    PATH = "/Volumes/Datasets/validation/*.svs"
    sift = cv.xfeatures2d.SIFT_create()
    # have dict relating order in filepath iteration in training to slide name fioepath parsed.
    test_order_to_label = {}
    index = 0
    testSize = 0 
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
        kp = sift.detect(gray,None)
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
            kp  = sift.detect(gray,None)

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
        #Lymph bounds in all 4 directions has been found and attached to the filename.
        #**************************************
    # test size counts the number of lymph we are dealing with
    dimension = (len(bounded_lymph), len(codebook))
    histTest = np.zeros(dimension)
    hist = np.zeros(dimension)
    # print("Bounded lymph nodes:")
    # print(len(bounded_lymph))
    # print(bounded_lymph)
    # print("-------------------")

    # lymph[0] ==> filename
    #lymph[1], lymph[2] ==> leftx, rightx
    #lymph[3], lymph[4] ==>  top y, bottom y
    lymph_index = 0
    #### for each file in filepath, this is not part of larger file loop
    pre_PATH = "/Volumes/Datasets/validation/"
    lymph_order_to_label = {}


    for lymph in bounded_lymph: 

        whole_path = pre_PATH + lymph[0]
        im = openslide.OpenSlide(whole_path)

        img = im.read_region((lymph[1], lymph[3]), RESOLUTION, (lymph[2] - lymph[1] + 100, lymph[4] - lymph[3]))
        #img.show()
        imgA = np.array(img)
        # print("array empty?")
        # print(np.sum(imgA))
        gray = cv.cvtColor(imgA, cv.COLOR_BGR2GRAY)
        #Normal sift 
        #kp, des = sift.detectAndCompute(gray,None)
        # DENSE! DENSE
        kp, des = dsift(gray)
        lymph_order_to_label[lymph_index] = lymph[0]
        # make histogram for the test set. Question: How sparse are the histograms?
        if des is not None:
            # print("len(des): AAAAA")
            print(len(des))
            for d in des:
                d = np.nan_to_num(np.reshape(d, (1, 128)))
                #predicted = which centroid is this feature closest to?
                predicted = kmeans.predict(d)
                # image i 
                histTest[lymph_index][predicted] += 1
                desCount += 1
                # if desCount % 100 == 0:
                #     print(desCount)
            hist[lymph_index] = np.nan_to_num(np.true_divide(histTest[lymph_index], desCount))
        lymph_index += 1
    # print("test hist len!!!!!!!!!!!!!!!!!!!!!:")
    # print(len(hist))
    return tr_histograms, hist, testSize, tr_order_to_label, labels, bounded_lymph, lymph_order_to_label


def classify(tr_histograms, hist, testSize, tr_order_to_label, labels, bounded_lymph, lymph_order_to_label):
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

    testSize = len(bounded_lymph)
    # NN performance on test set. 
    for i in range(testSize):
        # I think the problem is test_order_to_label isn't filled with all values. 
        #print("test_order_to_label")
        #print(test_order_to_label)
        truth = labels[lymph_order_to_label[i]]
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
            print("double check")
            print(type(pred))
            print(truth)
    
    # SVM performance on test set. 
    print("SVM")
    for i in range(testSize):
        name = lymph_order_to_label[i]
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
            print(type(pred))
            # print(truth)
            # print(pred)
            confusion_SVM["n-n"] += 1

    # logit performance on test set. 
    for i in range(testSize):
        truth = labels[lymph_order_to_label[i]]
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
    # print("in SVM training histogram:")
    # print(train_hist)
    # print("-------------")
    # print("in test histogram:")
    # print(len(test_hist))
    # print("-------------")
    model.fit(train_hist, [0, 1])
    prediction = model.predict(test_hist)

    return prediction



def logistic(train_hist, test_hist, testSize, tr_order_to_label, labels):
    model = LogisticRegression(random_state=0).fit(train_hist, [0, 1]) #should be 0,1
    prediction = model.predict(test_hist)
    # print("logit probabilities of test:")
    # print(model.predict_proba(test_hist))
    # print("in SVM training histogram:")
    # print(train_hist)
    # print("-------------")
    # print("in test histogram:")
    # print(test_hist)
    # print("-------------")
    return prediction

#  return each test image's nearest neighbor.
def nearest_neighbor(train_hist, histTest, testSize, tr_order_to_label, labels):
    nn = NearestNeighbors().fit(train_hist)
    prediction = nn.kneighbors(histTest, 1, return_distance=False)
    prediction = prediction.transpose().flatten()
    # print("NN prediction: ")
    # print(prediction.shape)
    return prediction


histograms, kmeans, codebook, labels, tr_order_to_label = train()
tr_histograms, hist, testSize, tr_order_to_label, labels,  bounded_lymph, lymph_order_to_label = test(histograms, kmeans, codebook, labels, tr_order_to_label)
print("K: ")
print(K)
print("Resolution: ")
print(RESOLUTION)
classify(tr_histograms, hist, testSize, tr_order_to_label, labels,  bounded_lymph, lymph_order_to_label)

