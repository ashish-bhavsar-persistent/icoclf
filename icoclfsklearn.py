import cv2
import numpy as np
from sklearn import svm
from sklearn.externals import joblib

# =============================================================================
# import matplotlib.pyplot as plt
# from skimage.feature import hog
# from skimage import exposure
# =============================================================================
#from common import mosaic

def load_dataset(n, fn):
    dataset = []
    for i in range(n):
        img = cv2.resize(cv2.imread('msg{}/{}{}.png'.format(fn, fn, i+1),0),(100,100))
        if fn == 'trainset':
            flip_img = cv2.flip(img, 1)
            dataset.append(flip_img)
        dataset.append(img)
    dataset = np.array(dataset).reshape(-1,100,100)
    return dataset

def get_hog() : 
    winSize = (20,20)
    blockSize = (10,10)
    blockStride = (5,5)
    cellSize = (10,10)
    nbins = 9
    derivAperture = 1
    winSigma = -1.
    histogramNormType = 0
    L2HysThreshold = 0.2
    gammaCorrection = 1
    nlevels = 64
    signedGradient = True

    hog = cv2.HOGDescriptor(winSize,blockSize,blockStride,cellSize,nbins,derivAperture,winSigma,histogramNormType,L2HysThreshold,gammaCorrection,nlevels, signedGradient)

    return hog

if __name__ is '__main__':

    print('Loading dataset ... ')
    # Load data.
    icons_train = load_dataset(100, 'trainset')
   
    labels = np.array([1] * 30 + [2] * 24 + [3] * 14 + [4] * 32 + [0] * 100)
    
    print('Defining HoG parameters ...')
    # HoG feature descriptor
    hog = get_hog();
    
    print('Calculating HoG descriptor for every icon ... ')
    hog_descriptors_train = []
    
    for ico in icons_train:
        hog_descriptors_train.append(hog.compute(ico))
    hog_descriptors_train = np.squeeze(hog_descriptors_train)
    
    print('Training SVM model ...')
    model = svm.SVC(kernel='rbf', C=12.5, gamma=0.0005)
    model.fit(hog_descriptors_train, labels)
                
    print('Saving SVM model ...')
    joblib.dump(model, 'icoclf_svm.sav')
# =============================================================================
#     plt.imshow(vis, cmap='gray')
#     plt.show()
# =============================================================================
    print('Model is trained!!!')
