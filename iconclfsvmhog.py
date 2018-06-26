import cv2
import numpy as np
import matplotlib.pyplot as plt
# =============================================================================
# from skimage.feature import hog
# from skimage import exposure
# =============================================================================
from common import mosaic


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

class StatModel(object):
    def load(self, fn):
        self.model.load(fn)
    def save(self, fn):
        self.model.save(fn)

class SVM(StatModel):
    def __init__(self, C = 12.5, gamma = 0.00050625):
        self.model = cv2.ml.SVM_create()
        self.model.setGamma(gamma)
        self.model.setC(C)
        self.model.setKernel(cv2.ml.SVM_RBF)
        self.model.setType(cv2.ml.SVM_C_SVC)

    def train(self, samples, responses):
        self.model.train(samples, cv2.ml.ROW_SAMPLE, responses)

    def predict(self, samples):

        return self.model.predict(samples)[1].ravel()


def evaluate_model(model, icons, samples, labels):
    resp = model.predict(samples)
    err = (labels != resp).mean()
    print('Accuracy: %.2f %%' % ((1 - err)*100))

    confusion = np.zeros((10, 10), np.int32)
    for i, j in zip(labels, resp):
        confusion[int(i), int(j)] += 1
    print('confusion matrix:')
    print(confusion)

    vis = []
    for img, flag in zip(icons, resp == labels):
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        if not flag:
            img[...,:2] = 0
        
        vis.append(img)
    return mosaic(5, vis)

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
    icons_test = load_dataset(20, 'testset')
   
    labels = np.array([1] * 30 + [2] * 24 + [3] * 14 + [4] * 32 + [0] * 100)
    labels_test = np.array([1] * 3 + [2] * 4 + [3] * 2 + [4] * 4 + [0] * 7)
    
    print('Defining HoG parameters ...')
    # HoG feature descriptor
    hog = get_hog();
    
    print('Calculating HoG descriptor for every icon ... ')
    hog_descriptors_train = []
    hog_descriptors_test = []
    
    for ico in icons_train:
        hog_descriptors_train.append(hog.compute(ico))
    hog_descriptors_train = np.squeeze(hog_descriptors_train)
    
    for ico in icons_test:
        hog_descriptors_test.append(hog.compute(ico))
    hog_descriptors_test = np.squeeze(hog_descriptors_test)
    
    print('Training SVM model ...')
    model = SVM()
    model.train(hog_descriptors_train, labels)
                
    print('Saving SVM model ...')
    model.save('icoclf_svm.dat')
    
    print('Evaluating model ... ')
    vis = evaluate_model(model, icons_test, hog_descriptors_test, labels_test)
    cv2.imwrite("digits-classification.jpg",vis)
# =============================================================================
#     plt.imshow(vis, cmap='gray')
#     plt.show()
# =============================================================================
    cv2.imshow("Vis", vis)
    cv2.waitKey(0)
