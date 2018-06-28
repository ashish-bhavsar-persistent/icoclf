import cv2
import numpy as np
import itertools as it
from sklearn.externals import joblib

def load_dataset(n, fn):
    dataset = []
    for i in range(n):
        img = cv2.resize(cv2.imread('msg{}/{}{}.png'.format(fn, fn, i+1),0),
                         (100,100))
        if fn == 'trainset':
            flip_img = cv2.flip(img, 1)
            dataset.append(flip_img)
        dataset.append(img)
    dataset = np.array(dataset).reshape(-1,100,100)
    return dataset

def grouper(n, iterable, fillvalue=None):
    '''grouper(3, 'ABCDEFG', 'x') --> ABC DEF Gxx'''
    args = [iter(iterable)] * n
    output = it.zip_longest(fillvalue=fillvalue, *args)
    return output

def mosaic(w, imgs):
    '''Make a grid from images.

    w    -- number of grid columns
    imgs -- images (must have same size and format)
    '''
    imgs = iter(imgs)
    img0 = next(imgs)
    pad = np.zeros_like(img0)
    imgs = it.chain([img0], imgs)
    rows = grouper(w, imgs, pad)
    return np.vstack(map(np.hstack, rows))

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
            img[:,:2] = 0
        
        vis.append(img)
    return mosaic(5, vis)

def get_hog() : 
    winSize = (32,32)
    blockSize = (16,16)
    blockStride = (8,8)
    cellSize = (8,8)
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

    print('loading dataset... ')
    # Load data.
    icons_test = load_dataset(20, 'testset')
    labels_test = np.array([1] * 3 + [2] * 4 + [3] * 2 + [4] * 4 + [0] * 7)
    
    print('defining HoG parameters...')
    # HoG feature descriptor
    hog = get_hog();
    
    print('calculating HoG descriptor for every icon... ')
    hog_descriptors_test = []
    
    for ico in icons_test:
        hog_descriptors_test.append(hog.compute(ico))
    hog_descriptors_test = np.squeeze(hog_descriptors_test)
    
    print('loading model...')
    model = joblib.load('icoclf.pkl')
    
    print('evaluating model...')
    vis = evaluate_model(model, icons_test, hog_descriptors_test, labels_test)
    cv2.imwrite("icoclfevaluate.png",vis)
    cv2.imshow("Vis", vis)
    cv2.waitKey(0)