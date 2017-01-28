import numpy as np
import copy
from scipy.ndimage import imread
from scipy.spatial.distance import cdist

# Parameters
nrun = 20 # number of classification runs
fname_label = 'class_labels.txt' # where class labels are stored for each run


def LoadImgAsPoints(fn):
    # Load image file and return coordinates of 'inked' pixels in the binary
    # image
    I = imread(fn, flatten=True)
    I = np.array(I, dtype=bool)
    I = np.logical_not(I)
    (row, col) = I.nonzero()
    D = np.array([row, col])
    D = np.transpose(D)
    D = D.astype(float)
    n = D.shape[0]
    mean = np.mean(D, axis=0)
    for i in range(n):
        D[i, :] = D[i, :] - mean
    return D


def ModHausdorffDistance(itemA, itemB):
    # Modified Hausdorff Distance for object matching
    D = cdist(itemA, itemB)
    mindist_A = D.min(axis=1)
    mindist_B = D.min(axis=0)
    mean_A = np.mean(mindist_A)
    mean_B = np.mean(mindist_B)
    return max(mean_A, mean_B)


def classification_run(folder, f_load, f_cost, ftype='cost'):
    assert ((ftype=='cost') | (ftype=='score'))

    # get file names
    with open(folder + '/' + fname_label) as f:
        content = f.read().splitlines()
    pairs = [line.split() for line in content]
    test_files = [pair[0] for pair in pairs]
    train_files = [pair[1] for pair in pairs]
    answers_files = copy.copy(train_files)
    train_files.sort()
    test_files.sort()
    ntrain = len(train_files)
    ntest = len(test_files)

    # load the images
    train_items = [f_load(f) for f in train_files]
    test_items = [f_load(f) for f in test_files]

    # compute cost matrix
    costM = np.zeros((ntest, ntrain), float)
    for i in range(ntest):
        for c in range(ntrain):
            costM[i, c] = f_cost(test_items[i], train_items[c])
    if ftype == 'cost':
        YHAT = np.argmin(costM, axis=1)
    elif ftype == 'score':
        YHAT = np.argmax(costM, axis=1)
    else:
        assert False

    # compute the error rate
    correct = 0.0
    for i in range(ntest):
        if train_files[YHAT[i]] == answers_files[i]:
            correct += 1.0

    pcorrect = 100 * correct / ntest
    print " correct " + str(pcorrect)
    perror = 100 - pcorrect
    return perror


if __name__ == '__main__':
    print 'One shot classification with Modified Hausdorff Distance'
    perror = np.zeros(nrun) # stores the error for each run

    for r in range(1, nrun + 1):
        rs = str(r)
        if len(rs) == 1:
            rs = '0' + rs
        perror[r-1] = classification_run('run' + rs, LoadImgAsPoints,
                                         ModHausdorffDistance, 'cost')
        print " run " + str(r) + " (error " + str(perror[r-1]) + "%)"
    total = np.mean(perror)
    print "average error " + str(total) + "%"
