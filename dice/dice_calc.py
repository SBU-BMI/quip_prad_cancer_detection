import os
import sys
from os import path

import numpy as np
from scipy.spatial.distance import dice


def apply_threshold(arr):
    arr[arr < 0.5] = 0  # Do below first
    arr[arr >= 0.5] = 1  # Then above
    return arr


def is_all_zeros(arr):
    if np.any(arr):
        return False
    return True


def mask(arr):
    # x_loc = arr[:, 0]
    # y_loc = arr[:, 1]
    grade3 = arr[:, 2]
    grade4_5 = arr[:, 3]
    benign = arr[:, 4]

    # apply threshold
    grade3 = apply_threshold(grade3)
    grade4_5 = apply_threshold(grade4_5)
    benign = apply_threshold(benign)

    A = is_all_zeros(grade3)
    B = is_all_zeros(grade4_5)
    C = is_all_zeros(benign)
    if A or B or C:
        print("Can't do it if the array is all zeros")
        exit(1)

    return grade3, grade4_5, benign


def mask1(arr):
    feature = arr[:, 2]
    feature = apply_threshold(feature)
    if is_all_zeros(feature):
        print("Can't do it if the array is all zeros")
        exit(1)
    return feature


# Dice similarity function
def compute_dice(prediction, truth):
    try:
        similarity = 1.0 - dice(prediction.astype(bool), truth.astype(bool))
    except ZeroDivisionError:
        similarity = 0
    except:
        print("Unexpected error:", sys.exc_info()[0])
        raise
    return similarity
    # return format(similarity, '.2f')


def compute1(prediction, truth, k=1):
    intersection = np.sum(prediction[truth == k]) * 2.0
    _dice = intersection / (np.sum(prediction) + np.sum(truth))
    return _dice


def files_exist(file1, file2):
    if path.exists(file1) and path.exists(file2):
        return True
    else:
        if not path.exists(file1):
            raise IOError("%s not found." % file1)
        else:
            raise IOError("%s not found." % file2)


def process(fname, f_pred, f_true, isSeparate):
    files_exist(f_pred, f_true)
    if isSeparate:
        y_pred = np.loadtxt(f_pred).astype(np.float32)
        y_true = np.loadtxt(f_true).astype(np.float32)
    else:
        y_pred = np.loadtxt(f_pred, skiprows=1).astype(np.float32)
        y_true = np.loadtxt(f_true, skiprows=1).astype(np.float32)

    if y_true.shape != y_pred.shape:
        print('The dimensions of the two sets are not equal', fname)
        exit(1)

    if isSeparate:
        pred = mask1(y_pred)
        truth = mask1(y_true)
        dice_score = compute_dice(pred, truth)
        print('{0},{1}'.format(fname.replace('prediction-', ''), dice_score))
    else:
        # A = prediction, B = ground truth
        grade3A, grade4_5A, benignA = mask(y_pred)
        grade3B, grade4_5B, benignB = mask(y_true)

        score1 = compute_dice(grade3A, grade3B)
        score2 = compute_dice(grade4_5A, grade4_5B)
        score3 = compute_dice(benignA, benignB)
        print('{0},{1},{2},{3}'.format(fname.replace('prediction-', ''), score1, score2, score3))


if __name__ == '__main__':
    isSeparateClass = int(sys.argv[1])
    folder1 = sys.argv[2]
    folder2 = sys.argv[3]

    print("Dice Similarity")
    if isSeparateClass:
        ind = folder1.rindex('_') + 1
        if ind < len(folder1):  # don't run off the end of the array
            str1 = folder1[ind:]
        else:
            str1 = folder1
        print("Slide,{0}".format(str1))
    else:
        print("Slide,Grade3,Grade4+5,Benign")

    for dirName, subdirList, fileList in os.walk(folder1):
        for fname in fileList:
            if fname.startswith('prediction'):
                path1 = os.path.join(folder1, fname)
                path2 = os.path.join(folder2, fname)
                process(fname, path1, path2, isSeparateClass)
