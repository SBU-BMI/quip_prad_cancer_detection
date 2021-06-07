import os
import sys

import numpy as np


def apply_threshold(arr):
    arr[arr >= 0.5] = int(1)
    arr[arr < 0.5] = int(0)
    return arr


def is_all_zeros(arr):
    if np.any(arr):
        return False
    return True


def mask(arr):
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


# Compute similarity
def similarity(arr1, arr2):
    len1 = len(arr1)
    len2 = len(arr2)
    total = len1 + len2
    new_arr = []
    count = 0
    for x in range(len(arr1)):
        if arr1[x] == arr2[x]:
            new_arr.append(1)
    for x in range(len(new_arr)):
        count += 1

    intersection = count * 2
    dice = intersection / total
    return dice


def print_arr(arr):
    for x in range(len(arr)):
        print(arr[x])


def process(fname, f_pred, f_true, isSeparate):
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
        pred = mask1(y_pred).astype(np.int)
        truth = mask1(y_true).astype(np.int)
        dice_score = similarity(pred, truth)
        print('{0},{1}'.format(fname.replace('prediction-', ''), dice_score))
    else:
        # A = prediction, B = ground truth
        grade3A, grade4_5A, benignA = mask(y_pred)
        grade3B, grade4_5B, benignB = mask(y_true)

        score1 = similarity(grade3A, grade3B)
        score2 = similarity(grade4_5A, grade4_5B)
        score3 = similarity(benignA, benignB)
        print('{0},{1},{2},{3}'.format(fname.replace('prediction-', ''), score1, score2, score3))


if __name__ == '__main__':
    isSeparateClass = int(sys.argv[1])
    folder1 = sys.argv[2]
    folder2 = sys.argv[3]

    print("Similarity scores")
    if isSeparateClass:
        ind = folder1.rindex('_') + 1
        str1 = folder1[ind:]
        print("Slide,{0}".format(str1))
    else:
        print("Slide,Grade3,Grade4+5,Benign")

    for dirName, subdirList, fileList in os.walk(folder1):
        for fname in fileList:
            if fname.startswith('prediction'):
                path1 = os.path.join(folder1, fname)
                path2 = os.path.join(folder2, fname)
                process(fname, path1, path2, isSeparateClass)
