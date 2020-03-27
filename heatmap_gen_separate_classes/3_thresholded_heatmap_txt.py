import os
import sys
import glob
import numpy as np
import multiprocessing as mp

in_fol = '../data/heatmap_txt'
thresholded_fol = '../data/heatmap_txt_3classes_separate_class/heatmap_txt_thresholded'
tumor_fol = '../data/heatmap_txt_3classes_separate_class/heatmap_txt_tumor'

if not os.path.exists(thresholded_fol):
    os.mkdir(thresholded_fol)

if not os.path.exists(tumor_fol):
    os.mkdir(tumor_fol)

probs = [0.5, 0.9, 0.1] # grade3, grade45, benign
files = glob.glob(in_fol + '/prediction*')

def process(file):
    print(file)
    slide_id = file.split('/')[-1]
    preds = [f.rstrip().split(' ') for f in open(file, 'r')]
    thresholded = open(os.path.join(thresholded_fol, slide_id), 'w')
    tumor = open(os.path.join(tumor_fol, slide_id), 'w')
    for pred in preds[1:]:
        grades = np.array([float(p) for p in pred[2:]])
        res = probs[np.argmax(grades)] if sum(grades) > 0 else 0
        thresholded.writelines('{} {} {} 0 \n'.format(pred[0], pred[1], res))

        benign_prob = grades[2]
        benign_prob_adjusted = 1
        if sum(grades) > 0:
            benign_prob_adjusted = benign_prob*(len(grades) - 1) / (sum(grades) + benign_prob*(len(grades) - 2))
        tumor.writelines('{} {} {} 0 \n'.format(pred[0], pred[1], 1 - benign_prob_adjusted))

    color_fn = 'color-' + slide_id.split('prediction-')[-1]
    os.system('cp {} {}'.format(os.path.join(in_fol, color_fn), os.path.join(thresholded_fol, color_fn)))
    os.system('cp {} {}'.format(os.path.join(in_fol, color_fn), os.path.join(tumor_fol, color_fn)))

    thresholded.close()
    tumor.close()

print(len(files))
pool = mp.Pool(processes=20)
pool.map(process, files)



