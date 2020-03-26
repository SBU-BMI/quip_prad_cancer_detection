import os
import sys
import glob
import numpy as np
import multiprocessing as mp

in_fol = 'heatmap_txt_3classes_header'
out_fol = 'heatmap_txt_3class_thresholded'
#b. 0.55 if it is grade 3
#c. 0.19 if it is grade 4
#a. 0.91 if it is benign
probs = [0.55, 0.19, 0.91]
files = glob.glob(in_fol + '/prediction*')
#files = glob.glob(in_fol + '/prediction-001738-000307_1_07_20180925-multires')

def process(file):
    print(file)
    slide_id = file.split('/')[-1]
    preds = [f.rstrip().split(' ') for f in open(file, 'r')]
    out = open(os.path.join(out_fol, slide_id), 'w')
    for pred in preds[1:]:
        grades = np.array([float(pred[2]), float(pred[3]), float(pred[4])])
        res = probs[np.argmax(grades)] if sum(grades) > 0 else 0
        out.writelines('{} {} {} 0 \n'.format(pred[0], pred[1], res))

    os.system('cp ' + os.path.join(in_fol, 'color-' + slide_id[11:]) + ' ' + os.path.join(out_fol,  'color-' + slide_id[11:]))

    out.close()

print(len(files))
pool = mp.Pool(processes=20)
pool.map(process, files)



