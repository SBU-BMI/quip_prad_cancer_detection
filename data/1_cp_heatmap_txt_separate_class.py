import os
import sys
import glob
import csv
import pandas as pd
import multiprocessing as mp

def process(file):
    slide_id = file.split('/')[-1]

    print(file)
    pred = pd.read_csv(file, delimiter=' ')
    zeros = [0]*len(pred)
    pred['tumor'] = pred['grade3'] + pred['grade4+5']
    pred['zeros'] = zeros

    columns = pred.columns
    for i in range(len(dest_fols)):
        pred[[columns[0], columns[1], columns[i + 2], 'zeros']].to_csv(os.path.join(out_fol, dest_fols[i], slide_id), sep=' ', header=False, index=False)
        os.system('cp ' + os.path.join(in_fol, 'color-' + slide_id[11:]) + ' ' + os.path.join(out_fol, dest_fols[i],  'color-' + slide_id[11:]))


in_fol = 'heatmap_txt_3classes_header'
out_fol = 'heatmap_txt_3classes_separate_class'

dest_fols = ['heatmap_txt_grade3', 'heatmap_txt_grade4_5', 'heatmap_txt_benign', 'heatmap_txt_tumor']
for fol in dest_fols:
    path = os.path.join(out_fol, fol)
    if not os.path.exists(path):
        os.mkdir(path)

files = glob.glob(in_fol + '/prediction-*')
pool = mp.Pool(processes=64)
pool.map(process, files)

