import os
import sys
import glob
import csv
import pandas as pd

def mkdir_fol(path):
    if not os.path.exists(path):
        os.mkdir(path)

out_dir = os.environ.get('OUT_DIR')
if out_dir is None:
   out_dir = "../data" 

in_fol  = str(out_dir)+'/heatmap_txt'
out_fol = str(out_dir)+'/heatmap_txt_3classes_separate_class'

dest_fols = ['heatmap_txt_grade3', 'heatmap_txt_grade45', 'heatmap_txt_benign']
mkdir_fol(out_fol)
for fol in dest_fols:
    mkdir_fol(os.path.join(out_fol, fol))

for file in glob.glob(in_fol + '/prediction-*'):
    slide_id = file.split('/')[-1]

    print(file)
    pred = pd.read_csv(file, delimiter=' ')
    zeros = [0]*len(pred)
    pred['zeros'] = zeros

    columns = pred.columns
    for i in range(len(dest_fols)):
        data = pred[[columns[0], columns[1], columns[i + 2], 'zeros']]
        data.to_csv(os.path.join(out_fol, dest_fols[i], slide_id), sep=' ', header=False, index=False)
        os.system('cp ' + os.path.join(in_fol, 'color-' + slide_id[11:]) + ' ' + os.path.join(out_fol, dest_fols[i],  'color-' + slide_id[11:]))

