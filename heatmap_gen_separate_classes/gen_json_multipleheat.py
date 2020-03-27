# Usage:
#   python gen_json_multipleheat.py ./prediction-TCGA-xxxx lym_v1 lymphocyte_4 lym 0.5 necrosis 0.5

import subprocess
import numpy as np
import math
import json
import sys
import datetime
import random
import os
import openslide
from bson import json_util
#from pymongo import MongoClient

is_shifted = False;
isHeader = True        # to skip the first line in the prediction-xxx files

n_argv = len(sys.argv);
pred_file_path = sys.argv[1];
pred_folder, pred_file_name = os.path.split(pred_file_path);
filename = pred_file_name;
heatmap_name = sys.argv[2];
svs_img_folder=sys.argv[3];
start_id_multiheat = 4; # additional input starts from 4th

# Load configs from ../conf/variables.sh
mongo_host = 'localhost';
mongo_port = 27017;
cancer_type = 'quip';
lines = [line.rstrip('\n') for line in open('../conf/variables.sh')];
for config_line in lines:
    if (config_line.startswith('MONGODB_HOST=')):
        parts = config_line.split('=');
        mongo_host = parts[1];
        print("Mongodb host ", mongo_host);

    if (config_line.startswith('MONGODB_PORT=')):
        parts = config_line.split('=');
        str_port = parts[1];
        mongo_port = int(str_port);
        print("Mongodb port ", mongo_port);

    if (config_line.startswith('CANCER_TYPE=')):
        parts = config_line.split('=');
        cancer_type = parts[1];
        slide_type = cancer_type;
        print("Cancer type ", cancer_type);

n_heat = (n_argv - start_id_multiheat) // 2;
heat_list = [];
weight_list = [];
for h_id, h_name in enumerate(sys.argv[start_id_multiheat::2]):
    heat_list.append(h_name);
    weight_list.append(sys.argv[(start_id_multiheat+1)+2*h_id]);

casename = filename.split('prediction-')[1].split('.low_res')[0];
if 'low_res' not in filename:
    casename = casename.split('.intersected')[0]

print("Casename ", casename);
imgfilename = svs_img_folder + '/' + casename + '.svs';
if not os.path.isfile(imgfilename):
    imgfilename = svs_img_folder + '/' + casename + '.tif';
if not os.path.isfile(imgfilename):
    print("{}.svs/tif does not exist".format(svs_img_folder + '/' + casename));
    print("Quit");
    sys.exit(0);
print("Doing {}".format(imgfilename));


# Retrieve case_id and subject_id from mongodb
# Read mongodb port

#mongo_client = MongoClient(mongo_host, mongo_port);
#db = mongo_client[cancer_type].images;
#query_filename = imgfilename;
#db_result = db.find_one({"filename":query_filename});
#caseid = db_result['case_id'];
#subjectid = db_result['subject_id'];
caseid = casename;
subjectid = casename[:-2];

json_out_fol = '../data/heatmap_jsons_3classes'
if not os.path.exists(json_out_fol):
    os.mkdir(json_out_fol)
json_out_fol = os.path.join(json_out_fol, pred_file_path.split('/')[-2].split('_')[-1])
if not os.path.exists(json_out_fol):
    os.mkdir(json_out_fol)

heatmapfile = json_out_fol + '/heatmap_' + filename.split('prediction-')[1] + '.json';
metafile = json_out_fol +'/meta_' + filename.split('prediction-')[1] + '.json';

oslide = openslide.OpenSlide(imgfilename);
slide_width_openslide = oslide.dimensions[0];
slide_height_openslide = oslide.dimensions[1];


print("Loaded caseid and subjectid ", caseid, subjectid);

analysis_execution_id = 'highlym';
x_arr = np.zeros(10000000, dtype=np.float64);
y_arr = np.zeros(10000000, dtype=np.float64);
score_arr = np.zeros(10000000, dtype=np.float64);
score_set_arr = np.zeros(shape=(10000000, n_heat), dtype=np.float64);
idx = 0;
with open(pred_file_path) as infile:
    for line in infile:
        if isHeader:
            isHeader = False
            continue

        parts = line.split(' ');
        x = int(parts[0]);
        y = int(parts[1]);
        score = float(parts[2]);
        score_set = [float(i) for i in parts[2:2+n_heat]];

        x_arr[idx] = x;
        y_arr[idx] = y;
        score_arr[idx] = score;
        score_set_arr[idx] = score_set;
        idx += 1;

x_arr = x_arr[:idx];
y_arr = y_arr[:idx];
score_arr = score_arr[:idx];
score_set_arr = score_set_arr[:idx];

patch_width = max(x_arr[1] - x_arr[0], y_arr[1] - y_arr[0]);
patch_height = patch_width;

slide_width = int(slide_width_openslide);
slide_height = int(slide_height_openslide);

x_arr = x_arr / slide_width;
y_arr = y_arr / slide_height;
patch_width = patch_width / slide_width;
patch_height = patch_height / slide_height;
patch_area = patch_width * slide_width * patch_height * slide_height;

dict_img = {};
dict_img['case_id'] = caseid;
dict_img['subject_id'] = subjectid;

dict_analysis = {};
dict_analysis['study_id'] = slide_type;
dict_analysis['execution_id'] = heatmap_name;
dict_analysis['source'] = 'computer';
dict_analysis['computation'] = 'heatmap';


if (is_shifted == True):
    shifted_x = -3*patch_width / 4.0;
    shifted_y = -3*patch_height / 4.0;
else:
    shifted_x = 0;
    shifted_y = 0;

with open(heatmapfile, 'w') as f:
    for i_patch in range(idx):
        dict_patch = {};
        dict_patch['type'] = 'Feature';
        dict_patch['parent_id'] = 'self';
        dict_patch['footprint'] = patch_area;
        dict_patch['x'] = x_arr[i_patch] + shifted_x;
        dict_patch['y'] = y_arr[i_patch] + shifted_y;
        dict_patch['normalized'] = 'true';
        dict_patch['object_type'] = 'heatmap_multiple';

        x1 = dict_patch['x'] - patch_width/2;
        x2 = dict_patch['x'] + patch_width/2;
        y1 = dict_patch['y'] - patch_height/2;
        y2 = dict_patch['y'] + patch_height/2;
        dict_patch['bbox'] = [x1, y1, x2, y2];

        dict_geo = {};
        dict_geo['type'] = 'Polygon';
        dict_geo['coordinates'] = [[[x1, y1], [x2, y1], [x2, y2], [x1, y2], [x1, y1]]];
        dict_patch['geometry'] = dict_geo;

        dict_prop = {};
        dict_prop['metric_value'] = score_arr[i_patch];
        dict_prop['metric_type'] = 'tile_dice';
        dict_prop['human_mark'] = -1;

        dict_multiheat = {};
        dict_multiheat['human_weight'] = -1;
        dict_multiheat['weight_array'] = weight_list;
        dict_multiheat['heatname_array'] = heat_list;
        dict_multiheat['metric_array'] = score_set_arr[i_patch].tolist();

        dict_prop['multiheat_param'] = dict_multiheat;
        dict_patch['properties'] = dict_prop;

        dict_provenance = {};
        dict_provenance['image'] = dict_img;
        dict_provenance['analysis'] = dict_analysis;
        dict_patch['provenance'] = dict_provenance;

        dict_patch['date'] = datetime.datetime.now();

        json.dump(dict_patch, f, default=json_util.default);
        f.write('\n');

with open(metafile, 'w') as mf:
    dict_meta = {};
    dict_meta['color'] = 'yellow';
    dict_meta['title'] = 'Heatmap-' + heatmap_name;
    dict_meta['image'] = dict_img;

    dict_meta_provenance = {};
    dict_meta_provenance['analysis_execution_id'] = heatmap_name;
    dict_meta_provenance['study_id'] = slide_type;
    dict_meta_provenance['type'] = 'computer';
    dict_meta['provenance'] = dict_meta_provenance;

    dict_meta['submit_date'] = datetime.datetime.now();
    dict_meta['randval'] = random.uniform(0,1);

    json.dump(dict_meta, mf, default=json_util.default);

