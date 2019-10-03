#!/bin/bash

# Variables
DEFAULT_OBJ=20
DEFAULT_MPP=0.5
CANCER_TYPE=quip
MONGODB_HOST=xxxx
MONGODB_PORT=27017

HEATMAP_VERSION=cancer-prad

# Base directory
BASE_DIR=/root/quip_prad_cancer_detection
DATA_DIR=/data

# The username you want to download heatmaps from
USERNAME=abc@test.test

# The list of case_ids you want to download heaetmaps from
CASE_LIST=${BASE_DIR}/raw_marking_to_download_case_list/case_list.txt

# Paths of data, log, input, and output
JSON_OUTPUT_FOLDER=${DATA_DIR}/output/heatmap_jsons
HEATMAP_TXT_OUTPUT_FOLDER=${DATA_DIR}/output/heatmap_txt
LOG_OUTPUT_FOLDER=${DATA_DIR}/output/log
SVS_INPUT_PATH=${DATA_DIR}/svs
PATCH_PATH=${DATA_DIR}/patches

PATCH_SAMPLING_LIST_PATH=${DATA_DIR}/patch_sample_list
RAW_MARKINGS_PATH=${DATA_DIR}/raw_marking_xy
MODIFIED_HEATMAPS_PATH=${DATA_DIR}/modified_heatmaps
TUMOR_HEATMAPS_PATH=${DATA_DIR}/tumor_labeled_heatmaps
TUMOR_GROUND_TRUTH=${DATA_DIR}/tumor_ground_truth_maps
TUMOR_IMAGES_TO_EXTRACT=${DATA_DIR}/tumor_images_to_extract
GRAYSCALE_HEATMAPS_PATH=${DATA_DIR}/grayscale_heatmaps
THRESHOLDED_HEATMAPS_PATH=${DATA_DIR}/thresholded_heatmaps
PATCH_FROM_HEATMAP_PATH=${DATA_DIR}/patches_from_heatmap
THRESHOLD_LIST=${DATA_DIR}/threshold_list/threshold_list.txt

CAE_TRAINING_DATA=${BASE_DIR}/training_data_cae
CAE_TRAINING_DEVICE=gpu0
CAE_MODEL_PATH=${BASE_DIR}/models_cae
LYM_CNN_TRAINING_DATA=${BASE_DIR}/training_data_cnn
LYM_CNN_TRAINING_DEVICE=gpu0
LYM_CNN_PRED_DEVICE=gpu0
LYM_NECRO_CNN_MODEL_PATH=${BASE_DIR}/models_cnn
NEC_CNN_TRAINING_DATA=${BASE_DIR}/training_data_cnn
NEC_CNN_TRAINING_DEVICE=gpu1
NEC_CNN_PRED_DEVICE=gpu0
EXTERNAL_LYM_MODEL=0

if [[ -z "${CUDA_VISIBLE_DEVICES}" ]]; then
	LYM_CNN_TRAINING_DEVICE=0
	LYM_CNN_PRED_DEVICE=0
else
	LYM_CNN_TRAINING_DEVICE=${CUDA_VISIBLE_DEVICES}
	LYM_CNN_PRED_DEVICE=${CUDA_VISIBLE_DEVICES}
fi

