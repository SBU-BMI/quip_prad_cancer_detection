#!/bin/bash
source ../conf/variables.sh

bash 1_cp_heatmaps_all.sh > ${LOG_OUTPUT_FOLDER}/log.1_cp_heatmaps_all.txt 2>&1
wait;

python -u 2_cp_heatmap_txt_separate_class.py >> ${LOG_OUTPUT_FOLDER}/log.2_cp_heatmap_txt_separate_class.txt 2>&1
wait;

python -u 3_thresholded_heatmap_txt.py >> ${LOG_OUTPUT_FOLDER}/log.3_thresholded_heatmap_txt.txt 2>&1
wait;

python -u run_gen_jsons_threads.py >> ${LOG_OUTPUT_FOLDER}/log.run_gen_jsons_threads.txt 2>&1
wait;

