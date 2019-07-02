#!/bin/bash

source ../conf/variables.sh

nohup bash save_svs_to_tiles.sh 0 8 &> ${LOG_OUTPUT_FOLDER}/log.save_svs_to_tiles.thread_0.txt &
nohup bash save_svs_to_tiles.sh 1 8 &> ${LOG_OUTPUT_FOLDER}/log.save_svs_to_tiles.thread_1.txt &
nohup bash save_svs_to_tiles.sh 2 8 &> ${LOG_OUTPUT_FOLDER}/log.save_svs_to_tiles.thread_2.txt &
nohup bash save_svs_to_tiles.sh 3 8 &> ${LOG_OUTPUT_FOLDER}/log.save_svs_to_tiles.thread_3.txt &
nohup bash save_svs_to_tiles.sh 4 8 &> ${LOG_OUTPUT_FOLDER}/log.save_svs_to_tiles.thread_4.txt &
nohup bash save_svs_to_tiles.sh 5 8 &> ${LOG_OUTPUT_FOLDER}/log.save_svs_to_tiles.thread_5.txt &
nohup bash save_svs_to_tiles.sh 6 8 &> ${LOG_OUTPUT_FOLDER}/log.save_svs_to_tiles.thread_6.txt &
nohup bash save_svs_to_tiles.sh 7 8 &> ${LOG_OUTPUT_FOLDER}/log.save_svs_to_tiles.thread_7.txt &
wait

#python -u back_ground_filter_all_folders.py ${PATCH_PATH} ${SVS_INPUT_PATH}
#wait

exit 0
