#!/bin/bash

source ../conf/variables.sh

cd color
nohup bash color_stats.sh ${PATCH_PATH} 0 8 &> ${LOG_OUTPUT_FOLDER}/log.color_stats_0.txt &
nohup bash color_stats.sh ${PATCH_PATH} 1 8 &> ${LOG_OUTPUT_FOLDER}/log.color_stats_1.txt &
nohup bash color_stats.sh ${PATCH_PATH} 2 8 &> ${LOG_OUTPUT_FOLDER}/log.color_stats_2.txt &
nohup bash color_stats.sh ${PATCH_PATH} 3 8 &> ${LOG_OUTPUT_FOLDER}/log.color_stats_3.xt &
nohup bash color_stats.sh ${PATCH_PATH} 4 8 &> ${LOG_OUTPUT_FOLDER}/log.color_stats_4.txt &
nohup bash color_stats.sh ${PATCH_PATH} 5 8 &> ${LOG_OUTPUT_FOLDER}/log.color_stats_5.txt &
nohup bash color_stats.sh ${PATCH_PATH} 6 8 &> ${LOG_OUTPUT_FOLDER}/log.color_stats_6.txt &
nohup bash color_stats.sh ${PATCH_PATH} 7 8 &> ${LOG_OUTPUT_FOLDER}/log.color_stats_7.txt &
    
cd ..

wait

exit 0
