#!/bin/bash

source ../conf/variables.sh
HEATMAP_VERSION="cancer-prad-beatrice-john-3classes_GRADE3_2020-01-20_han.le@stonybrook.edu"
HEATMAP_TXT_OUTPUT_FOLDER=${BASE_DIR}/data/heatmap_txt_3classes_separate_class/heatmap_txt_grade3

for files in ${HEATMAP_TXT_OUTPUT_FOLDER}/prediction-*; do
    if [[ "$files" == *.low_res* ]]; then
        python gen_json_multipleheat.py ${files} ${HEATMAP_VERSION}-low_res  ${SVS_INPUT_PATH} lym 0.5 necrosis 0.5 &>> ${LOG_OUTPUT_FOLDER}/log.gen_json_multipleheat.low_res.txt
    else
        python gen_json_multipleheat.py ${files} ${HEATMAP_VERSION}-high_res ${SVS_INPUT_PATH} lym 0.5 necrosis 0.5 &>> ${LOG_OUTPUT_FOLDER}/log.gen_json_multipleheat.txt    # HEATMAP_VERSION: neu_v1...
    fi
done

exit 0

