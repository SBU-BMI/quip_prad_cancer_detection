#!/bin/bash

source ../conf/variables.sh

COD_PARA=$1
MAX_PARA=$2
IN_FOLDER=${SVS_INPUT_PATH} # data/svs
OUT_FOLDER=${PATCH_PATH}    # data/patches

# CHECK FOLDER EXISTS AND IS NOT EMPTY
if [ -d "$IN_FOLDER" ]; then
    if find "$IN_FOLDER" -mindepth 1 | read; then
        LINE_N=0
        for files in ${IN_FOLDER}/*.*; do
            LINE_N=$((LINE_N + 1))
            if [ $((LINE_N % MAX_PARA)) -ne ${COD_PARA} ]; then continue; fi

            SVS=$(echo ${files} | awk -F'/' '{print $NF}')

            #    if [ ! -f ${OUT_FOLDER}/${SVS}/*_mask.png ]; then
            #        python -u back_ground_filter.py ${IN_FOLDER}/${SVS} ${OUT_FOLDER}/${SVS}
            #        wait
            #        echo 'Done extracting background'
            #    fi

            python save_svs_to_tiles.py ${SVS} ${IN_FOLDER} ${OUT_FOLDER}

            if [ $? -ne 0 ]; then
                echo "failed extracting patches for " ${SVS}
                rm -rf ${OUT_FOLDER}/${SVS}
            else
                touch ${OUT_FOLDER}/${SVS}/extraction_done.txt
            fi

        done
        exit 0
    else
        echo "Error: ${IN_FOLDER} is empty. Can not continue."
        exit 1
    fi
else
    echo "Error: ${IN_FOLDER} not found. Can not continue."
    exit 1
fi
