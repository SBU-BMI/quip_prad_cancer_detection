#!/bin/bash

source ../conf/variables.sh

FOLDER=${PATCH_PATH}     #/data/patches

PRED_VERSION=patch-level-3classes.txt

DIS_FOLDER=${HEATMAP_TXT_OUTPUT_FOLDER}
for files in ${FOLDER}/*/${PRED_VERSION}; do
    fname=`echo ${files} | awk -F '/' '{print $(NF-1)}'`;
    fprefix="${fname%.*}"
    dis="prediction-"$fprefix
    # dis=`echo ${files} | awk -F'/' '{print "prediction-"substr($(NF-1),1,length($(NF-1))-4);}'`
    cp ${files} ${DIS_FOLDER}/${dis}
    echo -e "x_loc y_loc grade3 grade4+5 benign\n$(cat ${DIS_FOLDER}/${dis})" > ${DIS_FOLDER}/${dis}
done

PRED_VERSION=patch-level-color.txt
DIS_FOLDER=${HEATMAP_TXT_OUTPUT_FOLDER}
for files in ${FOLDER}/*/${PRED_VERSION}; do
    fname=`echo ${files} | awk -F '/' '{print $(NF-1)}'`;
    fprefix="${fname%.*}"
    dis="color-"$fprefix
    # dis=`echo ${files} | awk -F'/' '{print "color-"substr($(NF-1),1,length($(NF-1))-4);}'`
    cp ${files} ${DIS_FOLDER}/${dis}
done

exit 0
