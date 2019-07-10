#!/bin/bash

FOLDER=${1}     #/data/patches

PRED_VERSION=patch-level-prad.txt
PRED_GRADES=patch-level-prad_grades.txt

DIS_FOLDER=./patch-level-lym/
DIS_FOLDER_GRADES=./patch-level-grades/
for files in ${FOLDER}/*/${PRED_VERSION}; do
    dis=`echo ${files} | awk -F'/' '{print "prediction-"substr($(NF-1),1,length($(NF-1))-4);}'`
    cp ${files} ${DIS_FOLDER}/${dis}
done

for files in ${FOLDER}/*/${PRED_GRADES}; do
    dis=`echo ${files} | awk -F'/' '{print "prediction_GRADES-"substr($(NF-1),1,length($(NF-1))-4);}'`
    cp ${files} ${DIS_FOLDER_GRADES}/${dis}
done



PRED_VERSION=patch-level-necrosis.txt
DIS_FOLDER=./patch-level-nec/
for files in ${FOLDER}/*/${PRED_VERSION}; do
    dis=`echo ${files} | awk -F'/' '{print "prediction-"substr($(NF-1),1,length($(NF-1))-4);}'`
    cp ${files} ${DIS_FOLDER}/${dis}
    #python -u reverse_nec_pred.py ${DIS_FOLDER}/${dis} ${DIS_FOLDER}/${dis}
done


PRED_VERSION=patch-level-color.txt
DIS_FOLDER=./patch-level-color/
for files in ${FOLDER}/*/${PRED_VERSION}; do
    dis=`echo ${files} | awk -F'/' '{print "color-"substr($(NF-1),1,length($(NF-1))-4);}'`
    cp ${files} ${DIS_FOLDER}/${dis}
done

exit 0
