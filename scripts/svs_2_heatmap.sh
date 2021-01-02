#!/bin/bash

cd ../
source ./conf/variables.sh

out_folders="heatmap_jsons heatmap_txt json log"
for i in ${out_folders}; do
	if [ ! -d ${OUT_DIR}/$i ]; then
		mkdir -p ${OUT_DIR}/$i
	fi
done
if [ ! -d ${DATA_DIR}/patches ]; then
	mkdir -p ${DATA_DIR}/patches;
fi
wait;

cd patch_extraction_tumor_40X
nohup bash start.sh &
cd ..

cd prediction_3classes
nohup bash start.sh &
cd ..

wait;

cd heatmap_gen_separate_classes
nohup bash start.sh &
cd ..

wait;

exit 0
