#!/bin/bash

cd ../
source ./conf/variables.sh

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
