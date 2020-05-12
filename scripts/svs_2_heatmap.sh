#!/bin/bash

cd ../
source ./conf/variables.sh

cd patch_extraction_tumor_20X
nohup bash start.sh &
cd ..

cd prediction_4classes
nohup bash start.sh &
cd ..

wait;

cd heatmap_gen_separate_classes
nohup bash start.sh &
cd ..

wait;

exit 0
