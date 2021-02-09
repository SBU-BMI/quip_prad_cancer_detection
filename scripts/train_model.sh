#!/usr/bin/env bash

source ../conf/variables.sh

[ -d "$LOG_OUTPUT_FOLDER" ] || mkdir "$LOG_OUTPUT_FOLDER"

cd "$BASE_DIR/training_codes"

nohup python train_prad_3classes.py &>"$LOG_OUTPUT_FOLDER/log.train_prad_3classes.txt" &

cd "$BASE_DIR"
wait
exit 0
