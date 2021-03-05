#!/usr/bin/env bash
# Example:
# ./run_dice_calc.sh "/prad_old_model/heatmap_txt" "/prad_new_model/heatmap_txt" "/prad_old_model/heatmap_txt_3classes_separate_class" "/prad_new_model/heatmap_txt_3classes_separate_class"

if [ $# -eq 0 ]; then
    echo 'Usage: ./'$(basename "$0") '/path/to/old_heatmap_txt /path/to/new_heatmap_txt /path/to/old_heatmap_txt_3classes_separate_class /path/to/new_heatmap_txt_3classes_separate_class'
fi

current_time=$(date "+%Y.%m.%d-%H.%M.%S")
dir_old="$1"
dir_new="$2"
old="$3"
new="$4"

heatmap_txt() {
    # Dice calculation (3 classes in one file)
    echo "3 classes"
    python dice_calc.py 0 "$1" "$2" >"prad_dice-$current_time.csv"
    if [ $? -eq 0 ]; then
        echo OK
    else
        echo FAIL
        exit 1
    fi
}
heatmap_txt $dir_old $dir_new

heatmap_txt_3classes_separate_class() {
    # Dice calculation (separate classes)
    subdirs=('heatmap_txt_benign' 'heatmap_txt_grade3' 'heatmap_txt_grade45' 'heatmap_txt_thresholded' 'heatmap_txt_tumor')
    for dir in "${subdirs[@]}"; do
        old="$1/$dir"
        new="$2/$dir"
        repl="heatmap_txt_"
        name="${dir//$repl/}" # 'benign', 'grade3', etc.
        echo "$name"
        # Dice calculation
        python dice_calc.py 1 "$old" "$new" >"prad_dice-$name-$current_time.csv"
        if [ $? -eq 0 ]; then
            echo OK
        else
            echo FAIL
            exit 1
        fi
    done
}
heatmap_txt_3classes_separate_class $old $new
