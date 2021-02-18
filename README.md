# Prostate Adenocarcinoma (PRAD) detection pipeline

This software implements the pipeline for the 3-classes (benign, grade3, grade45) Prostate cancer detection project. 

# Dependencies

 - [Pytorch 0.4.0](http://pytorch.org/)
 - Torchvision 0.2.0
 - cv2 (3.4.1)
 - [Openslide 1.1.1](https://openslide.org/api/python/)
 - [sklearn](https://scikit-learn.org/stable/)
 - [PIL](https://pillow.readthedocs.io/en/3.1.x/reference/Image.html)

# List of folders and functionalities are below: 

- scripts/: contains scripts that connect several sub-functionalities together for complete functionalities such as generating camicroscope heatmaps given svs images.

- conf/: contains configuration. 

- data/: a place where should contain all logs, input/output images, trained CNN models, and large files. 

- download_heatmap/: downloads grayscale lymphocyte or tumor heatmaps

- heatmap_gen/: generate json files that represents heatmaps for camicroscope, using the lymphocyte and necrosis CNNs' raw output txt files. 

- patch_extraction_tumor_40X/: extracts all patches from svs images. Mainly used in the test phase. 

- prediction/: CNN prediction code. 

- training_codes/: CNN training code. 

## Setup conf/variables.sh
- Change the BASE_DIR to the path of your folder after you clone the git repo

## Training
- Go to folder "training_codes", run python train_prad_3classes.py

## WSIs prediction
- Go to folder "scripts", run bash svs_2_heatmap.sh


# Docker Instructions

Build the docker image by: 

`docker build -t prad_detection .`  (Note the dot at the end). 

## Prediction
### Step 1:
Create folder named "data" and subfolders below on the host machine:

- data/svs: to contains *.svs files
- data/patches: to contain output from patch extraction
- data/log: to contain log files
- data/heatmap_txt: to contain prediction output
- data/heatmap_jsons: to contain prediction output as json files
- models_cnn: contains prediction models

### Step 2:
- Run the docker container as follows: 

```
nvidia-docker run --name prad-detection -itd -v <path-to-data>:/data -e CUDA_VISIBLE_DEVICES='<cuda device id>' prad_detection svs_2_heatmap.sh <model-name>
```
If you prefer to use the default model (in folder models_cnn), then simply run the above command without any model name.

CUDA_VISIBLE_DEVICES -- set to select the GPU to use 

The following example runs the cancer detection pipeline. It will process images in /home/user/data/svs and output the results to /home/user/data. 

```
nvidia-docker run --name prad-detection -itd -v /home/user/data:/data -e CUDA_VISIBLE_DEVICES='0' prad_detection svs_2_heatmap.sh
```

## Training
### Step 1:
Create folder named "data" and subfolders below on the host machine:

- data/input/training_data: to contain training data
- data/input/validation_data: to contain validation data
- data/output/checkpoint: to contain checkpoint models (the last file written will be the "best" trained model)
- data/output/log: to contain log files

### Step 2:
- Run the docker container as follows:

```
nvidia-docker run --name prad-cancer-detection --ipc=host -itd -v <path-to-data>:/data -e CUDA_VISIBLE_DEVICES='0' prad_cancer_detection train_model.sh
```

This will output prediction models to the `checkpoint` folder.  The one that was last written to the file system would be the one with the best F1 score.

Note the `--ipc=host` so that Torch can write to the model file.

> :warning: If you omit `--ipc=host` in the command, you will get an error like:

```
RuntimeError: unable to write to file </torch_XX_XXXXXXXXXX>
```

## Prediction After Training
Take the best model that was produced in the previous step, and put it into folder `models_cnn`.
Then, pass the file name as a parameter to `svs_2_heatmap.sh`, like this:

```
nvidia-docker run --name prad-cancer-detection -itd -v <path-to-data>:/data -e CUDA_VISIBLE_DEVICES='0' prad_cancer_detection svs_2_heatmap.sh <resnet34-model-name>
```
