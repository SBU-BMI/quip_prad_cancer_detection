import os
import torch
from torchvision import models, transforms
from PIL import Image
import numpy as np
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import DataLoader, Dataset
import time
import argparse
from torch.optim import lr_scheduler
import copy
import torch.nn.parallel
import torch.optim as optim
from sklearn.metrics import mean_squared_error, accuracy_score, hamming_loss, roc_curve, auc, f1_score
import sys
import torch.backends.cudnn as cudnn
import time
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True


APS = 175;
PS = 224
TileFolder = sys.argv[1] + '/';

mu = [0.6462,  0.5070,  0.8055]      # for Prostate cancer
sigma = [0.1381,  0.1674,  0.1358]

BatchSize = 96;
heat_map_out = sys.argv[3];

device = torch.device("cuda")
data_aug = transforms.Compose([
    transforms.Scale(PS),
    transforms.ToTensor(),
    transforms.Normalize(mu, sigma)])

def whiteness(png):
    wh = (np.std(png[:,:,0].flatten()) + np.std(png[:,:,1].flatten()) + np.std(png[:,:,2].flatten())) / 3.0;
    return wh;


def softmax_np(x):
    x = x - np.max(x, 1, keepdims=True)
    x = np.exp(x) / (np.sum(np.exp(x), 1, keepdims=True))
    return x

def iterate_minibatches(inputs, augs, targets):
    if inputs.shape[0] <= BatchSize:
        yield inputs, augs, targets;
        return;

    start_idx = 0;
    for start_idx in range(0, len(inputs) - BatchSize + 1, BatchSize):
        excerpt = slice(start_idx, start_idx + BatchSize);
        yield inputs[excerpt], augs[excerpt], targets[excerpt];
    if start_idx < len(inputs) - BatchSize:
        excerpt = slice(start_idx + BatchSize, len(inputs));
        yield inputs[excerpt], augs[excerpt], targets[excerpt];


def load_data(todo_list, rind):
    X = torch.zeros(size=(BatchSize*40, 3, PS, PS));
    inds = np.zeros(shape=(BatchSize*40,), dtype=np.int32);
    coor = np.zeros(shape=(200000, 2), dtype=np.int32);

    normalized = False  # change this to true if dont have images normalized and normalize on the fly
    parts = 4
    if normalized:
        parts = 4

    xind = 0;
    lind = 0;
    cind = 0;
    for fn in todo_list:
        lind += 1;
        full_fn = TileFolder + '/' + fn;
        if not os.path.isfile(full_fn):
            continue;
        if (len(fn.split('_')) != parts) or ('.png' not in fn):
            continue;

        try:
            x_off = float(fn.split('_')[0]);
            y_off = float(fn.split('_')[1]);
            svs_pw = float(fn.split('_')[2]);
            png_pw = float(fn.split('_')[3].split('.png')[0]);
        except:
            print('error reading image')
            continue

        png = np.array(Image.open(full_fn).convert('RGB'));
        for x in range(0, png.shape[1], APS):
            if x + APS > png.shape[1]:
                continue;
            for y in range(0, png.shape[0], APS):
                if y + APS > png.shape[0]:
                    continue;

                if (whiteness(png[y:y+APS, x:x+APS, :]) >= 12):
                    a = png[y:y + APS, x:x + APS, :]
                    a = Image.fromarray(a.astype('uint8'), 'RGB')
                    a = data_aug(a)
                    X[xind, :, :, :] = a
                    inds[xind] = rind
                    xind += 1

                coor[cind, 0] = np.int32(x_off + (x + APS/2) * svs_pw / png_pw);
                coor[cind, 1] = np.int32(y_off + (y + APS/2) * svs_pw / png_pw);

                cind += 1;
                rind += 1;
                if rind % 100 == 0: print('Processed: ', rind)
        if xind >= BatchSize:
            break;

    X = X[0:xind];
    inds = inds[0:xind];
    coor = coor[0:cind];

    return todo_list[lind:], X, inds, coor, rind;


def from_output_to_pred(output):
    pred = np.copy(output);
    pred = (pred >= 0.5).astype(np.int32);
    return pred;


def val_fn_epoch_on_disk(classn, val_fn):
    all_or = np.zeros(shape=(500000, classn), dtype=np.float32);
    all_inds = np.zeros(shape=(500000,), dtype=np.int32);
    all_coor = np.zeros(shape=(500000, 2), dtype=np.int32);
    rind = 0;
    n1 = 0;
    n2 = 0;
    n3 = 0;
    todo_list = os.listdir(TileFolder);
    processed = 0
    total = len(todo_list)
    start = time.time()
    coor_c = 0
    while len(todo_list) > 0:
        todo_list, inputs, inds, coor, rind = load_data(todo_list, rind);
        coor_c += len(coor)

        #if len(inputs) == 0:
        #    print('len of inputs is 0"')
        #    break;
        if inputs.size(0) < 2:
            print('len of inputs if less than 2')
        else:
            processed = total - len(todo_list)
            print('Processed: {}/{} \t Time Remaining: {}mins'.format(processed, total, (time.time() - start)/60*(total/processed - 1)))
            with torch.no_grad():
                inputs = Variable(inputs.to(device))
                output = val_fn(inputs)
            output = output.data.cpu().numpy()
            output = softmax_np(output)
            # output = output[:,0] + output[:,1]  # sum of probabilies of the 1st 2 classes Grade3 and Grade4-5
            # all_or[n1:n1+len(output)] = output.reshape(-1,1)
            all_or[n1:n1+len(output)] = output
            n1 += len(output)
            all_inds[n2:n2+len(inds)] = inds;
            n2 += len(inds);

        all_coor[n3:n3+len(coor)] = coor;
        n3 += len(coor);

    all_or = all_or[:n1];
    all_inds = all_inds[:n2];
    all_coor = all_coor[:n3];
    return all_or, all_inds, all_coor;

def confusion_matrix(Or, Tr, thres):
    tpos = np.sum((Or>=thres) * (Tr==1));
    tneg = np.sum((Or< thres) * (Tr==0));
    fpos = np.sum((Or>=thres) * (Tr==0));
    fneg = np.sum((Or< thres) * (Tr==1));
    return tpos, tneg, fpos, fneg;

def auc_roc(Pr, Tr):
    fpr, tpr, _ = roc_curve(Tr, Pr, pos_label=1.0);
    return auc(fpr, tpr);

def parallelize_model(model):
    if torch.cuda.is_available():
        model = model.to(device)
        model = torch.nn.DataParallel(model, device_ids=[0,1])
        cudnn.benchmark = True
        return model
def unparallelize_model(model):
    try:
        while 1:
            # to avoid nested dataparallel problem
            model = model.module
    except AttributeError:
        pass
    return model

# load model
print('start predicting...')
start = time.time()

old_model = '../../data/models_cnn/RESNET_34_prostate_trueVal_hard_train__0530_0015_0.954882634484846_1919.t7'

print("| Load pretrained at  %s..." % old_model)
checkpoint = torch.load(old_model, map_location=lambda storage, loc: storage)
model = checkpoint['model']
model = unparallelize_model(model)
model.to(device)
#model = parallelize_model(model)
model.train(False)
best_auc = checkpoint['f1-score']
print('previous best F1-score: \t%.4f'% best_auc)
print('=============================================')

classn = 4
Or, inds, coor = val_fn_epoch_on_disk(classn, model);    # 4 classes: grade3, grade4-5, benign, stroma
Or_all = np.zeros(shape=(coor.shape[0], classn), dtype=np.float32);
Or_all[inds] = Or

print('len of all coor: ', coor.shape)
print('shape of Or: ', Or.shape)
print('shape of inds: ', inds.shape)

fid = open(TileFolder + '/' + heat_map_out, 'w');
fid_grades = open(TileFolder + '/' + heat_map_out[:-4] + '_grades.txt', 'w');
for idx in range(0, Or_all.shape[0]):
    fid.write('{} {} {}\n'.format(coor[idx][0], coor[idx][1], Or_all[idx][0] + Or_all[idx][1]))
    fid_grades.write('{} {} {} {} {} {}\n'.format(coor[idx][0], coor[idx][1], Or_all[idx][0], Or_all[idx][1], Or_all[idx][2], Or_all[idx][3]))

fid.close();
fid_grades.close()


print('Elapsed Time: ', (time.time() - start)/60.0)
print('DONE!');
