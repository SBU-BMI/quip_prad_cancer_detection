import argparse
from torchvision import transforms
import time
import os
import sys
import glob
import copy
from time import strftime
from sklearn.metrics import mean_squared_error, accuracy_score, hamming_loss, roc_curve, auc, f1_score, confusion_matrix
from torch.utils.data import DataLoader, Dataset
import torchvision.models as models
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import numpy as np
import random
from utils import *

rand_seed = 26700
device = torch.device("cuda:0")
if rand_seed is not None:
    np.random.seed(rand_seed)
    torch.manual_seed(rand_seed)
    torch.cuda.manual_seed(rand_seed)


def get_args():
    parser = argparse.ArgumentParser(description='PyTorch Digital Mammography Training')
    parser.add_argument('--lr', default=1e-2, type=float, help='learning rate')
    parser.add_argument('--net_depth', default=34, type=int)
    parser.add_argument('--APS', default=175, type=int)
    parser.add_argument('--weight_decay', default=1e-4, type=float, help='weight decay')
    parser.add_argument('--batch_size', default=256, type=int)
    parser.add_argument('--num_workers', default=8, type=int)
    parser.add_argument('--num_epochs', default=100, type=int, help='Number of epochs in training')
    parser.add_argument('--check_after', default=2, type=int, help='check the network after check_after epoch')
    parser.add_argument('--n_class', default=3, type=int, help='number of classes in the classification problem')

    args = parser.parse_args()
    return args


def train_model(model, args, criterion, train_loader, val_loader):
    best_f1 = 0
    best_epoch = 0
    start_training = time.time()
    num_epochs = args.num_epochs
    model.train(True)

    for epoch in range(num_epochs):
        start = time.time()
        lr = adjust_learning_rate(args.lr, epoch)

        if epoch >= 10:
            for param in model.parameters():
                param.requires_grad = True

        optimizer = optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=lr, momentum=0.9,
                              weight_decay=args.weight_decay)

        print('Epoch {}/{}'.format(epoch + 1, num_epochs))
        print('lr: {:.6f}'.format(lr))
        print('-' * 50)

        running_loss = 0.0
        running_corrects = 0
        N_tot = 0
        labels_train = torch.zeros(0).type(torch.LongTensor)
        preds_train = torch.zeros(0).type(torch.LongTensor).to(device)
        for ix, data in enumerate(train_loader):
            if (len(train_loader.dataset) - N_tot) < 2: continue
            inputs, labels = data
            labels_train = torch.cat((labels_train, labels.type(torch.LongTensor)))

            inputs = Variable(inputs.to(device))
            labels = Variable(labels.type(torch.LongTensor).to(device))

            optimizer.zero_grad()
            outputs = model(inputs)
            if type(outputs) == tuple:  # for inception_v3 output
                outputs, _ = outputs

            _, preds = torch.max(outputs.data, 1)

            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            N_tot += outputs.size(0)
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)

            preds_train = torch.cat((preds_train, preds))

        unique, counts = np.unique(np.array(labels_train), return_counts=True)
        print('| Epoch:{}\tTrain_Loss: {:.4f}\tAccuracy: {:.4f}\tTrain_data: {}\tTime: {:.2f} mins'.format(
            epoch + 1,
            running_loss / N_tot,
            running_corrects.item() / N_tot,
            dict(zip(unique, counts)),
            (time.time() - start) / 60.0))

        try:
            conf_matrix = confusion_matrix(labels_train.to(device), preds_train,
                                           labels=[i for i in range(args.n_class)])
            print(conf_matrix)
        except:
            print('could not compute confusion matrix.')
        sys.stdout.flush()

        # VALIDATION ======================================================
        if (epoch + 1) % args.check_after == 0:
            model.eval()
            start = time.time()
            val_acc, f1, Pr, Tr, val_loss, labels_val = val_fn_epoch(model=model, criterion=criterion,
                                                                     val_loader=val_loader)
            print(
                "Epoch: {}\tVal_Loss: {:.4f}\tAccuracy: {:.4f}\tF1-score: {:.4f}\tVal_data: {}\tTime: {:.3f}mins".format(
                    (epoch + 1), val_loss, val_acc, f1, labels_val, (time.time() - start) / 60.0))

            try:
                conf_matrix = confusion_matrix(Tr, Pr, labels=[i for i in range(args.n_class)])
                print(conf_matrix)
            except:
                print('could not compute confusion matrix.')

            # deep copy the model
            if f1 > best_f1 and epoch > 2:
                print('Saving model')
                best_f1 = f1
                best_epoch = epoch + 1
                best_model = copy.deepcopy(model)
                state = {
                    'model': best_model,
                    'f1-score': best_f1,
                    'args': args,
                    'lr': lr,
                    'saved_epoch': epoch,
                }
                checkpoint = '/data/output/checkpoint'
                if not os.path.isdir(checkpoint):
                    os.mkdir(checkpoint)
                save_point = checkpoint
                if not os.path.isdir(save_point):
                    os.mkdir(save_point)

                saved_model_fn = 'resnet{}_{}_bestF1_{:.4f}_epoch_{}.t7'.format(args.net_depth,
                                                                        strftime('%m%d_%H%M'),
                                                                        best_f1,
                                                                        epoch)
                torch.save(state, os.path.join(save_point, saved_model_fn))
                print('=======================================================================')

    time_elapsed = time.time() - start_training
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best F1-score: {:4f} at epoch: {}'.format(best_f1, best_epoch))


def adjust_learning_rate(lr, epoch):
    if epoch < 5:
        lr = lr
    elif epoch < 10:
        lr = lr / 2
    elif epoch < 15:
        lr = lr / 10
    elif epoch < 20:
        lr = lr / 50
    else:
        lr = lr / 100
    return lr


def val_fn_epoch(model, criterion, val_loader):
    nline = 0
    running_loss = 0.0
    labels_val = torch.zeros(0).type(torch.LongTensor)
    preds_val = torch.zeros(0).type(torch.LongTensor).to(device)
    with torch.no_grad():
        for ix, batch in enumerate(val_loader):
            if (len(val_loader.dataset) - nline) < 2: continue
            inputs, targets = batch

            labels_val = torch.cat((labels_val, targets.type(torch.LongTensor)))
            inputs = Variable(inputs.to(device))
            targets = Variable(targets.type(torch.LongTensor).to(device))
            output = model(inputs)
            if type(output) == tuple:
                output, _ = output
            N = output.size(0)

            loss = criterion(output, targets)
            running_loss += loss.item() * N
            _, preds = torch.max(output.data, 1)  # get the argmax index along the axis 1
            preds_val = torch.cat((preds_val, preds))

    labels_val = labels_val.to(device)
    val_acc = accuracy_score(labels_val, preds_val)
    f1 = f1_score(labels_val, preds_val,
                  average='macro')  # Calculate metrics for each label, and find their average weighted
    print('F1-score of each class for validation: ', f1_score(labels_val, preds_val, average=None))

    unique, counts = np.unique(np.array(labels_val), return_counts=True)
    return val_acc, f1, preds_val, labels_val, running_loss / labels_val.size(0), dict(zip(unique, counts))


def get_model(net_depth, n_class):
    if net_depth == 34:
        model = models.resnet34(pretrained=True)
    elif net_depth == 50:
        model = models.resnet50(pretrained=True)
    elif net_depth == 101:
        model = models.resnet101(pretrained=True)
    elif net_depth == 152:
        model = models.resnet152(pretrained=True)
    else:
        model = models.resnet18(pretrained=True)

    for param in model.parameters():
        param.requires_grad = False

    num_in = model.fc.in_features
    model.fc = nn.Linear(num_in, n_class)  # benign vs. tumor
    model = model.to(device)
    if torch.cuda.device_count() >= 2:  # use multiple GPUs
        model = torch.nn.DataParallel(model, device_ids=[0, 1])
        cudnn.benchmark = True

    return model


def get_data_transforms(mean, std, APS, input_size=224):
    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomCrop(APS),  # perform random crop manually in the dataloader
            transforms.Scale(input_size),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)]),

        'val': transforms.Compose([
            transforms.CenterCrop(APS),
            transforms.Scale(input_size),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])
    }
    return data_transforms


# Check if the path specified is a valid directory that contains files.
def isEmpty(path):
    flag = 0
    if os.path.exists(path) and os.path.isdir(path):
        # Checking if the directory is empty or not
        if not os.listdir(path):
            print(path + ": This directory is empty.")
            flag = 1
    else:
        # if pgm continues, you will get division by zero.
        print(path + ": This path is either a file or is not valid.")
        flag = 1

    return flag


def main():
    args = get_args()
    print(args)

    use_gpu = torch.cuda.is_available()
    print('Using GPU: ', use_gpu)

    mean = [0.6462, 0.5070, 0.8055]  # for Prostate cancer
    std = [0.1381, 0.1674, 0.1358]

    data_transforms = get_data_transforms(mean, std, args.APS)

    # train_seer_fol = '/data10/shared/hanle/extract_prad_seer_john/patches_prad_seer_4classes'
    # train_beatrice_fol = '/data10/shared/hanle/extract_prad_seer_john/beatrice_training_4classes'
    # val_fol = '/data10/shared/hanle/extract_prad_seer_john/beatrice_validation_4classes'
    # img_trains = glob.glob(os.path.join(train_seer_fol, '*png')) + glob.glob(os.path.join(train_beatrice_fol, '*png'))
    # img_vals = glob.glob(os.path.join(val_fol, '*png'))

    train_fol = '/data/input/training_data'
    val_fol = '/data/input/validation_data'

    if isEmpty(train_fol):
        exit(1)
    if isEmpty(val_fol):
        exit(1)

    pattern = "*.png"
    img_trains = []
    img_vals = []

    for m_dir, _, _ in os.walk(train_fol):
        img_trains.extend(glob.glob(os.path.join(m_dir, pattern)))

    for m_dir, _, _ in os.walk(val_fol):
        img_vals.extend(glob.glob(os.path.join(m_dir, pattern)))

    print('len of train/val set: ', len(img_trains), len(img_vals))

    # for a quick demo training...
    random.shuffle(img_trains)
    random.shuffle(img_vals)
    img_trains, img_vals = img_trains[:20000], img_vals[:1000]
    print('len of train/val set: ', len(img_trains), len(img_vals))

    train_set = data_loader(img_trains, transform=data_transforms['train'])
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)

    val_set = data_loader(img_vals, transform=data_transforms['val'])
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    model = get_model(args.net_depth, args.n_class)
    print('Start training ... ')

    criterion = nn.CrossEntropyLoss().to(device)
    train_model(model, args, criterion, train_loader=train_loader, val_loader=val_loader)


if __name__ == "__main__":
    main()
