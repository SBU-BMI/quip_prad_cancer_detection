from torchvision import transforms
import time
from time import strftime
from sklearn.metrics import mean_squared_error, accuracy_score, hamming_loss, roc_curve, auc, f1_score, confusion_matrix
import copy
import collections
import torch
import numpy as np
from torch.utils.data import DataLoader, Dataset
import torch.optim as optim
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torchvision.models as models
from torch.autograd import Variable
import os
import sys
import torch.nn as nn
import random
import glob
import argparse
from PIL import Image


def parallelize_model(model):
    if torch.cuda.is_available():
        model = model.cuda()
        # model = torch.nn.DataParallel(model, device_ids=range(torch.cuda.device_count()))
        model = torch.nn.DataParallel(model, device_ids=[0, 1])
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


def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def cvt_to_gpu(X):
    return Variable(X.cuda()) if torch.cuda.is_available() \
    else Variable(X)


def val_fn_epoch(val_fn = None, crit = None, val_loader = None):
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
            output = val_fn(inputs)
            if type(output) == tuple:
                output,_ = output
            N = output.size(0)

            loss = crit(output, targets)
            running_loss += loss.item() * N
            _, preds = torch.max(output.data, 1)        # get the argmax index along the axis 1
            preds_val = torch.cat((preds_val, preds))

    labels_val = labels_val.to(device)
    val_acc = accuracy_score(labels_val, preds_val)
    f1 = f1_score(labels_val, preds_val, average='macro')    # Calculate metrics for each label, and find their average weighted
    print(f1_score(labels_val, preds_val, average=None))      # print F1-score for each class

    unique, counts = np.unique(np.array(labels_val), return_counts=True)
    return val_acc, f1, preds_val, labels_val, running_loss/labels_val.size(0), dict(zip(unique, counts))


def train_model(model, criterion = None, num_epochs=100, train_loader = None, val_loader = None):
    best_f1 = 0
    best_epoch = 0
    start_training = time.time()

    for epoch in range(num_epochs):
        start = time.time()

        if epoch < 8: lr = args.lr
        elif epoch < 12: lr = args.lr/2
        elif epoch < 15: lr = args.lr/10
        elif epoch < 20: lr = args.lr / 50
        else: lr = args.lr/100

        if epoch >= 2:
            for param in model.parameters():
                param.requires_grad = True

        optimizer = optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=lr, momentum=0.9, weight_decay=args.weight_decay)

        print('Epoch {}/{}'.format(epoch + 1, num_epochs))
        print('lr: {:.6f}'.format(lr))
        print('-' * 50)

        for phase in ['train']:
            if phase == 'train':
                data_loader = train_loader
                model.train(True)
            else:
                data_loader = val_loader
                model.train(False)

            running_loss = 0.0
            running_corrects = 0
            N_tot = 0
            labels_train = torch.zeros(0).type(torch.LongTensor)
            preds_train = torch.zeros(0).type(torch.LongTensor).to(device)
            for ix, data in enumerate(data_loader):
                if (len(data_loader.dataset) - N_tot) < 3: continue
                inputs, labels = data
                labels_train = torch.cat((labels_train, labels.type(torch.LongTensor)))

                inputs = Variable(inputs.to(device))
                labels = Variable(labels.type(torch.LongTensor).to(device))

                optimizer.zero_grad()
                outputs = model(inputs)
                if type(outputs) == tuple:  # for inception_v3 output
                    outputs,_ = outputs

                _, preds = torch.max(outputs.data, 1)   # preds are the index of the maximum element

                loss = criterion(outputs, labels)
                if phase == 'train':
                    loss.backward()
                    optimizer.step()

                N_tot += outputs.size(0)
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

                preds_train = torch.cat((preds_train, preds))

            unique, counts = np.unique(np.array(labels_train), return_counts=True)
            print('| Epoch:[{}][{}/{}]\tTrain_Loss: {:.4f}\tAccuracy: {:.4f}\tTrain_data: {}\tTime: {:.2f} mins'.format(epoch + 1, ix + 1,
                 len(data_loader.dataset)//args.batch_size,
                 running_loss / N_tot, running_corrects.item() / N_tot, dict(zip(unique, counts)), (time.time() - start)/60.0))

            try:
                conf_matrix = confusion_matrix(labels_train.to(device), preds_train, labels=[i for i in range(n_class)])
                print(f1_score(labels_train.to(device), preds_train, average=None))
                print(conf_matrix)
            except:
                print('could not compute confusion matrix.')
            sys.stdout.flush()

            ############ VALIDATION #############################################
            if (epoch + 1) % args.check_after == 0:
                model.eval()
                start = time.time()
                val_acc, f1, Pr, Tr, val_loss, labels_val = val_fn_epoch(val_fn = model, crit = criterion, val_loader = val_loader)
                print("Epoch: {}\tVal_Loss: {:.4f}\tAccuracy: {:.4f}\tF1-score: {:.4f}\tVal_data: {}\tTime: {:.3f}mins".format(
                    (epoch + 1), val_loss, val_acc, f1, labels_val, (time.time() - start)/60.0))

                try:
                    conf_matrix = confusion_matrix(Tr, Pr, labels=[i for i in range(n_class)])
                    print(conf_matrix)
                except:
                    print('could not compute confusion matrix.')

                start = time.time()


                save_point = './checkpoint/'
                if not os.path.isdir(save_point):
                    os.mkdir(save_point)
                saved_model_fn = args.net_type + '_' + strftime('%m%d_%H%M')
                best_model = copy.deepcopy(model)
                state = {
                    'model': best_model,
                    'f1-score': f1,
                    'args': args,
                    'lr': lr,
                    'saved_epoch': epoch,
                }

                # deep copy the model
                if f1 > best_f1 and epoch > 2:
                    print('Saving model')
                    best_f1 = f1
                    best_epoch = epoch + 1
                    torch.save(state, save_point + saved_model_fn + '_bestF1_' + str(f1) + '_' + str(epoch) + '.t7')
                    print('=======================================================================')
                else:
                    torch.save(state, save_point + saved_model_fn + '_F1_' + str(f1) + '_' + str(epoch) + '.t7')


    time_elapsed = time.time() - start_training
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best F1-score: {:4f} at epoch: {}'.format(best_f1, best_epoch))


def get_data_transforms(APS, input_size=224):
    mean = [0.6462,  0.5070,  0.8055]      # for Prostate cancer
    std = [0.1381,  0.1674,  0.1358]

    return {'train': transforms.Compose([           # 2 steps of data augmentation for training
            transforms.RandomCrop(APS),       # perform random crop manually in the dataloader
            #transforms.Scale(input_size),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)]),

        'val': transforms.Compose([
            transforms.CenterCrop(APS),
            #transforms.Scale(input_size),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
            ])}


def get_model(n_class):
    model = models.resnet34(pretrained=True)
    for param in model.parameters():
        param.requires_grad = False
    num_in = model.fc.in_features
    model.fc = nn.Linear(num_in, n_class)
    model = model.to(device)
    model = torch.nn.DataParallel(model, device_ids=[0,1])
    cudnn.benchmark = True
    return model


def find_slides_from_slides(slides):
    counters = collections.Counter(slides)
    out = []
    for slide, cnt in counters.items():
        if 300 > cnt > 10:
            out.append(slide)

    out.sort()
    random.shuffle(out)
    return out


def find_val_slides_from_fns(fns, num_slides_per_class=5):
    all_slides = [f.split('/')[-1].split('.')[0] for f in fns]
    class_slides_maps = collections.defaultdict(list)
    for i, fn in enumerate(fns):
        class_slides_maps[convert_lb(fn[-5])].append(all_slides[i])

    results = []
    for classID, slides in class_slides_maps.items():
        results.extend(find_slides_from_slides(slides)[:num_slides_per_class])
    return results


def enable_random_seed(seed):
    random.seed(args.random_seed)
    np.random.seed(args.random_seed)
    torch.manual_seed(args.random_seed)
    torch.cuda.manual_seed(args.random_seed)


def convert_lb(lb):
    lb = int(lb)
    return lb


def compute_stats(imgs, dataset_name=''):
    stats = collections.defaultdict(int)
    for fn in imgs:
        stats[convert_lb(fn[-5])] += 1
    print(dataset_name, stats)


class data_loader(Dataset):
    def __init__(self, imgs, transform=None):
        self.imgs = imgs
        self.transform = transform

    def __getitem__(self, index):
        img = Image.open(self.imgs[index]).convert('RGB')
        img = self.transform(img)

        lb = int(self.imgs[index][-5])
        return img, convert_lb(lb)    # *_{lb}.png --> extract the label

    def __len__(self):
        return len(self.imgs)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='PyTorch Digital Mammography Training')
    parser.add_argument('--lr', default=1e-2, type=float, help='learning rate')
    parser.add_argument('--net_type', default='', type=str, help='model')
    parser.add_argument('--net_depth', default=34, type=int)
    parser.add_argument('--weighted', type=str2bool, default=False, help="apply weight to the loss")
    parser.add_argument('--APS', default = 224, type = int)
    parser.add_argument('--weight_decay', default=1e-4, type=float, help='weight decay')
    parser.add_argument('--batch_size', default=256, type=int)
    parser.add_argument('--num_workers', default=8, type=int)
    parser.add_argument('--random_seed', default=294321, type=int)
    parser.add_argument('--num_epochs', default=100, type=int, help='Number of epochs in training')
    parser.add_argument('--check_after', default=2, type=int, help='check the network after check_after epoch')
    parser.add_argument('--note', type=str, default=None, help="note while running the code")
    parser.add_argument('--add_beatrice', type=str2bool, default=False, help="using sampler method")

    args = parser.parse_args()
    args.net_type = '{}_netDepth-{}_APS-{}_randomSeed-{}'.format(os.path.basename(__file__).split('.')[0],\
                                str(args.net_depth), str(args.APS), str(args.random_seed))
    if args.note is not None:
        args.net_type += '_note-' + args.note

    with open(os.path.basename(__file__)) as f:
        codes = f.readlines()
    print('\n\n' + '=' * 20 + os.path.basename(__file__) + '=' * 20)
    for c in codes:
        print(c[:-1])
    print(args)

    enable_random_seed(args.random_seed)
    use_gpu = torch.cuda.is_available()
    print('Using GPU: ', use_gpu)
    device = torch.device("cuda:0")

    train_seer_fol = '/data10/shared/hanle/extract_prostate_seer_john_grade5_subtypes/patches_prostate_seer_john_6classes'

    '''
    Benign:                 0
    Gleason 3:              1
    Gleason 4:              2
    Gleason 5-Single Cells: 3
    Gleason 5-Secretions:   4
    Gleason 5:              5
    '''
    classes = {'0', '1', '2', '3'}
    n_class = 4
    img_fns = [f for f in glob.glob(os.path.join(train_seer_fol, '*.png')) if f[-5] in classes]

    print('len of img_fns: ', len(img_fns))

    val_slides = find_val_slides_from_fns(img_fns, num_slides_per_class=5)
    val_slides = set(val_slides)
    print('val slides: ', val_slides)
    print('len of val slides: ', len(val_slides))

    img_vals = [f for f in img_fns if f.split('/')[-1].split('.')[0] in val_slides]
    img_trains = [f for f in img_fns if f.split('/')[-1].split('.')[0] not in val_slides]

    print('len of train/val set: ', len(img_trains), len(img_vals))
    compute_stats(img_trains, 'training data stats: ')
    compute_stats(img_vals, 'val data stats: ')

    data_transforms = get_data_transforms(args.APS)
    train_set = data_loader(img_trains, transform=data_transforms['train'])
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)

    val_set = data_loader(img_vals, transform = data_transforms['val'])
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    print('Start training ... ')


    model = get_model(n_class)
    criterion = nn.CrossEntropyLoss().to(device)
    train_model(model, criterion, num_epochs=args.num_epochs, train_loader=train_loader, val_loader=val_loader)
