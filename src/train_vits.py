import torch
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from PIL import Image, ImageFile
import os
import argparse
from utils import progress_bar
from os.path import abspath

# from vits import *
# from vits.vits_model import vit_test
from vits_models.vision_transformer import vit_tiny_patch16_224 as vit_test


ImageFile.LOAD_TRUNCATED_IMAGES = True


# USING_HOST_DATAPATH = True
# datapath_style = 'host'

# parsing CL str into python
parser = argparse.ArgumentParser(description='PyTorch fashion textile ViTs Training')
parser.add_argument('--data', default='fibre', type=str, help='type of data')
parser.add_argument('--data_parent_dir', default='../data', type=str, help='type of data')
parser.add_argument('--lr', default=5e-4, type=float, help='learning rate')
parser.add_argument('--batch_size', default=256, type=int, help='batch size')
parser.add_argument('--num_workers', default=0, type=int, help='number of workers')
parser.add_argument('--num_classes', default=10, type=int, help='classes')
parser.add_argument('--resume', '-r', action='store_true',
                    help='resume from checkpoint_vits')
parser.add_argument('--epochs', default=100, type=int, help='number of epochs to train')
args = parser.parse_args()

DATA = args.data
train_data = f'{args.data_parent_dir}/{DATA}/train'
test_data = f'{args.data_parent_dir}/{DATA}/test'


# else:
#     # cluster image dir
#     JOB_ID = os.getenv('JOB_ID')
#     #data = '/SAN/uclic/TextileNet/data_iMaterial/fiber/'
#     data = f'/scratch0/szhong/{JOB_ID}/data_iMaterial/fiber'
#     # data = abspath('/SAN/uclic/TextileNet/data_iMaterial/')
#     PATH_TO_TEST = ('/SAN/uclic/TextileNet/data_iMaterial/test')

# Load data
# resize the dataload image, DatasetFolder for data loader
def my_collate(batch):
    data = torch.stack([item[0] for item in batch])
    target = [item[1] for item in batch]
    target = torch.LongTensor(target)
    # print(data.shape)
    return [data, target]

# def show_image_batch(img_list, title=None):
#     num = len(img_list)
#     # num = 5
#     fig = plt.figure()
#     for i in range(num):
#         ax = fig.add_subplot(1, num, i+1)
#         ax.imshow(img_list[i].numpy().transpose([1,2,0]))
#         ax.set_title(title[i])
#     plt.show()

# use gpu, if not cpu
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch

# slip the ImageForlder data into dataloader


# transform PIL image to tensor, for train set, do some rotation or flip
data_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.RandomSizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
test_transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
])


# pytorch ImageFolder, dataset stores sample and its label;
# slip the ImageFolder data into dataloader
train_set = datasets.ImageFolder(root=train_data, transform=data_transform)
test_set = datasets.ImageFolder(root=test_data, transform=test_transform)

# dataloader wraps an iterable dataset, with batchsize and other infos 

train_loader = torch.utils.data.DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, collate_fn=my_collate)
test_loader = torch.utils.data.DataLoader(test_set, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, collate_fn=my_collate)

model = vit_test(num_classes=args.num_classes)
print(f"Number of devices: {torch.cuda.device_count()}")
if torch.cuda.device_count() >= 1:
    print("Let's use", torch.cuda.device_count(), "GPUs!")
    model = nn.DataParallel(model)
model.to(device)


if args.resume:
    # Load checkpoint.
    print('==> Resuming from checkpoint..')
    assert os.path.isdir(f'./{DATA}_results/checkpoint_vits'), 'Error: no checkpoint directory found!'
    checkpoint = torch.load(f'./{DATA}_results/checkpoint_vits/ckpt.pth')
    model.load_state_dict(checkpoint['model'])
    best_acc = checkpoint['acc']
    start_epoch = checkpoint['epoch']

# This criterion computes the cross entropy loss between input and target.
criterion = nn.CrossEntropyLoss()
# initial optimizer as SGD

# optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

# change lr from high to nearly 0 schedully
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)

def topk(output, target, k=(1, ), count=False, regression=False):
    try:
        _, pred = output.topk(max(k), 1, True, True)
    except:
        import pdb; pdb.set_trace()
        # if graph_mask is not None:
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    batch = 1 if count else target.size(0)
    return [float(correct[:k].sum()) / batch for i, k in enumerate(k)]

def train(epoch):
    print('\nEpoch: %d' % epoch)
    model.train() 
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(train_loader):
        # in pytorch, .to is transfer to (), here is move to device
        inputs, targets = inputs.to(device), targets.to(device)
        # except for the 1st loop, need to zero the gradient due to there is an auto differentian in . backward
        optimizer.zero_grad()
        # compute output, which is the label
        outputs = model(inputs)
        # use crossEntropyLoss to calculate loss
        loss = criterion(outputs, targets)
        # auto grad calculation
        loss.backward() 
        # w = w + wg*lr
        optimizer.step()

        train_loss += loss.item()
        # pick the index of the max output
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
        tops = topk(outputs, targets, (1, 3, 5))


        progress_bar(batch_idx, len(train_loader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                     % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))


def test(epoch):
    global best_acc
    model.eval()
    test_loss = 0
    correct = 0
    total = 0
    top_1s, top_3s, top_5s = [], [], []
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(test_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            tops = topk(outputs, targets, (1, 3, 5))
            top_1s.append(tops[0])
            top_3s.append(tops[1])
            top_5s.append(tops[2])
            progress_bar(batch_idx, len(test_loader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                         % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))

    # Save checkpoint.
    acc = 100.*correct/total
    if acc > best_acc:
        print('Saving..')
        state = {
            'model': model.state_dict(),
            'acc': acc,
            'epoch': epoch,
        }
        if not os.path.isdir(f'./{DATA}_results/checkpoint_vits'):
            os.mkdir(f'./{DATA}_results/checkpoint_vits')
        torch.save(state, f'./{DATA}_results/checkpoint_vits/ckpt.pth')
        best_acc = acc

    return sum(top_1s)/len(top_1s), sum(top_3s)/len(top_3s), sum(top_5s)/len(top_5s)


epoch_wise_top1s, epoch_wise_top3s, epoch_wise_top5s = [], [], []
for epoch in range(start_epoch, start_epoch+args.epochs):
    train(epoch)
    top1, top3, top5 = test(epoch)
    # scheduler.step()
    epoch_wise_top1s.append(top1)
    epoch_wise_top3s.append(top3)
    epoch_wise_top5s.append(top5)

with open(f'./{DATA}_results/top1_vit.txt', 'w') as f:
    for item in epoch_wise_top1s:
        f.write("%s\n" % item)

with open(f'./{DATA}_results/top3_vit.txt', 'w') as f:
    for item in epoch_wise_top3s:
        f.write("%s\n" % item)

with open(f'./{DATA}_results/top5_vit.txt', 'w') as f:
    for item in epoch_wise_top5s:
        f.write("%s\n" % item)



with open(f'./{DATA}_results/vits_acc.png', 'w') as f:
    import matplotlib.pyplot as plt
    plt.plot(range(start_epoch, start_epoch+args.epochs), epoch_wise_top1s)
    plt.plot(range(start_epoch, start_epoch+args.epochs), epoch_wise_top3s)
    plt.plot(range(start_epoch, start_epoch+args.epochs), epoch_wise_top5s)
    plt.savefig('vits_acc.png')


print('Finished Training')