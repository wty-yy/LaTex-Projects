import torch
import torch.nn as nn
import torch.optim as opt

from tqdm import tqdm
from torch.utils.data import DataLoader
from datasets import MyDataset
from funcation import *
from evaluater import *
from model import *

device = 'cuda:0'
batch_size = 32

# prepare dataset
root_path = 'data/256_ObjectCategories/'
test_dataset = MyDataset(root_path, 'test.txt', device)
train_dataset = MyDataset(root_path, 'train.txt', device)
test_dataloader = DataLoader(test_dataset, batch_size, shuffle=True)
train_dataloader = DataLoader(train_dataset, batch_size, shuffle=True)

# set trainning config
epoches = 200
weight_decay = 0
learning_rate = 1.5e-3
evaluater = SEvaluate()
loss_funcation = nn.CrossEntropyLoss()
#VGG_CBAM = VGG_CBAM(3, 256, 2).to(device)
VGG = VGG(3, 256).to(device)
#optimizer = opt.Adam(params=VGG_CBAM.parameters(), lr=learning_rate, weight_decay=weight_decay)
optimizer = opt.Adam(params=VGG.parameters(), lr=learning_rate, weight_decay=weight_decay)

# record trainning process
test_acc = []
train_acc = []
test_loss = []
train_loss = []

# trainning
for epoch in tqdm(range(epoches)):
    avg_loss = 0
    avg_acc = 0
    num = 0
    #VGG_CBAM.train()
    VGG.train()
    if epoch == 100:
        learning_rate /= 10
        #optimizer = opt.Adam(params=VGG_CBAM.parameters(), lr=learning_rate, weight_decay=weight_decay)
        optimizer = opt.Adam(params=VGG.parameters(), lr=learning_rate, weight_decay=weight_decay)
    for image, label in tqdm(train_dataloader):
        #out = VGG_CBAM.forward(image)
        out = VGG.forward(image)
        loss = loss_funcation(out, label.squeeze(dim=1))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        num += image.shape[0]
        avg_loss += loss.item()
        avg_acc += get_acc(out, label).item()

    avg_loss = avg_loss*batch_size/num
    avg_acc /= num
    train_loss.append(avg_loss)
    train_acc.append(avg_acc)
    print('epoch ' + str(epoch) + ': train_loss: ' + str(avg_loss) + ', train_acc: ' + str(avg_acc))
    if epoch%10 == 0:  
        avg_loss = 0
        avg_acc = 0
        num = 0
        #VGG_CBAM.eval()
        VGG.eval()
        for image, label in tqdm(test_dataloader):
            #out = VGG_CBAM.forward(image)
            out = VGG.forward(image)
            loss = loss_funcation(out, label.squeeze(dim=1))
            num += image.shape[0]
            avg_loss += loss.item()
            avg_acc += get_acc(out, label).item()
        avg_loss = avg_loss*batch_size/num
        avg_acc /= num
        test_loss.append(avg_loss)
        test_acc.append(avg_acc)
        print('epoch ' + str(epoch) + ': test_loss: ' + str(avg_loss) + ', test_acc: ' + str(avg_acc))
        print('epoch{} :test_loss: {}    test_acc: {}'.format(epoch, avg_loss, avg_acc))
    evaluater.visualize(train_loss, test_loss, 'loss', VGG)
    evaluater.visualize(train_acc, test_acc, 'acc', VGG)