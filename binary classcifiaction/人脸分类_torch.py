# 这是一个二分类任务的torch基本编程方法
# https://aistudio.baidu.com/aistudio/datasetdetail/33766


import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset,DataLoader
import torchvision
from torchvision import transforms
import os
import zipfile
import random
import json
import numpy as np
from PIL import Image
from torchvision import models
import time
train_set_dir = "face"

batch_size=32


train_transforms = transforms.Compose(
        [transforms.RandomResizedCrop(size=256, scale=(0.8, 1.0)),#随机裁剪到256*256
        transforms.RandomRotation(degrees=15),#随机旋转
        transforms.RandomHorizontalFlip(),#随机水平翻转
        transforms.ToTensor(),#转化成张量
])

train_datasets = torchvision.datasets.ImageFolder(train_set_dir,train_transforms)
train_data_size = len(train_datasets)
train_data = torch.utils.data.DataLoader(train_datasets, batch_size=batch_size, shuffle=True)
valid_data = None # 定义验证集读取
valid_data_size = len(valid_data)
print(train_data_size)

resnet_50 = models.resnet50(pretrained=True)
for param in resnet_50.parameters():
    param.requires_grad = False

fc_inputs = resnet_50.fc.in_features
resnet_50.fc = nn.Sequential(
    nn.Linear(fc_inputs,32),
    nn.ReLU(True),
    nn.Dropout(0.4),
    nn.Linear(32,2)
)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(resnet_50.parameters(),lr=0.001)


def train(model,criterion,optimizer):
    device = "cuda:0"
    model = model.to(device)
    record =[]
    best_acc = 0.0
    best_epoch =0
    for epoch in range(10):
        epoch_start = time.time()
        print("Epoch: {}/{}".format(epoch + 1, 10))
        model.train()
        train_loss= 0.0
        train_acc = 0.0

        valid_loss = 0.0
        valid_acc = 0.0
        for i,(inputs,labels) in enumerate(train_data):

            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            pre = torch.softmax(outputs,dim=1)
            loss = criterion(outputs,labels)
            loss.backward()
            optimizer.step()
            train_loss+=loss.item()*inputs.size(0)
            ret,predictions = torch.max(pre.data,1)
            correct_counts  = predictions.eq(labels.data.view_as(predictions))
            acc = torch.mean(correct_counts.type(torch.FloatTensor))
            train_acc+=acc.item()*inputs.size(0)
        #利用验证集进行验证
        with torch.no_grad():
            model.eval()  # 验证

            for j, (inputs, labels) in enumerate(valid_data):
                inputs = inputs.to(device)
                labels = labels.to(device)

                outputs = model(inputs)

                loss = criterion(outputs, labels)

                valid_loss += loss.item() * inputs.size(0)

                ret, predictions = torch.max(outputs.data, 1)
                correct_counts = predictions.eq(labels.data.view_as(predictions))

                acc = torch.mean(correct_counts.type(torch.FloatTensor))

                valid_acc += acc.item() * inputs.size(0)

        avg_train_loss = train_loss/train_data_size
        avg_train_acc = train_acc/train_data_size
        avg_valid_loss = valid_loss / valid_data_size
        avg_valid_acc = valid_acc / valid_data_size

        record.append([avg_train_loss, avg_valid_loss, avg_train_acc, avg_valid_acc])

        if avg_valid_acc > best_acc  :#记录最高准确性的模型
            best_acc = avg_valid_acc
            best_epoch = epoch + 1
        print("ACC:{:.4f},LOSS:{:.4f}".format(avg_train_acc,avg_train_loss))
        epoch_end = time.time()

        print("Epoch: {:03d}, Training: Loss: {:.4f}, Accuracy: {:.4f}%, \n\t\tValidation: Loss: {:.4f}, Accuracy: {:.4f}%, Time: {:.4f}s".format(
                epoch + 1, avg_valid_loss, avg_train_acc * 100, avg_valid_loss, avg_valid_acc * 100,
                epoch_end - epoch_start))
        print("Best Accuracy for validation : {:.4f} at epoch {:03d}".format(best_acc, best_epoch))

        torch.save(model, '' + str(epoch + 1) + '.pth')
    return model, record


mo,record = train(resnet_50,criterion,optimizer)