import os
import zipfile
import random
import json
import cv2
import numpy as np
from PIL import Image
import paddle
import paddle.fluid as fluid
from multiprocessing import cpu_count
from paddle.fluid.dygraph import Linear
from paddle.fluid.dygraph import Pool2D,Conv2D
import matplotlib.pyplot as plt

train_parameters = {
    "input_size": [1, 32, 32],                           #输入图片的shape
    "class_dim": -1,                                     #分类数
    "src_path":"face",          #原始数据集路径
    "train_list_path": "dataset/train_data.txt",              #train_data.txt路径
    "eval_list_path": "dataset/val_data.txt",                  #eval_data.txt路径
    "label_dict":{},                                    #标签字典
    "readme_path": "readme.json",   #readme.json路径
    "num_epochs": 2,                                    #训练轮数
    "train_batch_size": 8,                             #训练时每个批次的大小
    "learning_strategy": {                              #优化函数相关的配置
        "lr": 0.0001                                     #超参数学习率
    }
}

def get_data_list(data_path,train_list_path,eval_list_path):
    class_detail = []
    data_list_path = data_path
    class_dirs = os.listdir(data_list_path)

    # #总的图像数量
    all_class_images = 0
    # #存放类别标签
    class_label=0
    # #存放类别数目
    class_dim = 0
    # #存储要写进eval.txt和train.txt中的内容
    trainer_list=[]
    eval_list=[]

    for class_dir  in class_dirs:
        class_detail_list = {}
        eval_sum = 0
        trainer_sum = 0
        class_sum = 0
        path = os.path.join(data_path,class_dir)
        img_paths = os.listdir(path)
        for img_path in img_paths:
            name_path = os.path.join(path, img_path)  # 每张图片的路径
            if class_sum % 10 == 0:  # 每10张图片取一个做验证数据
                eval_sum += 1  # test_sum为测试数据的数目
                eval_list.append(name_path + "\t%d" % class_label + "\n")
            else:
                trainer_sum += 1
                trainer_list.append(name_path + "\t%d" % class_label + "\n")  # trainer_sum测试数据的数目
            class_sum += 1  # 每类图片的数目
            all_class_images += 1
            # 说明的json文件的class_detail数据

        class_detail_list['class_name'] = class_dir  # 类别名称
        class_detail_list['class_label'] = class_label  # 类别标签
        class_detail_list['class_eval_images'] = eval_sum  # 该类数据的测试集数目
        class_detail_list['class_trainer_images'] = trainer_sum  # 该类数据的训练集数目
        class_detail.append(class_detail_list)
        # 初始化标签列表
        train_parameters['label_dict'][str(class_label)] = class_dir
        class_label += 1
    # 初始化分类数
    train_parameters['class_dim'] = class_dim
    print(train_parameters)
    # 乱序
    random.shuffle(eval_list)
    with open(eval_list_path, 'a') as f:
        for eval_image in eval_list:
            f.write(eval_image)

    random.shuffle(trainer_list)
    with open(train_list_path, 'a') as f2:
        for train_image in trainer_list:
            f2.write(train_image)

            # 说明的json文件信息
    readjson = {}
    readjson['all_class_name'] = data_list_path  # 文件父目录
    readjson['all_class_images'] = all_class_images
    readjson['class_detail'] = class_detail
    jsons = json.dumps(readjson, sort_keys=True, indent=4, separators=(',', ': '))
    with open(train_parameters['readme_path'], 'w') as f:
        f.write(jsons)
    print('生成数据列表完成！')

data_path = train_parameters["src_path"]
train_list_path=train_parameters['train_list_path']
eval_list_path=train_parameters['eval_list_path']
batch_size=train_parameters['train_batch_size']

# 每次生成数据列表前，首先清空train.txt和eval.txt
with open(train_list_path, 'w') as f:
    f.seek(0)
    f.truncate()
with open(eval_list_path, 'w') as f:
    f.seek(0)
    f.truncate()

# 生成数据列表
get_data_list(data_path, train_list_path, eval_list_path)

def data_reader(file_list):

    def reader():
        with open(file_list,"r") as F:
            lines = [line.strip() for line in F]
            for line in lines:
                img_path,label = line.strip().split("\t")
                img = Image.open(img_path)
                img = img.resize((32,32),Image.ANTIALIAS)
                img = np.array(img).astype("float32")
                img = img/255.0
                yield img,label
    return reader

train_reader = paddle.batch(data_reader(train_list_path),
                            batch_size=batch_size,
                            drop_last=True)
eval_reader = paddle.batch(data_reader(eval_list_path),
                            batch_size=batch_size,
                            drop_last=True)



#定义网络
class MyCNN(fluid.dygraph.Layer):
    def __init__(self):
        super(MyCNN,self).__init__()
        self.hidden1 = Conv2D(1,32,3,1)
        self.hidden2 = Conv2D(32,64,3,1)
        self.hidden3 = Pool2D(pool_size=2,pool_type='max',pool_stride=2)
        self.hidden4 = Conv2D(64,128,3,1)
        self.hidden5 = Linear(128*12*12,2,act='softmax')
    def forward(self,input):
        x = self.hidden1(input)
        # print(x.shape)
        x = self.hidden2(x)
        # print(x.shape)
        x = self.hidden3(x)
        # print(x.shape)
        x = self.hidden4(x)
        # print(x.shape)
        x = fluid.layers.reshape(x, shape=[-1, 128*12*12])
        y = self.hidden5(x)
        return y


Batch=0
Batchs=[]
all_train_accs=[]
#定义draw_train_acc，绘制准确率变化曲线
def draw_train_acc(iters, train_accs):
    title="training accs"
    plt.title(title, fontsize=24)
    plt.xlabel("batch", fontsize=14)
    plt.ylabel("acc", fontsize=14)
    plt.plot(iters, train_accs, color='green', label='training accs')
    plt.legend()
    plt.grid()
    plt.show()
#定义draw_train_loss，绘制损失变化曲线
all_train_loss=[]
def draw_train_loss(iters, train_loss):
    title="training loss"
    plt.title(title, fontsize=24)
    plt.xlabel("batch", fontsize=14)
    plt.ylabel("loss", fontsize=14)
    plt.plot(iters, train_loss, color='red', label='training loss')
    plt.legend()
    plt.grid()
    plt.show()

with fluid.dygraph.guard():
    model = MyCNN()
    model.train()
    optimizer = paddle.fluid.optimizer.Adam(parameter_list=model.parameters(),learning_rate=train_parameters['learning_strategy']['lr'])

    EPOCH=10
    for epoch in range(EPOCH):
        for batch_id,data in enumerate(train_reader()):
            images = np.array([d[0].reshape(1,32,32) for d in data],dtype=np.float32)
            label = np.array([d[1] for d in data]).astype("int64")
            label = label[:,np.newaxis]
            image = fluid.dygraph.to_variable(images)
            label = fluid.dygraph.to_variable(label)
            predict = model(image)
            loss = fluid.layers.cross_entropy(predict, label)
            avg_loss = fluid.layers.mean(loss)  # 获取loss值
            acc = fluid.layers.accuracy(predict, label)  # 计算精度
            if batch_id != 0 and batch_id % 20 == 0:
                Batch = Batch + 20
                Batchs.append(Batch)
                all_train_loss.append(avg_loss.numpy()[0])
                all_train_accs.append(acc.numpy()[0])
                print(
                    "train_pass:{},batch_id:{},train_loss:{},train_acc:{}".format(epoch, batch_id, avg_loss.numpy(),
                                                                                  acc.numpy()))
            avg_loss.backward()
            optimizer.minimize(avg_loss)
            model.clear_gradients()
        fluid.save_dygraph(model.state_dict(), 'MyCNN')  # 保存模型

with fluid.dygraph.guard():
    accs = []
    model_dict, _ = fluid.load_dygraph('MyCNN')
    model = MyCNN()
    model.load_dict(model_dict) #加载模型参数
    model.eval() #训练模式
    for batch_id,data in enumerate(eval_reader()):#测试集
        images=np.array([x[0].reshape(1,32,32) for x in data],np.float32)
        labels = np.array([x[1] for x in data]).astype('int64')
        labels = labels[:, np.newaxis]
        image=fluid.dygraph.to_variable(images)
        label=fluid.dygraph.to_variable(labels)
        predict=model(image)
        acc=fluid.layers.accuracy(predict,label)
        accs.append(acc.numpy()[0])
        avg_acc = np.mean(accs)
    print(avg_acc)