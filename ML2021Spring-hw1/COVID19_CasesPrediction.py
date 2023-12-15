import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

import numpy as np
import csv
import os

import matplotlib.pyplot as plt
from matplotlib.pyplot import figure

import sklearn
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_regression
from sklearn import preprocessing

import pandas as pd

def get_device():
    '''torch.cuda.is_available()判断是否可用gpu，返回true或false'''
    return 'cuda' if torch.cuda.is_available() else 'cpu'

def plot_learning_curve(loss_record, title=''):
    ''' Plot learning curve of your DNN (train & dev loss) '''
    total_steps = len(loss_record['train'])
    x_1 = range(total_steps)
    x_2 = x_1[::len(loss_record['train']) // len(loss_record['dev'])]
    figure(figsize=(6, 4))
    plt.plot(x_1, loss_record['train'], c='tab:red', label='train')
    plt.plot(x_2, loss_record['dev'], c='tab:cyan', label='dev')
    plt.ylim(0.0, 5.)
    plt.xlabel('Training steps')
    plt.ylabel('MSE loss')
    plt.title('Learning curve of {}'.format(title))
    plt.legend()
    plt.show()

def plot_pred(dv_set, model, device, lim=35., preds=None, targets=None):
    ''' Plot prediction of your DNN '''
    if preds is None or targets is None:
        model.eval()
        preds, targets = [], []
        for x, y in dv_set:
            x, y = x.to(device), y.to(device)
            with torch.no_grad():
                pred = model(x)
                preds.append(pred.detach().cpu())
                targets.append(y.detach().cpu())
        preds = torch.cat(preds, dim=0).numpy()
        targets = torch.cat(targets, dim=0).numpy()

    figure(figsize=(5, 5))
    plt.scatter(targets, preds, c='r', alpha=0.5)
    plt.plot([-0.2, lim], [-0.2, lim], c='b')
    plt.xlim(-0.2, lim)
    plt.ylim(-0.2, lim)
    plt.xlabel('ground truth value')
    plt.ylabel('predicted value')
    plt.title('Ground Truth v.s. Prediction')
    plt.show()

tr_path = 'covid.train.csv'
tt_path = 'covid.test.csv'

myseed = 42069
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(myseed)
torch.manual_seed(myseed)   # torch.manual_seed()手动设置随机数种子
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(myseed)

device_count = torch.cuda.device_count()    # torch.cuda.device_count()返回可用cuda支持的gpu个数
print(f"Number of CUDA devices:{device_count}")

# 分割数据集为X和y
data = pd.read_csv(tr_path)
x = data[data.columns[1:94]]
y = data[data.columns[94]]
x = (x - x.min()) / (x.max() - x.min())     # 正则化

bestfeatures = SelectKBest(score_func=f_regression)     # 创建一个SelectKBest实例
fit = bestfeatures.fit(x,y)                             # fit()方法是传入数据集(x,y)计算每个特征的线性相关性得分并更新自身，返回的还是个SelectKBest实例
dfscores = pd.DataFrame(fit.scores_)                    # .scores_是SelectKBest类的一个属性，内容是一个包含各个特征线性相关性得分的数组，n行1列
dfcolumns = pd.DataFrame(x.columns)                     # 创建一个新的pandas表，包含x的列名(即特征名称)，n行1列
featureScores = pd.concat([dfcolumns,dfscores],axis=1)  # 水平合并，即合并后为n行*2列，其中第一列是特征名称，第二列是对应的得分
featureScores.columns = ['Specs','Score']               # 第一列列名为Specs，第二列列名为Score
print(featureScores.nlargest(20,'Score'))               # 取featureScores中score最高的20个打印
top_rows = featureScores.nlargest(20, 'Score').index.tolist()[:17]
print(top_rows)     # 取featureScores中20个分数最高的中的前17个的序号打印，称为最重要特征(这个17是看到值后手动决定的，因为前17个score均大于5000)


class COVID19Dataset(Dataset):
    ''' Dataset for loading and preprocessing the COVID19 dataset '''

    def __init__(self, path, mode='train', target_only=True):
        self.mode = mode
        # Read data into numpy arrays
        with open(path, 'r') as fp:
            data = list(csv.reader(fp))     # csv.reader()按行读取csv文件内容，返回一个迭代器，每次迭代返回文件的一行，其中每行是一个列表
            data = np.array(data[1:])[:, 1:].astype(float)  # 去掉第一行（标题）和第一列(序号)
        if not target_only:     # 如果target_only不为True，则使用所有(93个)特征
            feats = list(range(93))
        else:   # 如果target_only为True，则使用以下索引的这些特征(就是刚才调出来的17个)
            # TODO: Using 40 states & 2 tested_positive features (indices = 57 & 75)
            feats = [75, 57, 42, 60, 78, 43, 61, 79, 40, 58, 76, 41, 59, 77, 92, 74, 56]
        # 区分数据集和测试集
        if mode == 'test':
            # 如果是测试集
            # data: 893 x 93 (40 states + day 1 (18) + day 2 (18) + day 3 (17))
            data = data[:, feats]
            self.data = torch.FloatTensor(data) # 数据转化为torch中可以在gpu上运算的浮点张量
        else:   # 如果是训练集，就要有目标值target
            # data: 2700 x 94 (40 states + day 1 (18) + day 2 (18) + day 3 (18))
            target = data[:, -1]
            data = data[:, feats]
            # 如果是训练集
            if mode == 'train':
                indices = [i for i in range(len(data)) if i % 10 != 0]  # 取所有不为10倍数的索引作为训练集
            # 如果是开发集
            elif mode == 'dev':
                indices = [i for i in range(len(data)) if i % 10 == 0]  # 取所有为10倍数的索引作为开发验证集
            self.data = torch.FloatTensor(data[indices])
            self.target = torch.FloatTensor(target[indices])

        # Normalize features (you may remove this part to see what will happen)
        # 类似正态分布 标准化 x-u/s，前40列表示的是所在州，不用标准化
        self.data[:, 40:] = \
            (self.data[:, 40:] - self.data[:, 40:].mean(dim=0, keepdim=True)) \
            / self.data[:, 40:].std(dim=0, keepdim=True)
        # self.data[:, 40:].mean(dim=0, keepdim=True))计算每列（即每个特征）的平均值。dim=0 指定沿着行的方向（即对每列进行操作）
        # keepdim=True 输出[[1*n]]二维数组
        # self.data[:, 40:].std(dim=0, keepdim=True) 计算每列的标准差
        self.dim = self.data.shape[1]   # 取列数(特征数)
        # 输出构造数据集的结果
        print('Finished reading the {} set of COVID19 Dataset ({} samples found, each dim = {})'
              .format(mode, len(self.data), self.dim))

    def __getitem__(self, index):
        # Returns one sample at a time
        if self.mode in ['train', 'dev']:
            # For training
            return self.data[index], self.target[index]     # 对训练集就是取那一行数据和目标值
        else:
            # For testing (no target)
            return self.data[index]                         # 对测试集就是取数据

    def __len__(self):
        # Returns the size of the dataset
        return len(self.data)                               # 返回训练集长度


## **DataLoader**
def prep_dataloader(path, mode, batch_size, n_jobs=0, target_only=False):
    ''' Generates a dataset, then is put into a dataloader. '''
    dataset = COVID19Dataset(path, mode=mode, target_only=target_only)  # 构造数据集实例
    dataloader = DataLoader(
        # 创建了一个torch的DataLoader实例，封装了之前创建的dataset
        dataset, batch_size,
        shuffle=(mode == 'train'), drop_last=False,
        num_workers=n_jobs, pin_memory=True)                # Construct dataloader
    return dataloader


# **Deep Neural Network**

class NeuralNet(nn.Module): # nn.Module是pytorch的神经网络基类
    ''' A simple fully-connected deep neural network '''

    def __init__(self, input_dim):
        super(NeuralNet, self).__init__()
        self.net = nn.Sequential(nn.Linear(input_dim, 16), nn.BatchNorm1d(16), nn.Dropout(p=0.2), nn.ReLU(), nn.Linear(16, 1))

        # Mean squared error loss
        self.criterion = nn.MSELoss(reduction='mean')

    def forward(self, x):
        ''' Given input of size (batch_size x input_dim), compute output of the network '''
        return self.net(x).squeeze(1)

    def cal_loss(self, pred, target):
        ''' Calculate loss '''
        regularization_loss = 0
        for param in model.parameters():
            # TODO: you may implement L1/L2 regularization here
            regularization_loss += torch.sum(param ** 2)
        return self.criterion(pred, target) + 0.00075 * regularization_loss

#         # Calculate the MSE loss
#         loss = self.criterion(pred, target)

#         l1 = 0
#         l2 = 0
#         for param in self.net.parameters():
#             l1 += param.abs().sum()
#             l2 += param.pow(2).sum()

#         loss = loss + l1_lambda * l1 + l2_lambda * l2

#         return loss


## **Training**

def train(tr_set, dv_set, model, config, device):
    ''' DNN training '''

    n_epochs = config['n_epochs']  # 最大轮数

    # Setup optimizer
    optimizer = getattr(torch.optim, config['optimizer'])(model.parameters(), **config['optim_hparas'])

    min_mse = 1000.
    loss_record = {'train': [], 'dev': []}      # for recording training loss
    early_stop_cnt = 0
    epoch = 0
    while epoch < n_epochs:
        model.train()                           # set model to training mode
        for x, y in tr_set:                     # iterate through the dataloader
            optimizer.zero_grad()               # set gradient to zero
            x, y = x.to(device), y.to(device)   # move data to device (cpu/cuda)
            pred = model(x)                     # forward pass (compute output)
            mse_loss = model.cal_loss(pred, y)  # compute loss
            mse_loss.backward()                 # compute gradient (backpropagation)
            optimizer.step()                    # update model with optimizer
            loss_record['train'].append(mse_loss.detach().cpu().item())

        # After each epoch, test your model on the validation (development) set.
        dev_mse = dev(dv_set, model, device)
        if dev_mse < min_mse:
            # Save model if your model improved
            min_mse = dev_mse
            print('Saving model (epoch = {:4d}, loss = {:.4f})'
                .format(epoch + 1, min_mse))
            torch.save(model.state_dict(), config['save_path'])  # Save model to specified path
            early_stop_cnt = 0
        else:
            early_stop_cnt += 1

        epoch += 1
        loss_record['dev'].append(dev_mse)
        if early_stop_cnt > config['early_stop']:
            # Stop training if your model stops improving for "config['early_stop']" epochs.
            break

    print('Finished training after {} epochs'.format(epoch))
    return min_mse, loss_record

## **Validation**

def dev(dv_set, model, device):
    model.eval()                                # set model to evalutation mode
    total_loss = 0
    for x, y in dv_set:                         # iterate through the dataloader
        x, y = x.to(device), y.to(device)       # move data to device (cpu/cuda)
        with torch.no_grad():                   # disable gradient calculation
            pred = model(x)                     # forward pass (compute output)
            mse_loss = model.cal_loss(pred, y)  # compute loss
        total_loss += mse_loss.detach().cpu().item() * len(x)  # accumulate loss
    total_loss = total_loss / len(dv_set.dataset)              # compute averaged loss

    return total_loss


## **Testing**

def test(tt_set, model, device):
    model.eval()                                # set model to evalutation mode
    preds = []
    for x in tt_set:                            # iterate through the dataloader
        x = x.to(device)                        # move data to device (cpu/cuda)
        with torch.no_grad():                   # disable gradient calculation
            pred = model(x)                     # forward pass (compute output)
            preds.append(pred.detach().cpu())   # collect prediction
    preds = torch.cat(preds, dim=0).numpy()     # concatenate all predictions and convert to a numpy array
    return preds

# **Setup Hyper-parameters**

device = get_device()                 # get the current available device ('cpu' or 'cuda')
os.makedirs('models', exist_ok=True)  # The trained model will be saved to ./models/
target_only = True                   # TODO: Using 40 states & 2 tested_positive features

# TODO: How to tune these hyper-parameters to improve your model's performance?
config = {
    'n_epochs': 10000,                # maximum number of epochs
    'batch_size': 200,               # mini-batch size for dataloader
    'optimizer': 'Adam',              # optimization algorithm (optimizer in torch.optim)
    'optim_hparas': {                # hyper-parameters for the optimizer (depends on which optimizer you are using)
        'lr': 0.0005,                # learning rate of SGD ---0.00185; 0.00181; 0.001811
        # 'momentum': 0.9              # momentum for SGD
    },
    'early_stop': 1000,               # early stopping epochs (the number epochs since your model's last improvement)
    'save_path': 'models/model.pth'  # your model will be saved here
}

# **Load data and model**

tr_set = prep_dataloader(tr_path, 'train', config['batch_size'], target_only=target_only)
dv_set = prep_dataloader(tr_path, 'dev', config['batch_size'], target_only=target_only)
tt_set = prep_dataloader(tt_path, 'test', config['batch_size'], target_only=target_only)

model = NeuralNet(tr_set.dataset.dim).to(device)  # Construct model and move to device

## **Start Training**

model_loss, model_loss_record = train(tr_set, dv_set, model, config, device)

plot_learning_curve(model_loss_record, title='deep model')

plot_learning_curve(model_loss_record, title='deep model')

del model
model = NeuralNet(tr_set.dataset.dim).to(device)
ckpt = torch.load(config['save_path'], map_location='cpu')  # Load your best model

def save_pred(preds, file):
    ''' Save predictions to specified file '''
    print('Saving results to {}'.format(file))
    with open(file, 'w') as fp:
        writer = csv.writer(fp)
        writer.writerow(['id', 'tested_positive'])
        for i, p in enumerate(preds):
            writer.writerow([i, p])

preds = test(tt_set, model, device)  # predict COVID-19 cases with your model
save_pred(preds, 'pred.csv')         # save prediction file to pred.csv

