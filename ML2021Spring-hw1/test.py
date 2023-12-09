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

data = pd.read_csv(tr_path)
x = data[data.columns[1:94]]
y = data[data.columns[94]]
x = (x - x.min()) / (x.max() - x.min())     # 正则化

bestfeatures = SelectKBest(score_func=f_regression)     # 创建一个SelectKBest实例
fit = bestfeatures.fit(x,y)                             # fit()方法是传入数据集(x,y)计算每个特征的线性相关性得分并更新自身，返回的还是个SelectKBest实例
dfscores = pd.DataFrame(fit.scores_)                    # .scores_是SelectKBest类的一个属性，内容是一个包含各个特征线性相关性得分的数组，n行1列
dfcolumns = pd.DataFrame(x.columns)                     # 创建一个新的pandas表，包含x的列名(即特征名称)，n行1列
featureScores = pd.concat([dfcolumns,dfscores],axis=1)  # 水平合并为一个新表featureScores，即合并后为n行*2列，其中第一列是特征名称，第二列是对应的得分
featureScores.columns = ['Specs','Score']               # 第一列列名为Specs，第二列列名为Score
print(featureScores.nlargest(20,'Score'))               # 取featureScores中score最高的20个打印

top_rows = featureScores.nlargest(20, 'Score').index.tolist()[:17]
print(top_rows)
