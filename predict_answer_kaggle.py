import numpy as np
import scipy
import pandas as pd

from sklearn.metrics import roc_auc_score, log_loss
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import LabelEncoder

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from deepctr_torch.inputs import SparseFeat, get_feature_names
from deepctr_torch.models import DCNMix, NFM, DCN, DeepFM, xDeepFM, AutoInt

from dcnmixr import DCNMixR
from xdeepfmr import xDeepFMR
from autointr import AutoIntR
from dcnr import DCNR

import gc
from time import time
# from deepctr_torch.models import DeepFM

# class of dataframe: catgory(int), timestamp,(float) and sequence(list)
class catgory_preprocess:
    def __init__(self):
        self.max_len = 0
        self.dict_map = {}
    
    def fit(self, catgory_list):
        index = 1 
        for i in catgory_list:
            if i not in self.dict_map:
                self.dict_map[i] = index
                index += 1
        self.max_len = index + 1
        
    def transform(self, catgory_list):
        transform_list = []
        for i in catgory_list:
            if i in self.dict_map:
                transform_list.append(self.dict_map[i])
            else:
                transform_list.append(0)
        return transform_list

class timestamp_preprocess:
    def __init__(self):
        self.max = 0
        self.min = 0
        self.max_min = 0 
        
    def fit(self, float_list):
        for i in float_list:
            if i < self.min:
                self.min = i
            if i > self.max:
                self.max = i
        self.max_min = self.max - self.min # 寻找中位数
        
    def transform(self, float_list):
        transform_list = []
        for i in float_list:
            if i < self.min:
                transform_list.append(0)
            elif i > self.max:
                transform_list.append(1)
            else:
                transform_list.append(i/self.max_min) # 近似归一化
        return transform_list
    
class sequence_preprocess:
    def __init__(self,sep_len=6,sep=' '):
        self.sep_len = sep_len
        self.sep = sep
        self.max_len = 0
        self.dict_map = {}
    
    def fit(self, cat_seq_list):
        index = 1 
        for cat_seq_i in cat_seq_list:
            cat_seq_i = cat_seq_i.split(self.sep)
            for cat_i in cat_seq_i:        
                if cat_i not in self.dict_map:
                    self.dict_map[cat_i] = index
                    index += 1
        self.max_len = index + 1
        
    def transform(self, cat_seq_list):
        cat_transform_list = []
        for cat_seq_i in cat_seq_list:
            cat_seq_i = cat_seq_i.split(self.sep)
            len_cat_seq_i = len(cat_seq_i)
            if len_cat_seq_i >= self.sep_len:
                cat_seq_i = cat_seq_i[:self.sep_len]
            else:
                cat_seq_i = cat_seq_i + [0] * (self.sep_len - len_cat_seq_i)
            cat_seq_n = []
            for cat_i in cat_seq_i:
                if cat_i in self.dict_map:
                    cat_seq_n.append(self.dict_map[cat_i])
                else:
                    cat_seq_n.append(0)
            cat_transform_list.append(cat_seq_n)            
        return cat_transform_list

# read csv file
def deal_train_csv(limit=100):
    root_path = './datasets/riiid-test-answer-prediction/'
    train_file = 'train.csv'
    question_file = 'questions.csv'

    nrows = 1500*10000
    
    train = pd.read_csv(
        root_path + train_file, 
        usecols=['row_id', 'timestamp', 'user_id', 'content_id', 
            'content_type_id', 'task_container_id', 'answered_correctly',
            'prior_question_elapsed_time','prior_question_had_explanation'
        ],
        dtype={
            'row_id': 'int64',
            'timestamp': 'int64',
            'user_id': 'int32',
            'content_id': 'int16',
            'content_type_id': 'int8',
            'task_container_id': 'int8',
            'answered_correctly': 'int8',
            'prior_question_elapsed_time': 'float32',
            'prior_question_had_explanation': 'str',
        }
    )
    print('user nums:',len(train.user_id.value_counts()))
    question = pd.read_csv(
        root_path + question_file, 
        nrows=nrows,
        usecols=['question_id','bundle_id','part','tags'], 
        dtype={
            'question_id': 'int16',
            'bundle_id': 'int16',
            'part': 'int8',
            'tags': 'str',
        }
    )
    tag = question["tags"].str.split(" ", n = 10, expand = True) 
    tag.columns = ['tags1','tags2','tags3','tags4','tags5','tags6']

    question =  pd.concat([question,tag],axis=1).drop(['tags'],axis=1)
    question['tags1'] = pd.to_numeric(question['tags1'], errors='coerce',downcast='integer').fillna(-1)
    question['tags2'] = pd.to_numeric(question['tags2'], errors='coerce',downcast='integer').fillna(-1)
    question['tags3'] = pd.to_numeric(question['tags3'], errors='coerce',downcast='integer').fillna(-1)
    question['tags4'] = pd.to_numeric(question['tags4'], errors='coerce',downcast='integer').fillna(-1)

    # transform bool lean to 0/1, fill none with -1
    train['prior_question_had_explanation'] = train['prior_question_had_explanation'].map({'True':1,'False':0}).fillna(-1).astype(np.int8)

    # only save content_id==0(had_explanation==False) data
    train = train[train['content_type_id']==0]
    gc.collect()

    train = train.groupby(['user_id']).tail(limit)

    # link train data and question data into one pd dataframe
    train = pd.merge(
            left=train,
            right=question,
            how='left', # 对齐左边进行拼接
            left_on='content_id',
            right_on='question_id'
    )
    train = train.fillna(0)
    gc.collect()

    return train

# modify dataframe file
def deal_dataframe(train):    
    #  preprocess dataframe 'train'
    dict_catgory_class = {}
    dict_timestamp_class = {}
    # dict_sequence_class = {}

    for columns in ['user_id','content_id','task_container_id','prior_question_had_explanation','bundle_id','part']:
        dict_catgory_class[columns] = catgory_preprocess()
        dict_catgory_class[columns].fit(train[columns])
        train[columns] = dict_catgory_class[columns].transform(train[columns])
    
    for columns in ['timestamp','prior_question_elapsed_time']:
        dict_timestamp_class[columns] = timestamp_preprocess()
        dict_timestamp_class[columns].fit(train[columns])
        train[columns] = dict_timestamp_class[columns].transform(train[columns])
        train[columns] = train[columns].apply(lambda x: round(float(x),6))

    """   
    for columns in ['tags']:
        dict_sequence_class[columns] = sequence_preprocess(sep_len=6,sep=' ')
        dict_sequence_class[columns].fit(train[columns])
        
        train[columns] = dict_sequence_class[columns].transform(train[columns]) 
    """
    gc.collect()
    return train

import os
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.sampler import SubsetRandomSampler

def add_result(train):
    rst_content = train[['content_id','answered_correctly']].groupby(['content_id']).agg(['mean','std','skew'])
    rst_content.columns = ['content_mean','content_std','content_skew']

    rst_user = train[['user_id','answered_correctly']].groupby(['user_id']).agg(['mean','std','skew'])
    rst_user.columns = ['user_mean','user_std','user_skew']

    rst_timestamp = train[['timestamp','answered_correctly']].groupby(['timestamp']).agg(['mean','std','skew'])
    rst_timestamp.columns = ['timestamp_mean','timestamp_std','timestamp_skew']

    rst_question_elapsed_time = train[['prior_question_elapsed_time','answered_correctly']].groupby(['prior_question_elapsed_time']).agg(['mean','std','skew'])
    rst_question_elapsed_time.columns = ['prior_question_elapsed_time_mean','prior_question_elapsed_time_std','prior_question_elapsed_time_skew']

    train = pd.merge(
        left=train,
        right=rst_content,
        on=['content_id'],
        how='left',
    )
    train = pd.merge(
        left=train,
        right=rst_user,
        on=['user_id'],
        how='left',
    )
    train = pd.merge(
        left=train,
        right=rst_timestamp,
        on=['timestamp'],
        how='left',
    )
    train = pd.merge(
        left=train,
        right=rst_question_elapsed_time,
        on=['prior_question_elapsed_time'],
        how='left',
    )

    train = train.fillna(0)
    print(train)
    gc.collect()
    return train

# main function
def train():
    train = deal_train_csv(limit=1000)
    train = deal_dataframe(train)
    train = add_result(train)

    float_columns = ['timestamp','prior_question_elapsed_time']
    int_columns = ['user_id','content_id','task_container_id','prior_question_had_explanation','bundle_id','part','tags1','tags2','tags3','tags4']
    result_columns = ['content_mean','content_std','content_skew','user_mean','user_std','user_skew','timestamp_mean','timestamp_std','timestamp_skew',
                      'prior_question_elapsed_time_mean','prior_question_elapsed_time_std','prior_question_elapsed_time_skew']

    features_columns = float_columns + int_columns + result_columns
    target_columns = ['answered_correctly']
    
    # Label Encoding for sparse features,and do simple transformation for dense features
    for feat in features_columns:
        lbe = LabelEncoder() # lbe: label encoded
        train[feat] = lbe.fit_transform(train[feat])

    # count unique features to get feature dims
    sparse_feature_columns = [SparseFeat(feat, train[feat].nunique()) for feat in features_columns]
    linear_feature_columns = sparse_feature_columns
    dnn_feature_columns = sparse_feature_columns
    feature_names = get_feature_names(linear_feature_columns + dnn_feature_columns)

    # generate input data
    train, test = train_test_split(train, test_size=0.1)
    train_dataset = {name: train[name] for name in feature_names}
    test_dataset = {name: test[name] for name in feature_names}
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    print(device)
    epoch = 2
    batch_size = 1024

    model = DCNMixR(linear_feature_columns=linear_feature_columns, dnn_feature_columns=dnn_feature_columns,
                task='binary', 
                cross_num=4,
                low_rank=64,
                num_experts=4,
                dnn_hidden_units=(1024,1024,1024),
                l2_reg_embedding=1e-5,
                device=device)

    model.compile('adagrad','binary_crossentropy',['auc','logloss','acc'],)
    model.fit(x=train_dataset,y=train[target_columns].values,batch_size=batch_size,epochs=epoch,verbose=2,validation_split=0.2)

    pred_ans = model.predict(test_dataset, 256)
    print("test LogLoss", round(log_loss(test[target_columns].values, pred_ans), 5))
    print("test AUC", round(roc_auc_score(test[target_columns].values, pred_ans), 5))

if __name__ == '__main__':
    train()
