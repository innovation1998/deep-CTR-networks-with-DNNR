import pandas as pd
import numpy as np
import torch
from sklearn.metrics import mean_squared_error, log_loss, roc_auc_score, roc_curve, auc
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
from torch.nn.functional import cross_entropy, softmax

from deepctr_torch.inputs import SparseFeat, get_feature_names
from deepctr_torch.models import DCNMix, NFM, DCN, DeepFM, xDeepFM, AutoInt

from dcnmixr import DCNMixR
from xdeepfmr import xDeepFMR
from autointr import AutoIntR
from dcnr import DCNR

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

def preprocess():
    root = './datasets/ml-1m/'
    movies = pd.read_table(
        root+'movies.dat',
        header=None,
        names=['movie_id','title','genres'],
        sep="::",
        engine='python',
        )
    movies = movies.drop(['title'],axis=1)
    genres = movies['genres'].str.split('|',n=5,expand=True)
    genres.columns = ['genres1','genres2','genres3','genres4','genres5','genres6']
    movies =  pd.concat([movies,genres],axis=1).drop(['genres','genres5','genres6'],axis=1)

    ratings = pd.read_table(
        root+'ratings.dat',
        header=None,
        names=['user_id','movie_id','rating','timestamps'],
        sep="::",
        engine='python',
        )
    ratings['rating'] = ratings['rating'].apply(lambda x: 0 if x<3 else 1 if x>3 else None)
    ratings = ratings.dropna(subset=['rating'])

    users = pd.read_table(
        root+'users.dat',
        header=None,
        names=['user_id','gender','age','occupation','zip'],
        sep="::",
        engine='python',
        )
    zips = users['zip'].str.split('-',n=6,expand=True)
    zips.columns = ['zip1','zip2']
    users =  pd.concat([users,zips],axis=1).drop(['zip'],axis=1)
    users['zip2'] = pd.to_numeric(users['zip2'], errors='coerce',downcast='integer').fillna(0).astype(np.int8)
    
    data = pd.merge(
        left=ratings,right=movies,how='left',left_on='movie_id',right_on='movie_id'
    )
    data = pd.merge(
        left=data,right=users,how='left',left_on='user_id',right_on='user_id'
    )
    
    data['gender'] = data['gender'].map({'M':0,'F':1}).fillna(-1).astype(np.int8)

    dict_timestamp_class = {}
    for columns in ['timestamps']:
        dict_timestamp_class[columns] = timestamp_preprocess()
        dict_timestamp_class[columns].fit(data[columns])
        data[columns] = dict_timestamp_class[columns].transform(data[columns])
        data[columns] = data[columns].apply(lambda x: round(float(x),6))
    
    return data

def add_result(train):
    rst_movie = train[['movie_id','rating']].groupby(['movie_id']).agg(['mean','std','skew'])
    rst_movie.columns = ['movie_mean','movie_std','movie_skew']

    rst_user = train[['user_id','rating']].groupby(['user_id']).agg(['mean','std','skew'])
    rst_user.columns = ['user_mean','user_std','user_skew']

    rst_gender = train[['gender','rating']].groupby(['gender']).agg(['mean','std','skew'])
    rst_gender.columns = ['gender_mean','gender_std','gender_skew']

    rst_occupation = train[['occupation','rating']].groupby(['occupation']).agg(['mean','std','skew'])
    rst_occupation.columns = ['occupation_mean','occupation_std','occupation_skew']

    rst_age = train[['age','rating']].groupby(['age']).agg(['mean','std','skew'])
    rst_age.columns = ['age_mean','age_std','age_skew']

    train = pd.merge(
        left=train,
        right=rst_movie,
        on=['movie_id'],
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
        right=rst_gender,
        on=['gender'],
        how='left',
    )
    train = pd.merge(
        left=train,
        right=rst_occupation,
        on=['occupation'],
        how='left',
    )
    train = pd.merge(
        left=train,
        right=rst_age,
        on=['age'],
        how='left',
    )
    train = train.fillna(0)
    print(train)
    return train

if __name__ == "__main__":

    data = preprocess()
    data = add_result(data)
    sparse_features = ["movie_id", "user_id","gender", "age", "occupation","timestamps", "zip1","zip2"]
    rst_sparse_features = ['movie_mean','movie_std','movie_skew','user_mean','user_std','user_skew','age_mean','age_std','age_skew',
                            'gender_mean','gender_std','gender_skew','occupation_mean','occupation_std','occupation_skew']
    valen_sparse_features = ['genres1','genres2','genres3','genres4']#,'genres5','genres6']
    target = ['rating']
    sparse_features = sparse_features + rst_sparse_features
    all_features = sparse_features + valen_sparse_features
    # 1.Label Encoding for sparse features,and do simple Transformation for dense features
    for feat in sparse_features:
        lbe = LabelEncoder()
        data[feat] = lbe.fit_transform(data[feat])
    for feat in valen_sparse_features:
        lbe = LabelEncoder()
        data[feat] = lbe.fit_transform(data[feat].astype(str))
    # 2.count #unique features for each sparse field
    fixlen_feature_columns = [SparseFeat(feat, data[feat].nunique())
                              for feat in all_features]
    linear_feature_columns = fixlen_feature_columns
    dnn_feature_columns = fixlen_feature_columns
    feature_names = get_feature_names(linear_feature_columns + dnn_feature_columns)
    print(data)

    # 4.Define Model,train,predict and evaluate

    device = 'cpu'
    use_cuda = True
    if use_cuda and torch.cuda.is_available():
        device = 'cuda:0'

    m = 1
    for i in range(m):
        
        train, test = train_test_split(data, test_size=0.1)
        train_model_input = {name: train[name] for name in feature_names}
        test_model_input = {name: test[name] for name in feature_names}
        
        model = DCNMixR(linear_feature_columns=linear_feature_columns, dnn_feature_columns=dnn_feature_columns,
                    task='binary', 
                    cross_num=4,
                    low_rank=64,
                    num_experts=4,
                    dnn_hidden_units=(1024,1024,1024),
                    l2_reg_embedding=1e-5,
                    device=device)

        model.compile("adagrad", "binary_crossentropy",
                    metrics=["logloss", "auc", "acc"], )

        model.fit(train_model_input, train[target].values, batch_size=1024, epochs=3, verbose=2, validation_split=0.1)
        
        pred_ans = model.predict(test_model_input, 256)
        print('log_loss:',round(log_loss(test[target].values, pred_ans), 5))
        print('AUC:',round(roc_auc_score(test[target].values, pred_ans), 5))
