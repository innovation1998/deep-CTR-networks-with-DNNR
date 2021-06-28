# -*- coding: utf-8 -*-
import pandas as pd
import torch
import math
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

from deepctr_torch.inputs import SparseFeat, DenseFeat, get_feature_names
from deepctr_torch.models import DCNMix, NFM, DCN, DeepFM, xDeepFM, AutoInt

from dcnmixr import DCNMixR
from xdeepfmr import xDeepFMR
from autointr import AutoIntR
from dcnr import DCNR

if __name__ == "__main__":
    data = pd.read_csv('./datasets/criteo/criteo_sampled_data.csv')
    sparse_features = ['C' + str(i) for i in range(1, 27)]
    dense_features = ['I' + str(i) for i in range(1, 14)]
    target = ['label']
    sparse_features.pop(21)
    dense_features.pop(9)

    for feat in dense_features:
        if feat == 'I2':
            data[feat] = data[feat].apply(lambda x:math.log2(x+4))
        else:
            data[feat] = data[feat].apply(lambda x:math.log2(x+1))

    data[sparse_features] = data[sparse_features].fillna('-1', )
    data[dense_features] = data[dense_features].fillna(0, )

    # 1.Label Encoding for sparse features,and do simple Transformation for dense features
    for feat in sparse_features:
        lbe = LabelEncoder()
        data[feat] = lbe.fit_transform(data[feat])

    # 2.count #unique features for each sparse field,and record dense feature field name

    fixlen_feature_columns = [SparseFeat(feat, data[feat].nunique())
                              for feat in sparse_features] + [DenseFeat(feat, 1, )
                                                              for feat in dense_features]

    dnn_feature_columns = fixlen_feature_columns
    linear_feature_columns = fixlen_feature_columns

    feature_names = get_feature_names(
        linear_feature_columns + dnn_feature_columns)

    # 3.generate input data for model
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
                    use_bn=True,
                    activation = 'dice',
                    dnn_hidden_units=(1024,1024,1024),
                    l2_reg_embedding=1e-5,
                    device=device)

        model.compile("adagrad", "binary_crossentropy",
                    metrics=["logloss", "auc", "acc"], )

        model.fit(train_model_input, train[target].values, batch_size=1024, epochs=3, verbose=2, validation_split=0.1)
        
        pred_ans = model.predict(test_model_input, 256)
        print('log_loss:',round(log_loss(test[target].values, pred_ans), 5))
        print('AUC:',round(roc_auc_score(test[target].values, pred_ans), 5))

        
