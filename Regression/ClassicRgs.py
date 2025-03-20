import csv

import hpelm
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pickle

from sklearn.cross_decomposition import PLSRegression
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVR
from Evaluate.RgsEvaluate import ModelRgsevaluate

def Pls( X_train, X_test, y_train, y_test):
    tempRmse = 0
    tempR2 = 0
    tempMae = 0
    tempTRmse = 0
    tempTR2 = 0
    tempTMae = 0
    LVs = 0
    for i in range(1, 20):
        n_components = i+1
        model = PLSRegression(n_components)
        # fit the model
        model.fit(X_train, y_train)
        # predict the values
        pred_train = model.predict(X_train)
        y_pred = model.predict(X_test)
        TRmse, TR2, TMae = ModelRgsevaluate(pred_train, y_train)
        Rmse, R2, Mae = ModelRgsevaluate(y_pred, y_test)
        if R2 > tempR2:
            tempR2 = R2
            tempRmse = Rmse
            tempMae = Mae
            tempTR2 = TR2
            tempTRmse = TRmse
            tempTMae = TMae
            LVs = n_components
            # f = open('C://Users//analysis//Desktop//PLSmodelAMY.pkl', 'wb')
            # pickle.dump(model, f)
            # f.close()
    R2 = tempR2
    Rmse = tempRmse
    Mae = tempMae
    print('最佳潜在变量（LVs）数量:', LVs)
    print('The TRAIN rmse:{}, R2:{}, mae:{}'.format('%.4f' % (tempTRmse), '%.4f' % (tempTR2),'%.4f' % (tempTMae)))
    # plotpre(y_test, y_pred)
    # Writefile(model.x_loadings_)
    # print(model.x_loadings_)
    # with open("C:/Users/analysis/Desktop/真实值和预测值/PLS真实值和预测值/CT_Cars_train.csv", "w", newline='', encoding='utf-8') as file:
    #     writer = csv.writer(file, delimiter=',')
    #     writer.writerows(np.c_[y_train, pred_train])
    # with open("C:/Users/analysis/Desktop/真实值和预测值/PLS真实值和预测值/CT_Cars_test.csv", "w", newline='', encoding='utf-8') as file:
    #     writer = csv.writer(file, delimiter=',')
    #     writer.writerows(np.c_[y_test, y_pred])
    return Rmse, R2, Mae




def Svregression(X_train, X_test, y_train, y_test):
    model = SVR(C=1e7, gamma=0.001, kernel='rbf')
    model.fit(X_train, y_train)
    # predict the values
    # f = open('C://Users//analysis//Desktop//SVMmodelPRO.pkl', 'wb')
    # pickle.dump(model, f)
    # f.close()
    pred_train = model.predict(X_train)
    TRmse, TR2, TMae = ModelRgsevaluate(pred_train, y_train)
    y_pred = model.predict(X_test)
    print('The TRAIN RMSE:{}, R2:{}, MAE:{}'.format('%.4f' % (TRmse), '%.4f' % (TR2), '%.4f' % (TMae)))
    Rmse, R2, Mae = ModelRgsevaluate(y_pred, y_test)
    return Rmse, R2, Mae

def Anngression(X_train, X_test, y_train, y_test):


    model = MLPRegressor(
        hidden_layer_sizes=(20, 20), activation='relu', solver='adam', alpha=0.0001, batch_size='auto',
        learning_rate='constant', learning_rate_init=0.001, power_t=0.5, max_iter=400, shuffle=True,
        random_state=1, tol=0.0001, verbose=False, warm_start=False, momentum=0.9, nesterovs_momentum=True,
        early_stopping=False, beta_1=0.9, beta_2=0.999, epsilon=1e-08)

    model.fit(X_train, y_train)

    # predict the values
    y_pred = model.predict(X_test)
    Rmse, R2, Mae = ModelRgsevaluate(y_pred, y_test)

    return Rmse, R2, Mae

def ELM(X_train, X_test, y_train, y_test):

    model = hpelm.ELM(X_train.shape[1], 1)
    model.add_neurons(20, 'sigm')


    model.train(X_train, y_train, 'r')
    y_pred = model.predict(X_test)
    Rmse, R2, Mae = ModelRgsevaluate(y_pred, y_test)

    return Rmse, R2, Mae

def RF(X_train, X_test, y_train, y_test):
    # 训练模型
    forest = RandomForestRegressor()
    forest.fit(X_train, y_train)  # 训练集和训练集标签
    # 模型评估
    score = forest.score(X_test, y_test)
    print(score)  # 这里的score代表的R2分数
    # 模型预测
    y_pred = forest.predict(X_test)
    Rmse, R2, Mae = ModelRgsevaluate(y_pred, y_test)
    return Rmse, R2, Mae

