from Regression.ClassicRgs import Pls, Anngression, Svregression, ELM, RF
from Regression.CNN import CNNTrain
def  QuantitativeAnalysis(model, X_train, X_test, y_train, y_test):
    if model == "Pls":
        Rmse, R2, Mae = Pls(X_train, X_test, y_train, y_test)
    elif model == "ANN":
        Rmse, R2, Mae = Anngression(X_train, X_test, y_train, y_test)
    elif model == "SVR":
        Rmse, R2, Mae = Svregression(X_train, X_test, y_train, y_test)
    elif model == "ELM":
        Rmse, R2, Mae = ELM(X_train, X_test, y_train, y_test)
    elif model == "CNN":
        Rmse, R2, Mae = CNNTrain("AlexNet",X_train, X_test, y_train, y_test,  150)
    elif model == "RF":
        Rmse, R2, Mae = RF(X_train, X_test, y_train, y_test)
    else:
        print("no model of QuantitativeAnalysis")
    return Rmse, R2, Mae

