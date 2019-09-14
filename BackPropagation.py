import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import keras
from keras.models import Sequential
from keras.layers import Dense
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import joblib
import os
import tensorflow as tf
from neupy import algorithms
from neupy.layers import *


graph = tf.get_default_graph()

def BackProp(col_predict,no_of_output_para,input_par,link,epoch,units,tf):

    global graph
    with graph.as_default():
        dataset = pd.read_excel(link)

        #check for empty column
        cols_out = dataset.columns[col_predict:col_predict+1]
        for col in cols_out:
                if "Unnamed" in col:
                        return 0



        X = dataset.iloc[:,no_of_output_para + 1:dataset.values[0].size].values
        y = dataset.iloc[:,col_predict].values
        np.random.seed(0)


        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)


        sc = StandardScaler()
        X_train = sc.fit_transform(X_train)
        X_test = sc.transform(X_test)

        classifier = Sequential()


        classifier.add(Dense(units = units, kernel_initializer = 'uniform', activation = tf, input_dim = input_par))


        classifier.add(Dense(units = units, kernel_initializer = 'uniform', activation = tf))


        classifier.add(Dense(units = units, kernel_initializer = 'uniform', activation = tf))


        classifier.add(Dense(units = 1, kernel_initializer = 'uniform' ,activation = tf))


        classifier.compile(optimizer = 'adam', loss = 'mean_absolute_error', metrics = ['mean_absolute_error'])


        classifier.fit(X_train, y_train, batch_size = 10, epochs = epoch)
        #epochs=500, batch_size=10, validation_split = 0.2, callbacks=callbacks_list

        joblib.dump(classifier,link+"-"+str(col_predict)+".pkl")


def QuasiNewton(col_predict,no_of_output_para,input_par,link,epoch,units,tf):
    global graph
    with graph.as_default():

        dataset=pd.read_excel(link)

        #check for empty column
        cols_out = dataset.columns[col_predict:col_predict+1]
        for col in cols_out:
                if "Unnamed" in col:
                        return 0
        
        X=dataset.iloc[:,no_of_output_para + 1:dataset.values[0].size].values
        Y=dataset.iloc[:,col_predict].values
        np.random.seed(0)

        X_train = np.array(X)
        y_train = np.array(Y)
        X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.2, random_state = 0)

        sc = StandardScaler()
        X_train = sc.fit_transform(X_train)
        X_test = sc.transform(X_test)

        network = Input(input_par) >> Sigmoid(units) >> Relu(1)
        optimizer = algorithms.QuasiNewton([network],update_function='bfgs',verbose=False,shuffle_data=False)

        optimizer.train(X_train, y_train, epochs=epoch)

        joblib.dump(optimizer,link+"-"+str(col_predict)+".pkl")

def LevenbergMarquardt(col_predict,no_of_output_para,input_par,link,epoch,units,tf):
    global graph
    with graph.as_default():

        dataset=pd.read_excel(link)

        #check for empty column
        cols_out = dataset.columns[col_predict:col_predict+1]
        for col in cols_out:
                if "Unnamed" in col:
                        return 0
        
        X=dataset.iloc[:,no_of_output_para + 1:dataset.values[0].size].values
        Y=dataset.iloc[:,col_predict].values
        np.random.seed(0)

        X_train = np.array(X)
        y_train = np.array(Y)
        X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.2, random_state = 0)

        sc = StandardScaler()
        X_train = sc.fit_transform(X_train)
        X_test = sc.transform(X_test)

        network = Input(input_par) >> Sigmoid(units) >> Relu(1)
        optimizer = algorithms.LevenbergMarquardt([network],verbose=False,shuffle_data=False)

        optimizer.train(X_train, y_train, epochs=epoch)

        joblib.dump(optimizer,link+"-"+str(col_predict)+".pkl")


def MomentumAdaptation(col_predict,no_of_output_para,input_par,link,epoch,units,tf):
    global graph
    with graph.as_default():

        dataset=pd.read_excel(link)

        #check for empty column
        cols_out = dataset.columns[col_predict:col_predict+1]
        for col in cols_out:
                if "Unnamed" in col:
                        return 0
        
        X=dataset.iloc[:,no_of_output_para + 1:dataset.values[0].size].values
        Y=dataset.iloc[:,col_predict].values
        np.random.seed(0)

        X_train = np.array(X)
        y_train = np.array(Y)
        X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.2, random_state = 0)

        sc = StandardScaler()
        X_train = sc.fit_transform(X_train)
        X_test = sc.transform(X_test)

        network = Input(input_par) >> Sigmoid(units) >> Relu(1)
        optimizer = algorithms.Momentum([network],verbose=False,shuffle_data=False)

        optimizer.train(X_train, y_train, epochs=epoch)

        joblib.dump(optimizer,link+"-"+str(col_predict)+".pkl")




def predict(arr,col_predict,link,no_of_output_para):

    global graph
    with graph.as_default():
        
        dataset = pd.read_excel(link)

        #check for empty column
        cols_out = dataset.columns[col_predict:col_predict+1]
        for col in cols_out:
                if "Unnamed" in col:
                        return None


        X = dataset.iloc[:,no_of_output_para + 1:dataset.values[0].size].values
        y = dataset.iloc[:,col_predict].values
        np.random.seed(0)


        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)


        sc = StandardScaler()
        X_train = sc.fit_transform(X_train)
        X_test = sc.transform(X_test)
        classifier = joblib.load(link+"-"+str(col_predict)+".pkl")

        # y_pred = classifier.predict(X_test)

        # # p = np.array([[60,1559,9.72,20.67,70.25,3,6.7,5.26,22.42,9.30,23.67,79.44,10.63,9.5,25.17,73.75,0.09,0.059,0.962,1.274,0.01,0.039,18.538,8.27,0.235
        # # ]]) #10
        # p = np.array([[69,1563,9.72,19.50,72.25,4,6.67,5.07,22.21,9.25,23.17,69.75,10.42,9.5,25.17,73.75,0.09,0.053,0.975,1.196,0.008,0.037,18.211,8.11,0.203
        # ]]) #11

        p = np.array([arr])
        
        p = sc.transform(p)
        p = classifier.predict(p)
        return p.tolist()[0][0]



# no_of_output_para = 5
# input_par = 25
# # link = "https://github.com/Nd98/Neural-Network-Dataset/blob/master/Test_Properties_5.xls?raw=true"
# epoch = 100
# units = 100
# tf = 'relu'
# link = os.path.abspath("instance/Test.xls")
# x = no_of_output_para
# while(x>0):
#     col_predict = x
#     BackProp(col_predict,no_of_output_para,input_par,link,epoch,units,tf)
#     x-=1

# sc_array.reverse()
# x = no_of_output_para
# while(x>0):
#     col_predict = x
#     predict([],col_predict,link)
#     x-=1