import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import keras
from keras.models import Sequential
from keras.layers import Dense
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

def BackProp(col_predict,no_of_output_para,input_par,link,epoch):

    col_predict = 1
    no_of_output_para = 5
    input_par = 25
    link = "https://github.com/Nd98/Neural-Network-Dataset/blob/master/Test_Properties_5.xls?raw=true"
    epoch = 100

    # dataset = pd.read_csv("/content/drive/My Drive/nayan/Teaning_data_set.csv")
    # dataset = pd.read_csv("https://raw.githubusercontent.com/Nd98/Neural-Network-Dataset/master/Teaning%20Data%20set.csv")
    dataset = pd.read_excel(link)
    X = dataset.iloc[:,no_of_output_para + 1:dataset.values[0].size].values
    y = dataset.iloc[:,col_predict].values


    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)


    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)

    classifier = Sequential()


    classifier.add(Dense(units = 5, kernel_initializer = 'uniform', activation = 'relu', input_dim = input_par))


    classifier.add(Dense(units = 5, kernel_initializer = 'uniform', activation = 'relu'))


    classifier.add(Dense(units = 5, kernel_initializer = 'uniform', activation = 'relu'))


    classifier.add(Dense(units = 1, kernel_initializer = 'uniform' ,activation = 'relu'))


    classifier.compile(optimizer = 'adam', loss = 'mean_absolute_error', metrics = ['mean_absolute_error'])


    classifier.fit(X_train, y_train, batch_size = 10, epochs = epoch)
    #epochs=500, batch_size=10, validation_split = 0.2, callbacks=callbacks_list

    y_pred = classifier.predict(X_test)

    print(y_pred)

