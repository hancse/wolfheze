# -*- coding: utf-8 -*-
"""
Created on Tue Apr 20 16:43:59 2021

@author: TrungNguyen
"""

import torch
from Machine_learning.LSTM_model_struct import LSTM
from Machine_learning.read_data import data_preprocess
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('QTAgg')  # Set the backend to QTAgg
from joblib import load
import numpy as np
from torch.autograd import Variable


def main():
    ''' An example for running the prediction with an assumption that Q_solar and
        internal heat gain for the next hour are not known, all the pass values are known
        and will be used as input for the next predicted hour.

    '''

    # filename = 'Heavy_weight.txt'
    filename = 'Data_DHW_heating_Trung/Heat_and_DHW_profile.csv'
    data_columns = ['Qheat_profile', 'Toutdoor', 'hod']
    seq_length = 12

    # Prepare the data.
    #np_data = load_excel_or_csv_data(filename)
    dataX, dataY, trainX, trainY, testX, testY = data_preprocess(filename, data_columns, seq_length)

    Xtsq = testX.clone().detach()

    # Define and load the model.
    input_size = 3
    hidden_size = 50
    num_layers = 1
    num_classes = 1
    bidirectional = True

    # name of the save model
    PATH = "heat_demand.pt"

    # load the model.
    model = LSTM(num_classes, input_size, hidden_size, num_layers, bidirectional, seq_length)
    model.load_state_dict(torch.load(PATH))
    model.eval()

    # call prediction function
    predict_test = model(Xtsq)
    predict_data = predict_test.data.numpy()

    # Transform the data to its original form.
    sc_Y = load('sc_Y.bin')
    predict = sc_Y.inverse_transform(predict_data)
    predict[predict < 0] = 0
    Y_val = testY.data.numpy()
    Y_val_plot = sc_Y.inverse_transform(Y_val)

    # Plot the results

    fig, axs = plt.subplots(2, figsize=(20, 12))
    axs[0].plot(Y_val_plot, label='measured')
    axs[0].plot(predict, label='predicted')
    axs[1].plot(Y_val_plot, label='measured')
    axs[1].plot(predict, label='predicted')
    axs[0].title.set_text('Zoom_in')
    axs[1].title.set_text('Heat demand')
    # plt.figure(figsize=(17,6)) #plotting
    #axs[0].set_xlim([0, 2000])
    # axs[1].set_xlim([500,1000])
    # plt.xlim([0,0])
    # plt.plot(dataY_plot[:,0],label='measured')
    # plt.plot(data_predict[:,0],label = 'predict')
    # plt.suptitle('Heat-Demand Prediction')
    axs[0].legend()
    axs[1].legend()
    plt.show()


if __name__ == "__main__":
    main()  # temporary solution, recommended syntax