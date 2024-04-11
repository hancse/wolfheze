# -*- coding: utf-8 -*-
"""
Created on Mon Mar 18 9:21:53 2024

@authors: TrungNguyen, Maarten van den Berg

"""
#from train_model_with_DHW_heating_data import load_excel_or_csv_data, preprocess_training
from Machine_learning.LSTM_model_struct import LSTM
import torch
from Machine_learning.read_data import data_preprocess


# import torch.nn as nn
# from torch.autograd import Variable

# PATH = "heat_demand.pt"

def train(filename, data_columns, seq_length, num_epochs, learning_rate,
          input_size, hidden_size, num_layers, num_classes, bidirectional, PATH):
    """
    Function to train and save the model for prediction.

     Args:

        filename:       name of the data set.
        data_columns:   Array of the columns that contains the data
        seq_length:     the number of pass input points which needed
                            for predicting the future value.

        num_epochs:     number of times all the data has passed to the model.

        learning_rate:  step size at each iteration while moving
                            toward a minimum of a loss function
      input_size:       number of input features.
      hidden_size:      number of hidden layer.
      num_classes:      number of outputs.
      bidirectional:    True or False.
      PATH:             name of the save model *.pt (pytorch model)

    Returns:

        lstm:           A train lstm models with the structure define as input.
    """

    #np_data = load_excel_or_csv_data(filename)
    #dataX, dataY, trainX, trainY, testX, testY = preprocess_training(np_data, seq_length)
    dataX, dataY, trainX, trainY, testX, testY = data_preprocess(filename, data_columns, seq_length)

    lstm = LSTM(num_classes, input_size, hidden_size, num_layers, bidirectional, seq_length)
    lstm.train()

    criterion = torch.nn.MSELoss()  # mean-squared error for regression
    optimizer = torch.optim.Adam(lstm.parameters(), lr=learning_rate)
    # optimizer = torch.optim.SGD(lstm.parameters(), lr=learning_rate)

    # Train the model
    for epoch in range(num_epochs):
        outputs = lstm(trainX)  # lstm.forward(trainX) propagate all data through the network
        optimizer.zero_grad()  # reset the optimizer to zero

        # obtain the loss function
        loss = criterion(outputs, trainY)

        loss.backward()  # backward propagate the losses

        optimizer.step()  # do one optimization step for updating the parameters
        if epoch % 100 == 0:
            print("Epoch: %d, loss: %1.5f" % (epoch, loss.item()))

    # Save the model for prediction.

    # PATH = PATH
    torch.save(lstm.state_dict(), PATH)  # save the obtained model

    return lstm


if __name__ == "__main__":
    # data file name
    # filename = 'Heavy_weight.txt'
    filename = 'Data_DHW_heating_Trung/Heat_and_DHW_profile.csv'
    data_columns = ['Qheat_profile', 'Toutdoor', 'hod']

    seq_length = 12

    # number of training cycle.
    num_epochs = 2000
    # learning rate
    learning_rate = 0.1
    # Train the model

    input_size = 3
    hidden_size = 50
    num_layers = 1
    num_classes = 1
    bidirectional = True

    PATH = "heat_demand.pt"

    lstm = train(filename, data_columns, seq_length, num_epochs, learning_rate,
                 input_size, hidden_size, num_layers, num_classes, bidirectional, PATH)
    print(lstm)

