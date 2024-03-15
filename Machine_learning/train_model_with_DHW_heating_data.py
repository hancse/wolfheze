import pandas as pd
import numpy as np
import torch
#from pathlib import Path
from sklearn.preprocessing import StandardScaler
from torch.autograd import Variable


def load_excel_or_csv_data(filename):
    """
    Reads a CSV or Excel file and returns a pandas DataFrame.

    Parameters:
    - file_path: str, path to the CSV or Excel file

    Returns:
    - df: pandas DataFrame containing the file data
    """
    if filename.endswith('.csv'):
        df = pd.read_csv(filename)
    elif filename.endswith('.xlsx'):
        df = pd.read_excel(filename, engine='openpyxl')
    else:
        raise ValueError("The file must be a CSV or Excel file (.csv or .xlsx).")

    data_selection = df[['Q_house_profile', 'Toutdoor', 'hod']]  # select the columns to use for machine learning
    data_selection_np = data_selection.to_numpy()  # conversion of the data to numpy array

    return data_selection_np  # numpy arrays containing the data used for training and testing


def preprocess_training(np_data, sequence_length):
    """" convert the data in the numpy array into an input format that can be used by the LSTM model
        1. normalize the data, shift with mean and scale with standard deviation the normalization parameters need to
        be saved in order to be able to use the model on new date.
        2. create a sliding window of input history for each output value. The sliding window is of of size
        'sequence_length'
        3. split the data into training and test set.
        4. convert the dataset to tensors suitable for torch

        Args:
            np_data: data set as numpy array
            sequence_length: size of history of the number inputs needed for predicting the next output
        Returns:
            normalization_parameter set
            torch tensors for training and testing
    """
    # define the input and output datasets.
    input_data = np_data  # the full dataset will be used as input
    output_data = np_data[:, 0]  # the first column contains the heat demand, and is will be used to create the output

    # normalize the data
    input_scale = StandardScaler().fit(input_data)  # scaling factors for the input data
    output_scale = StandardScaler().fit(output_data.reshape(-1, 1))  # scaling factors for the output data,
    # reshape is needed to create a column input for the standard scaler

    scaled_input_data = input_scale.transform(input_data)  # scale the input data with the means and variances
    scaled_output_data = output_scale.transform(output_data.reshape(-1, 1))  # scale the output data

    # create a sliding window over the input and match the appropriate output value
    # method from Trungs code:
    x = []
    y = []

    for i in range(len(scaled_input_data) - sequence_length - 1):
        x_i = scaled_input_data[i:(i + sequence_length)]
        y_i = scaled_output_data[i + sequence_length]
        x.append(x_i)
        y.append(y_i)

    x, y = np.array(x), np.array(y)

    nr_samples = len(y)  # total number of samples in the data set
    training_fraction = 0.7  # fraction of the data set that will be used for training
    index_train = int(nr_samples * training_fraction)  # index for the last sample of the training set

    train_in_np = x[0:index_train, :]  # training set is the first part of the time series
    train_out_np = y[0:index_train]  #

    test_in_np = x[index_train:-1, :]  # test set is the last part of the time series
    test_out_np = y[index_train:-1]

    train_in = Variable(torch.Tensor(train_in_np))  # for use in torch data needs to be converted to tensors
    train_out = Variable(torch.Tensor(train_out_np))
    test_in = Variable(torch.Tensor(test_in_np))  # for use in torch data needs to be converted to tensors
    test_out = Variable(torch.Tensor(test_out_np))

    return train_in, train_out, test_in, test_out, input_scale, output_scale

if __name__ == "__main__":

    excel_file = 'Data_DHW_heating_Trung/Heat_and_DHW_profile.csv'
    sequence_length = 12

    dataset = load_excel_or_csv_data(excel_file)

    train_in_tensor, train_out_tensor, test_in_tensor, test_out_tensor, input_scaler, output_scaler = preprocess_training(dataset,
                                                                                                     sequence_length)

    # when model has been trained and tested the model can be used for prediction
    # test_in, test_out = preprocess_use_model(test_data, input_scaler, sequence_length)