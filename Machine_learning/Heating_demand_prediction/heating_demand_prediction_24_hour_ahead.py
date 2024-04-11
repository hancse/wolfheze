# -*- coding: utf-8 -*-
"""
Created on Thu Mar 28 16:43:59 2024

@author: Maarten van den Berg
"""

import torch
from Machine_learning.LSTM_model_struct import LSTM
from Machine_learning.read_data import data_preprocess, preprocess_use_model
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('QTAgg')  # Set the backend to QTAgg
from joblib import load
import numpy as np
from torch.autograd import Variable


def recursive_prediction(model, initial_input, steps=2):
    """
    Generate recursive predictions using the model for a specified number of steps.

    :param model: The trained PyTorch LSTM model.
    :param initial_input: The last known input data (as a PyTorch tensor) to start the predictions.
    :param steps: Number of future steps to predict.
    :return: A list of predictions.
    """
    model.eval()  # Ensure the model is in evaluation mode.
    predictions = []
    current_input = initial_input

    with torch.no_grad():  # No need to track gradients
        for _ in range(steps):
            # Predict the next step
            prediction = model(current_input)

            # Append prediction to the output
            predictions.append(prediction.item())  # Adjust depending on your output shape

            # Prepare the new input for the next prediction
            # This involves incorporating the prediction into the input sequence.
            # You might need to adjust this part based on your input shape and requirements.
            current_input = torch.roll(current_input, -1, dims=1)
            current_input[:, -1] = prediction.squeeze()  # Assuming prediction needs to be squeezed

    return predictions

if __name__ == "__main__":
    # Define and load the model.
    seq_length = 12
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

    #Load data sequence from a file, basically a snapshot of the data
    filename = 'Data_DHW_heating_Trung/Heat_and_DHW_profile.csv'
    data_columns = ['Qheat_profile', 'Toutdoor', 'hod']
    num_points = 14
    input_data_sequence, output_data_sequence, total_output_sequence = preprocess_use_model(filename, data_columns, num_points)

    # Transform the data to its scaled form.
    sc_X = load('sc_X.bin')
    sc_Y = load('sc_Y.bin')
    input_data_sequence_scaled = sc_X.transform(input_data_sequence)
    output_data_sequence = np.array(output_data_sequence)
    output_data_sequence = output_data_sequence.reshape(-1, 1)
    output_data_sequence_scaled = sc_Y.transform(output_data_sequence)

    # method from Trungs code:
    test_input = []
    # Make a sequence of the data input and turn it into a Tensor
    for i in range(len(input_data_sequence_scaled) - seq_length - 1):
        x_i = input_data_sequence_scaled[i:(i + seq_length)]
        test_input.append(x_i)

    test_input = np.array(test_input)
    test_in_tensor = Variable(torch.Tensor(test_input))

    # call prediction function
    predict_test = model(test_in_tensor)
    predict_data = predict_test.data.numpy()

    # Use the recursive prediction to forecast heat demand
    recursive_result = recursive_prediction(model, test_in_tensor)
    recursive_result = np.array(recursive_result)
    recursive_result = recursive_result.reshape(-1, 1)
    predict = sc_Y.inverse_transform(recursive_result)

    # Prepare the x-axis indices for plotting
    x_measured = range(len(output_data_sequence))
    x_predicted = range(len(output_data_sequence) - 1, len(output_data_sequence) - 1 + len(predict))
    x_combined = range(len(total_output_sequence))

    # Plotting
    plt.figure(figsize=(10, 5))  # Adjust the figure size as needed
    plt.plot(x_measured, output_data_sequence, 'b-', label='Measured')  # Blue line for measured data
    plt.plot(x_predicted, predict, 'r-', label='Predicted')  # Red line for predicted data
    plt.plot(x_combined, total_output_sequence, 'g-', label='Actual')

    # Adding labels and legend
    plt.xlabel('Time Point')
    plt.ylabel('Value')
    plt.title('Measured vs Predicted')
    plt.legend()

    # Show plot
    plt.show()

