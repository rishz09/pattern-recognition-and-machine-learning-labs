#importing my train.py file just to access functionalities of my Perceptron class
from B22CS090_train import Perceptron
import sys
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def getDataset(file_path):
    n_test = None
    arr = None

    with open(f'B22CS090_{file_path}', 'r') as file:
        n_test = int(file.readline())
        arr = np.loadtxt(f'B22CS090_{file_path}', skiprows=1)

    #reads the true class labels from data.txt
    y_test = np.loadtxt('B22CS090_data.txt', usecols=(4,), skiprows=7000)

    return n_test, arr, y_test

def getStatData(*, case=0):

    #reads mean and std for a certain case out of the 3 cases
    mean = np.genfromtxt('B22CS090_stats.txt', dtype=np.float64, skip_header=6 + case * 11, max_rows=1)
    std = np.genfromtxt('B22CS090_stats.txt', dtype=np.float64, skip_header=7 + case * 11, max_rows=1)

    return mean, std

def main():

    #checks for correct number of command line arguments
    if len(sys.argv) != 2:
        print('Incorrect number of arguments!')
        return
    
    file_path = sys.argv[1]

    n_test, x_test, y_test  = getDataset(file_path)

    synth_frac = [20, 50, 70]

    preds = np.empty((n_test, 3))

    for i in range(3):
        #reads weights and biases as made by training of the model for the specific case
        w_all = np.genfromtxt('B22CS090_stats.txt', dtype=np.float64, skip_header = 3 + i * 11, max_rows=1)

        mean, std = getStatData(case=i)

        x_test_norm = (x_test - mean) / std

        #weights and biases are passed on to the model, so standard normal initialization does not occur
        model = Perceptron(x_test_norm.shape[0], w_all=w_all)

        with open('B22CS090_stats.txt', 'a') as file:
            file.write(f'Training set: {synth_frac[i]}%\n')
        
        #predicts the class labels
        pred = model.predictions(x_test_norm)
        print(pred)
        
        preds[:, i] = pred

        #calculates accuracies and stores it in stats.txt
        model.evaluate(x_test_norm, y_test, saveAccuracy=True, training=False)

    #writes the predicted class labels in a new txt file
    with open('B22CS090_y_preds.txt', 'w') as file:
        for i in range(preds.shape[0]):
            for j in range(preds.shape[1]):
                file.write(str(int(preds[i][j])) + ' ')

            file.write('\n')
        


if __name__ == '__main__':
    main()
