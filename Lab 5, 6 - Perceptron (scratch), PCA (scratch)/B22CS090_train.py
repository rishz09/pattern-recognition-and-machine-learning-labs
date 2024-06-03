import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys

n = 10000
x_train = None
y_train = None
n_train = -1


def readTrainData(file_path):
    filename = 'B22CS090_' + file_path
    
    with open(filename, 'r') as file:
        global n_train
        n_train = int(file.readline())
    
    #reading training data
    arr = np.loadtxt(filename, skiprows=1)
    df = pd.DataFrame(arr)

    global x_train
    global y_train

    #separating training data into features and labels
    x_train  = df.iloc[:, :4].copy()
    y_train = df.iloc[:, 4].copy()
    x_train = x_train.to_numpy()
    y_train = y_train.to_numpy(dtype=np.int8)


#method to normalise training set
def standardize(x):
    mean, std = x.mean(axis=0), x.std(axis=0)
    x_norm = (x - mean) / std

    return mean, std, x_norm

#plots and saves bar graphs of distribution of class labels
def plotBarGraph(y, percentage):
    unique, counts = np.unique(y, return_counts=True)
    plt.figure()

    bars = plt.bar(unique, counts, color='orange', width=0.5)

    plt.xticks(unique)
    plt.xlabel('Class Labels')
    plt.ylabel('Count')
    plt.title(f'Distribution for {percentage}% Training Data')

    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval, str(yval), ha='center', va='bottom')


    #plt.savefig(f'distribution_training_{percentage}')


class Perceptron:
    def __init__(self, num_features, *, w_all = None):
        np.random.seed(1)
        self.num_features = num_features

        #initializes weights and biases
        if w_all is None:
            self.w_all = np.random.standard_normal(num_features + 1)
        else:
            self.w_all = w_all

        self.weights, self.bias = self.w_all[:4], self.w_all[4]
        self.w_all = None
        self.acc_list = []

    #calculates predicted class label of a sample
    def forward(self, x):
        linear = (x @ self.weights.T) + self.bias
        #applies threshold
        preds = np.where(linear >= 0., 1, 0)
        return preds
    
    def backward(self, x, y):
        preds = self.forward(x)
        #calculates error
        errors = y - preds
        return errors
    
    #saves the weights and biases for future use by test set
    def save_params(self, percentage):
        with open('B22CS090_stats.txt', 'a') as file:
            file.write(f'Synthetic data used for training: {percentage}%\n\n')
            file.write(f'Weights and Bias after Training:\n')

            for i in range(len(self.weights)):
                file.write(str(self.weights[i])+ ' ')

            file.write(str(self.bias[0]) + '\n\n')

    def train(self, x, y, *, epochs, get_acc_list=False, percentage=70):
        for e in range(epochs):
            #makes a list of accuracies to plot a graph
            if get_acc_list == True:
                self.acc_list.append(self.evaluate(x, y))
                
            for i in range(len(y)):
                errors = self.backward(x[i].reshape(1, self.num_features), y[i])
                #updates weights and biases from the error calculated
                self.weights += errors * x[i]
                self.bias += errors
        
        self.save_params(percentage)
        return self.acc_list

    #calculates predictions after training
    def predictions(self, x):
        return self.forward(x)
    
    #returns accuracy in fraction
    def evaluate(self, x, y, *, saveAccuracy = False, training = True):
        preds = self.forward(x)
        acc = np.sum(preds == y) / len(y)

        if saveAccuracy:
            self.saveAcc(acc, training)

        return acc

    #saves accuracy in stats.txt
    def saveAcc(self, acc, training=True):
        str = 'Training' if training else 'Test'

        with open('B22CS090_stats.txt', 'a') as file:
            file.write(f'{str} accuracy: {acc * 100}%\n\n')



def main():

    #checks if right number of command line arguments have been entered
    if len(sys.argv) != 2:
        print('Incorrect number of arguments!')
        return
    
    file_path = sys.argv[1]

    readTrainData(file_path)
    
    training_fraction = [0.28571425, 0.714285714286, 1.0]
    synth_frac = [20, 50, 70]
    acc_arr = []

    #newly creates the stats.txt file
    with open('B22CS090_stats.txt', 'w') as file:
        pass    

    for i in range(len(training_fraction)):

        #takes given fraction of data as training data
        x_train_frac = x_train[ : int(training_fraction[i] * len(x_train))]
        y_train_frac = y_train[ : int(training_fraction[i] * len(y_train))]

        plotBarGraph(y_train_frac, synth_frac[i])

        mean, std, x_train_norm_frac = standardize(x_train_frac)
        
        model = Perceptron(x_train_norm_frac.shape[1])
        acc_list = model.train(x_train_norm_frac, y_train_frac, epochs=10, get_acc_list=True, percentage=synth_frac[i])
        acc_arr.append(acc_list)

        #stores training set mean and std for future normalisation of test set
        with open('B22CS090_stats.txt', 'a') as file:
            file.write('Training set mean and standard deviations:\n')
        
            for val in mean:
                file.write(str(val) + ' ')
            file.write('\n')

            for val in std:
                file.write(str(val) + ' ')
            file.write('\n\n')
        
        print(f'Training Accuracy: {model.evaluate(x_train_norm_frac, y_train_frac, saveAccuracy=True)}')
        print('Training Over and Weights are saved')

    #plots accuracy vs epochs graph
    plt.figure()
    for i in range(3):
        plt.plot(list(range(len(acc_arr[i]))), acc_arr[i], label=f'{synth_frac[i]}%')
        plt.xlabel('Epochs')
        plt.ylabel(f'Training Accuracy')
        plt.legend(loc='best')
        #plt.savefig(f'acc_vs_epochs.png')
    

if __name__ == '__main__':
    main()




