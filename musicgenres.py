import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score


#--------------------------------------------------
# PROBLEM 1A
# Read dataset from CSV
dataset = pd.read_csv("SpotifyFeatures.csv")

# Counting how many sample songs are in the dataset
sample_songs = len(dataset.axes[0])
print("There are {} sample songs in the dataset".format(sample_songs))
#The count of sample songs is 232.725

# Counting how many features are in the dataset
features = len(dataset.axes[1])
print("There are {} features in the dataset".format(features))
#Removing the Genre, Artist Namee, Track Name and Track ID as features for classifying the genre - the count is 14.


# Extracting all rows for where the genre is Pop, assigning label = 1 and the only features are liveness and loudnes
pop_songs = dataset[dataset['genre'] == 'Pop'][['liveness', 'loudness']]
pop_songs.loc[:,'label'] = 1
print("There are {} songs listed as Pop and are labeled with 1.".format(len(pop_songs)))

# Extracting all rows for where the genre is Classical, assigning a column with label = 0 and the only features are liveness and loudnes
classical_songs = dataset[dataset['genre'] == 'Classical'][['liveness', 'loudness']]
classical_songs.loc[:,'label'] = 0
print("There are {} songs listed as Classical and are labeled with 0".format(len(classical_songs)))

#--------------------------------------------------
# PROBLEM 1C

# Learning input array/matrix: sample songs on rows and loudness+liveness as columns
# Convert the dataframes to single arrays
classical_songs_array = classical_songs.to_numpy()
pop_songs_array = pop_songs.to_numpy()

# Combine the arrays and create a new matrix of them
learning_matrix = np.vstack((pop_songs_array, classical_songs_array))

#Shuffle the matrix before splitting
shuffeled_matrix = np.random.permutation(learning_matrix)
print(shuffeled_matrix)

#Add bias to the matrix - a column of 1`s
bias = np.ones((shuffeled_matrix.shape[0], 1))
matrix_with_bias = np.hstack((bias, shuffeled_matrix))

print(matrix_with_bias)

# Define the size for the training set
training_set_size = int(0.8 * len(learning_matrix))

#Split the matrix into a training set and test set
training_set = matrix_with_bias[:training_set_size]
test_set = matrix_with_bias[training_set_size:]

#Remove label from training/test set and create the label vector
x_training = training_set[:, :3]
y_training = training_set[:, 3]

# print(x_training)
# print(y_training)

x_test = test_set[:, :3]
y_test = test_set[:, 3]

# print("TEST")
# print(x_test)
# print(y_test)

#--------------------------------------------------
# PROBLEM 1D
# Plotting the samples of the dataframe

#Seperate the data by extracting the correct column from the different genres
pop_loudness = pop_songs_array[:, 1] 
pop_liveness = pop_songs_array[:, 0] 

classical_loudness = classical_songs_array[:, 1]
classical_liveness = classical_songs_array[:, 0]

# Scatter the values in different colors
plt.scatter(pop_loudness, pop_liveness, color='red')
plt.scatter(classical_loudness, classical_liveness, color='blue')

# Set the labels for the plot
plt.xlabel('Loudness')
plt.ylabel('Liveness')

# Show the plot
#plt.show()

#--------------------------------------------------
# PROBLEM 2A
# Implement logistic discrimination classifier and use the training data to train our machine

# Define the sigmoid function
def sigmoid(z):
    return 1/ (1+np.exp(-z))

# Calculate log loss
def log_loss(true, prediction):

    #Can clip the prediction to avoid log(0)
    prediction = np.clip(prediction, 1e-15, 1- 1e-15)
    #Returns the mathemathical calculation
    return -np.mean(true * np.log(prediction) + (1 - true) * np.log(1 - prediction))

# function for calculating the accuracy of the training
def accuracy(y_pred_acc, y_test_acc):
    return np.sum(y_pred_acc==y_test_acc)/len(y_test)


#Define weight and bias
weight = np.zeros(x_training.shape[1])
bias = 0

#define learning rate and iterations
learning_rate = 0.0001 #changeable
epochs = 10 #changable

# Loop through all iterations to optain the optimal value of m and c
for iterations in range(epochs):

    for i in range(len(x_training)):

        #compute prediction
        z = np.dot(x_training[i], weight) + bias
        y_pred = sigmoid(z)
        
        # compute the gradients
        sgd_weight = (y_pred - y_training[i]) * x_training[i]
        sgd_bias = y_pred - y_training[i]

        # update weight and bias
        weight -= learning_rate * sgd_weight
        bias -= learning_rate * sgd_bias

        pred_training = sigmoid(np.dot(x_training, weight) + bias)
        loss = log_loss(y_training, pred_training)
        print(f'Epoch {iterations + 1}/{epochs}, Loss: {loss}')


# Calculate the predictions with the test set
pred = []

for i in range(len(x_test)):
    z = np.dot(weight, x_test[i]) + bias
    y_pred = sigmoid(z)

    if np.any(y_pred >= 0.5):
        pred.append(1)
    else:
        pred.append(0)

pred_test = np.array(pred)

# Calculate the accuracy of the training
acc = accuracy(pred, y_test)
print(f'The accuracy of the training is:" {acc}')