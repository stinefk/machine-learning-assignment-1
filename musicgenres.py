import pandas as pd
import numpy as np

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

# Learning input array/matrix: sample songs on rows and loudness+liveness as columns

# Convert the dataframes to single arrays
classical_songs_array = classical_songs.to_numpy()
pop_songs_array = pop_songs.to_numpy()

# Combine the arrays and create a new matrix of them
learning_matrix = np.vstack((pop_songs_array, classical_songs_array))

#Vector array of genre
genre_vector = learning_matrix[:, 2]

# Define the size for the training set
training_set_size = int(0.8 * len(learning_matrix))
