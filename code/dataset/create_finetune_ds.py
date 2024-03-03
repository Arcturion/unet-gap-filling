import matplotlib.pyplot as plt
from numpy import load
import numpy as np
import random

from sklearn.utils import shuffle


def make_sequences(data_raw, look_back):
    sequences = [data_raw[index: index + look_back] for index in range(len(data_raw) - look_back)]
    sequences = np.array(sequences)
    x_train = sequences
    y_train = sequences[:, -1]

    return x_train, y_train

# MinMax Scaler
def min_max_scaler(data):
    masked_data = data[data > 0]
    #min_val = np.min(masked_data)
    #max_val = np.max(data)
    min_val = 5
    max_val = 35
    data[data > 0] = (masked_data - min_val) / (max_val - min_val)
    return data


sat_data_1 = load('2015sampai2020.npy')
sat_data_2 = load('2020sampai2022.npy')

#scalling
data_scaled_1 = min_max_scaler(np.reshape(sat_data_1, (sat_data_1.shape[0],sat_data_1.shape[1]*sat_data_1.shape[2])))
sat_data_1 = np.reshape(data_scaled_1, (sat_data_1.shape[0],sat_data_1.shape[1],sat_data_1.shape[2]))

data_scaled_2 = min_max_scaler(np.reshape(sat_data_2, (sat_data_2.shape[0],sat_data_2.shape[1]*sat_data_2.shape[2])))
sat_data_2 = np.reshape(data_scaled_2, (sat_data_2.shape[0],sat_data_2.shape[1],sat_data_2.shape[2]))

# Make Sequence
sat_data_1, sat_data_1_y = make_sequences(sat_data_1, 3)
sat_data_1 = np.transpose(sat_data_1, (0, 2, 3, 1))

sat_data_2, sat_data_2_y = make_sequences(sat_data_2, 3)
sat_data_2 = np.transpose(sat_data_2, (0, 2, 3, 1))


random_number = 20   ### Change the state each iteration

#Join
sat_data = np.concatenate((sat_data_1, sat_data_2), axis=0)
sat_data_y = np.concatenate((sat_data_1_y, sat_data_2_y), axis=0)

sat_data = np.flip(sat_data, axis = 1)
sat_data_y = np.flip(sat_data_y, axis = 1)

#Multipy
#sat_data = np.repeat(sat_data, 3, axis=0)
#sat_data_y = np.repeat(sat_data_y, 3, axis=0)

sat_data, sat_data_y = shuffle(sat_data, sat_data_y, random_state=random_number)

real_value = np.expand_dims(sat_data[:,:,:,0], axis= -1)

gap_index = np.zeros(real_value.shape)
gap_index = np.concatenate((real_value, gap_index), axis=-1)


#Take Random pixel

def noising_min(image, num_pixels=5000):

    array = np.array(image[:,:,0])
    gap = np.array(image[:,:,1])

    # Select 5000 random indices for rows and columns
    i = np.random.choice(256, size=num_pixels, replace=True)
    j = np.random.choice(256, size=num_pixels, replace=True)

    # Set the selected pixels to 1
    gap[i, j] = array[i, j]
    array[i, j]= 0.0 
    
    gap = np.concatenate((np.expand_dims(array, axis=-1), np.expand_dims(gap, axis=-1)), axis=-1)
    
    return gap

for x in range(1):
  gap_index[:, :, :] = np.array([*map(noising_min, gap_index[:, :, :, :])])

#Take the random pixel as other data
#masukkan index ke dataset sat
sat_data[:,:,:,0] = gap_index[:,:,:,0]
gap_index =  np.expand_dims(gap_index[:,:,:,1], axis=-1)

#save it
np.save('sat_data', sat_data)
np.save('gap_index', gap_index)
