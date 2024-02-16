import matplotlib.pyplot as plt
from numpy import load
import numpy as np
import random

from sklearn.preprocessing import MinMaxScaler
from sklearn.utils import shuffle

def make_sequence(data_raw, look_back):
    data = []

    # create all possible sequences of length seq_len
    for index in range(len(data_raw) - look_back):
        data.append(data_raw[index: index + look_back])

    data = np.array(data);

    x_train = np.array(data[:,:,:,:])
    y_train = np.array(data[:,-1,:,:])

    return x_train, y_train


sat_data_1 = load('2015sampai2020.npy')
sat_data_2 = load('2020sampai2022.npy')

#scalling
scaler_1 = MinMaxScaler().fit(np.reshape(sat_data_1, (sat_data_1.shape[0],sat_data_1.shape[1]*sat_data_1.shape[2])))
data_scaled_1 = scaler_1.transform(np.reshape(sat_data_1, (sat_data_1.shape[0],sat_data_1.shape[1]*sat_data_1.shape[2])))
sat_data_1 = np.reshape(data_scaled_1, (sat_data_1.shape[0],sat_data_1.shape[1],sat_data_1.shape[2]))

scaler_2 = MinMaxScaler().fit(np.reshape(sat_data_2, (sat_data_2.shape[0],sat_data_2.shape[1]*sat_data_2.shape[2])))
data_scaled_2 = scaler_2.transform(np.reshape(sat_data_2, (sat_data_2.shape[0],sat_data_2.shape[1]*sat_data_2.shape[2])))
sat_data_2 = np.reshape(data_scaled_2, (sat_data_2.shape[0],sat_data_2.shape[1],sat_data_2.shape[2]))

# Make Sequence
sat_data_1, sat_data_1_y = make_sequence(sat_data_1, 3)
sat_data_1 = np.transpose(sat_data_1, (0, 2, 3, 1))

sat_data_2, sat_data_2_y = make_sequence(sat_data_2, 3)
sat_data_2 = np.transpose(sat_data_2, (0, 2, 3, 1))

#concat both dataset
sat_data = np.concatenate((sat_data_1, sat_data_2), axis=0)
sat_data_y = np.concatenate((sat_data_1_y, sat_data_2_y), axis=0)

#multiply numbers
#sat_data = np.repeat(sat_data, 3, axis=0)
#sat_data_y = np.repeat(sat_data_y, 3, axis=0)

sat_data, sat_data_y = shuffle(sat_data, sat_data_y, random_state=16)



real_value = np.expand_dims(sat_data[:,:,:,0], axis= -1)

gap_index = np.zeros(real_value.shape)
gap_index = np.concatenate((real_value, gap_index), axis=-1)


#Take Random pixel

def noising_min(image):
    array = np.array(image[:,:,0])
    gap = np.array(image[:,:,1])
    i = random.choice(range(0,256)) # x coordinate for the top left corner of the mask
    j = random.choice(range(0,256)) # y coordinate for the top left corner of the mask
    gap[i, j] = array[i, j]
    array[i, j]=0.0 # setting the pixels in the masked region to -1
    gap = np.concatenate((np.expand_dims(array, axis=-1), np.expand_dims(gap, axis=-1)), axis=-1)

    return gap

for x in range(5000):
  gap_index[:, :, :] = np.array([*map(noising_min, gap_index[:, :, :, :])])

#Take the random pixel as other data
#masukkan index ke dataset sat
sat_data[:,:,:,0] = gap_index[:,:,:,0]
gap_index =  np.expand_dims(gap_index[:,:,:,1], axis=-1)

#save it
np.save('sat_data', sat_data)
np.save('gap_index', gap_index)
