import numpy as np
from numpy import load
from sklearn.utils import shuffle



# --------------------------------------------
# ---------- CREATE CLOUD MASK ---------------
# --------------------------------------------


def make_sequences(data_raw, look_back):
    sequences = [data_raw[index: index + look_back] for index in range(len(data_raw) - look_back)]
    sequences = np.array(sequences)
    x_train = sequences
    y_train = sequences[:, -1]
    return x_train, y_train

def load_and_process_data(file_path, sequence_length):

    sat_data_iden = load(file_path)
    #sat_data_iden = np.repeat(sat_data_iden, 34, axis=0)

    sat_data_iden = shuffle(sat_data_iden, random_state=41)[:1128]
    sat_data_iden, sat_data_iden_y = make_sequence(sat_data_iden, sequence_length)
    sat_data_iden = np.transpose(sat_data_iden, (0, 2, 3, 1))

    return sat_data_iden, sat_data_iden_y


sat_data_iden, sat_data_iden_y = load_and_process_data('/content/drive/MyDrive/transfer_learning_SST/2015sampai2020_cloud_index.npy', 3)

sat_data_iden = np.flip(sat_data_iden, axis = 1)
sat_data_iden_y = np.flip(sat_data_iden_y, axis = 1)
sat_data_iden = sat_data_iden[:,-256:,:256]
sat_data_iden_y = sat_data_iden_y[:,-256:,:256]

print(sat_data_iden.shape, sat_data_iden_y.shape)





# -------------------------------------------------
# ------------ CREATE TRAINING DATA ---------------
# -------------------------------------------------


# Load data
data = np.flip(load('/content/drive/MyDrive/RES_OCEANMOVE/L4_GHRSST/HIMA_SST_L4.npy'), axis=1)
data[np.isnan(data)] = 0

# MinMax Scaler
def min_max_scaler(data):
    masked_data = data[data > 0]
    min_val = np.min(masked_data)
    max_val = np.max(data)
    data[data > 0] = (masked_data - min_val) / (max_val - min_val)
    return data

data_scaled = min_max_scaler(data)

# Create sequences and transpose
data_x, data_y = make_sequences(data_scaled, 3)
data_x = np.transpose(data_x, (0, 2, 3, 1))

# Repeat data
x_train = np.repeat(data_x, 3, axis=0)
y_train = np.repeat(data_y, 3, axis=0)

# Shuffle data
x_train, y_train = shuffle(x_train, y_train, random_state=16)

# Artificial cloud
x_train *= sat_data_iden

# Expand y_train dimension
y_train = np.expand_dims(y_train, axis=-1)

# Split train-test
x_test = x_train[1000:]
y_test = y_train[1000:]
x_train = x_train[:1000]
y_train = y_train[:1000]

print(x_train.shape, y_train.shape, x_test.shape, y_test.shape)
