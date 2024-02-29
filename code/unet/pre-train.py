from numpy import load
import numpy as np
from sklearn import preprocessing
from sklearn.utils import shuffle
import random
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow import keras
from keras.layers import Lambda, Reshape




#create 3-days stacked array
def make_sequence(data_raw, look_back):
    data = []
    
    # create all possible sequences of length seq_len
    for index in range(len(data_raw) - look_back): 
        data.append(data_raw[index: index + look_back])
    
    data = np.array(data);
    
    x_train = np.array(data[:,:,:,:])
    y_train = np.array(data[:,-1,:,:])
    
    return x_train, y_train

#MinMax Scaler func
def MinMax(masuk):
  min = np.min(masuk[masuk>0])
  max = np.max(masuk)

  print(min)
  print(max)

  masuk[masuk>0] = (masuk[masuk>0]-min)/(max-min)

  return masuk

#dice coef eval matrix func
def dice_coef(y_true, y_pred):
    y_true_f = keras.backend.flatten(y_true)
    y_pred_f = keras.backend.flatten(y_pred)
    intersection = keras.backend.sum(y_true_f * y_pred_f)
    return (2. * intersection) / (keras.backend.sum(y_true_f + y_pred_f))



## Cloud Mask

sat_data_iden = load('cloud_index.npy')
sat_data_iden = sat_data_iden[:1128]
sat_data_iden = np.flip(sat_data_iden, 1)
sat_data_iden = np.flip(sat_data_iden, 1)
sat_data_iden = sat_data_iden[:,:256,:256]
sat_data_iden, sat_data_iden_y = make_sequence(sat_data_iden, 3)

sat_data_iden = np.transpose(sat_data_iden, (0, 2, 3, 1)) 


## SST L4 Dataset

data = load('HIMA_SST_L4.npy')
data[np.isnan(data)]=0
data_scaled = MinMax(data)
data, data_y = make_sequence(data_scaled, 3)
data = np.transpose(data, (0, 2, 3, 1))

x_train = np.repeat(data, 3, axis=0)
y_train = np.repeat(data_y, 3, axis=0)

del data
del data_scaled
del data_y

x_train, y_train = shuffle(x_train, y_train, random_state=16)



#create artificial cloud

x_train = sat_data_iden*x_train

#free the memory
del sat_data_iden
del sat_data_iden_y

y_train = np.expand_dims(y_train, axis=-1)

#divide test and train
x_test = x_train[1000:, :, :, :]
y_test = y_train[1000:, :, :, :]
x_train = x_train[:1000, :, :, :]
y_train = y_train[:1000, :, :, :]



# ----------------------------------------------------
# --------------------- M O D E L --------------------
# ----------------------------------------------------

keras.backend.clear_session()
model = inpaintingModel().prepare_model()
model.compile(optimizer='adam', loss='mean_absolute_error', metrics=[dice_coef])
#keras.utils.plot_model(model, show_shapes=True, dpi=76, to_file='model_v1.png')
#model.summary()

model.load_weights('/content/drive/MyDrive/INPAINTING/weights.30-0.0104.hdf5')
#model.save_weights('/content/drive/MyDrive/INPAINTING_2/weights-1.hdf5')

early_stopping = keras.callbacks.EarlyStopping(monitor="val_loss", patience=10)
reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor="val_loss", patience=5)
checkpoint = tf.keras.callbacks.ModelCheckpoint(filepath='/content/drive/MyDrive/INPAINTING/weights.{epoch:02d}-{val_loss:.4f}.hdf5')

epochs = 100
batch_size = 16

# ----------------------------------------------------
# --------------------- T R A I N --------------------
# ----------------------------------------------------

model.fit(
    x_train,
    y_train,
    batch_size=batch_size,
    epochs=epochs,
    validation_data=(x_test, y_test),
    callbacks=[early_stopping, reduce_lr, checkpoint],
)
