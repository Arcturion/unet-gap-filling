import numpy as np
from numpy import load
import random
from sklearn.utils import shuffle
import os

import tensorflow as tf
from tensorflow import keras
from keras.layers import Lambda, Reshape, BatchNormalization
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D



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




sat_data_1 = load('tes_data.npy')
sat_data_1[np.isnan(sat_data_1)] = 0
data_scaled_1 = min_max_scaler(np.reshape(sat_data_1, (sat_data_1.shape[0],sat_data_1.shape[1]*sat_data_1.shape[2])))
sat_data = np.reshape(data_scaled_1, (sat_data_1.shape[0],sat_data_1.shape[1],sat_data_1.shape[2]))

sat_data, sat_data_y = make_sequences(sat_data, 3)
sat_data = np.transpose(sat_data, (0, 2, 3, 1))
sat_data = np.flip(sat_data, axis = 1)

gap_index = np.expand_dims(np.zeros(sat_data_y.shape), axis= -1)

sat_data = shuffle(sat_data, random_state=14)
gap_index = gap_index[:1000]




# ----------------------------------------------
# ----------------- M O D E L ------------------
# ----------------------------------------------

class inpaintingModel:

  def prepare_model(self, input_size=(256,256,3)):
    inputs = keras.layers.Input(input_size)

    conv1, pool1 = self.__ConvBlock(64, (5,5), (2,2), 'relu', 'same', inputs)
    conv2, pool2 = self.__ConvBlock(128, (3,3), (2,2), 'relu', 'same', pool1)
    conv3, pool3 = self.__ConvBlock(256, (3,3), (2,2), 'relu', 'same', pool2)
    conv4, pool4 = self.__ConvBlock(512, (3,3), (2,2), 'relu', 'same', pool3)

    conv5, up6 = self.__UpConvBlock(1024, 256, (3,3), (2,2), (2,2), 'relu', 'same', pool4, conv4)
    conv6, up7 = self.__UpConvBlock(512, 128, (3,3), (2,2), (2,2), 'relu', 'same', up6, conv3)
    conv7, up8 = self.__UpConvBlock(256, 64, (3,3), (2,2), (2,2), 'relu', 'same', up7, conv2)
    conv8, up9 = self.__UpConvBlock(128, 32, (3,3), (2,2), (2,2), 'relu', 'same', up8, conv1)

    conv9 = self.__ConvBlock(64, (3,3), (2,2), 'relu', 'same', up9, False)
    conv10 = self.__ConvBlock(32, (3,3), (2,2), 'relu', 'same', conv9, False)

    outputs = keras.layers.Conv2D(1, (3, 3), activation='linear', padding='same')(conv10)

    return keras.models.Model(inputs=[inputs], outputs=[outputs])

  def __ConvBlock(self, filters, kernel_size, pool_size, activation, padding, connecting_layer, pool_layer=True):
    conv = keras.layers.Conv2D(filters=filters, kernel_size=kernel_size, activation=activation, padding=padding)(connecting_layer)
    conv = keras.layers.Conv2D(filters=filters, kernel_size=kernel_size, activation=activation, padding=padding)(conv)
    conv = BatchNormalization()(conv, training=True)

    if pool_layer:
      pool = keras.layers.MaxPooling2D(pool_size)(conv)
      return conv, pool
    else:
      return conv

  def __UpConvBlock(self, filters, up_filters, kernel_size, up_kernel, up_stride, activation, padding, connecting_layer, shared_layer):
    conv = keras.layers.Conv2D(filters=filters, kernel_size=kernel_size, activation=activation, padding=padding)(connecting_layer)
    conv = keras.layers.Conv2D(filters=filters, kernel_size=kernel_size, activation=activation, padding=padding)(conv)
    up = keras.layers.Conv2DTranspose(filters=up_filters, kernel_size=up_kernel, strides=up_stride, padding=padding)(conv)
    up = keras.layers.concatenate([up, shared_layer], axis=3)

    return conv, up

  def select_channel(self, x, n):
    x = x[:, :, :, n]
    x = Reshape((x.shape[1], x.shape[2], 1))(x)
    return x





# ----------------------------------------------
# ----------------- WRAPPER --------------------
# ----------------------------------------------


keras.backend.clear_session()
wrapper_model = inpaintingModel().prepare_model()

image_input = Input(shape=(256, 256, 3))
outputs1 = wrapper_model(image_input)

image_input2 = tf.slice(image_input, [0, 0, 0, 1], [-1, 256, 256, 2]) # shape (1, 256, 256, 2), remove the first channel
arr = tf.concat([outputs1, image_input2], -1) # shape (1, 256, 256, 3), replace the first channel

outputs2 = wrapper_model(arr)



# --------------------------------------------
# ----------------- COMPILE ------------------
# --------------------------------------------


keras.backend.clear_session()
# load the pre-trained model
pretrained_model = Model(inputs=[image_input], outputs=[outputs2])

# set all layers except the last two to be untrainable
for layer in pretrained_model.layers[3:30]:
    layer.trainable = False

# Define the loss function
def gap_loss(gap_mask):
    def loss(y_true, y_pred):
        y_pred = y_pred[y_true>0]
        y_true = y_true[y_true>0]
        return tf.reduce_mean(tf.abs(y_true - y_pred))
    return loss

image_input = Input(shape=(256, 256, 3))
gap_mask_input = Input(shape=(256, 256, 1))

outputs = pretrained_model(image_input)

new_model = Model(inputs=[image_input, gap_mask_input], outputs=[outputs])
new_model.load_weights('fine-tuned.keras')

#COMPILE
opt = keras.optimizers.Adam(learning_rate=0.000001)
new_model.compile(optimizer=opt, loss=gap_loss(gap_mask_input))
keras.utils.plot_model(new_model, show_shapes=True, dpi=76, to_file='model_v1.png')
#model.summary()




# -----------------------------------------------
# --- Create Ocean Index for Plotting Purpose ---
# -----------------------------------------------


ocean_index = sat_data.copy()
ocean_index = np.sum(ocean_index[:,:,:,0], axis=0)
ocean_index[ocean_index > 0 ] = 1
ocean_index[ocean_index == 0 ] = np.nan



# ----------------------------------------------
# ----------------- INFERENCE ------------------
# ----------------------------------------------

new_prediction = new_model.predict([sat_data[:25], gap_index[:25]])




# -----------------------------------------
# ----------- Post Processing -------------
# -----------------------------------------

new_prediction[new_prediction<0.0001] = np.nan
real_sat = sat_data[:25, :, :, -1]
real_sat[real_sat<0.0001] = np.nan
diff = np.reshape(new_prediction, (25,256,256))-real_sat
filled = real_sat.copy()
filled[np.isnan(filled)] = new_prediction[np.isnan(filled)][:,0]
np.save('gapfilled.npy', filled)
