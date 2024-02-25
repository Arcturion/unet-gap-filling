import tensorflow as tf
from tensorflow import keras
from keras.layers import Lambda, Reshape


def dice_coef(y_true, y_pred):
    y_true_f = keras.backend.flatten(y_true)
    y_pred_f = keras.backend.flatten(y_pred)
    intersection = keras.backend.sum(y_true_f * y_pred_f)
    return (2. * intersection) / (keras.backend.sum(y_true_f + y_pred_f))

class inpaintingModel:
  '''
  Build UNET like model for image inpaining task.
  '''
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
      
      conv = keras.layers.Conv2D(filters=filters, kernel_size=kernel_size, activation=activation, padding=padding, kernel_regularizer=tf.keras.regularizers.l2(l2_reg), bias_regularizer=tf.keras.regularizers.l1(l1_reg))(connecting_layer)
      conv = keras.layers.Conv2D(filters=filters, kernel_size=kernel_size, activation=activation, padding=padding, kernel_regularizer=tf.keras.regularizers.l2(l2_reg), bias_regularizer=tf.keras.regularizers.l1(l1_reg))(conv)
      
      if pool_layer:
          pool = keras.layers.MaxPooling2D(pool_size)(conv)
      return conv, pool
      else:
      return conv

   def __UpConvBlock(self, filters, up_filters, kernel_size, up_kernel, up_stride, activation, padding, connecting_layer, shared_layer, l1_reg=0.001, l2_reg=0.001):
    
      conv = keras.layers.Conv2D(filters=filters, kernel_size=kernel_size, activation=activation, padding=padding, kernel_regularizer=tf.keras.regularizers.l2(l2_reg), bias_regularizer=tf.keras.regularizers.l1(l1_reg))(connecting_layer)
      conv = keras.layers.Conv2D(filters=filters, kernel_size=kernel_size, activation=activation, padding=padding, kernel_regularizer=tf.keras.regularizers.l2(l2_reg), bias_regularizer=tf.keras.regularizers.l1(l1_reg))(conv)
    
      up = keras.layers.Conv2DTranspose(filters=up_filters, kernel_size=up_kernel, strides=up_stride, padding=padding)(conv)
      up = keras.layers.concatenate([up, shared_layer], axis=3)

    return conv, up

    def select_channel(self, x, n):
        x = x[:, :, :, n]
        x = Reshape((x.shape[1], x.shape[2], 1))(x)
    return x




keras.backend.clear_session()
# load the pre-trained model
wrapper_model = inpaintingModel().prepare_model()
wrapper_model.load_weights('/content/drive/MyDrive/transfer_learning_SST/weights.30-0.0097.hdf5')

image_input = Input(shape=(256, 256, 3))

outputs1 = wrapper_model(image_input)

image_input2 = tf.slice(image_input, [0, 0, 0, 1], [-1, 256, 256, 2]) # shape (1, 256, 256, 2), remove the first channel
arr = tf.concat([outputs1, image_input2], -1) # shape (1, 256, 256, 3), replace the first channel

outputs2 = wrapper_model(arr)

#new_model = Model(inputs=[image_input], outputs=[outputs2])
