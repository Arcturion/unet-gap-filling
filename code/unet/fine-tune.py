import tensorflow as tf
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow import keras
import os

keras.backend.clear_session()
# load the pre-trained model
wrapper_model = inpaintingModel().prepare_model()


# create a new model based on the pre-trained model
image_input = Input(shape=(256, 256, 3))

# pass the image_input to the pre-trained model and get the first output
outputs1 = wrapper_model(image_input)

# add a zero to the begin and size arguments for tf.slice
image_input2 = tf.slice(image_input, [0, 0, 0, 1], [-1, 256, 256, 2]) # shape (1, 256, 256, 2), remove the first channel
arr = tf.concat([outputs1, image_input2], -1) # shape (1, 256, 256, 3), replace the first channel

# pass the arr to the pre-trained model and get the second output
outputs2 = wrapper_model(arr)

# load the pre-trained model
pretrained_model = Model(inputs=[image_input], outputs=[outputs2])
pretrained_model.load_weights('/content/drive/MyDrive/Double U-Net Start Over/weight/pre-trained-weight.keras')

for layer in pretrained_model.layers[3:30]:
    layer.trainable = False

def gap_loss(gap_mask):
    def loss(y_true, y_pred):
        y_pred = y_pred[y_true>0]
        y_true = y_true[y_true>0]
        return tf.reduce_mean(tf.square(y_true - y_pred))
    return loss

image_input = Input(shape=(256, 256, 3))
gap_mask_input = Input(shape=(256, 256, 1))

outputs = pretrained_model(image_input)

new_model = Model(inputs=[image_input, gap_mask_input], outputs=[outputs])

#COMPILE
opt = keras.optimizers.Adam(learning_rate=0.000001)
new_model.compile(optimizer=opt, loss=gap_loss(gap_mask_input))
keras.utils.plot_model(new_model, show_shapes=True, dpi=76, to_file='model_v1.png')
#model.summary()

checkpoint = tf.keras.callbacks.ModelCheckpoint(filepath='/content/drive/MyDrive/transfer_learning_SST/model_weight/Double-Unet-weights.{epoch:02d}-{val_loss:.4f}.hdf5')

# train the new model
new_model.fit([sat_data, gap_index], gap_index, batch_size=15, epochs=5, validation_data=([sat_data, gap_index], gap_index), callbacks=[checkpoint])
