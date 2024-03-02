import tensorflow as tf
from tensorflow.keras.layers import Input, Concatenate
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam



# ----------------------------------------------------
# --------------------- M O D E L --------------------
# ----------------------------------------------------


def define_model(pretrained_model_path):
    # Load the pre-trained model
    pretrained_model = inpaintingModel().prepare_model()

    # Define image input
    image_input = Input(shape=(256, 256, 3))

    # Pass the image_input to the pre-trained model and get the first output
    outputs1 = pretrained_model(image_input)

    # Extract channels 2 and 3 from image_input
    image_input2 = image_input[:, :, :, 1:]  # Remove the first channel

    # Concatenate outputs1 and image_input2 along the channel axis
    arr = Concatenate(axis=-1)([outputs1, image_input2])

    # Pass arr to the pre-trained model and get the second output
    outputs2 = pretrained_model(arr)

    # Define the new model with image_input as input and outputs2 as output
    new_model = Model(inputs=image_input, outputs=outputs2)

    # Load weights
    new_model.load_weights(pretrained_model_path)

    # Compile the model
    opt = Adam(learning_rate=0.00001)
    new_model.compile(optimizer=opt, loss='mean_squared_error')

    return new_model


pretrained_model_path = '/content/drive/MyDrive/Double U-Net Start Over/weight/weights.05-0.0032.keras' ### Fill the weight
new_model = define_model(pretrained_model_path)
new_model.summary()





# ----------------------------------------------------
# --------------------- T R A I N --------------------
# ----------------------------------------------------



# Define some callbacks to improve training.
early_stopping = keras.callbacks.EarlyStopping(monitor="val_loss", patience=10)
reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor="val_loss", patience=5)
#checkpoint = tf.keras.callbacks.ModelCheckpoint(filepath='/content/drive/MyDrive/Double U-Net Start Over/weight/weights.{epoch:02d}-{val_loss:.4f}.hdf5')
checkpoint = tf.keras.callbacks.ModelCheckpoint(filepath='/content/drive/MyDrive/Double U-Net Start Over/weight/weights.{epoch:02d}-{val_loss:.4f}.keras')


# Define modifiable training hyperparameters.
epochs = 50
batch_size = 16

# Fit the model to the training data.
new_model.fit(
    x_train,
    y_train,
    batch_size=batch_size,
    epochs=epochs,
    validation_data=(x_test, y_test),
    callbacks=[early_stopping, reduce_lr, checkpoint],
)
