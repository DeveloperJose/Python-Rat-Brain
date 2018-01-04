from keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, ZeroPadding2D
from keras.models import Model
from keras.callbacks import TensorBoard
from keras.datasets import mnist
from keras.wrappers.scikit_learn import KerasRegressor
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV
import numpy as np

np.random.seed(0)

sw_data = np.load('atlas_sw.npz')
x_train = sw_data['images'].astype('float32') / 255.
x_shape = (x_train.shape[0], x_train.shape[1], x_train.shape[2], 1)
x_train = np.reshape(x_train, x_shape)

print("X_Train: ", x_train.shape)

pw_data = np.load('atlas_pw.npz')
pw_im = pw_data['images'].astype('float32') / 255.
pw_shape = pw_im.shape[0], pw_im.shape[1], pw_im.shape[2], 1
pw_im = np.reshape(pw_im, pw_shape)

#x_train = x_train.append(pw_im)

x_test = pw_im
#x_test = np.array([pw_im[7],pw_im[10],pw_im[26],pw_im[39]])
print("X_Test: ", x_test.shape)
#x_train = x_train.astype('float32') / 255.
#x_test = x_test.astype('float32') / 255.
#x_train = np.reshape(x_train, (len(x_train), 28, 28, 1))  # adapt this if using `channels_first` image data format
#x_test = np.reshape(x_test, (len(x_test), 28, 28, 1))  # adapt this if using `channels_first` image data format

#np.save('x_train', x_train)

noise_factor = 0.4
x_train_noisy = x_train + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=x_train.shape)
x_test_noisy = x_test + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=x_test.shape)

x_train_noisy = np.clip(x_train_noisy, 0., 1.)
x_test_noisy = np.clip(x_test_noisy, 0., 1.)
print("X_Train_Noisy", x_train_noisy.shape, "Test", x_test_noisy.shape)

def create_network(optimizer='rmsprop'):
    input_img = Input(shape=(200, 120, 1))  # adapt this if using `channels_first` image data format
    x = Conv2D(16, (3, 3), activation='relu', padding='same')(input_img)
    x = MaxPooling2D((2, 2), padding='same')(x)
    x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
    x = MaxPooling2D((2, 2), padding='same')(x)
    x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
    encoded = MaxPooling2D((2, 2), padding='valid', name='encoder')(x)

    # at this point the representation is (4, 4, 8) i.e. 128-dimensional

    x = Conv2D(8, (3, 3), activation='relu', padding='same')(encoded)
    x = UpSampling2D((2, 2))(x)
    x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
    x = UpSampling2D((2, 2))(x)
    x = Conv2D(16, (3, 3), activation='relu', padding='same')(x)
    x = UpSampling2D((2, 2))(x)
    decoded = Conv2D(1, (3, 3), activation='sigmoid', padding='same')(x)
    autoencoder = Model(input_img, decoded)
    autoencoder.compile(optimizer=optimizer, loss='binary_crossentropy')
    return autoencoder

def train_model():
    autoencoder = create_network('adadelta')
    #epochs = [50, 60, 70, 80, 90, 100]
    #batches = [16, 32]
    #optimizers = ['rmsprop', 'adam', 'adadelta']
    
    autoencoder.fit(x_train_noisy, x_train,
                    epochs=100,
                    batch_size=16,
                    shuffle=True,
                    validation_data=(x_test_noisy, x_test),
                    callbacks=[TensorBoard(log_dir='/tmp/ae_both', histogram_freq=0, write_graph=False)]
                    )

    autoencoder.save('autoencoder.h5')

train_model()
