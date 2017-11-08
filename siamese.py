'''Train a Siamese MLP on pairs of digits from the MNIST dataset.

It follows Hadsell-et-al.'06 [1] by computing the Euclidean distance on the
output of the shared network and by optimizing the contrastive loss (see paper
for mode details).

[1] "Dimensionality Reduction by Learning an Invariant Mapping"
    http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf

Gets to 99.5% test accuracy after 20 epochs.
3 seconds per epoch on a Titan X GPU
'''
import numpy as np
np.random.seed(1337)  # for reproducibility

import pylab as plt
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Input, Lambda, Conv2D, MaxPooling2D, Activation, Flatten
from keras.layers.normalization import BatchNormalization
from keras.optimizers import RMSprop
from keras import backend as K
from keras.callbacks import EarlyStopping, ModelCheckpoint

from timeit import default_timer as timer

EPOCHS = 10
BATCH_SIZE = 32

sw_data = np.load('atlas_sw.npz')
x_train = sw_data['images']
x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], x_train.shape[2], 1)
#x_train = x_train.reshape(x_train.shape[0], x_train.shape[1] * x_train.shape[2])
y_train = sw_data['labels']

pw_data = np.load('atlas_pw.npz')
x_test = pw_data['images']
x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], x_test.shape[2], 1)
#x_test = x_test.reshape(x_test.shape[0], x_test.shape[1] * x_test.shape[2])
y_test = pw_data['labels']

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255

INPUT_SHAPE = (x_train.shape[1], x_train.shape[2], 1)
#INPUT_SHAPE = (x_train.shape[1],)

def euclidean_distance(vects):
    x, y = vects
    return K.sqrt(K.sum(K.square(x - y), axis=1, keepdims=True))

def eucl_dist_output_shape(shapes):
    shape1, shape2 = shapes
    return (shape1[0], 1)

def contrastive_loss(y_true, y_pred):
    '''Contrastive loss from Hadsell-et-al.'06
    http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
    '''
    margin = 1
    return K.mean(y_true * K.square(y_pred) + (1 - y_true) * K.square(K.maximum(margin - y_pred, 0)))

def create_base_network():
    '''Base network to be shared (eq. to feature extraction).
    '''
    model = Sequential()
    model.add(Conv2D(2, (1, 1), input_shape=INPUT_SHAPE))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.1))
    #
    # model.add(Conv2D(50, (5, 5)))
    # model.add(BatchNormalization())
    # model.add(Activation('relu'))
    # model.add(MaxPooling2D(pool_size=(3,3)))
    # model.add(Dropout(0.1))
    #
    model.add(Flatten())

    model.add(Dense(128))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.1))

    model.add(Dense(128))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.1))

    model.add(Dense(128))
    model.add(BatchNormalization())
    model.add(Activation('relu'))

    return model


def compute_accuracy(predictions, labels):
    '''Compute classification accuracy with a fixed threshold on distances.
    '''
    return labels[predictions.ravel() < 0.5].mean()


tr_pairs = []
tr_y = []
start_time = timer()
for i in range(73):
    x = x_train[i]

    # Self
    tr_pairs += [[x, x]]
    tr_y += [1.0]

    # Transformations

    indices = np.ones(73, dtype=np.bool)
    indices[i] = False
    # Plates above and below this one
    if (i + 1) < 73:
        x2 = x_train[i+1] # 0.75
        tr_pairs += [[x, x2]]
        tr_y += [0.75]
        indices[i+1] = False

    if (i - 1) >= 0:
        x3 = x_train[i-1] # 0.75
        tr_pairs += [[x, x3]]
        tr_y += [0.75]
        indices[i-1] = False

    # Plate two above or two below
    if (i + 2) < 73:
        x2 = x_train[i + 2]  # 0.75
        tr_pairs += [[x, x2]]
        tr_y += [0.50]
        indices[i+2] = False

    if (i - 2) >= 0:
        x3 = x_train[i - 2]  # 0.5
        tr_pairs += [[x, x3]]
        tr_y += [0.50]
        indices[i-2] = False

    # Plates three above or three below
    if (i + 3) < 73:
        x2 = x_train[i + 1]  # 0.25
        tr_pairs += [[x, x2]]
        tr_y += [0.25]
        indices[i+3] = False

    if (i - 3) >= 0:
        x3 = x_train[i - 1]  # 0.75
        tr_pairs += [[x, x3]]
        tr_y += [0.25]
        indices[i-3] = False

    # Add the rest as negative examples
    rest = x_train[indices]
    for x2 in rest:
        tr_pairs += [[x, x2]]
        tr_y += [0]

tr_pairs = np.array(tr_pairs)
tr_y = np.array(tr_y)

te_pairs = []
te_y = []

# Dr. Khan's table
te_pairs += [[x_test[8-1], x_train[6-1]]]
te_pairs += [[x_test[11-1], x_train[11-1]]]
te_pairs += [[x_test[26], x_train[23-1]]]

# PW #68 and SW #33
te_pairs += [[x_test[39], x_train[33-1]]]
#te_pairs += [[x_test[39], x_train[32-1]]]
#te_pairs += [[x_test[39], x_train[31-1]]]

te_y += [1]
te_y += [1]
te_y += [1]
te_y += [1]
#te_y += [0.75]
#te_y += [0.75]
te_y = np.array(te_y)
te_pairs = np.array(te_pairs)

ti_pairs = []
ti_pairs += [[x_test[39], x_train[31]]]
ti_pairs += [[x_test[39], x_train[30]]]
ti_pairs += [[x_test[39], x_train[33]]]
ti_pairs += [[x_test[39], x_train[34]]]
ti_pairs = np.array(ti_pairs)

duration = timer() - start_time
print("Creating pairs took %.2fs" % duration)
print("Pair shape", tr_pairs.shape)

# network definition
base_network = create_base_network()

left_input = Input(INPUT_SHAPE)
right_input = Input(INPUT_SHAPE)
# because we re-use the same instance `base_network`,
# the weights of the network
# will be shared across the two branches
encoded_left = base_network(left_input)
encoded_right = base_network(right_input)

distance = Lambda(euclidean_distance, output_shape=eucl_dist_output_shape)([encoded_left, encoded_right])

model = Model(inputs=[left_input, right_input], outputs=distance)

# train
rms = RMSprop()
model.compile(loss=contrastive_loss, optimizer=rms, metrics=['accuracy'])

model_check = ModelCheckpoint('train_net.e{epoch:02d}-vac{val_acc:.2f}.hdf5', monitor='val_acc')

from keras.models import load_model
model2 = load_model('siamese.hdf5', custom_objects={'contrastive_loss':contrastive_loss})

from keras.utils import plot_model
# plot_model(model2, to_file='model.png')

import pdb
pdb.set_trace()

start_time = timer()
history = model.fit([tr_pairs[:, 0], tr_pairs[:, 1]], tr_y,
          validation_data=([te_pairs[:, 0], te_pairs[:, 1]], te_y),
          batch_size=BATCH_SIZE,
          epochs=EPOCHS,
          callbacks=[model_check],
          verbose=1)

duration = timer() - start_time
print("Training took %.2fs" % duration)
model.save('siamese.hdf5')

# compute final accuracy on training and test sets
# Pred is closer to 0 if the numbers are similar
pred = model.predict([tr_pairs[:, 0], tr_pairs[:, 1]])
tr_acc = compute_accuracy(pred, tr_y)

#pred2 = model.predict([te_pairs[:, 0], te_pairs[:, 1]])
#te_acc = compute_accuracy(pred2, te_y)

print('* Accuracy on training set: %0.2f%%' % (100 * tr_acc))
#print('* Accuracy on testing set: %0.2f%%' % (100 * te_acc))

plt.figure(1)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

plt.figure(2)
plt.plot(history.history['acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train'], loc='upper left')
plt.show()

plt.figure(3)
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['test'], loc='upper left')
plt.show()