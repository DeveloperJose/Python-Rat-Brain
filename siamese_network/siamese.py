# '''Train a Siamese MLP on pairs of digits from the MNIST dataset.
#
# It follows Hadsell-et-al.'06 [1] by computing the Euclidean distance on the
# output of the shared network and by optimizing the contrastive loss (see paper
# for mode details).
#
# [1] "Dimensionality Reduction by Learning an Invariant Mapping"
#     http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
#
# Gets to 99.5% test accuracy after 20 epochs.
# 3 seconds per epoch on a Titan X GPU
# '''
# import numpy as np
# np.random.seed(1337)  # for reproducibility
#
# import pylab as plt
# from keras.models import Sequential, Model
# from keras.layers import Dense, Dropout, Input, Lambda, Conv2D, MaxPooling2D, Activation, Flatten
# from keras.layers.normalization import BatchNormalization
# from keras.optimizers import RMSprop
# from keras import backend as K
# from keras.callbacks import EarlyStopping, ModelCheckpoint
#
# from timeit import default_timer as timer
#
# EPOCHS = 10
# BATCH_SIZE = 32
#
# #INPUT_SHAPE = (x_train.shape[1], x_train.shape[2], 1)
# #INPUT_SHAPE = (x_train.shape[1],)
# INPUT_SHAPE = (x_train.shape[1]*x_train.shape[2],)
#
# def euclidean_distance(vects):
#     x, y = vects
#     return K.sqrt(K.sum(K.square(x - y), axis=1, keepdims=True))
#
# def eucl_dist_output_shape(shapes):
#     shape1, shape2 = shapes
#     return (shape1[0], 1)
#
# def contrastive_loss(y_true, y_pred):
#     '''Contrastive loss from Hadsell-et-al.'06
#     http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
#     '''
#     margin = 1
#     return K.mean(y_true * K.square(y_pred) + (1 - y_true) * K.square(K.maximum(margin - y_pred, 0)))
#
# def create_base_network():
#     '''Base network to be shared (eq. to feature extraction).
#     '''
#     model = Sequential()
#     #model.add(Conv2D(32, (7, 7), input_shape=INPUT_SHAPE))
#     #model.add(BatchNormalization())
#     #model.add(Activation('relu'))
#     #model.add(MaxPooling2D(pool_size=(2, 2)))
#     #model.add(Dropout(0.1))
#
#     # model.add(Conv2D(64, (5, 5)))
#     # model.add(BatchNormalization())
#     # model.add(Activation('relu'))
#     # model.add(MaxPooling2D(pool_size=(2, 2)))
#     #
#     # model.add(Conv2D(128, (3, 3)))
#     # model.add(BatchNormalization())
#     # model.add(Activation('relu'))
#     # model.add(MaxPooling2D(pool_size=(2, 2)))
#
#     #model.add(Conv2D(256, (1, 1)))
#     #model.add(BatchNormalization())
#     #model.add(Activation('relu'))
#     #model.add(MaxPooling2D(pool_size=(2, 2)))
#
#     #model.add(Conv2D(2, (1, 1)))
#     #model.add(MaxPooling2D(pool_size=(2, 2)))
#     #model.add(Flatten())
#
#     model.add(Dense(1024, input_shape=INPUT_SHAPE))
#     model.add(BatchNormalization())
#     model.add(Activation('relu'))
#     #model.add(Dropout(0.1))
#
#     model.add(Dense(2048))
#     model.add(BatchNormalization())
#     model.add(Activation('relu'))
#     #model.add(Dropout(0.1))
#
#     model.add(Dense(2))
#     #model.add(BatchNormalization())
#     #model.add(Activation('relu'))
#
#     return model
#
#
# def compute_accuracy(predictions, labels):
#     '''Compute classification accuracy with a fixed threshold on distances.
#     '''
#     return labels[predictions.ravel() < 0.5].mean()
#
#
# tr_pairs = []
# tr_y = []
# start_time = timer()
# print("X Shape: ", x_train[0].shape)
# for i in range(73):
#     x = x_train[i]
#     x_reg = x.reshape((200, 100))
#
#     #tr_pairs += [[x, x]]
#     #tr_y += [1.0]
#
#     # Transformations
#     #import pdb
#     #pdb.set_trace()
#     #plt.savefig()
#     tr_pairs += [[x, np.roll(x_reg, (1, 0))]] # Right
#     tr_y += [1.0]
#     tr_pairs += [[x, np.roll(x_reg, (1, 1))]] # Right-Up
#     tr_y += [1.0]
#     tr_pairs += [[x, np.roll(x_reg, (-1, 0))]] # Left
#     tr_y += [1.0]
#     tr_pairs += [[x, np.roll(x_reg, (-1, 1))]] # Left-Up
#     tr_y += [1.0]
#     tr_pairs += [[x, np.roll(x_reg, (1, -1))]] # Right-Down
#     tr_y += [1.0]
#     tr_pairs += [[x, np.roll(x_reg, (-1, -1))]] # Left-Down
#     tr_y += [1.0]
#     tr_pairs += [[x, np.roll(x_reg, (0, 1))]] # Up
#     tr_y += [1.0]
#     tr_pairs += [[x, np.roll(x_reg, (0, -1))]] # Down
#     tr_y += [1.0]
#     tr_pairs += [[x, np.fliplr(x_reg)]] # Left/Right Flip
#     tr_y += [1.0]
#     tr_pairs += [[x, np.flipud(x_reg)]] # Up/Down Flip
#     tr_y += [1.0]
#     #tr_pairs += [[x, np.rot90(x, 2)]]  # 90 Rotation
#     #tr_y += [1.0]
#     #tr_pairs += [[x, np.rot90(x, 2)]]  # 180 Rotation
#     #tr_y += [1.0]
#     #tr_pairs += [[x, np.rot90(x, 2)]]  # 270 Rotation
#     #tr_y += [1.0]
#
#     indices = np.ones(73, dtype=np.bool)
#     indices[i] = False
#     # Plates above and below this one
#     if (i + 1) < 73:
#         x2 = x_train[i+1] # 0.75
#         tr_pairs += [[x, x2]]
#         tr_y += [0.75]
#         indices[i+1] = False
#
#     if (i - 1) >= 0:
#         x3 = x_train[i-1] # 0.75
#         tr_pairs += [[x, x3]]
#         tr_y += [0.75]
#         indices[i-1] = False
#
#     # Plate two above or two below
#     if (i + 2) < 73:
#         x2 = x_train[i + 2]
#         tr_pairs += [[x, x2]]
#         tr_y += [0.50]
#         indices[i+2] = False
#
#     if (i - 2) >= 0:
#         x3 = x_train[i - 2]
#         tr_pairs += [[x, x3]]
#         tr_y += [0.50]
#         indices[i-2] = False
#
#     # Plates three above or three below
#     if (i + 3) < 73:
#         x2 = x_train[i + 1]  # 0.25
#         tr_pairs += [[x, x2]]
#         tr_y += [0.25]
#         indices[i+3] = False
#
#     if (i - 3) >= 0:
#         x3 = x_train[i - 1]
#         tr_pairs += [[x, x3]]
#         tr_y += [0.25]
#         indices[i-3] = False
#
#     # Add the rest as negative examples
#     rest = x_train[indices]
#     for x2 in rest:
#         tr_pairs += [[x, x2]]
#         tr_y += [0]
#
# tr_pairs = np.array(tr_pairs)
# tr_y = np.array(tr_y)
#
# te_pairs = []
# te_y = []
#
# # Dr. Khan's table
# te_pairs += [[x_test[8-1], x_train[6-1]]]
# te_pairs += [[x_test[11-1], x_train[11-1]]]
# te_pairs += [[x_test[26], x_train[23-1]]]
# te_y += [1]
# te_y += [1]
# te_y += [1]
#
# # PW #68 and SW #33
# te_pairs += [[x_test[39], x_train[33-1]]]
# te_y += [1]
#
# # PW #68 and SW #32/34
# te_pairs += [[x_test[39], x_train[32-1]]]
# te_pairs += [[x_test[39], x_train[34-1]]]
# te_y += [0.75]
# te_y += [0.75]
#
# te_pairs += [[x_test[39], x_train[31-1]]]
# te_pairs += [[x_test[39], x_train[33-1]]]
# te_y += [0.5]
# te_y += [0.5]
#
#
# te_y = np.array(te_y)
# te_pairs = np.array(te_pairs)
#
# ti_pairs = []
# ti_pairs += [[x_test[39], x_train[31]]]
# ti_pairs += [[x_test[39], x_train[30]]]
# ti_pairs += [[x_test[39], x_train[33]]]
# ti_pairs += [[x_test[39], x_train[34]]]
# ti_pairs = np.array(ti_pairs)
#
# duration = timer() - start_time
# print("Creating pairs took %.2fs" % duration)
# print("Pair shape", tr_pairs.shape)
#
# # For CNNs
#
# # network definition
# base_network = create_base_network()
#
# left_input = Input(INPUT_SHAPE)
# right_input = Input(INPUT_SHAPE)
# # because we re-use the same instance `base_network`,
# # the weights of the network
# # will be shared across the two branches
# encoded_left = base_network(left_input)
# encoded_right = base_network(right_input)
#
# distance = Lambda(euclidean_distance, output_shape=eucl_dist_output_shape)([encoded_left, encoded_right])
# model = Model(inputs=[left_input, right_input], outputs=distance)
#
# # train
# rms = RMSprop()
# model.compile(loss=contrastive_loss, optimizer=rms, metrics=['accuracy'])
#
# model_check = ModelCheckpoint('train_net.epoch{epoch:02d}-valacc{val_acc:.2f}.hdf5', monitor='val_acc')
#
# from keras.models import load_model
# #model2 = load_model('siamese.hdf5', custom_objects={'contrastive_loss':contrastive_loss})
#
# from keras.utils import plot_model
# # plot_model(model2, to_file='model.png')
#
# #import pdb
# #pdb.set_trace()
#
# start_time = timer()
# history = model.fit([tr_pairs[:, 0], tr_pairs[:, 1]], tr_y,
#           validation_data=([te_pairs[:, 0], te_pairs[:, 1]], te_y),
#           batch_size=BATCH_SIZE,
#           epochs=EPOCHS,
#           callbacks=[model_check],
#           verbose=1)
#
# duration = timer() - start_time
# print("Training took %.2fs" % duration)
# model.save('siamese.hdf5')
#
# # compute final accuracy on training and test sets
# # Pred is closer to 0 if the numbers are similar
# pred = model.predict([tr_pairs[:, 0], tr_pairs[:, 1]])
# tr_acc = compute_accuracy(pred, tr_y)
#
# #pred2 = model.predict([te_pairs[:, 0], te_pairs[:, 1]])
# #te_acc = compute_accuracy(pred2, te_y)
#
# print('* Accuracy on training set: %0.2f%%' % (100 * tr_acc))
# #print('* Accuracy on testing set: %0.2f%%' % (100 * te_acc))
#
# plt.figure(1)
# plt.plot(history.history['loss'])
# plt.plot(history.history['val_loss'])
# plt.title('model loss')
# plt.ylabel('loss')
# plt.xlabel('epoch')
# plt.legend(['train', 'test'], loc='upper left')
# plt.show()
#
# plt.figure(2)
# plt.plot(history.history['acc'])
# plt.title('model accuracy')
# plt.ylabel('accuracy')
# plt.xlabel('epoch')
# plt.legend(['train'], loc='upper left')
# plt.show()
#
# plt.figure(3)
# plt.plot(history.history['val_acc'])
# plt.title('model accuracy')
# plt.ylabel('accuracy')
# plt.xlabel('epoch')
# plt.legend(['test'], loc='upper left')
# plt.show()