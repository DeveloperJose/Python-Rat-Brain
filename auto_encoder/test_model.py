import numpy as np
from keras.models import Model
from keras.datasets import mnist
from keras.models import load_model
from sklearn.metrics import label_ranking_average_precision_score
import time
import cv2

t0 = time.time()

#(x_train, y_train), (x_test, y_test) = mnist.load_data()
sw_data = np.load('atlas_sw.npz')
x_train = sw_data['images'].astype('float32') / 255.
x_shape = (x_train.shape[0], x_train.shape[1], x_train.shape[2], 1)
x_train = np.reshape(x_train, x_shape)
y_train = sw_data['labels']
print("X_Train: ", x_train.shape)

print("===--- Paxinos/Watson Atlas")
pw_data = np.load('atlas_pw.npz')
pw_y = pw_data['labels']
pw_im = pw_data['images'].astype('float32') / 255.
pw_shape = pw_im.shape[0], pw_im.shape[1], pw_im.shape[2], 1
pw_im = np.reshape(pw_im, pw_shape)

x_test = np.array([pw_im[7], pw_im[10], pw_im[26], pw_im[39]])
y_test = np.array([pw_y[7], pw_y[10], pw_y[26], pw_y[39]])

x_test = pw_im
y_test = pw_y

print("X_Test: ", x_test.shape)

noise_factor = 0.4
x_train_noisy = x_train + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=x_train.shape)
x_test_noisy = x_test + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=x_test.shape)

x_train_noisy = np.clip(x_train_noisy, 0., 1.)
x_test_noisy = np.clip(x_test_noisy, 0., 1.)
t1 = time.time()
print('Dataset loaded in: ', t1-t0)

print('Loading model :')
t0 = time.time()
autoencoder = load_model('autoencoder.h5')
encoder = Model(inputs=autoencoder.input, outputs=autoencoder.get_layer('encoder').output)
t1 = time.time()
print('Model loaded in: ', t1-t0)

scores = []


def retrieve_closest_elements(test_code, test_label, learned_codes):
    distances = []
    for code in learned_codes:
        distance = np.linalg.norm(code - test_code)
        distances.append(distance)
    nb_elements = learned_codes.shape[0]
    distances = np.array(distances)
    learned_code_index = np.arange(nb_elements)
    labels = np.copy(y_train).astype('float32')
    labels[labels != test_label] = -1
    labels[labels == test_label] = 1
    labels[labels == -1] = 0
    distance_with_labels = np.stack((distances, labels, learned_code_index), axis=-1)
    sorted_distance_with_labels = distance_with_labels[distance_with_labels[:, 0].argsort()]

    sorted_distances = 28 - sorted_distance_with_labels[:, 0]
    sorted_labels = sorted_distance_with_labels[:, 1]
    sorted_indexes = sorted_distance_with_labels[:, 2]
    return sorted_distances, sorted_labels, sorted_indexes

def compute_average_precision_score(test_codes, test_labels, learned_codes, n_samples):
    out_labels = []
    out_distances = []
    retrieved_elements_indexes = []
    for i in range(len(test_codes)):
        sorted_distances, sorted_labels, sorted_indexes = retrieve_closest_elements(test_codes[i], test_labels[i], learned_codes)
        out_distances.append(sorted_distances[:n_samples])
        out_labels.append(sorted_labels[:n_samples])
        retrieved_elements_indexes.append(sorted_indexes[:n_samples])

    out_labels = np.array(out_labels)
    out_labels_file_name = 'computed_data/out_labels_{}'.format(n_samples)
    np.save(out_labels_file_name, out_labels)

    out_distances_file_name = 'computed_data/out_distances_{}'.format(n_samples)
    out_distances = np.array(out_distances)
    np.save(out_distances_file_name, out_distances)
    score = label_ranking_average_precision_score(out_labels, out_distances)
    scores.append(score)
    return score

INDEX = 0
def retrieve_closest_images(test_element, test_label, n_samples=10):
    global INDEX
    learned_codes = encoder.predict(x_train)
    learned_codes = learned_codes.reshape(learned_codes.shape[0],
                                          learned_codes.shape[1] * learned_codes.shape[2] * learned_codes.shape[3])

    test_code = encoder.predict(np.array([test_element]))
    test_code = test_code.reshape(test_code.shape[1] * test_code.shape[2] * test_code.shape[3])

    distances = []

    for code in learned_codes:
        distance = np.linalg.norm(code - test_code)
        distances.append(distance)
    nb_elements = learned_codes.shape[0]
    distances = np.array(distances)
    learned_code_index = np.arange(nb_elements)
    labels = np.copy(y_train).astype('float32')
    labels[labels != test_label] = -1
    labels[labels == test_label] = 1
    labels[labels == -1] = 0
    distance_with_labels = np.stack((distances, labels, learned_code_index), axis=-1)
    sorted_distance_with_labels = distance_with_labels[distance_with_labels[:, 0].argsort()]

    sorted_distances = 28 - sorted_distance_with_labels[:, 0]
    sorted_labels = sorted_distance_with_labels[:, 1]
    sorted_indexes = sorted_distance_with_labels[:, 2]
    kept_indexes = sorted_indexes[:n_samples]

    score = label_ranking_average_precision_score(np.array([sorted_labels[:n_samples]]), np.array([sorted_distances[:n_samples]]))
    
    kept_indexes = kept_indexes.astype(np.uint16)
    result_y = y_train[kept_indexes]
    result_distances = sorted_distances[kept_indexes]
    
    print("Plate {} - ".format(test_label), end='')
    for i in range(n_samples):
        match_y = result_y[i]
        match_d = result_distances[i]
        print("[{},{:.4f}] ".format(match_y, match_d), end='')
    
    print("")
    
    #print("Average precision ranking score for tested element is {}".format(score))

    original_image = test_element
    #cv2.imshow('original_image_' + str(INDEX), original_image)
    retrieved_images = x_train[int(kept_indexes[0]), :]
    for i in range(1, n_samples):
        retrieved_images = np.hstack((retrieved_images, x_train[int(kept_indexes[i]), :]))
        
    #cv2.imshow('Results_' + str(INDEX), retrieved_images)

    cv2.imwrite('test_results/plate_' + str(test_label) + '.jpg', 255 * cv2.resize(original_image, (0,0), fx=3, fy=3))
    cv2.imwrite('test_results/results' + str(test_label) + '.jpg', 255 * cv2.resize(retrieved_images, (0,0), fx=2, fy=2))

    #import pdb
    #pdb.set_trace()
    
    INDEX += 1
    return result_y

def test_model(n_test_samples, n_train_samples):
    learned_codes = encoder.predict(x_train)
    learned_codes = learned_codes.reshape(learned_codes.shape[0], learned_codes.shape[1] * learned_codes.shape[2] * learned_codes.shape[3])
    test_codes = encoder.predict(x_test)
    test_codes = test_codes.reshape(test_codes.shape[0], test_codes.shape[1] * test_codes.shape[2] * test_codes.shape[3])
    indexes = np.arange(len(y_test))
    np.random.shuffle(indexes)
    indexes = indexes[:n_test_samples]

    print('Start computing score for {} train samples'.format(n_train_samples))
    t1 = time.time()
    score = compute_average_precision_score(test_codes[indexes], y_test[indexes], learned_codes, n_train_samples)
    t2 = time.time()
    print('Score computed in: ', t2-t1)
    print('Model score:', score)


def plot_denoised_images():
    denoised_images = autoencoder.predict(x_test_noisy.reshape(x_test_noisy.shape[0], x_test_noisy.shape[1], x_test_noisy.shape[2], 1))
    test_img = x_test_noisy[0]
    resized_test_img = cv2.resize(test_img, (280, 280))
    cv2.imshow('input', resized_test_img)
    output = denoised_images[0]
    resized_output = cv2.resize(output, (280, 280))
    cv2.imshow('output', resized_output)
    cv2.imwrite('test_results/noisy_image.jpg', 255 * resized_test_img)
    cv2.imwrite('test_results/denoised_image.jpg', 255 * resized_output)


# To test the whole model
n_test_samples = 1000
n_train_samples = [10, 50, 100, 200, 300, 400, 500, 750, 1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000,
                   20000, 30000, 40000, 50000, 60000]


#for n_train_sample in n_train_samples:
#    test_model(n_test_samples, n_train_sample)

np.save('computed_data/scores', np.array(scores))

import pylab as plt
plt.xkcd()
plt.figure()
plt.title('SW Matching')
plt.xlabel('PW Plate')
plt.ylabel('SW Plate')
# To retrieve closest images
x = []
y = []
for i in range(len(x_test)):
#for i in range(3):
    x.append(y_test[i]) # Plate #
    predictions = retrieve_closest_images(x_test[i], y_test[i])
    y.append(predictions[0]) # Top Prediction
    
plt.plot(x, y)
plt.savefig('results.png')
plt.show(block=True)


# To plot a denoised image
#plot_denoised_images()