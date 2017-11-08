
import tensorflow as tf
import numpy as np
import os

#import helpers
import network
import visualize

ITERATIONS = 1000
STEP_INFO = 10
STEP_SAVE = 100
BATCH_SIZE = 73
N = 4

# prepare data and tf.session
# Load training dataset
train_data = np.load('atlas_sw.npz')
x_train = train_data['images'].reshape(73, 300*200)
y_train = train_data['labels'].astype(dtype=np.float16)

# Create tf dataset from training
train_dataset = tf.contrib.data.Dataset.from_tensor_slices((x_train, y_train))
train_dataset = train_dataset.shuffle(buffer_size=10000)
train_dataset = train_dataset.repeat()
train_dataset = train_dataset.batch(15)
train_iterator = train_dataset.make_one_shot_iterator()
sess = tf.InteractiveSession()

# setup siamese network
siamese = network.SiameseModel(training=True)
train_step = tf.train.GradientDescentOptimizer(0.01).minimize(siamese.loss)
saver = tf.train.Saver()
tf.initialize_all_variables().run()

# if you just want to load a previously trainmodel?
new = True
model_ckpt = '/tmp/model.ckpt'
if os.path.isfile(model_ckpt):
    input_var = None
    while input_var not in ['yes', 'no']:
        input_var = input("We found model.ckpt file. Do you want to load it [yes/no]?")
    if input_var == 'yes':
        new = False

# start training
if new:
    for step in [1, 2, 3, -1, -2, -3]:
        # Plates above and below
        batch_x2 = np.roll(x_train, step)
        batch_y2 = np.roll(y_train, step)
        batch_y = 1 - ((1 / 4) * np.abs(y_train - batch_y2))
        batch_y[batch_y < 0] = 0

        _, loss_v = sess.run([train_step, siamese.loss], feed_dict={
            siamese.x1: x_train,
            siamese.x2: batch_x2,
            siamese.y_: batch_y})

        print(batch_y, 'loss', loss_v)

        if loss_v == 0:
            print("Loss is 0...?")

        if np.isnan(loss_v):
            print('Model diverged with loss = NaN')

    saver.save(sess, '/tmp/model.ckpt')

    import pdb
    pdb.set_trace()

    # Random batches
    for step in range(ITERATIONS):
        batch_x1, batch_y1 = sess.run(train_iterator.get_next())
        batch_x2, batch_y2 = sess.run(train_iterator.get_next())
        #batch_y = (batch_y1 == batch_y2).astype('float')

        batch_y = 1 - ((1 / 4) * np.abs(batch_y1 - batch_y2))
        batch_y[batch_y < 0] = 0

        print('Matches', np.count_nonzero(batch_y))


        _, loss_v = sess.run([train_step, siamese.loss], feed_dict={
                            siamese.x1: batch_x1,
                            siamese.x2: batch_x2,
                            siamese.y_: batch_y})

        if loss_v == 0:
            print("Loss is 0...?")
            import pdb
            pdb.set_trace()
        #    saver.save(sess, '/tmp/model.ckpt')
        #    quit()

        if np.isnan(loss_v):
            print('Model diverged with loss = NaN')
            import pdb
            pdb.set_trace()

        if step % STEP_INFO == 0:
            print ('step %d: loss %.3f' % (step, loss_v))

        if step % STEP_SAVE == 0 and step > 0:
            saver.save(sess, '/tmp/model.ckpt')
            #embed = siamese.o1.eval({siamese.x1: mnist.test.images})
            #embed.tofile('embed.txt')
else:
    saver.restore(sess, '/tmp/model.ckpt')

# visualize result
# x_test = mnist.test.images.reshape([-1, 28, 28])
# visualize.visualize(embed, x_test)
