import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import tensorflow.contrib.slim as slim


BATCH_SIZE = 73
TRAIN_ITER = 5000
STEP = 500

# Which plates to consider similar
N = 4
PN = 1/N

c = ['#ff0000', '#ffff00', '#00ff00', '#00ffff', '#0000ff', '#ff00ff', '#990000', '#999900', '#009900', '#009999']

# Load training dataset
train_data = np.load('atlas_sw.npz')
x_train = train_data['images']
y_train = train_data['labels']

# Create tf dataset from training
train_dataset = tf.contrib.data.Dataset.from_tensor_slices((x_train, y_train))
train_dataset = train_dataset.shuffle(buffer_size=10000)
train_dataset = train_dataset.repeat()
train_dataset = train_dataset.batch(BATCH_SIZE)
train_iterator = train_dataset.make_one_shot_iterator()

left = tf.placeholder(tf.float32, [None, 300, 200, 1], name='left')
right = tf.placeholder(tf.float32, [None, 300, 200, 1], name='right')
with tf.name_scope("similarity"):
    label = tf.placeholder(tf.float32, [None, 1], name='label')  # 1 if same, 0 if different

margin = 0.2
left_output = SiameseNetwork(left, reuse=False)
right_output = SiameseNetwork(right, reuse=True)
loss = contrastive_loss(left_output, right_output, label, margin)
global_step = tf.Variable(0, trainable=False)

# starter_learning_rate = 0.0001
# learning_rate = tf.train.exponential_decay(starter_learning_rate, global_step, 1000, 0.96, staircase=True)
# tf.scalar_summary('lr', learning_rate)
# train_step = tf.train.RMSPropOptimizer(learning_rate).minimize(loss, global_step=global_step)

train_step = tf.train.MomentumOptimizer(0.01, 0.99, use_nesterov=True).minimize(loss, global_step=global_step)

saver = tf.train.Saver()
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    # setup tensorboard
    tf.summary.scalar('step', global_step)
    tf.summary.scalar('loss', loss)
    for var in tf.trainable_variables():
        tf.summary.histogram(var.op.name, var)
    merged = tf.summary.merge_all()
    writer = tf.summary.FileWriter('train.log', sess.graph)


    # train iter
    for i in range(TRAIN_ITER):
        print("Iteration", i, '/', TRAIN_ITER)
        batch_left_im, batch_left_label = sess.run(train_iterator.get_next())
        batch_right_im, batch_right_label = sess.run(train_iterator.get_next())

        #import pdb
        #pdb.set_trace()

        batch_sim = 1 - ((100 / 4) * np.abs(batch_left_label - batch_right_label))
        batch_sim[batch_sim < 0] = 0
        batch_sim = batch_sim.reshape(BATCH_SIZE, 1)

        #import pdb
        #pdb.set_trace()

        _, loss, summary_str = sess.run([train_step, loss, merged],
                                     feed_dict={
                                         left: batch_left_im.reshape(BATCH_SIZE, 300, 200, 1),
                                         right: batch_right_im.reshape(BATCH_SIZE, 300, 200, 1),
                                         label: batch_sim})

        writer.add_summary(summary_str, i)
        print("\r#%d - Loss" % i, loss)

        train_dataset = train_dataset.shuffle(buffer_size=10000)

        # if (i + 1) % STEP == 0:
        # 	#generate test
        # 	feat = sess.run(left_output, feed_dict={left:test_im})
        #
        # 	labels = mnist.test.labels
        # 	# plot result
        # 	f = plt.figure(figsize=(16,9))
        # 	for j in range(10):
        # 	    plt.plot(feat[labels==j, 0].flatten(), feat[labels==j, 1].flatten(),
        # 	    	'.', c=c[j],alpha=0.8)
        # 	plt.legend(['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'])
        # 	plt.savefig('img/%d.jpg' % (i + 1))

    saver.save(sess, "model/model.ckpt")
