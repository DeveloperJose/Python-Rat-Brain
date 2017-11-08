import tensorflow as tf

class SiameseModel:
    # Create model
    def __init__(self, training):
        self.x1 = tf.placeholder(tf.float32, [None, 300*200])
        self.x2 = tf.placeholder(tf.float32, [None, 300*200])

        with tf.variable_scope("siamese") as scope:
            self.net_left = self.network(self.x1, training)
            scope.reuse_variables()
            self.net_right = self.network(self.x2, training)

        # Create loss
        self.y_ = tf.placeholder(tf.float32, [None])
        self.loss = self.loss_with_spring()

    def network(self, x, training):
        with tf.variable_scope("ConvNet", reuse=False):
            x = tf.reshape(x, [-1, 300, 200, 1])
            conv1 = tf.layers.conv2d(x, 4, 3, padding='same', activation=tf.nn.relu)
            conv1_norm = tf.layers.batch_normalization(conv1)
            conv1_dropout = tf.layers.dropout(inputs=conv1_norm, rate=0.2, training=training)

            conv2 = tf.layers.conv2d(conv1_dropout, 8, 3, padding='same', activation=tf.nn.relu)
            conv2_norm = tf.layers.batch_normalization(conv2)
            conv2_dropout = tf.layers.dropout(inputs=conv2_norm, rate=0.2, training=training)
            #pool1 = tf.layers.max_pooling2d(conv1, pool_size=2, strides=2)

            #conv3 = tf.layers.conv2d(conv2_dropout, 8, 3, padding='same', activation=tf.nn.relu)
            #conv3_norm = tf.layers.batch_normalization(conv3)
            #conv3_dropout = tf.layers.dropout(inputs=conv3_norm, rate=0.2, training=training)

            flat = tf.contrib.layers.flatten(conv2_dropout)

            dense = tf.layers.dense(inputs=flat, units=256, activation=tf.nn.relu)
            dropout = tf.layers.dropout(inputs=dense, rate=0.2, training=training)

            output = tf.layers.dense(inputs=dropout, units=2)
            return output

    def loss_with_spring(self):
        margin = 5.0
        labels_t = self.y_
        labels_f = tf.subtract(1.0, self.y_, name="1-yi")          # labels_ = !labels;
        eucd2 = tf.pow(tf.subtract(self.net_left, self.net_right), 2)
        eucd2 = tf.reduce_sum(eucd2, 1)
        eucd = tf.sqrt(eucd2+1e-6, name="eucd")
        C = tf.constant(margin, name="C")
        # yi*||CNN(p1i)-CNN(p2i)||^2 + (1-yi)*max(0, C-||CNN(p1i)-CNN(p2i)||^2)
        pos = tf.multiply(labels_t, eucd2, name="yi_x_eucd2")
        # neg = tf.multiply(labels_f, tf.subtract(0.0,eucd2), name="yi_x_eucd2")
        # neg = tf.multiply(labels_f, tf.maximum(0.0, tf.subtract(C,eucd2)), name="Nyi_x_C-eucd_xx_2")
        neg = tf.multiply(labels_f, tf.pow(tf.maximum(tf.subtract(C, eucd), 0), 2), name="Nyi_x_C-eucd_xx_2")
        losses = tf.add(pos, neg, name="losses")
        loss = tf.reduce_mean(losses, name="loss")
        return loss

    def loss_with_step(self):
        margin = 5.0
        labels_t = self.y_
        labels_f = tf.subtract(1.0, self.y_, name="1-yi")          # labels_ = !labels;
        eucd2 = tf.pow(tf.subtract(self.net_left, self.net_right), 2)
        eucd2 = tf.reduce_sum(eucd2, 1)
        eucd = tf.sqrt(eucd2+1e-6, name="eucd")
        C = tf.constant(margin, name="C")
        pos = tf.multiply(labels_t, eucd, name="y_x_eucd")
        neg = tf.multiply(labels_f, tf.maximum(0.0, tf.subtract(C, eucd)), name="Ny_C-eucd")
        losses = tf.add(pos, neg, name="losses")
        loss = tf.reduce_mean(losses, name="loss")
        return loss