import numpy as np
import tensorflow as tf
import pickle

data = pickle.load(open('../data/multi_train_data', 'rb'))
label_all = pickle.load(open('../data/multi_train_label', 'rb'))
test_data = pickle.load(open('../data/multi_test_data', 'rb'))
test_label_all = pickle.load(open('../data/multi_test_label', 'rb'))
sizes = pickle.load(open('../data/multi_label_sizes', 'rb'))
rebuild_label = pickle.load(open('../data/rebuild_label.npz', 'rb'))

data_all = {
    'train_inputs': data,
    'train_targets': label_all,
    'valid_inputs': test_data,
    'valid_targets': test_label_all,
    'label_size': sizes
}

class parameter(object):

    def __init__(self, shape):
        self.shape = shape

    def normal(self):
        return tf.Variable(tf.random_normal(self.shape))

class layer(object):

    def __init__(self, shape):
        raise NotImplementedError

    def __str__(self):
        raise NotImplementedError



class lstm_layer(layer):

    def __init__(self, inputs, n_units, scope):

        self.n_units = n_units
        self.input_shape = inputs.shape
        self.outputs, self.states = tf.nn.dynamic_rnn(
            cell = tf.contrib.rnn.BasicLSTMCell(self.n_units),
            inputs = inputs,
            dtype = tf.float32,
            scope = scope)

    def __str__(self):
        print('|---------------[LSTM layer]---------------|')
        print('shape: ', self.input_shape, ' => ', self.outputs.shape)
        return '|_______________[LSTM layer]_______________|'

class affine_layer(layer):

    def __init__(self, inputs, shape, activation):

        self.shape = shape
        self.input_shape = inputs.shape
        self.activation = activation
        self.weights = parameter(self.shape).normal()
        self.biases = parameter([self.shape[-1]]).normal()
        self.outputs = self.activation(tf.add(tf.matmul(inputs, self.weights), self.biases))

    def __str__(self):
        print('|---------------[Affine layer]---------------|')
        print('shape: ', self.input_shape, ' => ', self.outputs.shape)
        return '|_______________[Affine layer]_______________|'

class reshape_layer(layer):

    def __init__(self, inputs, shape):

        self.input_shape = inputs.shape
        self.outputs = tf.reshape(inputs, shape)

    def __str__(self):
        print('|---------------[Reshape layer]---------------|')
        print('shape: ', self.input_shape, ' => ', self.outputs.shape)
        return '|_______________[Reshape layer]_______________|'

graph = tf.Graph()
learning_rate = 1e-3
iteration = 5000
interval = 5
keys = data.keys()
keys_valid = test_data.keys()
with graph.as_default():

    with tf.name_scope('multi_model1'):

        with tf.name_scope('placeholder'):
            with tf.name_scope('input_placeholder'):
                input_placeholder = tf.placeholder(tf.float32, [None, None, 16])

            with tf.name_scope('target_placeholder_part0'):
                target_placeholder_p0 = tf.placeholder(tf.float32, [None, 10])

            with tf.name_scope('target_placeholder_part1'):
                target_placeholder_p1 = tf.placeholder(tf.float32, [None, 75])

            with tf.name_scope('target_placeholder_part2'):
                target_placeholder_p2 = tf.placeholder(tf.float32, [None, 38])

            with tf.name_scope('target_placeholder_part3'):
                target_placeholder_p3 = tf.placeholder(tf.float32, [None, 10])

            with tf.name_scope('target_placeholder_part4'):
                target_placeholder_p4 = tf.placeholder(tf.float32, [None, 2])

            with tf.name_scope('target_placeholder_rebuild'):
                target_placeholder_rebuild = tf.placeholder(tf.float32, [None, 134])

        with tf.name_scope('model_0'):
            model_0_lstm_0 = lstm_layer(input_placeholder, 16, 'model_0')
            model_0_reshape_0 = reshape_layer(model_0_lstm_0.outputs[:,-1,:], [-1, 16])
            model_0_affine_0 = affine_layer(model_0_reshape_0.outputs, [16, 128], tf.nn.relu)
            model_0_affine_1 = affine_layer(model_0_affine_0.outputs, [128, 128], tf.nn.relu)
            model_0_affine_2 = affine_layer(model_0_affine_1.outputs, [128, 128], tf.nn.relu)
            model_0_output = affine_layer(model_0_affine_2.outputs, [128, 10], tf.identity)
            # tf.summary.image('model_0_classify_weights', reshape_layer(model_0_output.weights, [-1,128,10,1]).outputs)
        with tf.name_scope('model_1'):
            model_1_lstm_0 = lstm_layer(input_placeholder, 16, 'model_1')
            model_1_reshape_0 = reshape_layer(model_1_lstm_0.outputs[:,-1,:], [-1, 16])
            model_1_affine_0 = affine_layer(model_1_reshape_0.outputs, [16, 128], tf.nn.relu)
            model_1_affine_1 = affine_layer(model_1_affine_0.outputs, [128, 128], tf.nn.relu)
            model_1_affine_2 = affine_layer(model_1_affine_1.outputs, [128, 128], tf.nn.relu)
            model_1_output = affine_layer(model_1_affine_2.outputs, [128, 75], tf.identity)
            # tf.summary.image('model_1_classify_weights', reshape_layer(model_1_output.weights, [-1,128,75,1]).outputs)
        with tf.name_scope('model_2'):
            model_2_lstm_0 = lstm_layer(input_placeholder, 16, 'model_2')
            model_2_reshape_0 = reshape_layer(model_2_lstm_0.outputs[:,-1,:], [-1, 16])
            model_2_affine_0 = affine_layer(model_2_reshape_0.outputs, [16, 128], tf.nn.relu)
            model_2_affine_1 = affine_layer(model_2_affine_0.outputs, [128, 128], tf.nn.relu)
            model_2_affine_2 = affine_layer(model_2_affine_1.outputs, [128, 128], tf.nn.relu)
            model_2_output = affine_layer(model_2_affine_2.outputs, [128, 38], tf.identity)
            # tf.summary.image('model_2_classify_weights', reshape_layer(model_2_output.weights, [-1,128,38,1]).outputs)
        with tf.name_scope('model_3'):
            model_3_lstm_0 = lstm_layer(input_placeholder, 16, 'model_3')
            model_3_reshape_0 = reshape_layer(model_3_lstm_0.outputs[:,-1,:], [-1, 16])
            model_3_affine_0 = affine_layer(model_3_reshape_0.outputs, [16, 128], tf.nn.relu)
            model_3_affine_1 = affine_layer(model_3_affine_0.outputs, [128, 128], tf.nn.relu)
            model_3_affine_2 = affine_layer(model_3_affine_1.outputs, [128, 128], tf.nn.relu)
            model_3_output = affine_layer(model_3_affine_2.outputs, [128, 10], tf.identity)
            # tf.summary.image('model_3_classify_weights', reshape_layer(model_3_output.weights, [-1,128,10,1]).outputs)
        with tf.name_scope('model_4'):
            model_4_lstm_0 = lstm_layer(input_placeholder, 16, 'model_4')
            model_4_reshape_0 = reshape_layer(model_4_lstm_0.outputs[:,-1,:], [-1, 16])
            model_4_affine_0 = affine_layer(model_4_reshape_0.outputs, [16, 128], tf.nn.relu)
            model_4_affine_1 = affine_layer(model_4_affine_0.outputs, [128, 128], tf.nn.relu)
            model_4_affine_2 = affine_layer(model_4_affine_1.outputs, [128, 128], tf.nn.relu)
            model_4_output = affine_layer(model_4_affine_2.outputs, [128, 2], tf.identity)
            # tf.summary.image('model_4_classify_weights', reshape_layer(model_4_output.weights, [-1,128,2,1]).outputs)

        with tf.name_scope('combined_models'):
            combine = tf.concat([model_0_affine_0.outputs,
                                 model_1_affine_0.outputs,
                                 model_2_affine_0.outputs,
                                 model_3_affine_0.outputs,
                                 model_4_affine_0.outputs], 1)
#             print(combine)
            # tf.summary.image('combine', reshape_layer(combine, [1, -1, 1280, 1]).outputs)
#             reschedule = tf.sigmoid(combine)
#             print(reschedule)
            affine0 = affine_layer(combine, [640, 512], tf.nn.relu)
            affine1 = affine_layer(affine0.outputs, [512, 256], tf.nn.relu)
            affine2 = affine_layer(affine1.outputs, [256, 128], tf.nn.relu)
            output = affine_layer(affine2.outputs, [128, 134], tf.identity)

        with tf.name_scope('result_0'):
            model_0_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=model_0_output.outputs, labels=target_placeholder_p0))
            model_0_optimizer = tf.train.AdamOptimizer(learning_rate).minimize(model_0_loss)
            model_0_accuracy = tf.reduce_sum(tf.cast(tf.equal(tf.argmax(model_0_output.outputs, 1), tf.argmax(target_placeholder_p0, 1)), tf.float32))
            model_0_mean_acc = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(model_0_output.outputs, 1), tf.argmax(target_placeholder_p0, 1)), tf.float32))
            tf.summary.scalar('model_0_accuracy', model_0_mean_acc)
            tf.summary.scalar('model_0_loss', model_0_loss)
        with tf.name_scope('result_1'):
            model_1_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=model_1_output.outputs, labels=target_placeholder_p1))
            model_1_optimizer = tf.train.AdamOptimizer(learning_rate).minimize(model_1_loss)
            model_1_accuracy = tf.reduce_sum(tf.cast(tf.equal(tf.argmax(model_1_output.outputs, 1), tf.argmax(target_placeholder_p1, 1)), tf.float32))
            model_1_mean_acc = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(model_1_output.outputs, 1), tf.argmax(target_placeholder_p1, 1)), tf.float32))
            tf.summary.scalar('model_1_accuracy', model_1_mean_acc)
            tf.summary.scalar('model_1_loss', model_1_loss)
        with tf.name_scope('result_2'):
            model_2_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=model_2_output.outputs, labels=target_placeholder_p2))
            model_2_optimizer = tf.train.AdamOptimizer(learning_rate).minimize(model_2_loss)
            model_2_accuracy = tf.reduce_sum(tf.cast(tf.equal(tf.argmax(model_2_output.outputs, 1), tf.argmax(target_placeholder_p2, 1)), tf.float32))
            model_2_mean_acc = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(model_2_output.outputs, 1), tf.argmax(target_placeholder_p2, 1)), tf.float32))
            tf.summary.scalar('model_2_accuracy', model_2_mean_acc)
            tf.summary.scalar('model_2_loss', model_2_loss)
        with tf.name_scope('result_3'):
            model_3_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=model_3_output.outputs, labels=target_placeholder_p3))
            model_3_optimizer = tf.train.AdamOptimizer(learning_rate).minimize(model_3_loss)
            model_3_accuracy = tf.reduce_sum(tf.cast(tf.equal(tf.argmax(model_3_output.outputs, 1), tf.argmax(target_placeholder_p3, 1)), tf.float32))
            model_3_mean_acc = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(model_3_output.outputs, 1), tf.argmax(target_placeholder_p3, 1)), tf.float32))
            tf.summary.scalar('model_3_accuracy', model_3_mean_acc)
            tf.summary.scalar('model_3_loss', model_3_loss)
        with tf.name_scope('result_4'):
            model_4_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=model_4_output.outputs, labels=target_placeholder_p4))
            model_4_optimizer = tf.train.AdamOptimizer(learning_rate).minimize(model_4_loss)
            model_4_accuracy = tf.reduce_sum(tf.cast(tf.equal(tf.argmax(model_4_output.outputs, 1), tf.argmax(target_placeholder_p4, 1)), tf.float32))
            model_4_mean_acc = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(model_4_output.outputs, 1), tf.argmax(target_placeholder_p4, 1)), tf.float32))
            tf.summary.scalar('model_4_accuracy', model_4_mean_acc)
            tf.summary.scalar('model_4_loss', model_4_loss)
        with tf.name_scope('result_all'):
            loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=output.outputs, labels=target_placeholder_rebuild))
            optimizer = tf.train.AdamOptimizer(learning_rate).minimize(loss)
            accuracy = tf.reduce_sum(tf.cast(tf.equal(tf.argmax(output.outputs, 1), tf.argmax(target_placeholder_rebuild, 1)), tf.float32))
            mean_acc = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(output.outputs, 1), tf.argmax(target_placeholder_rebuild, 1)), tf.float32))
            tf.summary.scalar('model_accuracy', mean_acc)
            tf.summary.scalar('model_loss', loss)

        sess = tf.Session()
        merged = tf.summary.merge_all()
        writer = tf.summary.FileWriter('../combined_outputs', sess.graph)
        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver(write_version=tf.train.SaverDef.V1)


        with tf.name_scope('train'):

            for i in range(iteration):
                accs = 0.0
                errs = 0.0
                ns = 0
                for key in keys:

                    feed_dict = {
                        input_placeholder: data[key],
                        target_placeholder_p0: label_all[0][key],
                        target_placeholder_p1: label_all[1][key],
                        target_placeholder_p2: label_all[2][key],
                        target_placeholder_p3: label_all[3][key],
                        target_placeholder_p4: label_all[4][key],
                        target_placeholder_rebuild: rebuild_label['train_targets'][key]
                    }
                    ns += len(data[key])
                    merge, _, acc, err = sess.run([merged, optimizer, accuracy, loss], feed_dict=feed_dict)
                    writer.add_summary(merge, i)
                    accs += acc
                    errs += err
                accs /= ns
                errs /= ns
                print('Training Epoch: %d, acc: %f, loss: %f '%(i, accs, errs))
                if i % interval == 0:
                    accs = 0.0
                    errs = 0.0
                    ns = 0
                    for key in keys_valid:

                        feed_dict = {
                            input_placeholder: test_data[key],
                            target_placeholder_p0: test_label_all[0][key],
                            target_placeholder_p1: test_label_all[1][key],
                            target_placeholder_p2: test_label_all[2][key],
                            target_placeholder_p3: test_label_all[3][key],
                            target_placeholder_p4: test_label_all[4][key],
                            target_placeholder_rebuild: rebuild_label['valid_targets'][key]
                        }
                        ns += len(test_data[key])
                        acc, err = sess.run([accuracy, loss], feed_dict=feed_dict)
                        accs += acc
                        errs += err
                    accs /= ns
                    errs /= ns
                    print('-----------------------------------------------------------------')
                    print('Testing Epoch: %d, acc: %f, loss: %f '%(i, accs, errs))
                    print('-----------------------------------------------------------------')
                if i%10 == 0:
                    saver_path = saver.save(sess, '../combined_models/model_'+str(i)+'.ckpt', global_step=i)
                    print("Model saved in file: ", saver_path)
