import tensorflow as tf
import numpy as np
import pickle

data = pickle.load(open('../data/data.npz', 'rb'))

class layer(object):
    def __init__(self):
        raise NotImplementedError
    
    def __str__(self):
        raise NotImplementedError
    
class affine_layer(layer):
    def __init__(self, name, inputs, shape, activation, reuse=False):
        with tf.variable_scope('affine', reuse=reuse):
            self.weights = tf.get_variable(name = name+'_weights', initializer=tf.truncated_normal(shape, stddev=0.05), dtype=tf.float32)
            self.bias = tf.get_variable(name = name+'_bias', initializer=tf.zeros([shape[-1]]))
            self.outputs = activation(tf.add(tf.matmul(inputs, self.weights), self.bias))
    def __str__(self):
        return self.outputs.__str__()
    
class conv_layer(layer):
    def __init__(self, name, inputs, shape, activation, reuse=False):
        with tf.variable_scope('conv', reuse=reuse):
            self.kernel = tf.get_variable(name = name+'_kernel', initializer=tf.truncated_normal(shape, stddev=0.05), dtype=tf.float32)
            self.bias = tf.get_variable(name = name+'_bias', initializer=tf.zeros([shape[-1]]))
            self.outputs = activation(tf.add(tf.nn.conv2d(inputs, self.kernel, padding='VALID', strides=[1,1,1,1]), self.bias))
    def __str__(self):
        return self.outputs.__str__()
    
class pooling_layer(layer):
    def __init__(self, name, inputs, reuse=False):
        with tf.variable_scope('pooling', reuse=reuse):
            self.outputs = tf.nn.max_pool(name = name+'_maxpooling', value=inputs, ksize=[1,2,2,1], padding='VALID', strides=[1,2,2,1])
    def __str__(self):
        return self.outputs.__str__()
        
class reshape_layer(layer):
    def __init__(self, name, inputs, shape, reuse=False):
        with tf.variable_scope('reshape', reuse=reuse):
            self.outputs = tf.reshape(name=name, tensor=inputs, shape=shape)
    def __str__(self):
        return self.outputs.__str__()
        
class deconv_layer(layer):
    def __init__(self, name, inputs, kernel_shape, output_shape, activation, reuse=False):
        with tf.variable_scope('deconv', reuse=reuse):
            self.kernel = tf.get_variable(name = name+'_kernel', initializer=tf.truncated_normal(kernel_shape, stddev=0.05), dtype=tf.float32)
            self.bias = tf.get_variable(name = name+'_bias', initializer=tf.zeros([kernel_shape[-2]]))
            self.outputs = activation(tf.add(tf.nn.conv2d_transpose(inputs, self.kernel, output_shape=output_shape, padding='VALID', strides=[1,2,2,1]), self.bias))
    def __str__(self):
        return self.outputs.__str__()
    
class lstm_layer(layer):
    def __init__(self, name, inputs, n_units, reuse=False):
        with tf.variable_scope('lstm', reuse=reuse):
            self.outputs, self.states = tf.nn.dynamic_rnn(
                cell = tf.contrib.rnn.BasicLSTMCell(n_units),
                inputs = inputs,
                dtype = tf.float32)
    def __str__(self):
        return self.outputs.__str__()
    
train_inputs = data['inputs']
train_targets = data['targets']
batch_sizes = data['batch_size']
n_units = data['units']
n_hidden = 300


graph = tf.Graph()
with graph.as_default():
    
    input_placeholder = tf.placeholder(tf.float32, [None, None, n_units])
    target_placeholder = tf.placeholder(tf.float32, [None, None, n_units])
    batch_size_placeholder = tf.placeholder(tf.int32)
    step_size_placeholder = tf.placeholder(tf.int32)
    
    
    reshape1 = reshape_layer('reshape_1', input_placeholder, [-1, n_units])
    affine1 = affine_layer('affine_1', reshape1.outputs, [n_units, n_hidden], tf.nn.relu)
    reshape2 = reshape_layer('reshape_2', affine1.outputs, [batch_size_placeholder, step_size_placeholder, n_hidden])
    
    
    lstm1 = lstm_layer('lstm_1', reshape2.outputs, n_hidden, False)
    reshape3 = reshape_layer('reshape_3', lstm1.outputs, [-1, n_hidden])
    affine2 = affine_layer('affine_2', reshape3.outputs, [n_hidden, n_units], tf.identity)
    reshape4 = reshape_layer('reshape_4', affine2.outputs, [batch_size_placeholder, step_size_placeholder, n_units])
    
#     affine_layer_2 = affine_layer('affine_2', affine_layer_1.outputs, [200, n_units], tf.identity)

    output_layer = reshape4
    
    loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=output_layer.outputs, labels=target_placeholder))
    optimizer = tf.train.AdamOptimizer(1e-3).minimize(loss)
    
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    
    for i in range(1):
        errs = 0.0
        c = 0
        for key in batch_sizes.keys():

            input_batch = train_inputs[key]
            target_batch = train_targets[key]
#             print(key, input_batch.shape)
            step_size = key
            batch_size = input_batch.shape[0]
            
            feed_dict = {step_size_placeholder:step_size, batch_size_placeholder:batch_size, input_placeholder:input_batch, target_placeholder:target_batch}
            _, err = sess.run([optimizer, loss], feed_dict=feed_dict)
            
            errs += err
            c += 1
        print(errs / c)