import numpy as np
import pickle
import tensorflow as tf

    
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

    
train = pickle.load(open('anouymous_train_data_50.npz', 'rb'))
valid = pickle.load(open('anouymous_valid_data_50.npz', 'rb'))
batched_inputs = train['inputs']
batched_targets = train['targets']
v_batched_inputs = valid['inputs']
v_batched_targets = valid['targets']

graph = tf.Graph()
with graph.as_default():
    
    input_placeholder = tf.placeholder(tf.float32, [None, None, 200])
    target_placeholder = tf.placeholder(tf.float32, [None, 134])
    
    lstm_layer1 = lstm_layer('lstm1', input_placeholder, 200)
    affine1 = affine_layer('affine1', lstm_layer1.outputs[:,-1,:], [200, 256], tf.nn.relu)
    affine2 = affine_layer('affine2', affine1.outputs, [256, 256], tf.nn.relu)
    affine3 = affine_layer('affine3', affine2.outputs, [256, 134], tf.identity)
    
    loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=affine3.outputs, labels=target_placeholder))
    optimizer = tf.train.AdamOptimizer(1e-3).minimize(loss)
    accuracy = tf.reduce_sum(tf.cast(tf.equal(tf.argmax(affine3.outputs, 1), tf.argmax(target_placeholder, 1)), tf.float32))
    
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    
    for i in range(500):
        
        accs = 0.0
        errs = 0.0
        n = 0
        for key in batched_inputs.keys():
            inputs_batch = batched_inputs[key]
            targets_batch = batched_targets[key]
#             print(inputs_batch.shape)
#             print(targets_batch.shape)
            feed_dict = {input_placeholder:inputs_batch, target_placeholder:targets_batch}
            _, acc, err = sess.run([optimizer, accuracy, loss], feed_dict=feed_dict)
            accs += acc
            errs += err
            n += len(inputs_batch)
        print('#Training Epoch %d, acc: %f, err: %f, n: %d'%(i, accs/n, errs/n, n))
        if (i+1)%5 == 0:
            for key in v_batched_inputs.keys():
                inputs_batch = v_batched_inputs[key]
                targets_batch = v_batched_targets[key]
    #             print(inputs_batch.shape)
    #             print(targets_batch.shape)
                feed_dict = {input_placeholder:inputs_batch, target_placeholder:targets_batch}
                acc, err = sess.run([accuracy, loss], feed_dict=feed_dict)
                accs += acc
                errs += err
                n += len(inputs_batch)
            print('__________________________________________________________________________')
            print('#Testing Epoch %d, acc: %f, err: %f, n: %d'%(i, accs/n, errs/n, n))
            print('__________________________________________________________________________')

            
            
            
            
            
            