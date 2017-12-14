import tensorflow as tf
import numpy as np
import pickle

lines = open('../data/cleaned_corpus.txt', encoding='utf-8').readlines()
word_map = pickle.load(open('../data/word_map.npz', 'rb'))

line = lines[0]
print(line)
def getIDs(line, word_map):
    ids = []
    for i in line.split():
        if i not in word_map.keys():
            ids.append(word_map['UNKNOWN'])
        else:
            ids.append(word_map[i])
    return np.array(ids, dtype=np.int32)

print(getIDs(line, word_map))
print(len(word_map))
print(len(lines))
voc_size = len(word_map)

labels = [i.split()[0] for i in open('../data/TrainSet-eCarX-171019.txt', encoding='gbk').readlines()]
target_set = set([l for l in labels])
target_set = list(target_set)
# print(target_set)
# print(len(target_set))
target_size = len(target_set)
train_inputs = np.array([getIDs(i, word_map) for i in lines])
train_targets = np.zeros((len(labels), target_size))

contexts = open('../data/TrainSet-eCarX-171019.txt', encoding='gbk').readlines()
for line, i in zip(contexts, range(len(contexts))):
    target = line.split()[0]
    train_targets[i][target_set.index(target)] = 1.0
    
    
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
    
batched_inputs = dict()
batched_targets = dict()
batched_keys = dict()
for input_, target_ in zip(train_inputs, train_targets):
#     print(input_.shape)
    if len(input_) not in batched_inputs.keys():
        batched_inputs[len(input_)] = []
        batched_targets[len(input_)] = []
        batched_keys[len(input_)] = 0
    batched_inputs[len(input_)].append(input_)
    batched_targets[len(input_)].append(target_)
    batched_keys[len(input_)] += 1
#     break
for key in batched_keys.keys():
    batched_inputs[key] = np.array(batched_inputs[key])
    batched_targets[key] = np.array(batched_targets[key])
    batched_keys[key] = np.array(batched_keys[key])
    
    
graph = tf.Graph()
with graph.as_default():
    
    with tf.name_scope('embedding_lstm'):
#     if 1 == 1:

        
        layers = {}
        input_placeholder = tf.placeholder(tf.int32, [None, None])
        target_placeholder = tf.placeholder(tf.float32, [None, 134])
        batchSize_placeholder = tf.placeholder(tf.int32)
        
        layers['input_placeholder'] = input_placeholder
        layers['target_placeholder'] = target_placeholder
#         stepSize_placeholder = tf.placeholder(tf.int32)

        voc_size = 1816
        embedding_size = 300
        with tf.variable_scope('embedding'):
            layers['embedding'] = tf.get_variable(name='embedding', initializer=tf.random_uniform([voc_size, embedding_size], -1.0, 1.0), dtype=tf.float32)
        layers['input_embedding'] = tf.nn.embedding_lookup(layers['embedding'], input_placeholder)

        with tf.name_scope('lstm_layer1'):
            layers['lstm1'] = lstm_layer('lstm_layer1', layers['input_embedding'], embedding_size)
#         print(layers['lstm1'].outputs)
            
        with tf.name_scope('reshape_layer1'):
            layers['reshape1'] = reshape_layer('reshape_layer1', layers['lstm1'].outputs[:,-1,:], [batchSize_placeholder, 1, embedding_size, 1])
#         print(layers['reshape1'])
        
        with tf.name_scope('conv_layer1'):
            layers['conv1'] = conv_layer('conv_layer1', layers['reshape1'].outputs, [1, 50, 1, 1], tf.nn.relu)

        with tf.name_scope('reshape_layer2'):
            layers['reshape2'] = reshape_layer('reshape_layer2', layers['conv1'].outputs, [batchSize_placeholder, 251])

        with tf.name_scope('affine_layer1'):
            layers['affine1'] = affine_layer('affine_layer1', layers['reshape2'].outputs, [251, 128], tf.nn.relu)

        with tf.name_scope('affine_layer2'):
            layers['out'] = affine_layer('affine_layer2', layers['affine1'].outputs, [128, 134], tf.identity)


        for layer in layers.keys():
            try:
                print(layer.outputs)
            except:
                continue
        
        sess = tf.Session()
#         sess.run(tf.variables_initializer())
        sess.run(tf.global_variables_initializer())

        loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=layers['out'].outputs, labels=target_placeholder))
    #     print(loss)
        accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(layers['out'].outputs, 1), tf.argmax(target_placeholder, 1)), tf.float32))
        optimizer = tf.train.GradientDescentOptimizer(1e-3).minimize(loss)

        for i in range(100):
            errs = 0.0
            accs = 0.0
            n = 0
            for key in batched_keys.keys():
#                 inputs = inputs_batch.reshape(1, -1)
#                 targets = targets_batch.reshape(1, -1)
                inputs = batched_inputs[key]
                targets = batched_targets[key]
                batchSize = batched_keys[key]
#                 print(batchSize)
#                 print(inputs.shape)
#                 print(targets.shape)
                feed_dict = {input_placeholder: inputs, target_placeholder:targets, batchSize_placeholder:batchSize}
                _, err, acc, lookup = sess.run([optimizer, loss, accuracy, layers['input_embedding']], feed_dict=feed_dict)
#                 print(lookup.shape)
                n += 1
            errs = errs/n
            accs = accs/n
            print('#Training Epoch %d, ACC:%f, ERR:%f'%(i, accs, errs))