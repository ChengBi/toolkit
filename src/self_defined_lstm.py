import tensorflow as tf
import numpy as np
import pickle
train_data = pickle.load(open('../word2vec/anouymous_train_data_200.npz', 'rb'))
train_inputs = train_data['inputs']
train_targets = train_data['targets']
print(train_inputs['13_0'].shape)
print(train_targets['13_0'].shape)
padded_inputs = dict()
padded_targets = dict()
max_batch = 300
max_step = 40
feature = 200
for key in train_inputs.keys():
    temp = train_inputs[key]
    temp2 = train_targets[key]
    if temp.shape[0] != max_batch and temp.shape[0] != 0:
#         print(temp.shape)
#         print(50-len(temp))
#         print(np.zeros((50 - temp.shape[0], temp.shape[1], 200)).shape)
        temp = np.concatenate((temp, np.zeros((max_batch - temp.shape[0], temp.shape[1], feature))), 0)
        temp2 = np.concatenate((temp2, np.zeros((max_batch - temp2.shape[0], 134))), 0)
#         print(temp.shape)
        temp = np.concatenate((temp, np.zeros((max_batch, max_step - temp.shape[1], feature))), 1)
        padded_inputs[key] = temp
        padded_targets[key] = temp2
    elif temp.shape[0] == max_batch and temp.shape[0] != 0:
        temp = np.concatenate((temp, np.zeros((max_batch, max_step - temp.shape[1], feature))), 1)
        padded_inputs[key] = temp
        padded_targets[key] = train_targets[key]
print(padded_inputs['13_1'].shape)
print(padded_targets['13_1'].shape)

def init(shape):
    return tf.Variable(tf.truncated_normal(shape), tf.float32)

def lstm_layer(inputs, batch_size, step_size):
    shape = inputs.shape.as_list()
    batch_size = shape[0]
    step_size = shape[1]
    n_units = shape[2]
#     print(batch_size)
#     print(step_size)

    wf = init([n_units, n_units])
    bf = init([n_units])

    wi = init([n_units, n_units])
    bi = init([n_units])

    wc = init([n_units, n_units])
    bc = init([n_units])

    wo = init([n_units, n_units])
    bo = init([n_units])

    ht = init([batch_size, n_units])
    ct = init([1, n_units])

    ct_tiled = tf.tile(ct, [batch_size, 1])
    
    W = tf.transpose(tf.tile(tf.concat([wf, wi, wc, wo], 0), [1, 2]), [1, 0])
    B = tf.concat([bf, bi, bc, bo], 0)
    
        
    for i in range(step_size):
        xt = inputs[:, i, :]
#         ft = tf.sigmoid(tf.matmul(ht, wf) + tf.matmul(xt, wf) + bf)
#         it = tf.sigmoid(tf.matmul(ht, wi) + tf.matmul(xt, wi) + bi)
#         c = tf.tanh(tf.matmul(ht, wc) + tf.matmul(xt, wc) + bc)
#         ot = tf.sigmoid(tf.matmul(ht, wo) + tf.matmul(xt, wo) + bo)
        X = tf.concat([ht, xt], 1)
#         print(ht.shape)
#         print(xt.shape)
#         print(X.shape)
        V = tf.sigmoid(tf.matmul(X, W) + B)
#         print(W.shape)
#         print(B.shape)
        ft, it, c, ot = tf.split(V, num_or_size_splits=4, axis=1)
#         print(ft.shape)
#         print(it.shape)
#         print(c.shape)
#         print(ot.shape)
        
        ct = tf.reshape(tf.reduce_mean(ft * ct_tiled + it * c, 0), [1, n_units])
        ht = ot * tf.tile(tf.tanh(ct), [batch_size, 1])
        
    return ht, ct


graph = tf.Graph()
with graph.as_default():
    
    batch_size = max_batch
    step_size = max_step
    
    inputs_placeholder = tf.placeholder(tf.float32, [batch_size, step_size, feature])
    targets_placeholder = tf.placeholder(tf.float32, [batch_size, 134])
#     batch_size = tf.placeholder(tf.int32)
#     step_size = tf.placeholder(tf.int32)
    
#     outputs, states = tf.nn.dynamic_rnn(
#             cell = tf.contrib.rnn.BasicLSTMCell(num_units=200),
#             inputs = inputs_placeholder,
#             dtype = tf.float32)
        
#     print(outputs)
#     print(cells)
    outputs, states = lstm_layer(inputs_placeholder, batch_size, step_size)
    
    w1 = init([200, 300])
    b1 = init([300])
    
    w2 = init([300, 300])
    b2 = init([300])
    
    w3 = init([300, 134])
    b3 = init([134])
    
    o1 = tf.nn.relu(tf.add(tf.matmul(outputs, w1), b1))
    o2 = tf.nn.relu(tf.add(tf.matmul(o1, w2), b2))
    o3 = tf.nn.relu(tf.add(tf.matmul(o2, w3), b3))
    
    loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=o3, labels=targets_placeholder))
    sum_loss = tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(logits=o3, labels=targets_placeholder))
    optimizer = tf.train.AdamOptimizer().minimize(loss)
    accuracy = tf.reduce_sum(tf.cast(tf.equal(tf.argmax(o3, 1), tf.argmax(targets_placeholder, 1)), tf.float32))
    
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    
    keys = padded_inputs.keys()
    
    for i in range(100):
        accs = 0.0
        errs = 0.0
        n = 0
        for key in keys:
            batch_inputs = padded_inputs[key]
            batch_targets = padded_targets[key]
            batch_size = len(train_inputs)
#             step_size = key.split('_')[0]
#             print(batch_inputs.shape)
#             print(batch_targets.shape)
            n += batch_size
            feed_dict = {inputs_placeholder:batch_inputs, targets_placeholder:batch_targets}#, batch_size:batch_size, step_size:step_size}
            _, acc, err = sess.run([optimizer, accuracy, sum_loss], feed_dict=feed_dict)
            accs += acc
            errs += err
        print('#Trainig Epoch: %d, ACC:%f ERR:%f'%(i, accs/n, errs/n))