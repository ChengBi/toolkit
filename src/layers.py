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
    
    def __init__(self, inputs, n_units):
        
        self.n_units = n_units
        self.input_shape = inputs.shape
        self.outputs, self.states = tf.nn.dynamic_rnn(
            cell = tf.contrib.rnn.BasicLSTMCell(self.n_units),
            inputs = inputs,
            dtype = tf.float32)
        
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