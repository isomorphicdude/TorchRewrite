"""
Implement the stochastic descent in matrix form.

"""

import random

import numpy as np

#import scipy.linalg

#When loading data, it is also possible to convert the list of tuples to a 2 dimensional
#array with first entry as a list of ndarrays of inputs and second entry as a list of outputs
#using zip(*list(training_data))


class Network(object):
    np.random.seed(1234)
    def __init__(self,sizes,batch_size=10):
        self.num_layers = len(sizes)

        self.sizes = sizes

        self.biases = [np.random.randn(y, ) for y in sizes[1:]]

        #self.biases_batch=[np.zeros([batch_size,len(bias)])+bias for bias in self.biases]
        # broadcasting the values, biases fill the matrix of shape [batch_size,len(bias)]
        self.weights = [np.random.randn(x, y)
                        for x, y in zip(sizes[:-1], sizes[1:])]

        self.batch_size=batch_size

    def feedforward(self,a):
        """
        Implement feedforward with matrix batch.

        Parameter: network and initial activation a
        """
        weights=[x.transpose() for x in self.weights]
        for b, w in zip(self.biases, weights):
            a = sigmoid(np.dot(w, np.ravel(a))+b)
        return a

    def SGD(self, training_data, epochs, eta, test_data=None):
            #A mini batch can be inputted directly using a matrix
            #But a conversion is also acceptable
            """
            Implement matrix approach of stochastic descent.
            """
            mini_batch_size=self.batch_size

            #training_data = list(training_data)
            n_training = len(training_data)

            if test_data:
                test_data = list(test_data)
                n_test = len(test_data)

            for j in range(epochs):
                random.shuffle(training_data)
                mini_batches = [
                    training_data[k:k+mini_batch_size]
                    for k in range(0, n_training, mini_batch_size)]
                #mini_batches=np.array(mini_batches)
                
                for mini_batch in mini_batches:
                    self.update_mini_batch(mini_batch, eta)
                #A change is made here to use the matrix form
                if test_data:
                    print("Epoch {} : {} / {}".format(j,self.evaluate(test_data),n_test))
                    #print(f"Weight: {(self.weights[0][0]==og_weight[0][0]).all()}")
                    print(f"Biase: {self.biases[0][0]}")
                else:
                    print("Epoch {} complete".format(j))
    

    def update_mini_batch(self, mini_batch, eta):

        mini_batch_size=len(mini_batch)

        x=np.array([np.ravel(a[0]) for a in mini_batch]) # x is the input array of the batches

        y=np.array([np.ravel(b[1])for b in mini_batch]) # y is the response vector of the elements in this batch

        delta_nabla_b, delta_nabla_w = self.backprop(x, y)

        delta_nabla_b_sum=[sum(x) for x in delta_nabla_b]

        self.weights=[w-(eta/mini_batch_size)*dw  for w, dw in zip(self.weights, delta_nabla_w)]


        self.biases=[b-(eta/mini_batch_size)*nb for b, nb in zip(self.biases, delta_nabla_b_sum)]

            
    def backprop(self, x, y):

        biases_batch=[np.zeros([self.batch_size,len(bias)])+bias for bias in self.biases]

        """Return a tuple ``(nabla_b, nabla_w)`` representing the
        gradient for the cost function C_x.
        """
        #nabla_b = [np.zeros(b.shape) for b in self.biases]
        #nabla_w = [np.zeros(w.shape) for w in self.weights]

        nabla_b = [np.zeros(b.shape) for b in biases_batch]
        #the bias derivative matrix should store the mini_batch_size*no_of_nodes
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        # the weight derivative matrix should store only the sum of weight matrices produced
        # The lists of matrices defined above store derivatives for every mini-batch

        # feedforward
        activation = x
        activations = [x]

        zs = [] # list to store all the z vectors, layer by layer
        for b, w in zip(biases_batch, self.weights):
            # Weights need not to be full matrix form
            # They are not altered during one batch pass through
            z = np.dot(activation,w)+b #changed here as weight is transposed
            zs.append(z)
            activation = sigmoid(z)
            activations.append(activation)
        # backward pass
        delta = self.cost_derivative(activations[-1], y) * \
            sigmoid_prime(zs[-1])
            
        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(activations[-2].transpose(), delta)


        for l in range(2, self.num_layers):
            z = zs[-l]
            sp = sigmoid_prime(z)
            delta = np.dot(delta, self.weights[-l+1].transpose()) * sp
            nabla_b[-l] = delta
            # each element is of the form M*10
            nabla_w[-l] = np.dot(activations[-l-1].transpose(),delta)
            # each element is of the form 10*10 which are sums of M weight derivative matrices
        return (nabla_b, nabla_w)

#################################### Other Functions #######################################
    def evaluate(self, test_data):
        """Return the number of test inputs for which the neural
        network outputs the correct result. Note that the neural
        network's output is assumed to be the index of whichever
        neuron in the final layer has the highest activation."""
        test_results = [(np.argmax(self.feedforward(x)), y)
                        for (x, y) in test_data]
        return sum(int(x == y) for (x, y) in test_results)

    def cost_derivative(self, output_activations, y):
        """Return the vector of partial derivatives \partial C_x /
        \partial a for the output activations."""
        return (output_activations-y)

def sigmoid(z):
    """The sigmoid function."""
    return 1.0/(1.0+np.exp(-z))

def sigmoid_prime(z):
    """Derivative of the sigmoid function."""
    return sigmoid(z)*(1-sigmoid(z))