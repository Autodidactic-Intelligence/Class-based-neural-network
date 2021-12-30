import math
import random
from sklearn import datasets
import numpy as np

class Node:
    def __init__(self):
        self.output = 0
        self.input = 0
        self.derivative = 0
        self.backwards_conns = []
        self.error = 0
        self.bias = 1
    def activate(self):
        self.output = 1/(1+np.exp(-self.input))
    def derive(self):
        self.derivative = self.output * (1-self.output)
    def back_prop(self, learning_rate):
        for c in self.backwards_conns:
            c.weight += learning_rate * c.innode.output * self.error * self.derivative
            c.innode.error += self.error * c.weight
        self.bias += learning_rate * self.error * self.derivative
        self.error = 0
    def feed(self):
        self.input = 0
        for c in self.backwards_conns:
            self.input += c.innode.output * c.weight
        self.input += self.bias
        self.activate()
        self.derive() 
    
class Connection:
    def __init__(self, innode, outnode):
        self.innode = innode
        self.outnode = outnode
        self.weight = random.uniform(-0.5, 0.5)
   
class Network:
    def __init__(self, seed):
        self.seed = seed
        self.node_list = []
        self.learning_rate = 0.1
        for i in range(len(seed)):
            x = []
            for _ in range(seed[i]):
                x.append(Node())
            self.node_list += [x]
        for i in range(len(seed)-1):
            for n in self.node_list[i+1]:
                for k in self.node_list[i]:
                    n.backwards_conns.append(Connection(k, n))
    def run(self, x):
        for i, value in enumerate(x):
            self.node_list[0][i].output = value
        for i in range(1, len(self.seed)):
            for n in self.node_list[i]:
                n.feed()
        outputs = []
        for n in self.node_list[len(self.seed)-1]:
            outputs.append(n.input)
        np_out = np.array(outputs)
        expon = np.exp(outputs)
        probs = expon / np.sum(expon)
        return probs
    def train(self, error):
        for i, value in enumerate(error):
            self.node_list[len(self.seed)-1][i].error = value
        for i in range(len(self.seed)-1, 0, -1):
            for n in self.node_list[i]:
                n.back_prop(self.learning_rate)



def digits():
    X, y = datasets.load_digits(return_X_y=True)
    loops = int((len(X)/2) + 100)
    epochs = 10
    #scale the values to between 0 and 1
    for x in X:
        x *= (1/16)

    #Scramble the digits
    for i in range(loops):
        index_1 = random.randint(0, len(X)-1)   
        index_2 = random.randint(0, len(X)-1)
        
        #scramble the instances and training data
        X[[index_2, index_1]] = X[[index_1, index_2]]
        y[[index_1, index_2]] = y[[index_2, index_1]]
        
    #slice into training and test sets
    slice_index = int(len(X) *0.8)
    train_X = X[0 : slice_index]
    test_X = X[slice_index : len(X)]
    train_y = y[0: slice_index]
    test_y = y[slice_index: len(X)]

    #train_X = X[65:67]
    #test_X = X[63:65]
    #train_y = y[65:67]
    #test_y = y[63:65]

    #make and train the network
    net = Network([64,12,12,10])
    for _ in range(epochs):
        for i, x in enumerate(train_X):
            eo = [0.0 for _ in range(10)]
            cost = [0.0 for _ in range(10)]
            result = net.run(x)
            eo[train_y[i]] = 1
            for i, r in enumerate(result):
                cost[i] = (eo[i] - r)
            net.train(cost)
    
    #test the network on new data
    total = 0
    correct = 0
    for index, x in enumerate(test_X):
        result = net.run(x)
        chosen = result[0]
        chosen_i = 0
        for i, r in enumerate(result):
             if r > chosen:
                 chosen = r
                 chosen_i = i
        if chosen_i == test_y[index]:
            correct += 1
        total += 1

    print("correct: ",correct)
    print("total: ", total)
    print("accuracy: ", correct/total)

def get_instance():
    X = []
    for _ in range(2):
        X.append(random.randint(0, 1))
    y = []
    y.append(abs(X[0]- X[1]))
    return X, y

def get_instance_2_out():
    X = []
    for _ in range(2):
        X.append(random.randint(0, 1))
    b = abs(X[0]- X[1])
    if b == 1:
        y = [0.0,1]
    else:
        y = [1,0]
    return X, y

def xor():
    net = Network([2,2,1])
    for _ in range(20000):
        X, y = get_instance()
        outputs = net.run(X)
        out = outputs[0]
        for i in y:
            error = (i-out)
        net.train([error])
    print(net.run([1.0, 0]))
    print(net.run([0, 1.0]))
    print(net.run([1.0, 1.0]))
    print(net.run([0.0, 0]))
    """for n in net.node_list:
        for node in n:
            print("bias: ",node.bias)
            for c in node.backwards_conns:
                print(c.weight)
            print()"""

def xor_2_out():
    net = Network([2,2,2])
    error = []
    for _ in range(20000):
        X, y = get_instance_2_out()
        outputs = net.run(X)
        for i, value in enumerate(y):
            error.append((value-outputs[i]))
        net.train(error)
        error.clear()
    print(net.run([1.0, 0]))
    print(net.run([0, 1.0]))
    print(net.run([1.0, 1.0]))
    print(net.run([0.0, 0]))

xor_2_out()