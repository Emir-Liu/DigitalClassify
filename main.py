import numpy as np

class Network(object):
    def __init__(self,sizes):
        self.num_layers = len(sizes)
        self.sizes = sizes
        self.biases = [np.random.randn(y,1) for y in sizes[1:]]
        self.weights = [np.random.randn(y,x) for x,y in zip(sizes[:-1],sizes[1:])]
        '''
        建立神经网络，假设第0层为输入层，对于神经元都有偏置
        假设，所有都从0开始编号，即从第0层开始计数，每一层从第0个神经元开始

        num_layers为神经元层数。
        sizes为包含各层中神经元的数量的数组。
        biases为层与层之间的偏置：
        biases[i][j]:第i层的神经元到第i+1层神经元的第j个神经元的偏置。
        weights为两层之间节点的权值:
        weights[i][j,k]:第i层第j个神经元到第i+1层第k个神经元之间的权重。
        因此，如果我们需要设置一个网络对象，第一层2个神经元，第二层3个神经元，第三层1个神经元
        a = Network(2,3,1)

        上面使用了Numpy中的np.random.randn(x,y),产生X*Y的矩阵，矩阵中的数据是按照平均值为0，方差为1的正态分布。
        上面还使用了内置函数zip函数，又来将一个或多个迭代器合并到一起，然后返回一个对象。
        >>>a.biases
        [array([[-0.31202914],
               [ 0.61242964],
               [ 0.77307146]]), array([[-0.6499739]])]
        >>a.weights
        [array([[-1.31596366, -1.51535716],
               [ 1.64330833,  0.7670224 ],
               [ 0.87469726,  0.47140375]]), array([[-1.91929887,  0.40883907,  0.56018541]])]
        '''

    def SGD(self,training_data,epochs,mini_batch_size,eta,test_data=None):
        if test_data:
            n_test = len(test_data)
        n = len(list(training_data))
        for j in range(epochs):
            np.random.shuffle(training_data)
            mini_batches = [
                training_data[k:k+mini_batch_size]
                for k in range(0,n,mini_batch_size)]
            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch,eta)
            if test_data:
                print("Epoch {0}:{1}/{2}".format(j,self.evaluate(test_data),n_test))
            else:
                print("Epoch {0} complete".format(j))
        '''
        Stochastic Gradient Descent 随机梯度下降
        其中,training_data是包含元组(X,Y)的列表，分别表示训练输入和相应的期望输出
        epochs是训练阶段的期望数目，可以看作训练的次数
        mini_batch_size是采样的小批量的数目
        eta是学习速率
        如果提供了可选参数test_data，那么程序将会在每次训练之后评估网络，并且打印出部分进度，有利于跟踪进度，但是会降低速率
        
        '''

    def update_mini_batch(self,mini_batch,eta):
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        for x,y in mini_batch:
            delta_nabla_b,delta_nabla_w = self.backprop(x,y)
            nabla_w = [nb+dnb for nb,dnb in zip(nabla_b,delta_nabla_b)]
            nabla_w = [nw+dnw for nw,dnw in zip(nabla_w,delta_nabla_w)]
        self.weights = [w-(eta/len(mini_batch))*nw
                        for w,nw in zip(self.weights,nabla_w)]
        self.biases = [b-(eta/len(mini_batch))*nb
                        for b,nb in zip(self.biases,nabla_b)]
        '''
        在梯度下降的迭代中，更新权值和偏置，只使用训练数据mini_batch
        其中包含了一个backprop函数，用来计算代价函数的梯度,这个函数比较复杂，里面包含对激活函数的求导和代价函数，可以最后再看。
        
        nabla_b,nabla_w 是偏置和权重的微分算子
        delta_nabla_b,delta_nabla_w是每个采样数据返回的梯度
        详细过程为：计算出mini_batch中所有样本的梯度，然后得到所有样本权重和偏置的梯度和，然后根据梯度下降原理，计算出迭代之后的偏置和权重
        '''

    def feedforward(self, a):
        for b, w in zip(self.biases, self.weights):
            a = sigmoid(np.dot(w, a)+b)
        return a
    '''
    输入a，输出经过激活函数的结果
    '''

    def backprop(self, x, y):
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        # feedforward
        activation = x
        activations = [x] 
        zs = [] 
        for b, w in zip(self.biases, self.weights):
            z = np.dot(w, activation)+b
            zs.append(z)
            activation = sigmoid(z)
            activations.append(activation)
        # backward pass
        delta = self.cost_derivative(activations[-1], y) * sigmoid_prime(zs[-1])
        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(delta, activations[-2].transpose())

        for l in range(2, self.num_layers):
            z = zs[-l]
            sp = sigmoid_prime(z)
            delta = np.dot(self.weights[-l+1].transpose(), delta) * sp
            nabla_b[-l] = delta
            nabla_w[-l] = np.dot(delta, activations[-l-1].transpose())
        return (nabla_b, nabla_w)
    '''
    输入：一个训练输入和一个期望输出x,y
    返回：偏置梯度和权重梯度nabla_b, nabla_w，这两个是逐层列表，和self.biases和self.weights类似
    
    l:代表倒数第几层，l=1时，代表最后一层，l=2时，倒数第二层，
    利用了python可以在列表中使用负索引
    activation:激活，一开始就为输入的训练数据
    activations:逐层记录所有的激活
    zs:逐层记录所有的z向量
    '''

    def evaluate(self, test_data):
        """Return the number of test inputs for which the neural
        network outputs the correct result. Note that the neural
        network's output is assumed to be the index of whichever
        neuron in the final layer has the highest activation."""
        test_results = [(np.argmax(self.feedforward(x)), y)
                        for (x, y) in test_data]
        return sum(int(x == y) for (x, y) in test_results)

    def cost_derivative(self, output_activations, y):
        return (output_activations-y)

def sigmoid(z):
    return 1.0/(1.0+np.exp(-z))
    '''
    这就是激活函数，每一层的输出就是:sigmoid(当前层的数据*权值+偏置)
    '''

def sigmoid_prime(z):
    return sigmoid(z)*(1-sigmoid(z))
'''
这就是激活函数的导数
'''

import mnist_loader
training_data, validation_data, test_data = mnist_loader.load_data_wrapper()
net = Network([784, 10, 10])
net.SGD(list(training_data), 30, 10, 3.0, test_data=list(test_data))
