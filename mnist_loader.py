'''
MNIST原始网站
http://yann.lecun.com/exdb/mnist/
MNIST装载
加载MNIST图像数据的库文件，
数据结构 load_data load_data_wrapper(这个通常由神经网络调用)
'''

import pickle
import gzip

import numpy as np

def load_data():
    '''
    返回MNIST数据的格式为：训练数据，验证数据，测试数据
    训练数据由包含两个实体的元组组成：
    第一个实体包含了训练图像，这是一个有50000个实体的numpy ndarrary(n维数组对象，是一系列同类型数据的集合)
    里面的每个实体都是一个有784个数值的数组，代表一个MNIST图像中的28*28=784个像素。
    第二个实体就是包含50000个实体的数组，里面每个实体是从0到9的数值，对应着第一个实体中的图像所代表的数字
    
    验证数据和测试数据类似，除了每一个都只包含10000个图片。

    对于神经网络,将其通过包装器函数load_data_wrapper()修改格式
    '''
    f = gzip.open('mnist.pkl.gz', 'rb')
    training_data,validation_data,test_data = pickle.load(f,encoding='bytes')
    f.close()
    return (training_data, validation_data, test_data)

def load_data_wrapper():
    '''
    返回一个元组包含(训练数据，验证数据，测试数据)
    以load_data为基础，但是格式更加便于我们在神经网络中使用

    训练数据是包含50000个元组(x,y):
    x是784维数组，包含输入图片
    y是一个10维的数组代表x的正确的数字

    验证数据和测试数据都是包含10000个元组(x,y)的列表
    x是784维数组，包含输入图片
    y是一个数字代表对应图片表示的数字
    '''
    tr_d, va_d, te_d = load_data()
    training_inputs = [np.reshape(x, (784, 1)) for x in tr_d[0]]
    training_results = [vectorized_result(y) for y in tr_d[1]]
    training_data = zip(training_inputs, training_results)
    validation_inputs = [np.reshape(x, (784, 1)) for x in va_d[0]]
    validation_data = zip(validation_inputs, va_d[1])
    test_inputs = [np.reshape(x, (784, 1)) for x in te_d[0]]
    test_data = zip(test_inputs, te_d[1])
    return (list(training_data), list(validation_data), list(test_data))

def vectorized_result(j):
    '''
    将0-9的数字转换为一个对应的10维的向量
    '''
    e = np.zeros((10, 1))
    e[j] = 1.0
    return e
