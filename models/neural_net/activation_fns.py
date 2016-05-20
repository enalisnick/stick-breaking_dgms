import theano.tensor as T

def ReLU(x):
    y = T.maximum(0.0, x)
    return(y)

def Sigmoid(x):
    y = T.nnet.sigmoid(x)
    return(y)

def Softplus(x):
    y = T.nnet.softplus(x)
    return(y)

def Softmax(x):
    y = T.nnet.softmax(x)
    return(y)

def Identity(x):
    return(x)

def Beta_fn(a, b):
    return T.exp(T.gammaln(a) + T.gammaln(b) - T.gammaln(a+b))
