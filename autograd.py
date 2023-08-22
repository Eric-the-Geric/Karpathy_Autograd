# followed this tutorial and added some other functionality like apply_gradients, and zero_grad()
# See the link below for the full tutorial by Andrej Karpathy
# https://www.youtube.com/watch?v=VMj-3S1tku0&list=PLAqhIrjkxbuWI23v9cThsA9GvCAUhRvKZ
import numpy as np
import math

class Value:
    def __init__(self, data, children=(), operation="", label=""):
        self.data = data
        self.children = set(children)
        self.grad = 0
        self.operation = operation
        self.label = label
        self._backprop = lambda: None

    def backward(self):
        topo = []
        visted = set()
        def build_topo(v):
            if v not in visted:
                visted.add(v)
                for child in v.children:
                    build_topo(child)
                topo.append(v)
        build_topo(self)
        self.grad = 1.0
        for node in reversed(topo):
            node._backprop()
    
    def apply_gradients(self, lr):
        topo = []
        visted = set()
        def build_topo(v):
            if v not in visted:
                visted.add(v)
                for child in v.children:
                    build_topo(child)
                topo.append(v)
        build_topo(self)
        for node in topo:
            node.data += -lr*node.grad

    def zero_grad(self):
        topo = []
        visted = set()
        def build_topo(v):
            if v not in visted:
                visted.add(v)
                for child in v.children:
                    build_topo(child)
                topo.append(v)
        build_topo(self)
        for node in topo:
            node.grad *= 0.0


    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out= Value(self.data + other.data, children=(self, other), operation="+")
        
        def _backprop():
            self.grad += 1.0*out.grad
            other.grad += 1.0*out.grad
        out._backprop = _backprop
        return out

    def tanh(self):
        x = self.data
        t = (math.exp(2*x) - 1)/(math.exp(2*x) + 1)
        out = Value(t, children=(self, ), operation='tanh')
    
        def _backprop():
          self.grad += (1 - t**2) * out.grad
        out._backprop = _backprop
        return out 

    def __sub__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        return self + (-other)

    def __rsub__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        return other - self

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data*other.data, children=(self, other), operation="*")
        def _backprop():
            self.grad += other.data*out.grad
            other.grad += self.data*out.grad
        out._backprop = _backprop
        return out

    def __truediv__(self, other):
        return self * other**-1

    def __pow__(self, other):
        assert isinstance(other, (int, float)), "only supping int/float powers fn"
        out = Value(self.data**other, children=(self,), operation=f"**{other}")
        def _backprop():
            self.grad += other*(self.data**(other -1))*out.grad
        out._backprop = _backprop
        return out

    def __rmul__(self, other):
        return self * other
    
    def __neg__(self):
        return self*-1
    
    def __radd__(self, other):
        return self + other
    
    def exp(self):
        x = self.data
        out = Value(math.exp(x), children=(self,), operation="exp")
        def _backprop():
            self.grad += out.data*out.grad
        out._backprop = _backprop
        return out

    #def __str__(self):
    #    return f"data = {str(self.data)} \n Gradient = {str(self.grad)} \n Number children = {len(self.children)} \n operation = {self.operation} \n label={self.label}"
    def __repr__(self):
        return f"Value(data={self.data}, grad={self.grad})"

class Neuron:
    def __init__(self, nin):
        self.w = [Value(np.random.uniform(-1,1)) for _ in range(nin)]
        self.b = Value(np.random.uniform(-1,1))

    def __call__(self, x):
        return (sum((wi*xi for wi, xi in zip(x, self.w)),self.b).tanh())

    def get_params(self):
        return self.w + [self.b]

class Layer:
    def __init__(self, nin, nout):
        self.neurons = [Neuron(nin) for _ in range(nout)]

    def __call__(self, x):
        out = [n(x) for n in self.neurons]

        return out[0] if len(out) == 1 else out

    def get_params(self):
        return [n.get_params() for n in self.neurons]

class MLP:
    def __init__(self, nin, nouts):
        sz = [nin] + nouts
        self.layers = [Layer(sz[i], sz[i+1]) for i in range(len(nouts))]

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    def get_params(self):
        return [layer.get_params() for layer in self.layers]
        

    
