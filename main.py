from autograd import Value, Neuron, Layer, MLP
import numpy as np
import math
import random

def sigmoid(x):
    return Value(1)/(Value(1) + (-x).exp())

def cost(ys, y_pred):
    return sum(((y-y_p)**2) for y, y_p in zip(ys, y_pred))

def main():

    xs = [[1.0, 1.0],
          [1.0, 0.0],
          [0.0, 1.0],
          [0.0, 0.0]]
    ys = [0, 1, 1, 0]

    model = MLP(1, [2, 1])

    for epoch in range(10):
        y_pred = [model(x) for x in xs]
        loss = cost(ys, y_pred)*0.5
        loss.zero_grad()
        loss.backward()
        loss.apply_gradients(0.5)
        print("============")
        print(loss.data)
        print("___________")
        print([y.data for y in y_pred])

if __name__=="__main__":
    main()
