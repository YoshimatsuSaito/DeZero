class Variable:
    def __init__(self, data):
        self.data=data

import numpy as np 
data=np.array(1.0)
x=Variable(data)
print(x.data)

class Function:
    def __call__(self, input):
        x = input.data
        y = x**2
        output = Variable(y)
        return output

