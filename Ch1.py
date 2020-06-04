#step1
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

x = Variable(np.array(10))
f = Function()
y = f(x)

print(type(y))
print(y.data)

#step2
class Function:
    def __call__(self, input):
        x=input.data
        y=self.forward(x)
        output=Variable(y)
        return output

    def forward(self, x):
        raise NotImplementedError()


class Square(Function):
    def forward(self, x):
        return x**2

x=Variable(np.array(10))
f=Square()
y=f(x)
print(type(y))
print(y.data)

#step3
class Exp(Function):
    def forward(self,x):
        return np.exp(x)

A=Square()
B=Exp()
C=Square()

x=Variable(np.array(0.5))
a=A(x)
b=B(a)
y=C(b)
print(y.data)

#step4
