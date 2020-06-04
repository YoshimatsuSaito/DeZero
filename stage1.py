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
def numerical_diff(f,x,eps=1e-4):
    x0=Variable(x.data-eps)
    x1=Variable(x.data+eps)
    y0=f(x0)
    y1=f(x1)
    return(y1.data - y0.data)/(2*eps)

f=Square()
x=Variable(np.array(2.0))
dy=numerical_diff(f, x)
print(dy)

def f(x):
    A=Square()
    B=Exp()
    c=Square()
    return C(B(A(x)))

x = Variable(np.array(0.5))
dy = numerical_diff(f, x)
print(dy)

#step6
class Variable:
    def __init__(self, data):
        self.data=data
        self.grad=None

class Function:
    def __call__(self, input):
        x=input.data
        y=self.forward(x)
        output=Variable(y)
        self.input=input
        return output
    
    def forward(self, x):
        raise NotImplementedError

    def backward(self,gy):
        raise NotImplementedError()

class Square(Function):
    def forward(self, x):
        y=x**2
        return y
    
    def backward(self,gy):
        x=self.input.data
        gx=2*x*gy
        return gx

class Exp(Function):
    def forward(self,x):
        y=np.exp(x)
        return y
    
    def backward(self,gy):
        x=self.input.data
        gx=np.exp(x)*gy
        return gx

A=Square()
B=Exp()
C=Square()

x=Variable(np.array(0.5))
a=A(x)
b=B(a)
y=C(b)

y.grad=np.array(1.0)
b.grad=C.backward(y.grad) 
a.grad=B.backward(b.grad)
x.grad=A.backward(a.grad)
print(x.grad)

#step7
class Variable:
    def __init__(self, data):
        self.data = data
        self.grad = None
        self.creator = None

    def set_creator(self, func):
        self.creator = func

class Function:
    def __call__(self, input):
        x = input.data
        y = self.forward(x)
        output = Variable(y)
        output.set_creator(self)  # Set parent(function)
        self.input = input
        self.output = output  # Set output
        return output

class Square(Function):
    def forward(self, x):
        y=x**2
        return y
    
    def backward(self,gy):
        x=self.input.data
        gx=2*x*gy
        return gx

class Exp(Function):
    def forward(self,x):
        y=np.exp(x)
        return y
    
    def backward(self,gy):
        x=self.input.data
        gx=np.exp(x)*gy
        return gx

A = Square()
B = Exp()
C = Square()

x = Variable(np.array(0.5))
a = A(x)
b = B(a)
y = C(b)

assert y.creator == C
assert y.creator.input == b
assert y.creator.input.creator == B
assert y.creator.input.creator.input == a
assert y.creator.input.creator.input.creator == A
assert y.creator.input.creator.input.creator.input == x

y.grad = np.array(1.0)
C=y.creator
b=C.input
b.grad=C.backward(y.grad)
B=b.creator
a=B.input
a.grad=B.backward(b.grad)
A=a.creator
x=A.input
x.grad=A.backward(a.grad)
print(x.grad)

class Variable:
    def __init__(self,data):
        self.data=data
        self.grad=None
        self.creator=None

    def set_creator(self,func):
        self.creator=func

    def backward(self):
        f=self.creator
        if f is not None:
            x=f.input
            x.grad=f.backward(self.grad)
            x.backward()

A=Square()
B=Exp()
C=Square()

x=Variable(np.array(0.5))
a=A(x)
b=B(a)
y=C(b)

y.grad=np.array(1.0)
y.backward()
print(x.grad)


#step8
class Variable:
    def __init__(self,data):
        self.data=data
        self.grad=None
        self.creator=None

    def set_creator(self,func):
        self.creator=func

    def backward(self):
        funcs=[self.creator]
        while funcs:
            f=funcs.pop()
            x,y=f.input, f.output
            x.grad=f.backward(y.grad)

            if x.creator is not None:
                funcs.append(x.creator)

A=Square()
B=Exp()
C=Square()

x=Variable(np.array(0.5))
a=A(x)
b=B(a)
y=C(b)

y.grad=np.array(1)
y.backward()
print(x.grad)


#step9
def square(x):
    f=Square()
    return f(x)

def exp(x):
    f=Exp()
    return f(x)

x=Variable(np.array(0.5))
a=square(x)
b=exp(a)
y=square(b)


x=Variable(np.array(0.5))
y=square(exp(square(x)))

y.grad=np.array(1.0)
y.backward()
print(x.grad)


class Variable:
    def __init__(self,data):
        self.data=data
        self.grad=None
        self.creator=None

    def set_creator(self,func):
        self.creator=func

    def backward(self):
        if self.grad is None:
            self.grad=np.ones_like(self.data)

        funcs=[self.creator]
        while funcs:
            f=funcs.pop()
            x,y=f.input, f.output
            x.grad=f.backward(y.grad)

            if x.creator is not None:
                funcs.append(x.creator)

x=Variable(np.array(0.5))
y=square(exp(square(x)))
y.backward()
print(x.grad)

class Variable:
    def __init__(self,data):
        if data is not None:
            if not isinstance(data, np.ndarray):
                raise TypeError('{} is not supported'.format(type(data)))

        self.data=data
        self.grad=None
        self.creator=None

    def set_creator(self,func):
        self.creator=func

    def backward(self):
        if self.grad is None:
            self.grad=np.ones_like(self.data)

        funcs=[self.creator]
        while funcs:
            f=funcs.pop()
            x,y=f.input, f.output
            x.grad=f.backward(y.grad)

            if x.creator is not None:
                funcs.append(x.creator)

def as_array(x):
    if np.isscaler(x):
        return np.array(x)
    return x

class Function:
    def __call__(self, input):
        x = input.data
        y = self.forward(x)
        output = Variable(as_array(y))
        output.set_creator(self)  # Set parent(function)
        self.input = input
        self.output = output  # Set output
        return output

#step10
import unittest
class SquareTest(unittest.TestCase):
    def test_forward(self):
        x=Variable(np.array(2.0))
        y=square(x)
        expected=np.array(4.0)
        self.assertEqual(y.data,expected)
    
    def test_backward(self):
        x=Variable(np.array(3.0))
        y=square(x)
        y.backward()
        expected=np.array(6.0)
        self.assertEqual(x.grad, expected)

    def test_gradient_check(self):
        x=Variable(np.random.rand(1))
        y=square(x)
        y.backward()
        num_grad=numerical_diff(square, x)
        flg=np.allclose(x.grad, num_grad)
        self.assertTrue(flg)
        