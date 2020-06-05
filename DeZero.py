#This is master file
import numpy as np

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
    if np.isscalar(x):
        return np.array(x)
    return x


class Function:
    def __call__(self, inputs):
        xs=[x.data for x in inputs]
        ys=self.forward(xs)
        outputs=[Variable(as_array(y)) for y in ys]

        for output in outputs:
            output.set_creator(self)
        self.inputs=inputs
        self.outputs=outputs
        return outputs

    def forward(self,x):
        raise NotImplementedError()
    
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

class Add(Function):
    def forward(self, xs):
        x0,x1=xs
        y=x0+x1
        return (y,)


def square(x):
    f=Square()
    return f(x)

def exp(x):
    f=Exp()
    return f(x)
