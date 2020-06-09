import numpy as np
import weakref
import contextlib

class Variable:
    def __init__(self,data,name=None):
        if data is not None:
            if not isinstance(data, np.ndarray): #data (argument of variable class) must be np.ndarray
                raise TypeError('{} is not supported'.format(type(data)))

        self.data=data #when user set argument (ex.Variable(np.array(3)), Variable class save the value (or vector))
        self.grad=None 
        self.name=name
        self.creator=None #grad and creator will be defined lator.
        self.generation=0 #At first layer, generation is 0.

    __array_priority__ = 200 #

    def set_creator(self,func): #each variable could have creator that is function.
        self.creator=func #This define creator of the variable. This method is called in "Function class". So, func is that Function. 
        self.generation=func.generation+1 #each variable has generation and that is creator's generation + 1.

    def backward(self, retain_grad=False, create_graph=False): 
        if self.grad is None: #backprobagation starts from final layer. and derivative of the value itself is 1.
            self.grad = Variable(np.ones_like(self.data))
        
        funcs=[] #this list will have function of each steps. and will not have all functions used in the flow at the same time.
        seen_set=set()
        def add_func(f): #get function and append the function to funcs list and seen_set(i.e., function set)
            if f not in seen_set:
                funcs.append(f)
                seen_set.add(f)
                funcs.sort(key=lambda x: x.generation) #sort of funcs list by their generation. this is possible because funcs has function class which have multiple internal state such as generation.
        add_func(self.creator) #add creator of this variable (if not this variable first input, this is not None)

        while funcs: #while len(funcs) > 0. In this loop, all derivative is calculated finally.
            f=funcs.pop() #take function at the particular node. first function is added above.
            gys=[output().grad for output in f.outputs]  #this weired output() is related to weakref. Meaning is same(=output) 
            
            with using_config('enable_backprop', create_graph): #using_config -> Config class (in __call__ method) (see p235)
                gxs=f.backward(*gys) #main backward process
                if not isinstance(gxs, tuple): 
                    gxs=(gxs,)

                for x, gx in zip(f.inputs, gxs): #using result of derivative (=gxs), savign of the derivative for the variable(f.inputs)
                    if x.grad is None:
                        x.grad=gx
                    else:
                        x.grad = x.grad + gx #this code is for derivative of add note (see p107)

                    if x.creator is not None: #if node is not first (i.e., all variable without first input has that creator)
                        add_func(x.creator) #add function to funcs list and sort them (see above).
                        #sort only has meanings when function is like add

            if not retain_grad:
                for y in f.outputs:
                    y().grad = None #y has () because y is weakref
                    #Because user is interested in input variable which is truly target of derivative,
                    #all f.outputs is ok to be removed after derivative calculation (position of this code is important)

    def cleargrad(self):
        self.grad=None #initialize self.grad when you wanna use same variable multiple times.

    @property
    def shape(self): #@property make it possible to output like x.shape (not x.shape())
        return self.data.shape

    @property
    def ndim(self): 
        return self.data.ndim
    
    @property
    def size(self):
        return self.data.size

    @property
    def dtype(self):
        return self.data.dtype

    def __len__(self): #special method with __func__ related to len function
        return len(self.data)

    def __repr__(self): #special method with __func__ related to print function
        if self.data is None:
            return 'variable(None)'
        p=str(self.data).replace('\n', '\n'+' '*9)
        return 'variable('+p+')'

    def __mul__(self,other): #this method should be in Variable class (not outside of this class)
        return mul(self,other) #this mul method is defined outside of this class (we defined this)

def as_array(x): #util functon used in Function class
    if np.isscalar(x):
        return np.array(x)
    return x


class Function: #this class is assumed to be succeeded
    def __call__(self, *inputs): #__call__ function gives method like usage of Function class. 
        #when coding Function(*inputs), this __call__ method works. 
        
        inputs=[as_variable(x) for x in inputs]
        
        xs=[x.data for x in inputs]  #assuming inputs is Variable class instance and multiple argument could be passed (ex. add function)
        ys=self.forward(*xs) #forward calculation is coded in succeeding class (ex. Square)
        if not isinstance(ys, tuple): #ys (output of forward calculation) must be tuple
            ys=(ys,)
        outputs=[Variable(as_array(y)) for y in ys] #outputs could be multiple and they must be list of array
        
        if Config.enable_backprop: #when doing derivative, code below will run. This code is because of memory problem
            self.generation=max([x.generation for x in inputs]) #as of Variable class, Function class has generation defined using variable class instance (input)
            for output in outputs: #output has creator (function)
                output.set_creator(self) #this set creator of Variable function (Function class does not have creator off course)
            self.inputs=inputs #function has inputs
            self.outputs=[weakref.ref(output) for output in outputs] #and outputs
        return outputs if len(outputs) > 1 else outputs[0] #return outputs (or scalar)

    def forward(self,x): #succeeding is needed
        raise NotImplementedError()
    
    def backward(self,gy): #likewise
        raise NotImplementedError()



class Add(Function):
    def forward(self, x0, x1):
        y=x0+x1
        return (y,)
    
    def backward(self,gy):
        return gy,gy

class Mul(Function):
    def forward(self, x0, x1):
        y=x0*x1
        return y
    
    def backward(self,gy):
        #x0,x1=self.inputs[0].data, self.inputs[1].data 
        x0,x1=self.inputs #directly refer Variable instances (until now refereded np.array instances like above)
        return gy*x1, gy*x0

class Neg(Function):
    def forward(self,x):
        return -x
    
    def backward(self,gy):
        return -gy

class Sub(Function):
    def forward(self, x0, x1):
        y=x0-x1
        return y
    
    def backward(self, gy):
        return gy, -gy

class Div(Function):
    def forward(self,x0,x1):
        y=x0/x1
        return y

    def backward(self,gy):
        #x0,x1=self.inputs[0].data, self.inputs[1].data
        x0,x1=self.inputs
        gx0=gy/x1
        gx1=gy*(-x0/x1**2)
        return gx0,gx1

class Pow(Function):
    def __init__(self,c):
        self.c=c

    def forward(self,x):
        y=x**self.c
        return y
    
    def backward(self,gy):
        #x=self.inputs[0].data
        x,=self.inputs
        c=self.c
        gx=c*x**(c-1)*gy
        return gx

def add(x0,x1):
    x1=as_array(x1)
    return Add()(x0,x1)

def mul(x0,x1):
    x1=as_array(x1)
    return Mul()(x0,x1)

def neg(x):
    return Neg()(x)

def sub(x0,x1):
    x1=as_array(x1)
    return Sub()(x0,x1)

def rsub(x0,x1):
    x1=as_array(x1)
    return Sub()(x1,x0)

def div(x0,x1):
    x1=as_array(x1)
    return Div()(x0,x1)

def rdiv(x0,x1):
    x1=as_array(x1)
    return Div()(x1,x0)

def pow(x,c):
    return Pow(c)(x)

#whether doing derivative or not
class Config:
    enable_backprop = True

@contextlib.contextmanager 
#this decolator make it enable that first preprocess (in this case, #1 and 2, before yield) 
#next other process (defined by user)
#finally final process (#3)
def using_config(name, value):
    old_value=getattr(Config, name) #1
    setattr(Config, name, value) #2
    try:
        yield
    finally:
        setattr(Config, name, old_value) #3

def no_grad(): #util function. Coding "with using_config~~~" is wasteful. (see p130)
    return using_config('enable_backprop', False)

def as_variable(obj): #util function converting ndarray to Variable instance
    if isinstance(obj, Variable):
        return obj
    return Variable(obj)

def setup_variable():
    Variable.__mul__=mul
    Variable.__rmul__=mul
    Variable.__add__=add
    Variable.__radd__=add
    Variable.__neg__=neg
    Variable.__sub__=sub
    Variable.__rsub__=rsub
    Variable.__truediv__=div
    Variable.__rtruediv__=div
    Variable.__pow__=pow