#This is master file
import numpy as np
import weakref
import contextlib

#class variable has a variable (ex. np.array(2)).
#Variable class will be called for each variable (x=Variable(np.array(2), y=Variable(np.array(3))))
class Variable:
    def __init__(self,data):
        if data is not None:
            if not isinstance(data, np.ndarray): #data (argument of variable class) must be np.ndarray
                raise TypeError('{} is not supported'.format(type(data)))

        self.data=data #when user set argument (ex.Variable(np.array(3)), Variable class save the value (or vector))
        self.grad=None 
        self.creator=None #grad and creator will be defined lator.
        self.generation=0 #At first layer, generation is 0.

    def set_creator(self,func): #each variable could have creator that is function.
        self.creator=func #This define creator of the variable. This method is called in "Function class". So, func is that Function. 
        self.generation=func.generation+1 #each variable has generation and that is creator's generation + 1.

    #derivative is calculated for "each variable" (not for function). 
    #Ex. dy/dx is derivative of "x"
    #In this sense, variable class has backward method
    def backward(self, retain_grad=False): 
        if self.grad is None: #backprobagation starts from final layer. and derivative of the value itself is 1.
            self.grad=np.ones_like(self.data)
        
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
            #function (named 'f' extracted above) has instance "outputs" defined below. 
            #Depending on function, number of outout could be multiple.
            #Because output is variable instance, that has derivative(grad).
            #Derivative of this node is calculated by gys which is previous derivative and input of next variable (see p.41)
            gxs=f.backward(*gys) 
            #above code is main part of this loop. 
            #All what is needed for derivative is gys (previous derivative and input which is stored in function and concrete procedure of the particular function)
            if not isinstance(gxs, tuple): #derivative must be tuple
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

def as_array(x): #util functon used in Function class
    if np.isscalar(x):
        return np.array(x)
    return x


class Function: #this class is assumed to be succeeded
    def __call__(self, *inputs): #__call__ function gives method like usage of Function class. 
        #when coding Function(*inputs), this __call__ method works. 
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


class Square(Function): #concrete function which is assumed to succeed Function class
    def forward(self, x):
        y=x**2
        return y
    
    def backward(self,gy):
        x=self.inputs[0].data
        gx=2*x*gy
        return gx

class Exp(Function):
    def forward(self,x):
        y=np.exp(x)
        return y
    
    def backward(self,gy):
        x=self.inputs[0].data
        gx=np.exp(x)*gy
        return gx

class Add(Function):
    def forward(self, x0, x1):
        y=x0+x1
        return (y,)
    
    def backward(self,gy):
        return gy,gy


def square(x): #util function
    f=Square()
    return f(x)

def exp(x):
    f=Exp()
    return f(x)

def add(x0,x1):
    return Add()(x0,x1)

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

