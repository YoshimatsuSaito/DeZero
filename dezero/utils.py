
import os
import subprocess
import numpy as np
from dezero import as_variable
from dezero import Variable


def _dot_var(v, verbose=False): #all variables in the model is shown from this function
    dot_var='{} [label="{}", color=orange, style=filled]\n'

    name='' if v.name is None else v.name
    if verbose and v.data is not None:
        if v.name is not None:
            name+=': '
        name+=str(v.shape) + ' ' + str(v.dtype)
    return dot_var.format(id(v), name)



def _dot_func(f): #all functions int the model is shown from this function. and all connections between variables and functions is also shown from this function.
    dot_func='{} [label="{}", color=lightblue, style=filled, shape=box]\n'
    txt=dot_func.format(id(f), f.__class__.__name__)

    dot_edge='{} -> {}\n'
    for x in f.inputs:
        txt+=dot_edge.format(id(x),id(f))
    for y in f.outputs:
        txt+=dot_edge.format(id(f),id(y())) #because y is weakref, y has ()
    return txt


def get_dot_graph(output, verbose=True):
    txt=''
    funcs=[]
    seen_set=set()

    def add_func(f):
        if f not in seen_set:
            funcs.append(f)
            seen_set.add(f)
        
    add_func(output.creator) #output.creator is function off course. and output is finall result of the model. this code starts backpropagation like backflow with while lopp below
    txt+=_dot_var(output, verbose)

    while funcs: #while len(funcs) > 0
        func=funcs.pop() #get last one function
        txt+=_dot_func(func) #add function to dot representation
        for x in func.inputs: #take inputs of the function
            txt+=_dot_var(x, verbose) #add the variable to dot representation

            if x.creator is not None: #until x.creator is None (i.e., start of the model)
                add_func(x.creator) #add function to the func list
    return 'digraph g{\n' + txt + '}'


def plot_dot_graph(output, verbose=True, to_file='graph.png'):
    dot_graph=get_dot_graph(output, verbose)

    tmp_dir=os.path.join(os.path.expanduser('~'),'.dezero')
    if not os.path.exists(tmp_dir):
        os.mkdir(tmp_dir)
    graph_path=os.path.join(tmp_dir, 'tmp_graph.dot')

    with open(graph_path,'w') as f:
        f.write(dot_graph) #just writing the txt to the file

    extenstion = os.path.splitext(to_file)[1][1:] #file type (=png, jpg and so on)
    cmd='dot {} -T {} -o {}'.format(graph_path, extenstion, to_file) 
    subprocess.run(cmd, shell=True) #run

    try:
        from IPython import display
        return display.Image(filename=to_file)
    except:
        pass

def reshape_sum_backward(gy, x_shape, axis, keepdims):
    ndim = len(x_shape)
    tupled_axis = axis
    if axis is None:
        tupled_axis = None
    elif not hasattr(axis, 'len'):
        tupled_axis = (axis,)

    if not (ndim == 0 or tupled_axis is None or keepdims):
        actual_axis = [a if a >= 0 else a + ndim for a in tupled_axis]
        shape = list(gy.shape)
        for a in sorted(actual_axis):
            shape.insert(a, 1)
    else:
        shape = gy.shape

    gy = gy.reshape(shape)  
    return gy