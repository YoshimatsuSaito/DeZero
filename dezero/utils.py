
import os
import subprocess
import numpy as np
from dezero import as_variable
from dezero import Variable
import urllib.request
from dezero import cuda


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

def sum_to(x, shape):
    ndim = len(shape)
    lead = x.ndim - ndim
    lead_axis = tuple(range(lead))

    axis = tuple([i + lead for i, sx in enumerate(shape) if sx == 1])
    y = x.sum(lead_axis + axis, keepdims=True)
    if lead > 0:
        y = y.squeeze(lead_axis)
    return y

def logsumexp(x, axis=1):
    xp = cuda.get_array_module(x)
    m = x.max(axis=axis, keepdims=True)
    y = x - m
    xp.exp(y, out=y)
    s = y.sum(axis=axis, keepdims=True)
    xp.log(s, out=s)
    m += s
    return m

def max_backward_shape(x, axis):
    if axis is None:
        axis = range(x.ndim)
    elif isinstance(axis, int):
        axis = (axis,)
    else:
        axis = axis

    shape = [s if ax not in axis else 1 for ax, s in enumerate(x.shape)]
    return shape
    

def pair(x):
    if isinstance(x, int):
        return (x, x)
    elif isinstance(x, tuple):
        assert len(x) == 2
        return x
    else:
        raise ValueError
    
def show_progress(block_num, block_size, total_size):
    bar_template = "\r[{}] {:.2f}%"

    downloaded = block_num * block_size
    p = downloaded / total_size * 100
    i = int(downloaded / total_size * 30)
    if p >= 100.0: p = 100.0
    if i >= 30: i = 30
    bar = "#" * i + "." * (30 - i)
    print(bar_template.format(bar, p), end='')


cache_dir = os.path.join(os.path.expanduser('~'), '.dezero')


def get_file(url, file_name=None):
    """Download a file from the `url` if it is not in the cache.

    The file at the `url` is downloaded to the `~/.dezero`.

    Args:
        url (str): URL of the file.
        file_name (str): Name of the file. It `None` is specified the original
            file name is used.

    Returns:
        str: Absolute path to the saved file.
    """
    if file_name is None:
        file_name = url[url.rfind('/') + 1:]
    file_path = os.path.join(cache_dir, file_name)

    if not os.path.exists(cache_dir):
        os.mkdir(cache_dir)

    if os.path.exists(file_path):
        return file_path

    print("Downloading: " + file_name)
    try:
        urllib.request.urlretrieve(url, file_path, show_progress)
    except (Exception, KeyboardInterrupt) as e:
        if os.path.exists(file_path):
            os.remove(file_path)
        raise
    print(" Done")

    return file_path


# =============================================================================
# others
# =============================================================================
def get_deconv_outsize(size, k, s, p):
    return s * (size - 1) + k - 2 * p


def get_conv_outsize(input_size, kernel_size, stride, pad):
    return (input_size + pad * 2 - kernel_size) // stride + 1

