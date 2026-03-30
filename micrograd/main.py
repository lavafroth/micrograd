import marimo

__generated_with = "0.21.1"
app = marimo.App()

with app.setup:
    import marimo as mo
    import math
    from enum import Enum
    import matplotlib.pyplot as plt
    import numpy as np
    from dataclasses import dataclass, field
    from itertools import count
    from typing import Callable
    from mermaid_builder.flowchart import Chart, Node, ChartDir


@app.function
def f(x: float) -> float:
    return 4 * x**2 - 5 * x + 9


@app.cell
def _():
    f(2.5)
    return


@app.cell
def _():
    _xs = np.arange(-4, 4, 0.2)
    _ys = f(_xs)
    plt.plot(_xs, _ys)
    return


@app.function
def derivative(x = 2.5):
    h = 1e-3
    print(f"{f(x)=}")
    print(f"{f(x + h)=:0.4f}")
    print(f"{f(x + h) - f(x)=:0.4f}")
    print(f"{(f(x + h) - f(x)) / h=:0.4f}")

    print(f"{8 * x - 5 = }")


@app.cell
def _():
    derivative()
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    Finding the critical point in my head and plugging in the value of x
    """)
    return


@app.cell
def _():
    derivative(5/8)
    return


@app.cell
def _():
    a = 2.0
    b = -3.0
    c = 11.0
    d = a + b * c
    d
    return


@app.cell
def _():
    h = 0.001
    a_1 = 2.0
    b_1 = -3.0
    c_1 = 11.0
    def d_1(a, b, c):
        return a * b + c

    def partial_derivative(for_axis=0):
        params = [a_1, b_1, c_1]
        params_nudged = params.copy()
        params_nudged[for_axis] = params_nudged[for_axis] + h
        print(f'd(*params) = {d_1(*params)!r}')
        print(f'd(*params_nudged) = {d_1(*params_nudged)!r}')
        print(f'slope (d(*params_nudged) - d(*params)) / h = {(d_1(*params_nudged) - d_1(*params)) / h:0.03f}')

    return (partial_derivative,)


@app.cell
def _(partial_derivative):
    partial_derivative()
    return


@app.cell
def _(partial_derivative):
    partial_derivative(for_axis=1)
    return


@app.cell
def _(partial_derivative):
    partial_derivative(for_axis=2)
    return


@app.class_definition
class Op(Enum):
    Leaf = ''
    Add = '+'
    Mul = '*'
    Tanh = "tanh"
    Exp = "exp"
    Pow = "pow"


@app.class_definition
@dataclass
class Value:
    identifier: int = field(default_factory=count().__next__, init=False)
    data: float
    children: tuple = ()
    op: Op = Op.Leaf
    grad: float = 0.0
    label: str = ''

    def __repr__(self) -> str:
        return f"Value({self.data})"

    def gather_children(self, other) -> tuple['Value']:
        if self == other:
            return (self,)
        return (self, other)
    def __add__(self, other: 'Value') -> 'Value':
        return Value(self.data + other.data, self.gather_children(other), Op.Add)

    def __mul__(self, other: 'Value') -> 'Value':
        return Value(self.data * other.data, self.gather_children(other), Op.Mul)

    def __hash__(self):
        return self.identifier

    def tanh(self):
        x = self.data
        o = ((math.exp(2*x) - 1)/(math.exp(2*x) + 1))
        return Value(o, (self,), Op.Tanh)

    def set_label(self, label: str) -> 'Value':
        self.label = label
        return self


@app.cell
def _():
    a_2 = Value(2.0, label='a')
    b_2 = Value(-5.0, label='b')
    c_2 = Value(-3.0, label='c')
    e = (a_2 * b_2).set_label('e')
    d_2 = (e + c_2).set_label('d')
    f_1 = Value(-2.0, label='f')
    L = (d_2 * f_1).set_label('L')
    (d_2.children, d_2.op)
    return L, a_2, b_2, c_2, f_1


@app.function
def trace(root):
    nodes, edges = set(), set()
    stack = [root]

    while stack:
        v = stack.pop()
        if v in nodes:
            continue
        nodes.add(v)
        stack.extend(v.children)
        for child in v.children:
            edges.add((child, v))
    return nodes, edges


@app.function
def draw_dot(root):
    G = Chart("computational graph", direction=ChartDir.LR)
    nodes, edges = trace(root)
    for n in nodes:
        uid = str(id(n))
        G.add_node(Node(f"{n.label}\ndata {n.data:0.4f}\ngrad {n.grad:0.4f}", id=uid))
        if n.op != Op.Leaf:
            op = n.op.value
            G.add_node(Node(op, id=uid + op))
            G.add_link_between(uid + op, uid)

    for n1, n2 in edges:
        G.add_link_between(str(id(n1)), str(id(n2)) + n2.op.value)

    return mo.mermaid(str(G))


@app.cell
def _(L):
    draw_dot(L)
    return


@app.cell
def _(a_2, b_2, c_2, f_1):
    lr = 0.01
    a_2.data = a_2.data + lr * a_2.grad
    b_2.data = b_2.data + lr * b_2.grad
    c_2.data = c_2.data + lr * c_2.grad
    f_1.data = f_1.data + lr * f_1.grad
    e_1 = (a_2 * b_2).set_label('e')
    d_3 = (e_1 + c_2).set_label('d')
    L3 = (d_3 * f_1).set_label('L')
    print(L3)
    return d_3, e_1


@app.cell
def _():
    def _():
        h = 1e-3
        a = Value(2.0, label='a')
        b = Value(-5.0, label='b')
        c = Value(-3.0, label='c')
        e = (a * b).set_label('e')
        d = (e + c).set_label('d')
        f = Value(-2.0, label="f")
        L1 = (d * f).set_label('L')

        a = Value(2.0, label='a')
        b = Value(-5.0 + h, label='b')
        c = Value(-3.0, label='c')


        # c.data += h

        e = (a * b).set_label('e')

        # e.data += h
        d = (e + c).set_label('d')
        f = Value(-2.0, label="f")    
        L2 = (d * f).set_label('L')

        print((L2.data - L1.data)/h)

    _()
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # Manual backpropagation
    """)
    return


@app.cell
def _(L):
    L.grad = 1.0
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    $$
    L = d \times f
    $$

    $$
    \frac{dL}{dd} = f
    $$
    """)
    return


@app.cell
def _(d_3, f_1):
    d_3.grad = -2.0
    f_1.grad = -13.0
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    $$
    \frac{dL}{dc} = \frac{dL}{dd} \times \frac{dd}{dc}
    $$

    Given $d = c + e$,

    $$
    \frac{dd}{dc} = 1
    $$

    and $\frac{dL}{dd} = f$

    $\implies \frac{dL}{dc} = f$
    """)
    return


@app.cell
def _(c_2, e_1):
    c_2.grad = -2.0
    e_1.grad = -2.0
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    $$
    \frac{dL}{da} = \frac{dL}{dd} \times \frac{dd}{de} \times \frac{dd}{da}
    $$
    """)
    return


@app.cell
def _(a_2, b_2, e_1):
    a_2.grad = -5.0 * e_1.grad
    b_2.grad = 2.0 * e_1.grad
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # Backpropagating through a neuron
    """)
    return


@app.cell
def _():
    _xs = np.arange(-5, 5, 0.25)
    _ys = np.tanh(_xs)
    plt.plot(_xs, _ys)
    plt.grid()
    return


@app.cell
def _():
    # inputs
    x1 = Value(2.0, label='x1')
    x2 = Value(5.0, label='x2')
    w1 = Value(-4.2, label='w1')
    # weights
    w2 = Value(3.0, label='w2')
    b_3 = Value(-5.725, label='b')
    x1w1 = (x1 * w1).set_label('x1 * w1')
    # bias
    x2w2 = (x2 * w2).set_label('x2 * w2')
    x1w1x2w2 = (x1w1 + x2w2).set_label('x1 * w1 + x2 * w2')
    # dot product
    n = (x1w1x2w2 + b_3).set_label('n')
    # add the bias
    o = n.tanh().set_label('o')
    return b_3, n, o, w1, w2, x1, x1w1, x1w1x2w2, x2, x2w2


@app.cell
def _(o):
    draw_dot(o)
    return


@app.cell
def _(o):
    o.grad = 1.0
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    $$
    \begin{align}
    o &= tanh(n) \\
    &= \frac{e^{2n} - 1}{e^{2n} + 1} \\
    \frac{do}{dn} &= \frac{(e^{2n} + 1) (2e^{2n}) - (e^{2n} - 1) (2e^{2n})}{(e^{2n} + 1)^2} \\
    &= \frac{2e^{2n} \times ((e^{2n} + 1) - (e^{2n} - 1))}{(e^{2n} + 1)^2} \\
    &= \frac{4e^{2n}}{(e^{2n} + 1)^2} \\
    &= (\frac{2e^n}{e^{2n} + 1})^2 \\
    &= sech^2(n) \\
    &= 1 - tanh^2(n) \\
    &= 1 - o^2 \\
    \end{align}
    $$
    """)
    return


@app.cell
def _(b_3, n, o, x1w1x2w2):
    n.grad = 1 - o.data ** 2
    b_3.grad = n.grad
    x1w1x2w2.grad = n.grad
    return


@app.cell
def _(x1w1, x1w1x2w2, x2w2):
    x1w1.grad = x1w1x2w2.grad
    x2w2.grad = x1w1x2w2.grad
    return


@app.cell
def _(w1, w2, x1, x1w1, x2, x2w2):
    x2.grad = w2.data * x2w2.grad
    w2.grad = x2.data * x2w2.grad

    x1.grad = w1.data * x1w1.grad
    w1.grad = x1.data * x1w1.grad
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # Autograd (yay)
    """)
    return


@app.class_definition
@dataclass
class Value_1:
    identifier: int = field(default_factory=count().__next__, init=False)
    data: float
    children: tuple = ()
    op: Op = Op.Leaf
    grad: float = 0.0
    backward: Callable = lambda: None
    label: str = ''

    def __repr__(self) -> str:
        return f'Value_1({self.data})'

    def gather_children(self, other) -> tuple['Value']:
        if self == other:
            return (self,)
        return (self, other)

    def __add__(self, other: 'Value') -> 'Value':
        out = Value_1(self.data + other.data, self.gather_children(other), Op.Add)
  # * 1.0
        def closure_backward():  # * 1.0
            self.grad = out.grad
            other.grad = out.grad
        out.backward = closure_backward
        return out

    def __mul__(self, other: 'Value') -> 'Value':
        out = Value_1(self.data * other.data, self.gather_children(other), Op.Mul)

        def closure_backward():
            self.grad = out.grad * other.data
            other.grad = out.grad * self.data
        out.backward = closure_backward
        return out

    def __hash__(self):
        return self.identifier

    def tanh(self):
        x = self.data
        tanh = (math.exp(2 * x) - 1) / (math.exp(2 * x) + 1)
        out = Value_1(tanh, (self,), Op.Tanh)

        def closure_backward():
            self.grad = (1 - tanh ** 2) * out.grad
        out.backward = closure_backward
        return out

    def set_label(self, label: str) -> 'Value':
        self.label = label
        return self


@app.cell
def _():
    # inputs
    x1_1 = Value_1(2.0, label='x1')
    x2_1 = Value_1(5.0, label='x2')
    w1_1 = Value_1(-4.2, label='w1')
    # weights
    w2_1 = Value_1(3.0, label='w2')
    b_4 = Value_1(-5.725, label='b')
    x1w1_1 = (x1_1 * w1_1).set_label('x1 * w1')
    # bias
    x2w2_1 = (x2_1 * w2_1).set_label('x2 * w2')
    x1w1x2w2_1 = (x1w1_1 + x2w2_1).set_label('x1 * w1 + x2 * w2')
    # dot product
    n_1 = (x1w1x2w2_1 + b_4).set_label('n')
    # add the bias
    o_1 = n_1.tanh().set_label('o')
    return n_1, o_1, x1w1_1, x1w1x2w2_1, x2w2_1


@app.cell
def _(o_1):
    draw_dot(o_1)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Try it one last time by hand
    """)
    return


@app.cell
def _(n_1, o_1, x1w1_1, x1w1x2w2_1, x2w2_1):
    o_1.grad = 1.0
    o_1.backward()
    n_1.backward()
    x1w1x2w2_1.backward()
    x2w2_1.backward()
    x1w1_1.backward()
    return


@app.cell
def _(o_1):
    draw_dot(o_1)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Automate it
    """)
    return


@app.function
def topsort(root: Value_1) -> list[Value_1]:
    top = [root]
    visited = set()
    stack = [root]
    while stack:
        v = stack.pop()
        if v in visited:
            continue
        for child in v.children:
            stack.append(child)
            top.append(child)
        visited.add(v)
    return top


@app.cell
def _(o_1):
    topsort(o_1)
    return


@app.cell
def _(o_1):
    o_1.grad = 1.0
    for node in topsort(o_1):
        node.backward()
    return


@app.cell
def _(o_1):
    draw_dot(o_1)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # Putting it together
    """)
    return


@app.class_definition
@dataclass
class Value_2:
    identifier: int = field(default_factory=count().__next__, init=False)
    data: float
    children: tuple = ()
    op: Op = Op.Leaf
    grad: float = 0.0
    backward: Callable = lambda: None
    label: str = ''

    @staticmethod
    def cast(other):
        if isinstance(other, Value_2):
            return other
        return Value_2(other)

    def __repr__(self) -> str:
        return f'Value_2({self.data})'

    def gather_children(self, other) -> tuple['Value']:
        if self == other:
            return (self,)
        return (self, other)

    def __add__(self, other: 'Value') -> 'Value':
        other = Value_2.cast(other)
        out = Value_2(self.data + other.data, self.gather_children(other), Op.Add)

        def closure_backward():  # * 1.0
            self.grad = self.grad + out.grad  # * 1.0
            other.grad = other.grad + out.grad
        out.backward = closure_backward
        return out

    def __truediv__(self, other) -> 'Value':
        return self * other ** (-1)

    def __rmul__(self, other):
        return self * other

    def __radd__(self, other):
        return self + other

    def __rsub__(self, other):
        return self - other

    def __neg__(self):
        return self * -1

    def __sub__(self, other):
        return self + -other

    def __pow__(self, other):
        assert isinstance(other, (int, float)), 'only supporting float and int exponents for now'
        out = Value_2(self.data ** other, (self,), Op.Pow)

        def closure_backward():
            self.grad = self.grad + other * self.data ** (other - 1) * out.grad
        out.backward = closure_backward
        return out

    def __mul__(self, other: 'Value') -> 'Value':
        other = Value_2.cast(other)
        out = Value_2(self.data * other.data, self.gather_children(other), Op.Mul)

        def closure_backward():
            self.grad = self.grad + out.grad * other.data
            other.grad = other.grad + out.grad * self.data
        out.backward = closure_backward
        return out

    def __hash__(self):
        return self.identifier

    def tanh(self) -> 'Value':
        x = self.data
        import math
        tanh = (math.exp(2 * x) - 1) / (math.exp(2 * x) + 1)
        out = Value_2(tanh, (self,), Op.Tanh)

        def closure_backward():
            self.grad = self.grad + (1 - tanh ** 2) * out.grad
        out.backward = closure_backward
        return out

    def exp(self):
        x = self.data
        out = Value_2(math.exp(x), (self,), Op.Exp)

        def closure_backward():
            self.grad = self.grad + out.data * out.grad
        out.backward = closure_backward
        return out

    def backpropagate(self):
        self.grad = 1.0
        self.backward()
        visited = set()
        stack = [self]
        while stack:
            v = stack.pop()
            if v in visited:
                continue
            for child in v.children:
                stack.append(child)
                child.backward()
            visited.add(v)

    def set_label(self, label: str) -> 'Value':
        self.label = label
        return self


@app.cell
def _():
    a_3 = Value_2(2.0)
    a_3.exp()
    return


@app.cell
def _():
    x1_2 = Value_2(2.0, label='x1')
    x2_2 = Value_2(5.0, label='x2')
    w1_2 = Value_2(-4.2, label='w1')
    # weights
    w2_2 = Value_2(3.0, label='w2')
    b_5 = Value_2(-5.725, label='b')
    x1w1_2 = (x1_2 * w1_2).set_label('x1 * w1')
    # bias
    x2w2_2 = (x2_2 * w2_2).set_label('x2 * w2')
    x1w1x2w2_2 = (x1w1_2 + x2w2_2).set_label('x1 * w1 + x2 * w2')
    # dot product
    n_2 = (x1w1x2w2_2 + b_5).set_label('n')
    # add the bias
    o_2 = n_2.tanh().set_label('o')
    return (o_2,)


@app.cell
def _(o_2):
    draw_dot(o_2)
    return


@app.cell
def _(o_2):
    o_2.backpropagate()
    draw_dot(o_2)
    return


@app.cell
def _():
    a_4 = Value_2(3.0, label='a')
    b_6 = (a_4 + a_4).set_label('b')
    b_6.backpropagate()
    draw_dot(b_6)
    return (a_4,)


@app.cell
def _(a_4):
    a_4 + 1
    return


@app.cell
def _(a_4):
    2 * a_4
    return


@app.cell
def _():
    # inputs
    x1_3 = Value_2(2.0, label='x1')
    x2_3 = Value_2(5.0, label='x2')
    w1_3 = Value_2(-4.2, label='w1')
    # weights
    w2_3 = Value_2(3.0, label='w2')
    b_7 = Value_2(-5.725, label='b')
    x1w1_3 = (x1_3 * w1_3).set_label('x1 * w1')
    # bias
    x2w2_3 = (x2_3 * w2_3).set_label('x2 * w2')
    x1w1x2w2_3 = (x1w1_3 + x2w2_3).set_label('x1 * w1 + x2 * w2')
    # dot product
    n_3 = (x1w1x2w2_3 + b_7).set_label('n')
    two_n_exp = (2 * n_3).exp()
    # add the bias
    o_3 = ((two_n_exp - 1) / (two_n_exp + 1)).set_label('o')
    return (o_3,)


@app.cell
def _(o_3):
    o_3.backpropagate()
    return


@app.cell
def _(o_3):
    draw_dot(o_3)
    return


@app.cell
def _():
    import torch

    return (torch,)


@app.cell
def _(torch):
    def _():
        x1 = torch.tensor([2.0], requires_grad=True, dtype=torch.double)
        x2 = torch.tensor([5.0], requires_grad=True, dtype=torch.double)
        w1 = torch.tensor([-4.2], requires_grad=True, dtype=torch.double)
        # weights
        w2 = torch.tensor([3.0], requires_grad=True, dtype=torch.double)
        b = torch.tensor([-5.725], requires_grad=True, dtype=torch.double)
        x1w1 = x1 * w1

        x2w2 = x2 * w2
        x1w1x2w2 = x1w1 + x2w2

        n = x1w1x2w2 + b
        o = n.tanh()

        print('o', o.data.item())
        print('o', o.item())
        o.backward()

        print('gradients:')

        print('x1', x1.grad.item())
        print('x2', x2.grad.item())
        print('w1', w1.grad.item())
        print('w2', w2.grad.item())


    _()
    return


@app.cell
def _(torch):
    print(torch.tensor([2.0]).dtype)
    print(torch.tensor([2.0]).double().dtype)
    return


@app.cell
def _():
    def _():
        xs = np.arange(0, 1, 0.1)
        ys = 3*(xs**2) - 4*xs + 5
        return plt.plot(xs, ys)

    _()
    return


@app.class_definition
class Neuron:
    def __init__(self, in_size):
        import random
        self.w = tuple(Value_2(random.uniform(-1, 1)) for _ in range(in_size))
        self.b = Value_2(random.uniform(-1, 1))

    def __call__(self, x):
        return sum((w*v for w, v in zip(self.w, x)), self.b).tanh()

    def parameters(self):
        return self.w + (self.b,)


@app.cell
def _():
    def _():
        x = [2.0, 3.0]
        n = Neuron(2)
        return n(x)
    _()
    return


@app.class_definition
class Layer:
    def __init__(self, in_size, out_size):
        self.neurons = tuple(Neuron(in_size) for _ in range(out_size))
    def __call__(self, x):
        out = [n(x) for n in self.neurons]

        if len(out) == 1:
            return out[0]
        return out

    def parameters(self):
        return sum((l.parameters() for l in self.neurons), tuple())


@app.cell
def _():
    def _():
        x = [2.0, 3.0]
        n = Layer(2, 3)
        return n(x)

    _()
    return


@app.class_definition
class MLP:
    def __init__(self, in_size: int, out_sizes: list[int]):
        sizes = [in_size] + out_sizes
        self.layers = [Layer(sizes[i], sizes[i+1]) for i in range(len(out_sizes))]

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    def parameters(self):
        return sum((l.parameters() for l in self.layers), tuple())


@app.cell
def _():
    net_0 = MLP(3, [3, 4, 1])
    def _():
        x = [2.0, 3.0, 5.6, 0.1]
        forward = net_0(x)
        return draw_dot(forward)
    _()
    return (net_0,)


@app.cell
def _(net_0):
    from types import SimpleNamespace
    def train_setup():

        xs = [
            [2.0, 3.0, -1.0],
            [3.0, -1.0, 0.5],
            [0.5, 1.0, 1.0],
            [1.0, 1.0, -1.0],
        ]
        ys = [1.0, -1.0, -1.0, 1.0]
        ypred = [net_0(x) for x in xs]
        howfar = [(ygt-yout)**2 for ygt, yout in zip(ys, ypred)]

        print("how far")
        for v in howfar:
            print(f"  {v}")

        loss = sum(howfar)
        print("loss", loss.data)


        loss.backpropagate()

        return SimpleNamespace(xs=xs, ys=ys)
    fn_get_loss = train_setup()
    return (fn_get_loss,)


@app.cell
def _(net_0):
    def backprop_loss(train):
        ypred = [net_0(x) for x in train.xs]
        sse = sum((y_ground-y_predicted)**2 for y_ground, y_predicted in zip(ypred, train.ys))
        print(ypred)
        print(sse.data)
        sse.backpropagate()

    return (backprop_loss,)


@app.cell
def _(net_0):
    print(net_0.layers[0].neurons[0].w[0].grad)
    params = net_0.parameters()
    print(len(params))
    return


@app.cell
def _(backprop_loss, fn_get_loss):
    backprop_loss(fn_get_loss)
    return


@app.cell
def _(net_0):
    def update_params():
        for p in net_0.parameters():
            p.data -= 0.01 * p.grad

    return (update_params,)


@app.cell
def _(update_params):
    update_params()
    return


@app.cell
def _(backprop_loss, fn_get_loss):
    backprop_loss(fn_get_loss)
    return


@app.cell
def _(net_0):
    net_0.parameters()
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Let's make the training loop
    """)
    return


@app.cell
def _():
    def _():
        net_1 = MLP(3, [3, 4, 1])
        lr = 0.01
        xs = [
                [2.0, 3.0, -1.0],
                [3.0, -1.0, 0.5],
                [0.5, 1.0, 1.0],
                [1.0, 1.0, -1.0],
            ]
        ys = [1.0, -1.0, -1.0, 1.0]

        for _ in range(100):
            y_pred = list(map(net_1, xs))
            sse = sum((ygt-yout)**2 for ygt, yout in zip(ys, y_pred))
            print(sse.data)

            for p in net_1.parameters():
                p.grad = 0.0

            sse.backpropagate()

            for p in net_1.parameters():
                p.data -= lr * p.grad

        return locals()
    loc0 = _()
    return (loc0,)


@app.cell
def _(loc0):
    loc0["y_pred"]
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
