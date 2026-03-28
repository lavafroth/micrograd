import marimo

__generated_with = "0.21.1"
app = marimo.App()

with app.setup:
    import marimo as mo
    import math


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # micrograd exercises

    1. watch the [micrograd video](https://www.youtube.com/watch?v=VMj-3S1tku0) on YouTube
    2. come back and complete these exercises to level up :)
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## section 1: derivatives
    """)
    return


@app.cell
def _():
    # here is a mathematical expression that takes 3 inputs and produces one output
    from math import sin, cos

    def f(a, b, c):
      return -a**3 + sin(3*b) - 1.0/c + b**2.5 - a**0.5

    print(f(2, 3, 4))
    return cos, f


@app.cell
def _(cos):
    # write the function df that returns the analytical gradient of f
    # i.e. use your skills from calculus to take the derivative, then implement the formula
    # if you do not calculus then feel free to ask wolframalpha, e.g.:
    # https://www.wolframalpha.com/input?i=d%2Fda%28sin%283*a%29%29%29
    def gradf(a, b, c):
        df_da = -3*(a**2) - 0.5 * a**(-0.5)
        df_db = 2.5 * (b**1.5) + cos(3*b) * 3
        df_dc =  - (-1) * c**-2
        return [df_da, df_db, df_dc]

    ans = [-12.353553390593273, 10.25699027111255, 0.0625]  # todo, return [df/da, df/db, df/dc]
    yours = gradf(2, 3, 4)
    # expected answer is the list of
    for _dim in range(3):
        _ok = 'OK' if abs(yours[_dim] - ans[_dim]) < 1e-05 else 'WRONG!'
        print(f'{_ok} for dim {_dim}: expected {ans[_dim]}, yours returns {yours[_dim]}')
    return (ans,)


@app.cell
def _(ans, f):
    # now estimate the gradient numerically without any calculus, using
    # the approximation we used in the video.
    # you should not call the function df from the last cell
    params = [2, 3, 4]

    h = 1e-6


    def nudge_slightly(params: list[float], index: int, h: float) -> list[float]:
        params_nudged = params.copy()
        params_nudged[index] = params[index] + h
        return params_nudged


    numerical_grad = [
        (f(*nudge_slightly(params, index, h)) - f(*params)) / h
        for index, param in enumerate(params)
    ]

    # -----------
    for _dim in range(3):
        _ok = "OK" if abs(numerical_grad[_dim] - ans[_dim]) < 1e-05 else "WRONG!"
        print(
            f"{_ok} for dim {_dim}: expected {ans[_dim]}, yours returns {numerical_grad[_dim]}"
        )
    return h, nudge_slightly, params


@app.cell
def _(ans, f, h, nudge_slightly, params):
    # there is an alternative formula that provides a much better numerical
    # approximation to the derivative of a function.
    # learn about it here: https://en.wikipedia.org/wiki/Symmetric_derivative
    # implement it. confirm that for the same step size h this version gives a
    # better approximation.
    numerical_grad2 = [
        (
            f(*nudge_slightly(params, index, h))
            - f(*nudge_slightly(params, index, -h))
        )
        / (2 * h)
        for index in range(len(params))
    ]

    # -----------
    for _dim in range(3):
        _ok = "OK" if abs(numerical_grad2[_dim] - ans[_dim]) < 1e-05 else "WRONG!"
        print(
            f"{_ok} for dim {_dim}: expected {ans[_dim]}, yours returns {numerical_grad2[_dim]}"
        )
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## section 2: support for softmax
    """)
    return


@app.cell
def _():
    # Value class starter code, with many functions taken out
    from math import exp, log


    class Value:
        def __init__(self, data, _children=(), _op="", label=""):
            self.data = data
            self.grad = 0.0
            self._backward = lambda: None
            self._prev = set(_children)
            self._op = _op
            self.label = label

        def __repr__(self):
            return f"Value(data={self.data})"

        @staticmethod
        def cast(other):
            return other if isinstance(other, Value) else Value(other)

        def __radd__(self, other):
            return self + other

        def __add__(self, other):  # exactly as in the video
            other = Value.cast(other)
            out = Value(self.data + other.data, (self, other), "+")

            def _backward():
                self.grad = self.grad + 1.0 * out.grad
                other.grad = other.grad + 1.0 * out.grad

            out._backward = _backward
            return out

        def exp(self):
            out = Value(math.exp(self.data), (self,), "exp")

            def _backward():
                self.grad += out.grad * out.data

            out._backward = _backward
            return out

        def log(self):
            out = Value(math.log(self.data), (self,), "log")
            def _backward():
                self.grad += out.grad / self.data
            out._backward = _backward
            return out

        def __pow__(self, other):
            assert isinstance(other, (int, float)), (
                "supporting int and float scalars for now"
            )
            out = Value(self.data**other, (self,), f"**{other}")

            def _backward():
                self.grad += out.grad * other * self.data ** (other - 1)

            out._backward = _backward
            return out

        def __truediv__(self, other):
            return self * other**-1

        def __neg__(self):
            return self * -1.0

        def __sub__(self, other):
            return other + (-self)

        def __mul__(self, other):
            other = Value.cast(other)
            out = Value(self.data * other.data, (self, other), "*")
            def _backward():
                self.grad += out.grad * other.data
                other.grad += out.grad * self.data
            out._backward = _backward
            return out

        def backward(self):

            topo = []  # ------
            visited = set()

            # your code here
            def build_topo(v):  # TODO
                if v not in visited:  # ------
                    visited.add(v)
                    for child in v._prev:  # exactly as in video
                        build_topo(child)
                    topo.append(v)

            build_topo(self)
            # print(topo[:5])
            self.grad = 1.0
            for node in reversed(topo):
                node._backward()

    return (Value,)


@app.cell
def _(Value):
    def softmax(logits):
        counts = [logit.exp() for logit in logits]
        denominator = sum(counts)
        out = [c / denominator for c in counts]
        return out
    logits = [Value(0.0), Value(3.0), Value(-2.0), Value(1.0)]
    probs = softmax(logits)
    loss = -probs[3].log()
    loss.backward()
    print(loss.data)
    ans_1 = [0.041772570515350445, 0.8390245074625319, 0.005653302662216329, -0.8864503806400986]
    for _dim in range(4):
        _ok = 'OK' if abs(logits[_dim].grad - ans_1[_dim]) < 1e-05 else 'WRONG!'
        print(f'{_ok} for dim {_dim}: expected {ans_1[_dim]}, yours returns {logits[_dim].grad}')
    return


@app.cell
def _():
    # verify the gradient using the torch library
    # torch should give you the exact same gradient
    import torch

    return (torch,)


@app.cell
def _(torch):
    def _():
        logits = torch.tensor([0.0, 3.0, -2.0, 1.0], dtype=torch.double, requires_grad=True)
        probs = torch.softmax(logits, dim=0)
        true_class = 3
        ce_loss = -probs[true_class].log()
        print(ce_loss)

        ce_loss.backward()
        print(logits.grad)
    _()
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
