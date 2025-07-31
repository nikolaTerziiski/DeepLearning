import graphviz as graphviz
import numpy as np


def tanh(v):
    out = Value(np.tanh(v.data))
    out._prev = {v}
    out._op = 'tanh'
    return out


class Value:

    def __init__(self, data, label=''):
        self.data = data
        self.label = label
        self.grad = 0.0
        self._prev = set()
        self._op = ''

    def __repr__(self):
        return f"Value(data={self.data})"

    def __add__(self, other):
        out = Value(self.data + other.data)
        out._prev = {self, other}
        out._op = '+'
        return out

    def __mul__(self, other):
        out = Value(self.data * other.data)
        out._prev = {self, other}
        out._op = '*'
        return out


def trace(root):
    nodes = set()
    edges = set()

    def dfs(v):
        if v in nodes: return
        nodes.add(v)
        for p in v._prev:
            edges.add((p, v))
            dfs(p)

    dfs(root)
    return nodes, edges


def draw_dot(root: Value) -> graphviz.Digraph:
    dot = graphviz.Digraph(filename='01_result',
                           format='svg',
                           graph_attr={'rankdir': 'LR'})  # LR = left to right

    nodes, edges = trace(root)

    for n in nodes:
        uid = str(id(n))
        # for any value in the graph, create a rectangular ('record') node
        dot.node(
            name=uid,
            label=f'{{ {n.label} | data: {n.data:.4f} | grad: {n.grad:.4f}}}',
            shape='record')
        if n._op:
            # if this value is a result of some operation, create an "op" node for the operation
            dot.node(name=uid + n._op, label=n._op)
            # and connect this node to the node of the operation
            dot.edge(uid + n._op, uid)

    for n1, n2 in edges:
        # connect n1 to the "op" node of n2
        dot.edge(str(id(n1)), str(id(n2)) + n2._op)

    return dot


def main() -> None:
    x1 = Value(2.0, 'x1')
    w1 = Value(-3.0, 'w1')
    x2 = Value(0.0, 'x2')
    w2 = Value(1.0, 'w2')

    x2w2 = x2 * w2
    x2w2.label = 'x2w2'
    x1w1 = x1 * w1
    x1w1.label = 'x1w1'

    x1w1x2w2 = x1w1 + x2w2
    x1w1x2w2.label = 'x1w1 + x2w2'

    bias = Value(6.8813735870195432, 'bias')

    logit = x1w1x2w2 + bias
    logit.label = 'logit'

    out = tanh(logit)
    out.label = 'L'

    #Manual calculation for the gradient
    out.grad = 1.0
    logit.grad = 1.0 - out.data**2
    bias.grad = 1 * logit.grad
    x1w1x2w2.grad = 1 * logit.grad
    x1w1.grad = x1w1x2w2.grad * 1
    x2w2.grad = x1w1x2w2.grad * 1

    w1.grad = x1w1.grad * x1.data
    x1.grad = x1w1.grad * w1.data
    w2.grad = x2w2.grad * x2.data
    x2.grad = x2w2.grad * w2.data
    # This will create a new directory and store the output file there.
    # With "view=True" it'll automatically display the saved file.
    draw_dot(out).render(directory='./graphviz_output', view=True)


if __name__ == "__main__":
    main()
