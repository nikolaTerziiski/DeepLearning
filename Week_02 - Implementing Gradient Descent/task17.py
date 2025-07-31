import graphviz as graphviz
import numpy as np


def tanh(v):
    out = Value(np.tanh(v.data))
    out._prev = {v}
    out._op = 'tanh'

    def _backward():
        v.grad += (1 - out.data**2)

    out._backward = _backward
    return out


class Value:

    def __init__(self, data, label=''):
        self.data = data
        self.label = label
        self.grad = 0.0
        self._prev = set()
        self._op = ''
        self._backward = lambda: None

    def __repr__(self):
        return f"Value(data={self.data})"

    def __add__(self, other):
        out = Value(self.data + other.data)
        out._prev = {self, other}
        out._op = '+'

        #Because we have here "+" basically the grad is easily calculate as 1.0 * out.grad
        def _backward():
            self.grad += 1.0 * out.grad
            other.grad += 1.0 * out.grad

        out._backward = _backward

        return out

    def __mul__(self, other):
        out = Value(self.data * other.data)
        out._prev = {self, other}
        out._op = '*'

        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad

        out._backward = _backward
        return out


def backward(nodeRoot):
    topo = top_sort(nodeRoot)
    nodeRoot.grad = 1.0
    for node in reversed(topo):
        #so in this task, only the initiliazed function will be called for "z", because it is a node made by addition of x and y
        node._backward()


def top_sort(root):
    visited, seenByOrder = set(), []

    def build(v):
        if v not in visited:
            visited.add(v)
            for parentNode in v._prev:
                build(parentNode)
            seenByOrder.append(v)

    build(root)
    return seenByOrder


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
    x = Value(10, 'x')
    y = x + x

    backward(y)
    # This will create a new directory and store the output file there.
    # With "view=True" it'll automatically display the saved file.
    draw_dot(y).render(directory='./graphviz_output', view=True)


if __name__ == "__main__":
    main()
