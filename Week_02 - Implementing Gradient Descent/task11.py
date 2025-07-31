import numpy as np
import graphviz as graphviz


def manual_der(node, a, b, c, f):
    original = node.data
    eps = 0.001
    node.data = original + eps
    e = a * b
    d = e + c
    L_bigger = f * d

    node.data = original - eps
    e = a * b
    d = e + c
    L_lower = f * d

    node.data = original

    return (L_bigger.data - L_lower.data) / (2 * eps)


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

    counter = 0
    for n in nodes:
        uid = str(id(n))
        # for any value in the graph, create a rectangular ('record') node
        print(n.grad)
        dot.node(name=uid,
                 label=f'{{ {n.label} | data: {n.data} | grad: {n.grad:.4f}}}',
                 shape='record')
        counter += 1
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
    a = Value(2.0, 'a')
    b = Value(-3.0, 'b')
    c = Value(10.0, 'c')
    f = Value(-2.0, 'f')

    e = a * b
    e.label = 'e'

    d = e + c
    d.label = 'd'
    result = f * d

    result.label = 'L'

    print(f"Old L = {result.data}")

    # #Manual calculations:
    # 1 * L = 1
    result.grad = 1.0
    # 1(dL = 1) * f = 1
    d.grad = f.data
    # 1(dL = 1) * d = 1
    f.grad = d.data
    e.grad = 1 * d.grad
    c.grad = 1 * d.grad
    a.grad = e.grad * b.data
    b.grad = e.grad * a.data

    a.data += 0.01 * a.grad
    b.data += 0.01 * b.grad
    c.data += 0.01 * c.grad
    f.data += 0.01 * f.grad

    e = a * b
    d = e + c
    newLoss = f * d
    print(f"New L = {newLoss.data:.6f}")

    # This will create a new directory and store the output file there.
    # With "view=True" it'll automatically display the saved file.
    #draw_dot(result).render(directory='./graphviz_output', view=True)


if __name__ == "__main__":
    main()
