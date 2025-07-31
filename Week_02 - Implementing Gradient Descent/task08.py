import numpy as np
import graphviz as graphviz


class Value:

    def __init__(self, data, label=''):
        self.data = data
        self.label = label
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
        dot.node(name=uid,
                 label=f'{{ {n.label} | data: {n.data} }}',
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
    # x = Value(2.0)
    # y = Value(-3.0)
    # z = Value(10.0)
    # result = x * y + z
    # print(result._op)

    #Task 6
    # x = Value(2.0)
    # y = Value(-3.0)
    # z = Value(10.0)
    # result = x * y + z

    # nodes, edges = trace(x)
    # print('x')
    # print(f'{nodes=}')
    # print(f'{edges=}')

    # nodes, edges = trace(y)
    # print('y')
    # print(f'{nodes=}')
    # print(f'{edges=}')

    # nodes, edges = trace(z)
    # print('z')
    # print(f'{nodes=}')
    # print(f'{edges=}')

    # nodes, edges = trace(result)
    # print('result')
    # print(f'{nodes=}')
    # print(f'{edges=}')

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

    # This will create a new directory and store the output file there.
    # With "view=True" it'll automatically display the saved file.
    draw_dot(result).render(directory='./graphviz_output', view=True)


if __name__ == "__main__":
    main()
