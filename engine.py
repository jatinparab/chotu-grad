import math


class Value:
    data = None

    def __init__(self, data, children=(), op='', label='') -> None:
        self.data = data
        self._children = set(children)
        self._op = op
        self.grad = 0
        self.label = label
        self._backward = lambda: None

    def __repr__(self) -> str:
        return f'Value(data={self.data})'

    def __add__(self, other: 'Value') -> 'Value':
        other = other if isinstance(other, Value) else Value(data=other)
        output = self.data + other.data

        def _backward():
            self.grad += out.grad
            other.grad += out.grad
        out = Value(data=output, children=(self, other), op='+')
        out._backward = _backward
        return out

    def __mul__(self, other: 'Value'):
        other = other if isinstance(other, Value) else Value(data=other)
        out = Value(data=self.data * other.data,
                    children=(self, other), op='*')

        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad
        out._backward = _backward
        return out

    def __pow__(self, other):
        assert isinstance(other, (int, float)
                          ), "only supporting int/float powers for now"
        out = Value(data=(self.data ** other),
                    children=(self,), op=f'**{other}')

        def _backward():
            self.grad += (other * (self.data ** (other - 1))) * out.grad
        out._backward = _backward
        return out

    def exp(self):
        out = Value(math.exp(self.data), (self,), 'e')

        def _backward():
            self.grad += out.data * out.grad
        out._backward = _backward
        return out

    def __radd__(self, other: 'Value') -> 'Value':
        return self + other

    def __rmul__(self, other: 'Value') -> 'Value':
        return self * other

    def __neg__(self) -> 'Value':
        return self * -1

    def __sub__(self, other: 'Value'):
        return self + -other

    def __rsub__(self, other):
        return -self + other

    def __truediv__(self, other):
        return self * (other**-1)

    def __rtruediv__(self, other):
        return other * (self**-1)

    def backward(self):
        topo = []
        visited = set()

        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._children:
                    build_topo(child)
                topo.append(v)
        build_topo(self)
        self.grad = 1
        for n in reversed(topo):
            n._backward()
