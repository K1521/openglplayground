import numpy as np

class vec2:
    def __init__(self, a, b=None):
        if isinstance(a,vec2):
            b=a.y
            a=a.x
        if b is None:  # If only one argument is given, use it for both x and y
            b = a
        self.data = np.array([a, b], dtype=np.float32)

    @property
    def x(self):
        return self.data[0]

    @x.setter
    def x(self, value):
        self.data[0] = value

    @property
    def y(self):
        return self.data[1]

    @y.setter
    def y(self, value):
        self.data[1] = value

    # String representation for debugging
    def __repr__(self):
        return f"vec2({self.x}, {self.y})"

    # Addition
    def __add__(self, other):
        if isinstance(other, vec2):
            return vec2(self.x + other.x, self.y + other.y)
        else:  # Scalar addition
            return vec2(self.x + other, self.y + other)

    # Subtraction
    def __sub__(self, other):
        if isinstance(other, vec2):
            return vec2(self.x - other.x, self.y - other.y)
        else:  # Scalar subtraction
            return vec2(self.x - other, self.y - other)

    # Multiplication
    def __mul__(self, other):
        if isinstance(other, vec2):
            return vec2(self.x * other.x, self.y * other.y)
        else:  # Scalar multiplication
            return vec2(self.x * other, self.y * other)

    # Division
    def __truediv__(self, other):
        if isinstance(other, vec2):
            return vec2(self.x / other.x, self.y / other.y)
        else:  # Scalar division
            return vec2(self.x / other, self.y / other)

    # Negation
    def __neg__(self):
        return vec2(-self.x, -self.y)

    # Equality
    def __eq__(self, other):
        return np.allclose(self.data, other.data)

    # Length (magnitude) of the vector
    def length(self):
        return np.sqrt(self.x**2 + self.y**2)



def length(x):
    return x.length()