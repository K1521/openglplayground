import numpy as np
import math

class MonomXYZ:
    X: "MonomXYZ" = None  
    Y: "MonomXYZ" = None
    Z: "MonomXYZ" = None
    ONE: "MonomXYZ" = None
    __slots__=("powers","_hash")

    @property
    def x(self):return self.powers[0]

    @property
    def y(self):return self.powers[1]

    @property
    def z(self):return self.powers[2]
    
    def __init__(self, powers):
        self.powers = powers
        self._hash = hash(powers)

    def __mul__(self, other):
        return MonomXYZ(tuple(a + b for a, b in zip(self.powers, other.powers)))

    def __pow__(self, exponent):
        return MonomXYZ(tuple(v * exponent for v in self.powers))

    def __hash__(self):
        return self._hash

    def __eq__(self, other):
        return self.powers == other.powers

    def __str__(self):
        terms = [f"{var}**{exp}".removesuffix("**1") for var, exp in zip("xyz", self.powers) if exp > 0]
        return "*".join(terms) or "1"

    def degree(self):
        return sum(self.powers)

    def __lt__(self, other):
        return (-self.degree(),self.powers) > (-other.degree(),other.powers)
MonomXYZ.X = MonomXYZ((1, 0, 0))
MonomXYZ.Y = MonomXYZ((0, 1, 0))
MonomXYZ.Z = MonomXYZ((0, 0, 1))
MonomXYZ.ONE = MonomXYZ((0, 0, 0))

class PolyTrace:
    X: "PolyTrace" = None  
    Y: "PolyTrace" = None
    Z: "PolyTrace" = None

    def __init__(self, coeffs=None):
        if coeffs is None:
            coeffs = {}
        if isinstance(coeffs, MonomXYZ):
            coeffs = {coeffs: 1}
        self.coeffs = {k: v for k, v in coeffs.items() if v != 0}

    def convert(self, value):
        if isinstance(value, PolyTrace):
            return value
        if isinstance(value, MonomXYZ):
            return PolyTrace({value: 1})
        if value == 0:
            return PolyTrace()
        return PolyTrace({MonomXYZ.ONE: value})

    def __add__(self, other):
        other = self.convert(other)
        keys = self.coeffs.keys() | other.coeffs.keys()
        return PolyTrace({k: self.coeffs.get(k, 0) + other.coeffs.get(k, 0) for k in keys})

    def __radd__(self, other):
        return self.__add__(other)

    def __sub__(self, other):
        other = self.convert(other)
        keys = self.coeffs.keys() | other.coeffs.keys()
        return PolyTrace({k: self.coeffs.get(k, 0) - other.coeffs.get(k, 0) for k in keys})

    def __mul__(self, other):
        other = self.convert(other)
        s = {}
        for ka, va in self.coeffs.items():
            for kb, vb in other.coeffs.items():
                v = va * vb
                if v != 0:
                    k = ka * kb
                    s[k] = s.get(k, 0) + v
        return PolyTrace(s)

    def __rmul__(self, other):
        return self.__mul__(other)

    def __truediv__(self, other):
        return self * (1 / other)

    def __str__(self):
        return " ".join(f"{c:+}*{m}".removesuffix("*") for m, c in self.coeffs.items())
    
PolyTrace.X=PolyTrace(MonomXYZ.X)
PolyTrace.Y=PolyTrace(MonomXYZ.Y)
PolyTrace.Z=PolyTrace(MonomXYZ.Z)


class TraceBasis:
    def __init__(self, point_fn):
        self.point = point_fn(PolyTrace.X,PolyTrace.Y, PolyTrace.Z)
        self.poly_basis_monoms = self._compute_poly_basis_monoms()
        self.poly_basis_vec = np.array([PolyTrace({b: 1}) for b in self.poly_basis_monoms])
        self.poly_basis_index = {b: i for i, b in enumerate(self.poly_basis_monoms)}
        self.point_mat = self._compute_point_matrix()

    def _compute_poly_basis_monoms(self):
        poly_basis_monoms = set()
        for b in self.point.blades.values():
            poly_basis_monoms.update(b.coeffs.keys())
        return sorted(poly_basis_monoms)

    def _compute_point_matrix(self):
        point_mat = np.zeros((len(self.point.blades), len(self.poly_basis_monoms)))
        for i, b in enumerate(self.point.blades.values()):
            for m, c in b.coeffs.items():
                j = self.poly_basis_index[m]
                point_mat[i, j] = c
        return point_mat

    def trace_linear(self, func):
        """Traces a linear function like `fun_to_visualize`."""
        eye = np.eye(len(self.point.blades))
        eyepoint=self.point.frombasis({basis: eye[i] for i, basis in enumerate(self.point.blades)})
        fun_mat = np.stack(list(func(eyepoint).blades.values()))
        return fun_mat

    def move_basis(self, ox, oy, oz):
        """Makes a matrix for moving a basis (shift matrix)."""
        shift_mat = np.zeros((len(self.poly_basis_monoms), len(self.poly_basis_monoms)))
        for basis_monom in self.poly_basis_monoms:
            i, j, k = basis_monom.powers
            source_index = self.poly_basis_index[basis_monom]
            for p in range(i + 1):
                for q in range(j + 1):
                    for r in range(k + 1):
                        factor = (math.comb(i, p) * math.comb(j, q) * math.comb(k, r) *
                                  ox ** (i - p) * oy ** (j - q) * oz ** (k - r))
                        target_monom = MonomXYZ((p, q, r))
                        target_index = self.poly_basis_index[target_monom]
                        shift_mat[source_index, target_index] += factor
        return shift_mat


if __name__=="__main__":
    import sys
    sys.path.append('./')
    import algebra.dcga as dcga
    tracer=TraceBasis(dcga.point)

    def funtoviz(point):
        obj=dcga.toroid(2,0.5)
        #obj=dcga.Plane(1,1,1,1)
        #obj=dcga.Plane(1,1,1,0.01).outer(dcga.toroid(2,0.5))

        #obj=sanwich(t,obj)
        prod=point.inner(obj)
        return prod
    funmat=tracer.trace_linear(funtoviz)
    shiftmat=tracer.move_basis(1,0,0)@tracer.move_basis(0,1,0)
    print(funmat)
    print(tracer.point_mat)
    print(str((funmat@tracer.point_mat@shiftmat@tracer.poly_basis_vec)[0]).replace("**","^"))