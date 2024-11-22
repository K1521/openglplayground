

class SimpleAlgebraBase:
    def convert(self,x)->"SimpleAlgebraBase":
        raise NotImplementedError("abstract method not implemented")
     
    def __mul__(self, other)->"SimpleAlgebraBase":
        return self.mul(self.convert(other)) 
    def __rmul__(self, other)->"SimpleAlgebraBase":
        return self.convert(other).mul(self)
    def mul(a, b)->"SimpleAlgebraBase":
        raise NotImplementedError("abstract method not implemented")
    
    def __add__(self, other)->"SimpleAlgebraBase":
        return self.add(self.convert(other)) 
    def __radd__(self, other)->"SimpleAlgebraBase":
        return self.convert(other).add(self)
    def add(a, b)->"SimpleAlgebraBase":
        raise NotImplementedError("Static abstract method not implemented")
    
    def __sub__(self, other)->"SimpleAlgebraBase":
        #return self.sub(self, self.convert(other)) 
        return self.sub(self.convert(other)) 
    def __rsub__(self, other)->"SimpleAlgebraBase":
        #return self.sub(self.convert(other), self)
        return self.convert(other).sub(self)
    def sub(a, b)->"SimpleAlgebraBase":
        return a.add(-b)
    def __neg__(self)->"SimpleAlgebraBase":
        return self*(-1)
    
    def __xor__(self, other)->"SimpleAlgebraBase":
        return self.xor(self.convert(other)) 
    def __rxor__(self, other)->"SimpleAlgebraBase":
        return self.convert(other).xor(self)
    def xor(a, b)->"SimpleAlgebraBase":
        raise NotImplementedError("Static abstract method not implemented")
    
    def __matmul__(self, other)->"SimpleAlgebraBase":
        return self.matmul(self.convert(other)) 
    def __rmatmul__(self, other)->"SimpleAlgebraBase":
        return self.convert(other).matmul(self)
    def matmul(a, b)->"SimpleAlgebraBase":
        raise NotImplementedError("Static abstract method not implemented")
    
    def __truediv__(self,other)->"SimpleAlgebraBase":
        return self*(1/other)
    
    def __pow__(self,e)->"SimpleAlgebraBase":
        if   e == 0:return self.convert(1)
        elif e == 1:return self  # Any base raised to the power of 1 is itself
        elif e > 1:
            half_pow = self ** (e // 2)
            pow=half_pow*half_pow
            return pow if e % 2 == 0 else pow * self
        else:
            raise ValueError("Exponent must be a non-negative integer")