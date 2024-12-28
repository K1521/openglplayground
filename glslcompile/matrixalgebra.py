

import numpy as np

class monom:
    def __init__(self,powerdict={}):
        if isinstance(powerdict,str):
            powerdict={powerdict:1}
        self.powerdict=powerdict
        self._hash=hash(frozenset(self.powerdict.items()))
    def __mul__(self,other):
        keys=self.powerdict.keys()|other.powerdict.keys()
        return monom({k:self.powerdict.get(k,0)+other.powerdict.get(k,0)for k in keys})
    def __hash__(self):
        return self._hash
    def __eq__(self,other):
        return (self is other) or (self.powerdict==other.powerdict)
    def __str__(self):
        return "*".join(f"{k}**{v}".removesuffix("**1") for k,v in self.powerdict.items())
class polytrace:
    def __init__(self,coeffs={}):
        if isinstance(coeffs,str):
            coeffs=monom(coeffs)
        if isinstance(coeffs,monom):
            coeffs={coeffs:1}
        for k,v in list(coeffs.items()):#remove 0s
            if v==0:
                del coeffs[k]
        self.coeffs=coeffs
    def convert(self,value):
        if isinstance(value,polytrace):
            return value
        if isinstance(value,(monom,str)):
            return polytrace(value)
        if value==0:
            return polytrace() 
        return polytrace({monom():value})
    def __add__(self,other):
        other=self.convert(other)
        keys=self.coeffs.keys()|other.coeffs.keys()
        return polytrace({k:self.coeffs.get(k,0)+other.coeffs.get(k,0)for k in keys})
    def __radd__(self,other):
        return self.__add__(other)
    def __sub__(self,other):
        other=self.convert(other)
        keys=self.coeffs.keys()|other.coeffs.keys()
        return polytrace({k:self.coeffs.get(k,0)-other.coeffs.get(k,0)for k in keys})
    def __mul__(self,other):
        other=self.convert(other)
        s={}
        for ka,va in self.coeffs.items():
            for kb,vb in other.coeffs.items():
                v=va*vb
                if v!=0:
                    k=ka*kb
                    s[k]=s.get(k,0)+v

        return polytrace(s)
    def __rmul__(self,other):
        return self.__mul__(other)
    def __truediv__(self,other):
        return self*(1/other)
    def __str__(self):
        return " ".join(f"{c:+}*{m}".removesuffix("*") for m,c in self.coeffs.items())

import sys
sys.path.append('./')
import algebra.dcga as dcga


t=dcga.Translator(3,4,5)
#print((t*t.reverse()).blades)


#xyz=(opg.VarNode("x"),opg.VarNode("y"),opg.VarNode("z"))
x=polytrace("x")
y=polytrace("y")
z=polytrace("z")   
point=dcga.point(x,y,z)

#obj=dcga.toroid(2,0.5)
#obj=dcga.Plane(1,1,1,1)
#obj=dcga.Plane(1,1,1,0.01).outer(dcga.toroid(2,0.5))

#obj=sanwich(t,obj)
#iprod=point.inner(obj)
#list(iprod.blades.values())
print(point.blades.values())
for b in point.blades.values():
    print(str(b))


pointbasis=set()
for b in point.blades.values():
    pointbasis.update(b.coeffs.keys())

pointmat=np.zeros((len(point.blades),len(pointbasis)))
print(len(pointbasis))
pointbasismat=np.array([polytrace(b)for b in pointbasis])[:,None]
pointbasis={b:i for i,b in enumerate(pointbasis)}
for i,b in enumerate(point.blades.values()):
    for (m,c) in b.coeffs.items():
        j=pointbasis[m]
        pointmat[i,j]=c
print(pointmat)

def funtoviz(point):
    obj=dcga.toroid(2,0.5)
    #obj=dcga.Plane(1,1,1,1)
    #obj=dcga.Plane(1,1,1,0.01).outer(dcga.toroid(2,0.5))

    #obj=sanwich(t,obj)
    prod=point.inner(obj)
    return prod

print(np.linalg.matrix_rank(pointmat))


eye=np.eye(len(point.blades))

funmat=np.stack(list(funtoviz(point.frombasis({basis:eye[i]for i,basis in enumerate(point.blades)})).blades.values()))
print(funmat)

print(str((funmat@pointmat@pointbasismat)[0,0]).replace("**","^"))





# 1,
# x,y,z,
# xx,xy,xz,yy,yz,zz,
# xxx,xxy,xxz,xyy,xyz,xzz,
# yyy,yyz,yzz,
# zzz,
# xxxx,xxxy,xxxz,xxyy,xxyz,xxzz,xyyy,xyyz,xyzz,xzzz,
# yyyy,yyyz,yyzz,yzzz,
# zzzz
