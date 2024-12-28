

import numpy as np
import math

class monom:
    def __init__(self,powerdict={}):
        if isinstance(powerdict,str):
            powerdict={powerdict:1}
        self.powerdict={k:v for k,v in sorted(powerdict.items()) if v>0}

        self._hash=hash(frozenset(self.powerdict.items()))
    def __mul__(self,other):
        keys=self.powerdict.keys()|other.powerdict.keys()
        return monom({k:self.powerdict.get(k,0)+other.powerdict.get(k,0)for k in keys})
    def __pow__(self,exponent):
        return monom({k:v*exponent for k,v in self.powerdict.items()})
    def __hash__(self):
        return self._hash
    def __eq__(self,other):
        return (self is other) or (self.powerdict==other.powerdict)
    def __str__(self):
        return "*".join(f"{k}**{v}".removesuffix("**1") for k,v in sorted(self.powerdict.items())) or "1"
    def degree(self):
        return sum(self.powerdict.values())
    def __lt__(self,other):
        keys=sorted(self.powerdict.keys()|other.powerdict.keys())
        sortkey=lambda x:(-x.degree(),[x.powerdict.get(v,0) for v in keys])
        return sortkey(self)>sortkey(other)


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

#def makepointmat():
x=polytrace("x")
y=polytrace("y")
z=polytrace("z")   
point=dcga.point(x,y,z)
polybasismonoms=set()
for b in point.blades.values():
    polybasismonoms.update(b.coeffs.keys())
polybasismonoms=sorted(polybasismonoms)
polybasisvec=np.array([polytrace(b)for b in polybasismonoms])#[:,None]
#pointbasismat2=np.array([polytrace(b)for b in pointbasis])[:,None]


pointmat=np.zeros((len(point.blades),len(polybasismonoms)))
print(len(polybasismonoms))

polybasisindex={b:i for i,b in enumerate(polybasismonoms)}
for i,b in enumerate(point.blades.values()):
    for (m,c) in b.coeffs.items():
        j=polybasisindex[m]
        pointmat[i,j]=c


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

print(str((funmat@pointmat@polybasisvec)[0]).replace("**","^"))
#print(str((funmat@pointmat@pointbasismat2)[0,0]).replace("**","^"))
print(funmat@pointmat)#@polybasisvec)
for b in polybasisvec:
    print(b)


ox=1
oy=2
oz=3
def makeshiftmatrix(ox,oy,oz):
    shiftmat=np.zeros((len(polybasismonoms),len(polybasismonoms)))
    for basismonom in polybasismonoms:
        i,j,k=[basismonom.powerdict.get(v,0)for v in "xyz"]
        source_index = polybasisindex[basismonom]
        for p in range(i+1):
            for q in range(j+1):
                for r in range(k+1):
                    factor=math.comb(i,p)*math.comb(j,q)*math.comb(k,r)*ox**(i-p)*oy**(j-q)*oz**(k-r)
                    #shiftmat[polybasisindex[basismonom],polybasisindex[x**p*y**q*z**r]]+=factor
                    target_monom = monom({"x":p,"y":q,"z":r})#x**p * y**q * z**r
                    #print(str(target_monom))
                    target_index = polybasisindex[target_monom]
                    shiftmat[source_index,target_index] += factor
    return shiftmat

#print(shiftmat)
import cProfile
profiler=cProfile.Profile()
with profiler:
    shiftmat=makeshiftmatrix(1,0,0)@makeshiftmatrix(0,1,0)
    print(funmat)
    print(pointmat)
    print(str((funmat@pointmat@shiftmat@polybasisvec)[0]).replace("**","^"))
profiler.print_stats()
#print(str((makeshiftmatrix(1,0,0)@polybasisvec)).replace("**","^"))
print(funmat@pointmat)

# for b1,b2 in zip(makeshiftmatrix(1,1,0)@polybasisvec,polybasisvec):
#     print(str(b1),"    ",str(b2))

#print(makeshiftmatrix(0,1,0))
# 1,
# x,y,z,
# xx,xy,xz,yy,yz,zz,
# xxx,xxy,xxz,xyy,xyz,xzz,
# yyy,yyz,yzz,
# zzz,
# xxxx,xxxy,xxxz,xxyy,xxyz,xxzz,xyyy,xyyz,xyzz,xzzz,
# yyyy,yyyz,yyzz,yzzz,
# zzzz
