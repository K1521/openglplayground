import numpy as np
class algebraspecification:

    def __init__(self,pnz,bladenames=None,bladenamesep=None,reversebits=True):
        if set(pnz)>{0,1,-1}:
            raise ValueError("pnz must be a list of 1/-1/0")
        self.positive=pnz.count(1)
        self.negative=pnz.count(-1)
        self.neutral=pnz.count(0)
        
        def makebitmask(f):
            maskstr="".join("1" if x==f else "0" for i,x in enumerate(pnz))
            if reversebits:
                maskstr=maskstr[::-1]
            return int(maskstr,2)
        self.positivemask=makebitmask(1)
        self.negativemask=makebitmask(-1)
        self.neutralmask=makebitmask(0)
        
        self.pnz=pnz
        self.dim=self.positive+self.negative+self.neutral
        if bladenames is None:
            bladenames=[str(x)for x in range(self.dim)]
        self.bladenames=bladenames
        if bladenamesep is None:
            bladenamesep= "" if max(map(len,bladenames))==1 else ";"
        self.bladenamesep=bladenamesep
        self.reversebits=reversebits#if true "001"=e1 else "001"=e3

    def bladenamefrombasis(self,basis):
        basisstr=f"{basis:0{self.dim}b}"
        if self.reversebits:
            basisstr=basisstr[::-1]
        return self.bladenamesep.join(self.bladenames[i] for i,x in enumerate(basisstr) if x=="1")



class tablealgebra(algebraspecification):
    def __init__(self, pnz, bladenames=None, bladenamesep=None):
        super().__init__(pnz, bladenames, bladenamesep, reversebits=True)
        self.geotable  =self.makegeotable()#np.ones([2**self.dim]*2)
        self.innertable=self.makeinnertable()#np.zeros_like(self.geotable)
        self.outertable=self.makeoutertable()#np.zeros_like(self.geotable)

    @staticmethod
    def accumulate(basis):
        basacc=basis>>1
        if isinstance(basis,int):
            #basis=int(basis)
            n=basis.bit_length().bit_length()
        elif isinstance(basis,np.ndarray):
            n=basis.dtype.itemsize.bit_length()+3
        else:
            print(type(basis))
            raise ValueError("basis must be int or intarray")
        for i in range(n):#lol
            basacc^=(basacc>>(1<<i))
        return basacc
    
    def makegeotable(self):#,basis1=None,basis2=None):
        basis1=np.arange(2**self.dim,dtype=np.uint)[:,None]
        basis2=basis1.T
        basis1acc=tablealgebra.accumulate(basis1).astype(basis1.dtype)
        geotablepnz=np.ones([2**self.dim]*2,dtype=np.int8)

        #invert=np.bitwise_count(basis1acc&basis2)&1
        #invert^=np.bitwise_count(self.negativemask&basis1&basis2)&1
        
        invert=np.bitwise_count(
            (basis1acc&basis2)^(self.negativemask&basis1&basis2)
            )&1
        #invert=np.bitwise_count(
        #    ((basis1acc)^(self.negativemask&basis1))&basis2
        #    )&1

        geotablepnz[invert==1]*=-1
        geotablepnz[(self.neutralmask&basis1&basis2)!=0]=0

        return geotablepnz
    
    def makeoutertable(self):
        outer=self.makegeotable()
        basis1=np.arange(2**self.dim,dtype=np.uint)[:,None]
        basis2=basis1.T
        outer[(basis1&basis2)!=0]=0
        return outer
        
    def makeinnertable(self):
        inner=self.makegeotable()
        basis1=np.arange(2**self.dim,dtype=np.uint)[:,None]
        basis2=basis1.T
        inner[(((~basis1)&basis2)!=0) & ((basis1&(~basis2))!=0)]=0
        return inner
    def geo_at(self,basis1,basis2):
        return self.geotable[basis1,basis2]
    def inner_at(self,basis1,basis2):
        return self.innertable[basis1,basis2]
    def outer_at(self,basis1,basis2):
        return self.outertable[basis1,basis2]
    def grade(self,basis):
        return basis.bit_count()
    def involute(self,basis):
        g=self.grade(basis)
        return (-1)**g
    def conjugate(self,basis):
        g=self.grade(basis)
        return (-1)**(g*(g+1)//2)
    def reverse(self,basis):
        g=self.grade(basis)
        return (-1)**(g*(g-1)//2)

    # def accumulate(self, blade1):
    #     bas1acc=blade1.basis>>1
    #     for i in range(blade1.basis.bit_length().bit_length()):#lol
    #         bas1acc^=(bas1acc>>(1<<i))
    #     return bas1acc
    # def outer(self,blade1,blade2):
    #     if blade1.basis&blade2.basis:
    #         return self.zero
    #     return self.geo(blade1,blade2)
    # def inner(self,blade1,blade2):
    #     if  (((~blade1.basis)&blade2.basis) and (blade1.basis&(~blade2.basis))):#works analog zu 
    #         return self.zero
    #     return self.geo(blade1,blade2)
# #alg=algebraspecification([1]*3)
# alg=tablealgebra([1,1,-1])
# table2=alg.makeinnertable()


# #from algebra import dcga
# from algebra import blademul
# dcga=blademul.algebra(2,1)
# multivec=blademul.sortgeo(dcga)
# blades=multivec.allblades(sort=False)
# #blades=dcga.multivec.allblades()
# import numpy as np
# table=np.zeros([len(blades)]*2,dtype=int)
# for i,a in enumerate(blades):
#     if a.lst and a.lst[0].basis!=i:
#         print("?????????????")
#     if a.lst:
#         print(a.lst[0].basis)
        
#     for j,b in enumerate(blades):
#         #print(a.geo(b))
#         lst=a.inner(b).lst
#         if lst:
#             table[i,j]=int(lst[0].magnitude)
#         #table[i,j]="+-"[blade.magnitude<0]+str(blade.basis)

# import sys
# import numpy
# numpy.set_printoptions(threshold=sys.maxsize)

# print(table)
# print(table2)    
# print(((table==table2).astype(int)==0).sum())
# print(np.array_equal(table,table2))


# print(dcga.negamask,alg.negativemask)
# print(dcga.posimask,alg.positivemask)
# print((table==table2).astype(int))



# basis1=np.arange(2**3,dtype=np.uint)[:,None]
# basis2=basis1.T
# print(((((~basis1)&basis2)!=0) & ((basis1&(~basis2))!=0)).astype(int))

# print((((~basis1)&basis2)).astype(int))
# print(((basis1^basis2)&~(basis1&basis2)).astype(int))

# print(~1)
from collections import defaultdict
class bladedict:
    def __init__(self,algebra,blades):
        self.blades=blades
        self.algebra=algebra

    def convert(self,other):
        if isinstance(other,bladedict):
            if other.algebra!=self.algebra:
                raise ValueError("can't add bladedict of different algebras")
            else:
                return other
        return self.frombasis(0,other)
    def frombasis(self,basis=0,value=1):
        return bladedict(self.algebra,{basis:value})
    def __add__(self,other):
        other=self.convert(other)
        newdict=self.blades|other.blades#blades which are in both dicts get overridden later
        for k in self.blades.keys()&other.blades.keys():
            newdict[k]=self.blades[k]+other.blades[k]
        return bladedict(self.algebra,newdict)
    def __sub__(self,other):
        other=self.convert(other)
        newdict=self.blades.copy()
        for k in self.blades.keys()&other.blades.keys():
            newdict[k]=newdict[k]-other.blades[k]#-=doesnt work because self.blades.copy() is a shallow copy
        for k in other.blades.keys()-self.blades.keys():
            newdict[k]=-other.blades[k]
        return bladedict(self.algebra,newdict)
    def _tableproduct(self,other,table):
        other=self.convert(other)
        newdict=defaultdict(int)
        for base1,value1 in self.blades.items():
            for base2,value2 in other.blades.items():
                value=table(base1,base2)
                if value==0:continue
                elif value==1: newdict[base1^base2]+=value1*value2
                elif value==-1:newdict[base1^base2]-=value1*value2
                else:newdict[base1^base2]+=value1*value2*value
        return bladedict(self.algebra,newdict)
    def geo(self,other):
        return self._tableproduct(other,self.algebra.geo_at)
    def inner(self,other):
        return self._tableproduct(other,self.algebra.inner_at)
    def outer(self,other):
        return self._tableproduct(other,self.algebra.outer_at)
    def __radd__(self,other):
        return self.__add__(other)
    def __rsub__(self,other):
        return self.convert(other).__sub__(self)
    def __mul__(self,other):
        return self.geo(other)
    def __rmul__(self,other):
        return self.geo(other)#scalarmul
    def __xor__(self,othe):
        return self.outer(othe)
    def toscalar(self,raiseifinconvertible=True):
        if len(self.blades)==0:
            return 0
        if len(self.blades)==1 and 0 in self.blades:
            return self.blades[0]
        if raiseifinconvertible:
            raise Exception("not convertible")
    def __truediv__(self,other):
        other=self.convert(other).toscalar(False)
        if other is None:
            raise Exception("currently only integer/float division is supported")
        newdict={k:v/other for k,v in self.blades.items()}
        return bladedict(self.algebra,newdict)
    def _1dtableproduct(self,table):
        def transform(basis,value):
            v=table(basis)
            if v==1:return value
            if v==-1:return -value
            return v*value
        newdict={b:transform(b,v)for b,v in self.blades}
        return bladedict(self.algebra,newdict)
    def involute(self):
        return self._1dtableproduct(self,self.algebra.involute)
    def conjugate(self):
        return self._1dtableproduct(self,self.algebra.conjugate)
    def reverse(self):
        return self._1dtableproduct(self,self.algebra.reverse)
    def __neg__(self):
        newdict={b:-v for b,v in self.blades.items()}
        return bladedict(self.algebra,newdict)
    def monoblades(self):
        return [self.frombasis(1<<i)for i in range(self.algebra.dim)]

        




# alg=tablealgebra([1,1,1,-1])
# b1=bladedict(alg,{0:1})
# b2=bladedict(alg,{0:1})
# print((b1+b2).blades)

