import itertools
from collections import defaultdict

class blade:
    __slots__ = ['basis', 'magnitude']
    def __init__(self,basis,magnitude=1) -> None:
        self.magnitude=magnitude
        self.basis=basis
    def __repr__(self):
        return f"blade({self.basis:b},{self.magnitude})"
    

class algebra:
    def __init__(self,posi=3,nega=0,neut=0,order="pnz"):
        self.zero=blade(0,0)
        self.posi=posi
        self.nega=nega
        self.neut=neut
        self.dim=posi+nega+neut

        self.bladenames=[str(i) for i in range(self.dim)]#list(map(str,range(self.dim)))

        s=0
        if sorted(order)!=sorted("pnz"):
            raise Exception("order must be a permutatioon of pnz like pzn or npz")
        for o in order:
            if o=="p":
                self.posimask=(2**posi-1)<<s
                s+=posi
            if o=="n":
                self.negamask=(2**nega-1)<<s
                s+=nega
            if o=="z":
                self.neutmask=(2**neut-1)<<s
                s+=neut
    
    def basedecode(self,basis):
        return tuple(i for i,x in enumerate(bin(basis)[:1:-1],0)if x=="1")
    def bladestr(self,blade:blade):
        try:
            mag=f"{blade.magnitude:+}"
        except TypeError:
            mag=f"+{blade.magnitude}"
        if blade.basis==0:
            return mag
        trenner="," if max(map(len,self.bladenames))>1 else ""
        return  mag+"*e"+trenner.join(self.bladenames[i] for i in self.basedecode(blade.basis))
    
    def geo(self,blade1:blade,blade2:blade):
        #if both have alligning neutral vector return 0
        if self.neutmask&blade1.basis&blade2.basis:
            return self.zero
        
        #count inversions
        #bas1acc=blade1.basis^(blade1.basis.bit_count()&1)
        #i=1
        #l=min(blade1.basis.bit_length(),blade2.basis.bit_length())
        #mask=(2<<l)-1
        #while i<=l:
        #    bas1acc^=(bas1acc<<i)&mask
        #    i<<=1
        #invert=(bas1acc&blade2.basis).bit_count()&1

        bas1acc = self.accumulate(blade1)
        invert=(bas1acc&blade2.basis).bit_count()&1

        
        #calculate negative alligned "inversions"
        invert^=(self.negamask&blade1.basis&blade2.basis).bit_count()&1

        #calculate magnitude
        magnitude=blade1.magnitude*blade2.magnitude
        if invert:
            magnitude=-magnitude
        return blade(blade1.basis^blade2.basis, magnitude)

    def accumulate(self, blade1):
        bas1acc=blade1.basis>>1
        for i in range(blade1.basis.bit_length().bit_length()):#lol
            bas1acc^=(bas1acc>>(1<<i))
        return bas1acc
    def outer(self,blade1:blade,blade2:blade):
        if blade1.basis&blade2.basis:
            return self.zero
        return self.geo(blade1,blade2)
    def inner(self,blade1:blade,blade2:blade):
        if  (((~blade1.basis)&blade2.basis) and (blade1.basis&(~blade2.basis))):#works analog zu 
            return self.zero
        return self.geo(blade1,blade2)
    
    def reverse(self,blade):
        if blade.basis.bit_count()&1:
            return blade(blade.basis, -blade.magnitude)
        else:
            return blade
    
    def bladesortkey(self,blade):
        l=self.basedecode(blade.basis)
        return len(l),l
    
    def grade(self,bladeo):
        return bladeo.basis.bit_count()
    def reverse(self,bladeo):
        g=self.grade(bladeo)
        sign=(-1)**(g*(g-1)//2)
        if sign==1:
            return bladeo
        else:
            return blade(bladeo.basis,-bladeo.magnitude)
    def conjugate(self,bladeo):
        g=self.grade(bladeo)
        sign=(-1)**(g*(g+1)//2)
        if sign==1:
            return bladeo
        else:
            return blade(bladeo.basis,-bladeo.magnitude)
    def involute(self,bladeo):
        g=self.grade(bladeo)
        sign=(-1)**g
        if sign==1:
            return bladeo
        else:
            return blade(bladeo.basis,-bladeo.magnitude)
    

class sortgeo:
    #@staticmethod
    #def filterzero(it):
    #    return list(i for i in it if i.magnitude!=0)

    def __init__(self,algebra:algebra,lst=None,compress=False) -> None:
        if lst is None:
            lst=[]
        self.algebra=algebra
        self.lst=[i for i in lst if i!=self.algebra.zero]
        if compress:
            self.compress()
    def allblades(self,sort=True):
        lst=[sortgeo(self.algebra,[blade(i)]) for i in range(2**self.algebra.dim)]
        if sort:
            return sorted(lst,key=lambda x:self.algebra.bladesortkey(x.lst[0]))
        return lst
    def monoblades(self):
        return [sortgeo(self.algebra,[blade(1<<i)])for i in range(self.algebra.dim)]
    def scalars(self,num):#makes a list of skalars from float/list of float
        return[sortgeo(self.algebra,[blade(0,x)]) for x in num]
    def scalar(self,num):
        return sortgeo(self.algebra,[blade(0,num)])
        

    def __str__(self) -> str:
        if not self.lst:
            return self.algebra.bladestr(self.algebra.zero)
        return " ".join(self.algebra.bladestr(b) for b in sorted(self.lst,key=self.algebra.bladesortkey))
    
    def __repr__(self):
        return self.__str__()

    def compress(self):
        
        lstnew=[]
        getblades=lambda x:x.basis
        self.lst.sort(key=getblades)
        for k,g in itertools.groupby((i for i in self.lst if self.algebra.zero!=0),key=getblades):
            lstnew.append(blade(k,sum(x.magnitude for x in g)))
        self.lst=lstnew
        #actblades=lst[0].blades
        #actmagnitude=lst[0].blades#i have desided to make blades imutable
        #for i in range(1,len(lst)):
        #    if lst[i].blades==lst[ilast].blades:
    #def __matmul__(self,othe): 
    #    return sortgeo(self.algebra,(self.algebra.geo(x,y) for x in self.lst for y in othe.lst),compress=True)
    def inner(self,othe):
        return sortgeo(self.algebra,(self.algebra.inner(x,y) for x in self.lst for y in othe.lst),compress=True)
    def __xor__(self,othe):
        return self.outer(othe)
    def outer(self,othe):
        return sortgeo(self.algebra,(self.algebra.outer(x,y) for x in self.lst for y in othe.lst),compress=True)
    def __add__(self,othe):
        if isinstance(othe,sortgeo):
            return sortgeo(self.algebra,self.lst+othe.lst,compress=True)
        else:
            return self+self.scalar(othe)
    def __sub__(self,othe):
        return self+ (-othe)
    def __rsub__(self,othe):
        return (-self)+ othe
    def __neg__(self):
        return sortgeo(self.algebra,(blade(b.basis,-b.magnitude) for b in self.lst),compress=False)
    def reverse(self):
        return sortgeo(self.algebra,(self.algebra.reverse(b) for b in self.lst),compress=False)
    def geo(self,othe):
        if isinstance(othe,sortgeo):
            return sortgeo(self.algebra,(self.algebra.geo(x,y) for x in self.lst for y in othe.lst),compress=True)#geometric product
        return sortgeo(self.algebra,(blade(b.basis,b.magnitude*othe) for b in self.lst),compress=False)#skalarmul
    def __mul__(self,othe):
        self.geo(othe)
    def __rmul__(self,othe):#skalarmul
        self.geo(othe)
    def __truediv__(self,othe):
        if isinstance(othe,sortgeo):
            raise Exception("currently only integer/float division is supported")
        return sortgeo(self.algebra,(blade(b.basis,b.magnitude/othe) for b in self.lst),compress=False)
    def toscalar(self):
        self.compress()
        if not self.lst:
            return 0
        if len(self.lst)==1:
            if self.lst[0].basis==0:
                return self.lst[0].magnitude
        raise Exception("not convertible")
    
    def involute(self):
        return sortgeo(self.algebra,(self.algebra.involute(b) for b in self.lst),compress=False)
    def conjugate(self):
        return sortgeo(self.algebra,(self.algebra.conjugate(b) for b in self.lst),compress=False)
    def reverse(self):
        return sortgeo(self.algebra,(self.algebra.reverse(b) for b in self.lst),compress=False)
