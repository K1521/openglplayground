

from matplotlib import pyplot as plt
import numpy as np


class Dual:
    def __init__(self,f,f_=0):
        self.f=f
        self.f_=f_

    @staticmethod
    def convert(x):
        if type(x)==Dual:
            return x
        return Dual(x)
    
    def __add__(self,other):
        other=Dual.convert(other)
        return Dual(self.f+other.f,self.f_+other.f_)

    def __mul__(self,other):
        other=Dual.convert(other)
        return Dual(self.f*other.f,self.f_*other.f+self.f*other.f_)

    def __sub__(self,other):
        other=Dual.convert(other)    
        return Dual(self.f-other.f,self.f_-other.f_)

    def __div__(self,other):
        other=Dual.convert(other)
        return Dual(self.f/other.f,(self.f_*other.f-self.f*other.f_)/(other.f*other.f))
    
    def __rmul__(self,other):
        return self*other

    def __radd__(self,other):
        return self+other
    
    def __rsub__(self,other):
        return -self+other
    
    def __neg__(self):
        return Dual(-self.f,-self.f_)
    
    def __pow__(self,other):
        #integer power
        if type(other)==int:
            if other==0:
                return Dual(1)
            if other==1:
                return self
            if other>1:
                return self*self**(other-1)
    
    def __str__(self):
        return str(self.f)+" + "+str(self.f_)+"e"


def Aberth2(initial,f,f_,iterations):
    x=np.linspace(0,5,100)

    history=[]
    roots=initial[:]
    for it in range(iterations):
        for r in range(len(roots)):
            root=roots[r]
            ff_=f(Dual(root,1))
            p=f(root)
            # if abs(p)<1e-15:
            #     continue
            p_=f_(root)
            s=0
            for j in range(len(roots)):
                if r!=j:
                    s+=1/(root-roots[j])+1/(root-roots[j].conjugate())
            #s+=1/(root-root.conjugate())
            n=p/p_
            roots[r]-=n/(1-n*s)
        history.append(roots[:])
        print(roots)
    return roots,history
    

def f(x):
    return (-1 + 12.73333*x - 12.55*x**2 + 4.266667*x**3 - 0.45*x**4)**2+(15 - 20.68333*x + 8.741667*x**2 - 1.066667*x**3 + 0.008333333*x**4)**2


import math
golden_ratio=(1+math.sqrt(5))/2
golden_angle=2*math.pi/golden_ratio
golden_complex=math.cos(golden_angle)+math.sin(golden_angle)*1j

#poly=f(np.polynomial.Polynomial([0,1]))

def randompoints(x):  

    return np.vstack([x,np.random.uniform(-1,1,len(x))])

def randpoly2(numcommonpoints=0):
    degree=4
    numpoints=degree+1-numcommonpoints
    x=np.linspace(-1,1,degree+1)
    #split x in x1 x2 with numpoints/numcommonpoints
    x1=x[:numpoints]
    x2=x[numpoints:]
    #print(numpoints,x1,x2)


    commonpoints=randompoints(x2)
    p=0
    for i in range(3):
        x,y=np.hstack([commonpoints,randompoints(x1)])
        #print(x,y)
        p+=np.poly1d(np.polyfit(x,y,degree))**2
    print(p)
    return p
f=poly=randpoly2()

initial=[golden_complex**i for i in range(1,3+1)]

roots,history=Aberth2(initial,poly,poly.deriv(),100)

def froots(x,roots):
    return math.prod(x-root for root in roots)

#plot trajectory of roots in the complex plane
history=np.array(history)
# for i in range(len(history[0])):
#     for j in range(len(history)-1):
#         start=history[j][i]
#         end=history[j+1][i]
#         plt.plot([start.real,end.real],[start.imag,end.imag],color='black')
for i in range(len(history[0])):
    # i want to plot with points
    plt.plot(history[:,i].real,history[:,i].imag,marker='o')
plt.show()

x=np.linspace(0,5,100)


plt.plot(x,f(x),color='green')
#plt.plot(x,f(Dual(x,1)).f_,color='blue')
plt.plot(x,f(x)/froots(x,poly.roots),color='red')
plt.plot(x,f(x)/froots(x,roots+[z.conjugate() for z in roots]),color='purple')

#plt.plot(x,f(x)/froots(x,roots),color='purple')

#plt.plot(x,poly(x),color='blue')
print(poly.roots)
plt.show()