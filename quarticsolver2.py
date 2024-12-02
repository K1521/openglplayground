
import math
import numpy as np
a=3
b=1
c=-4
d=2
e=1
def sign(x):
    return np.sign(x)
quartic=lambda x:abs(a*x**4+b*x**3+c*x**2+d*x+e)
quarticd1=lambda x:(4*a*x**3+3*b*x**2+2*c*x+d)*sign(a*x**4+b*x**3+c*x**2+d*x+e)
quarticd2=lambda x:(12*a*x**2+6*b*x+2*c)*sign(a*x**4+b*x**3+c*x**2+d*x+e)

#fitcurve=lambda p,x:quartic(p)+(x-p)*quarticd1(p)+(x-p)**2*quarticd2(p)/2
fitcurve=lambda p,x:quartic(p)+(x-p)*quarticd1(p)-(x-p)**2*abs(quarticd2(p))/2
# def findroots(p):
#     C=quartic(p)
#     B=quarticd1(p)
#     A=quarticd2(p)/2
#     #y=(x-a)
#     #f(a)+y*f′(a)+y**2/2*​f′′(a)=0
#     #y=-B/(2*A)+-sqrt((B/(2*A))**2-C/A)
#     discriminant=((B/(2*A))**2 - C/A)
#     mid=p-B / (2 * A)
#     #return [mid]
#     if discriminant<0:
#         return []
#     s=math.sqrt(discriminant)
#     return [mid+s,mid-s]

def findroots(p):
    C=quartic(p)
    B=quarticd1(p)
    A=-abs(quarticd2(p)/2)
    #y=(x-a)
    #f(a)+y*f′(a)+y**2/2*​f′′(a)=0
    #y=-B/(2*A)+-sqrt((B/(2*A))**2-C/A)
    discriminant=((B/(2*A))**2 - C/A)
    mid=p-B / (2 * A)
    #return [mid]
    if discriminant<0:
        return []
    s=math.sqrt(discriminant)
    return [mid+s,mid-s]

def findroots2(p):
    A=quartic(p)
    B=quarticd1(p)
    return [p-A/B]

def findrootsm(p):
    C=quartic(p)
    B=quarticd1(p)
    A=quarticd2(p)/2
    #y=(x-a)
    #f(a)+y*f′(a)+y**2/2*​f′′(a)=0
    #y=-B/(2*A)+-sqrt((B/(2*A))**2-C/A)
    discriminant=(B/(2*A))**2 - C/A
    mid=p-B / (2 * A)
    return mid

# def findroots(p):
#     A = quarticd2(p) / 2   # f''(p) / 2
#     B = quarticd1(p)       # f'(p)
#     C = quartic(p)         # f(p)

#     discriminant = B**2 - 4 * A * C  # Correct discriminant
#     mid = -B / (2 * A)  # Vertex of the parabola

#     if discriminant < 0:
#         # No real roots, return the minimum
#         return [p + mid]  # The minimum is at p + mid (vertex in terms of x)
#     else:
#         # Calculate the roots using the quadratic formula
#         s = math.sqrt(discriminant)
#         root1 = p + mid + s
#         root2 = p + mid - s
#         return [root1, root2]

def stepsize(p):return min(abs(r-p) for r in findroots(p)+findroots2(p))
#def stepsize(p):return min(abs(x-p) for x in findroots(p))

X=np.linspace(-5,5,1000)
S=[stepsize(x)for x in X]
Y=quartic(X)

import matplotlib.pyplot as plt

plt.plot(X,S)
plt.plot(X,Y)
#plt.plot(X,fitcurve(0,X))
#plt.plot(X,[abs(findrootsm(x)-x)for x in X])
#plt.plot(X,[abs(findroots2(x)[0]-x)for x in X])
plt.plot(X,[min(abs(r-x) for r in findroots(x)+[10000])for x in X])
#plt.plot(X,[abs((findroots(x) or [10000,10000])[0]-x)for x in X])
#plt.plot(X,[abs((findroots(x) or [10000,10000])[1]-x)for x in X])
plt.plot(X,fitcurve(0.7,X))
#plt.plot(X,fitcurve(0.8,X))
#plt.plot(X,fitcurve(1.2,X))
plt.ylim(-10,10)
plt.show()

print(findroots(0))