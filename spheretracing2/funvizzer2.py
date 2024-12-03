import numpy as np
import sys
sys.path.append('./')
from variable_listener import Client
#from scene import scene
import time
def normalize(v):
    norm = np.linalg.norm(v)
    return v / norm if norm != 0 else v

from math import sqrt,pow
from scene3 import compilepolys
# Define a dummy scene function, replace with your own function
# def scene(p):
#     # Assume some quartic function for this example, returning a vec4
#     # Here, scene(p) returns a scalar field value as 'w' (quartic value).
#     a, b, c, d = 1, 0, 0, 0  # Example coefficients
#     x, y, z = p
#     w = a*x**4 + b*x**3 + c*x**2 + d*x  # Example quartic function
#     return np.array([x, y, z, w])

# EPSILON_DERIV, a small value for numerical derivatives
EPSILON_DERIV = 1e-4
poly=None

def calcpoly(x):
    return poly[0]+poly[1]*x+poly[2]*x**2+poly[3]*x**3+poly[4]*x**4



client=Client()

def solveQuartic(a,b,c,d,e):
    #//https://en.wikipedia.org/wiki/Quartic_function
    delta0=c*c-3*b*d+12*a*e;
    delta1=2*c*c*c-9*b*c*d+27*b*b*e+27*a*d*d-72*a*c*e;
    p=(8*a*c-3*b*b)/(8*a*a);
    q=(b*b*b-4*a*b*c+8*a*a*d)/(8*a*a*a);
    print(delta0,delta1,p,q,delta1*delta1-4*delta0*delta0*delta0)
    if delta1*delta1-4*delta0*delta0*delta0<0:
        return None
    Q=pow(0.5*(delta1+sqrt(delta1*delta1-4*delta0*delta0*delta0)),1./3);
    S=0.5*sqrt(-p*2/3+(Q+delta0/Q)/(3*a));

    xpart1=-b/(4*a);
    xpart2=-4*S*S-2*p;
    xpart3=q/S;
    print(q,S,xpart2,xpart3)
    xsqrt1=float("nan") if xpart2+xpart3<0 else 0.5*sqrt(xpart2+xpart3);
    xsqrt2=float("nan") if xpart2-xpart3<0 else 0.5*sqrt(xpart2-xpart3);

    return (xpart1-S+xsqrt1,xpart1-S-xsqrt1,xpart1+S+xsqrt2,xpart1+S-xsqrt2);


raydyinit=client["raydyinit"]=1
rayfactor=client["rayfactor"]=1
rayalpha=client["rayalpha"]=0
rayepsilon=client["rayepsilon"]=0.1
raypow=client["raypow"]=1

camera_matrix=client["camera_matrix"]
camera_pos=client["camera_pos"]
forward=camera_matrix@np.array([0,0,1])

print(forward,camera_pos)
from matplotlib import pyplot as plt

fn=normalize(forward)

lin=np.linspace(0,500,1000)
plt.plot(lin,lin*0)

poly=compilepolys(camera_pos,fn)
print(poly)

poly=[poly[0],poly[1],poly[2],poly[3],poly[4]]

#print(solveQuartic(poly[4],poly[3],poly[2],poly[1],poly[0]))

#print(solveQuartic(poly[4]/24,poly[3]/6,poly[2]/2,poly[1],poly[0]))

#print(ray)
e=calcpoly(lin)



#print(y)
plt.plot(lin,e,label="e")
#plt.plot(lin,calcstep(fn.T,ray[:,None]).ravel(),label="step")

#plt.plot(lin,2*e/(np.sum(xyz*fn,axis=-1)+np.linalg.norm(xyz,axis=-1)),label="stepnewton")

dy=raydyinit
rayOrigin=camera_pos.copy()
rayDir=fn.copy()
pointsx=[]
pointse=[]
x=0



#print(pointse,pointsx)



#plt.plot(pointsx,pointse,label="iterations")



plt.ylim((-5,5))
plt.xlim((0,50))
plt.legend()
plt.show()


while True:
    camera_matrix=client["camera_matrix"]
    camera_pos=client["camera_pos"]
    forward=camera_matrix@np.array([0,0,1])
    poly=compilepolys(camera_pos,forward)
    print(np.array(poly))
    solveQuartic(*poly[::-1])
    time.sleep(0.1)