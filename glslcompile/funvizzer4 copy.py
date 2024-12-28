import numpy as np
import sys
from glslhelper import *
sys.path.append('./')
from variable_listener import Client
#from scene import scene
import time
def normalize(v):
    norm = np.linalg.norm(v)
    return v / norm if norm != 0 else v

from numpy import sqrt,cos,arccos,sign



    
dcomplex=vec2

def complex_multiply(a, b):
    # Complex multiplication: (a.x + i*a.y) * (b.x + i*b.y)
    return dcomplex(
        a.x * b.x - a.y * b.y, # Real part
        a.x * b.y + a.y * b.x  # Imaginary part
    );

def complex_divide(a, b):
    # Complex division: (a.x + i*a.y) / (b.x + i*b.y)
    denominator = b.x * b.x + b.y * b.y;
    return dcomplex(
        (a.x * b.x + a.y * b.y) / denominator, # Real part
        (a.y * b.x - a.x * b.y) / denominator  # Imaginary part
    );
degree=4
def initial_roots(center=dcomplex(0, 0)):
    golden_angle = (3 - np.sqrt(5)) * np.pi  # Golden angle in radians
    r1 = dcomplex(np.cos(golden_angle), np.sin(golden_angle))  # Base complex number
    
    # Initialize the roots array
    roots = [dcomplex(1, 0)]
    for i in range(1, degree):
        roots.append(complex_multiply(roots[i - 1], r1))  # Multiply by r1 iteratively
    
    # Add the center to all roots
    for i in range(degree):
        roots[i] = dcomplex(roots[i].x + center.x, roots[i].y + center.y)
    
    return roots

def aberth_method(roots,coefficients,pdegree):
    def evaluatePolynomial(x, coefficients):
        result = dcomplex(coefficients[degree],0);
        for i in reversed(range(degree)):
            result = complex_multiply(result , x) + dcomplex(coefficients[i],0);
        return result;
    def evaluatePolynomialDerivative(x,coefficients):
        result = dcomplex(coefficients[degree]*degree,0);
        for i in reversed(range(1,degree)):
            result = complex_multiply(result , x) + dcomplex(coefficients[i]*i,0);
        return result;

    threshold = 1e-8; # Convergence threshold
    max_iterations = 40; # Maximum iterations to avoid infinite loop

    for iter in range(max_iterations):
        max_change = 0.0; # Track the largest change in roots

        for k in range(pdegree):
            # Evaluate the polynomial and its derivative at the current root
            a = complex_divide(
                evaluatePolynomial(roots[k], coefficients),
                evaluatePolynomialDerivative(roots[k], coefficients)
            );

            s = dcomplex(0.0); # Summation term
            for j in range(pdegree):
                if (j != k): # Avoid self-interaction
                    diff = roots[k] - roots[j];
                    #dnumcomplex denom = dot(diff, diff); # Squared magnitude
                    #if (denom > threshold) { # Check against threshold
                    s += complex_divide(dcomplex(1.0, 0.0), diff);
                    #}

            # Compute the correction term
            w = complex_divide(a, dcomplex(1.0, 0.0) - complex_multiply(a, s));
            roots[k] -= w; # Update the root
            print(length(w))

            # Track the maximum change in root
            max_change = float(max(max_change, length(w)));

        # If the maximum change is smaller than the threshold, stop early
        if (max_change < threshold):
            #print(iter)
            break # Converged, exit the loop
        
    return roots













client=Client()

print(client["pyfun"])

exec(client["pyfun"],globals())
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
def calcpolyderiv(x):
    return poly[1]+poly[2]*x/2+poly[3]*x**2/3+poly[4]*x**3/4
def calcpoly2deriv(x):
    return poly[2]/2+poly[3]*x/6+poly[4]*x**2/12

def newtoniter():
    x=0
    for i in range(20):
        x-=calcpolyderiv(x)/calcpoly2deriv(x)
    return x
EPSILON_ROOTS=0.001;
def newtoniter(iter=20,x=0):
    shift=x;
    bestval=float("inf");
    for i in range(iter):
        f =calcpoly(x);
        df=calcpolyderiv(x);
        if(x>=0.0 and abs(f)<bestval):
            bestval=abs(f);
            shift=x;
        #print(x)
        if(bestval<EPSILON_ROOTS):break;
        x-=f/df;
    
    return vec2(shift,bestval);





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

def solveQuartic(a,b,c,d,e):
    #//https://en.wikipedia.org/wiki/Quartic_function
    delta0=c*c-3*b*d+12*a*e;
    delta1=2*c*c*c-9*b*c*d+27*b*b*e+27*a*d*d-72*a*c*e;
    p=(8*a*c-3*b*b)/(8*a*a);
    q=(b*b*b-4*a*b*c+8*a*a*d)/(8*a*a*a);
    
    det=delta1*delta1-4*delta0*delta0*delta0
    if det<0:
        phi=arccos(delta1/(2*sqrt(delta0*delta0*delta0)))
        S=0.5*sqrt(-p*2/3+(2*sqrt(delta0)*cos(phi/3))/(3*a));
    elif det!=0 and delta0==0:
        Q=pow(0.5*(delta1+sign(delta1)*sqrt(delta1*delta1)),1./3);
        S=0.5*sqrt(-p*2/3+(Q+delta0/Q)/(3*a));
    else:
        Q=pow(0.5*(delta1+sqrt(delta1*delta1-4*delta0*delta0*delta0)),1./3);
        S=0.5*sqrt(-p*2/3+(Q+delta0/Q)/(3*a));
        #if S==0:


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

camera_matrix=client["window.camera_matrix"].astype(np.float32)
camera_pos=client["window.camera_pos"].astype(np.float32)
forward=camera_matrix@np.array([0,0,1])

print(forward,camera_pos)
from matplotlib import pyplot as plt

fn=normalize(forward).astype(np.float32)

lin=np.linspace(0,50,10000).astype(np.float32)
print(lin.dtype)
plt.plot(lin,lin*0)

poly=compilepolys(camera_pos,fn)
print(poly)

#poly=[poly[0],poly[1],poly[2],poly[3],poly[4]]



print(solveQuartic(poly[4],poly[3],poly[2],poly[1],poly[0]))

#print(solveQuartic(poly[4]/24,poly[3]/6,poly[2]/2,poly[1],poly[0]))

#print(ray)
e=calcpoly(lin.astype(np.float64))
e2=calcpoly(lin.astype(np.float32))




#print(y)
plt.plot(lin,e,label="e64")
plt.plot(lin,e2,label="e32")

shift=newtoniter()
print(shift)
#shift=-poly[3]/(4*poly[4])
# poly=compilepolys(camera_pos+fn*shift,fn)
# print(poly,shift)
# e2=calcpoly(lin.astype(np.float32))
# plt.plot(lin+shift,e2,label="es32")
#plt.plot(lin,calcstep(fn.T,ray[:,None]).ravel(),label="step")

#plt.plot(lin,2*e/(np.sum(xyz*fn,axis=-1)+np.linalg.norm(xyz,axis=-1)),label="stepnewton")

dy=raydyinit
rayOrigin=camera_pos.copy()
rayDir=fn.copy()
pointsx=[]
pointse=[]
x=0

xs=[]
ys=[]


l=0.8585858585858587


#for l in np.linspace(-1,1,100):
l2=l/100
print(l2)
poly=compilepolys(camera_pos+(fn*l2).astype(np.float32),fn)
# e2=calcpoly(lin.astype(np.float32))
# plt.plot(lin,e2,label="e32")
pdegree=degree;
while(abs(poly[pdegree])<1E-12):pdegree-=1;
center=dcomplex(newtoniter().x,0)
#center=dcomplex(0,0)
#print("center",center)
#print(initial_roots())
roots=aberth_method(initial_roots(center),poly,pdegree)
print("roots",roots)
x=float("inf");
for i in range(pdegree):
    r=vec2(roots[i]);
    if((r.y*r.y)<(1E-2) and r.x>0):
        x=min(r.x,x)
xs.append(l)
ys.append(x+l2)
print(x)


ys=np.array(ys)
print(ys)
ys=ys-ys.mean()
# print(ys.max())
# ys*=1/ys.max()
plt.plot(xs,ys)
plt.show()
exit()
    #print(x)
# x=np.array([0,1,2,3,4])
# # Define more points for higher precision
# x = np.linspace(-1, 1, 10)  # 10 points in the range [-1, 1]
# y = np.array([compilepolys(camera_pos + fn * i, fn)[0] for i in x])[:, None]  # Example function calls

# # Create the Vandermonde matrix
# A = np.vander(x, 5, increasing=True)
# A_pinv = np.linalg.pinv(A)
# c = A_pinv @ y
# c = c.squeeze(-1)  # Flatten the coefficients array

# # Print the results
# print("Pseudo-Inverse of Vandermonde Matrix:\n", A_pinv)
# print("Polynomial Coefficients:\n", c)
# print(poly)
#plt.plot(pointsx,pointse,label="iterations")



plt.ylim((-5,5))
plt.xlim((0,50))
plt.legend()
plt.show()


# while True:
#     camera_matrix=client["camera_matrix"]
#     camera_pos=client["camera_pos"]
#     forward=camera_matrix@np.array([0,0,1])
#     poly=compilepolys(camera_pos,forward)
#     print(np.array(poly))
#     solveQuartic(*poly[::-1])
#     time.sleep(0.1)