import numpy as np
import sys
sys.path.append('./')
from variable_listener import Client
from scene import scene

def normalize(v):
    norm = np.linalg.norm(v)
    return v / norm if norm != 0 else v


import numpy as np

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

# Define the scene2 function for computing derivatives
def scene2(rayDir, p):
    # Get the scene values at three points (p - rayDir * epsilon, p, p + rayDir * epsilon)
    w1 = scene((p - rayDir * EPSILON_DERIV).T)[3]
    w2 = scene(p.T)[3]
    w3 = scene((p + rayDir * EPSILON_DERIV).T)[3]

    # Derivatives of the scene function (value, first, second derivative)
    derivs = np.array([
        w2,                     # C = quartic(p)
        (w3 - w1) / (2 * EPSILON_DERIV),  # B = quartic'(p)
        (w3 - 2*w2 + w1) / (EPSILON_DERIV**2)  # A = quartic''(p)/2
    ])
    
    return derivs

# Define the calcstep function
def calcstep(rayDir, p):
    derivs = scene2(rayDir, p)
    
    C = derivs[0]  # C = quartic(p)
    B = derivs[1]  # B = quartic'(p)
    A = -np.abs(derivs[2])  # A = -abs(quartic''(p)/2)

    newtonstep = np.abs(C / B)  # Newton step
    
    discriminant = (B / (2 * A))**2 - C / A
    mid = -B / (2 * A)

    # Handle the conditional check for discriminant < 0 using np.where
    discriminant_condition = discriminant < 0
    s = np.sqrt(np.maximum(discriminant, 0))  # Ensure that sqrt doesn't get negative values
    
    # Use np.where to handle the conditional logic
    step = np.where(discriminant_condition, newtonstep, np.min([newtonstep, np.abs(mid + s), np.abs(mid - s)], axis=0))

    return step



client=Client()



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

lin=np.linspace(0,50,1000)
plt.plot(lin,lin*0)
ray=fn*lin[...,None]+camera_pos
#print(ray)
x,y,z,e=scene(ray.T)
xyz=np.vstack([x,y,z]).T
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


for i in range(128):
    s=scene((rayOrigin+x*rayDir))
    magnitude =s[-1]
    pointsx.append(x)
    pointse.append(magnitude)
    #dy=0.5*dy+2*abs(dot(s.xyz,rayDir));
    dy=rayalpha*dy+rayfactor*np.linalg.norm(s[:-1])
    x +=  (magnitude/(dy+rayepsilon))
#print(pointse,pointsx)



#plt.plot(pointsx,pointse,label="iterations")



plt.ylim((-5,5))
plt.xlim((0,50))
plt.legend()
plt.show()


e=scene((fn*np.array([0,1,2,3,4,5])[...,None]+camera_pos).T)[-1]
print("{"+",".join(f"({x},{y})" for x,y in zip([0,1,2,3,4,5],e))+"}")