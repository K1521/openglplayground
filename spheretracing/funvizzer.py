import numpy as np
import sys
sys.path.append('./')
from variable_listener import Client
from scene import scene

def normalize(v):
    norm = np.linalg.norm(v)
    return v / norm if norm != 0 else v




client=Client()



raydyinit=client["raydyinit"]=1
rayfactor=client["rayfactor"]=1
rayalpha=client["rayalpha"]=0
rayepsilon=client["rayepsilon"]=0.01
raypow=client["raypow"]=1

camera_matrix=client["camera_matrix"]
camera_pos=client["camera_pos"]
forward=camera_matrix@np.array([0,0,1])

print(forward,camera_pos)
from matplotlib import pyplot as plt

fn=normalize(forward)

lin=np.linspace(0,50,1000)
ray=fn*lin[...,None]+camera_pos
#print(ray)
x,y,z,e=scene(ray.T)
xyz=np.vstack([x,y,z]).T
#print(y)
plt.plot(lin,e,label="e")
plt.plot(lin,e**.5,label="sqrte")
plt.plot(lin,np.linalg.norm(xyz,axis=-1),label="norm")
plt.plot(lin,e/np.linalg.norm(xyz,axis=-1),label="stepnorm")
plt.plot(lin,e/np.sum(xyz*fn,axis=-1),label="stepnewton")
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
print(pointse,pointsx)



# fdx=0.01
# hlast=1
# dhlast=0
# dx=0
# for i in range(128):
#     s=scene((rayOrigin+x*rayDir))
#     h =s[-1]
#     sxyz=s[:-1]
#     dh=np.dot(sxyz,rayDir)
#     error=abs((dhlast*dx+hlast)/h)
#     #print(error)
#     print(h,error,fdx,x)
#     if 1.01>error>0.99 and fdx<1:
#         fdx*=1.1
#     else:
#         fdx/=1.3
#     pointsx.append(x)
#     pointse.append(dx)
#     #dy=0.5*dy+2*abs(dot(s.xyz,rayDir));
#     dy=rayalpha*dy+rayfactor*np.linalg.norm(sxyz)
#     dx=(h/(dy+rayepsilon))*fdx
#     x+=dx
#     hlast=h
#     dhlast=dh
plt.plot(pointsx,pointse,label="iterations")



plt.ylim((-5,5))
plt.xlim((0,50))
plt.legend()
plt.show()
