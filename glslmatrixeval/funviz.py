import numpy as np
import sys
sys.path.append('./')
from variable_listener import Client
#from scene import scene
import time
def normalize(v):
    norm = np.linalg.norm(v)
    return v / norm if norm != 0 else v

from numpy import sqrt,cos,arccos,sign

client=Client()

# print(np.array([1,2,3])[None])
# print(np.array([1,2,3])[None,:])
# print(np.array([1,2,3])[:,None])
# exit()
polypowmat=client["polypowmat"].astype(int)
coefficients=client["matupdater.mat"]
#print(polypowmat)
#print(coefficients)
def lerp(a, b, t):
    return (1 - t) * a + t * b
"""
ivec3 basis=polybasis[j];
float px=powersx[basis.x];
float py=powersy[basis.y];
float pz=powersz[basis.z];
float pdx = (basis.x > 0) ? basis.x * powersx[basis.x - 1] : 0.0;
float pdy = (basis.y > 0) ? basis.y * powersy[basis.y - 1] : 0.0;
float pdz = (basis.z > 0) ? basis.z * powersz[basis.z - 1] : 0.0;

m[j]=vec4(pdx*py*pz,px*pdy*pz,px*py*pdz,px*py*pz);
        """
def gaussnewton(pos):
    m=np.zeros((len(polypowmat),4))
    #print(pos)
    #pos**polypowmat()
    for i,power in enumerate(polypowmat):
        p=pos**polypowmat[i]
        pd=polypowmat[i]*pos**np.maximum(polypowmat[i]-1,0)
        m[i]+=np.array([p[0]*p[1]*p[2],pd[0]*p[1]*p[2],p[0]*pd[1]*p[2],p[0]*p[1]*pd[2]])
    
    fxyz=np.zeros((len(coefficients),4))
    xx=xy=xz=yy=yz=zz=0
    JTf=0
    for i,c in enumerate(coefficients):

        f,fdx,fdy,fdz=fxyz[i]=m.T@c
        xx+=fdx**2
        xy+=fdx*fdy
        xz+=fdx*fdz
        yy+=fdy**2
        yz+=fdy*fdz
        zz+=fdz**2
        JTf+=f*np.array([fdx,fdy,fdz])
    J=fxyz[:,1:]
    f=fxyz[:,0]

    #print((J.T@J)/np.array([[xx,xy,xz],[xy,yy,yz],[xz,yz,zz]])-1)
    #print("-------------------")
    #print(fxyz)
    
    #return J[0]
    JTJ=J.T@J+np.eye(3)*1e-5
    #print(JTJ)
    JTF=J.T@f

    #print(JTF/JTf-1)
    dpos=np.linalg.inv(JTJ)@JTF
    return pos-dpos



print(coefficients[0])
#print(polypowmat)
#exit()
lin=np.linspace(-1,1,100)*2

z=lin[:,None,None]
y=lin[None,:,None]
x=lin[None,None,:]

scalar_field=0
for i in range(len(polypowmat)):
    px,py,pz=polypowmat[i]
    scalar_field+=x**px*y**py*z**pz*coefficients[0,i]

import pyvista as pv

# Create a structured grid
grid = pv.StructuredGrid(*np.meshgrid(
    lin,lin,lin,
    indexing='ij'
))
grid["scalar_field"] = scalar_field.ravel()

# Isosurface extraction and visualization
isosurface = grid.contour(isosurfaces=np.array([-0.3,-0.2,-0.1,0,0.3,0.2,0.1])*4)

# Plot the isosurfaces
lin=np.linspace(0,1,10)
field=lerp(np.array([0,0,0]),np.array([0,0,2]),lin[:,None])[None]+lerp(np.array([-1,0,0]),np.array([1,0,0]),lin[:,None])[:,None]+lerp(np.array([0,-1,0]),np.array([0,1,0]),np.linspace(0,1,5)[:,None])[:,None,None]
plotter = pv.Plotter()
points=field.reshape((-1,3))#np.array([[0,0,0]])
plotter.add_arrows(points, np.array([gaussnewton(p)for p in points])-points, mag=1,opacity=0.99)
plotter.add_points(np.array([[0,0,0]]), render_points_as_spheres=True, point_size=15, opacity=0.99)
plotter.add_mesh(isosurface, cmap="viridis", opacity=0.6)
plotter.add_axes()
plotter.show_grid()

pos=np.array([0,0,0])
for i in range(20):
    pos=gaussnewton(pos)
    print(pos)


plotter.show()



exit()





lin=np.linspace(0,1,20)


print(lerp(np.array([0,0,-1]),np.array([0,0,1]),lin[:,None]))
field=lerp(np.array([0,0,0]),np.array([0,0,2]),lin[:,None])[None,None,:]+lerp(np.array([-1,0,0]),np.array([1,0,0]),lin[:,None])[None,:,None]+lerp(np.array([0,-1,0]),np.array([0,1,0]),np.linspace(0,1,20)[:,None])[:,None,None]
points=field.reshape((-1,3))
vectors=points
vectors=np.array([gaussnewton(p)for p in vectors])-points
# vectors=np.array([jakobi(p)for p in vectors])
# vectors=np.array([jakobi(p)for p in vectors])
# vectors=np.array([jakobi(p)for p in vectors])
# vectors-=points
    #drawvector from p to p2

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

x, y, z = points.T
u, v, w = vectors.T

fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
ax.quiver(x, y, z, u, v, w, length=0.2, normalize=True, color="blue")

plt.show()