import numpy as np
import sys
sys.path.append('./')
from variable_listener import Client
#from scene import scene

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
poly=None

def calcpoly(x):
    return poly[0]+poly[1]*x+poly[2]*x**2/2+poly[3]*x**3/6+poly[4]*x**4/24

def compilepolys(p,d):
    a=0
    ox=p[0]
    oy=p[1]
    oz=p[2]
    dx=d[0]
    dy=d[1]
    dz=d[2]
    n0=ox
    n1=a
    n2=dx
    n3=(n1*n2)
    n4=(n0+n3)
    n5=(n4*n4)
    n6=-16
    n7=(n5*n6)
    n8=oy
    n9=dy
    n10=(n1*n9)
    n11=(n8+n10)
    n12=(n11*n11)
    n13=(n12*n6)
    n14=oz
    n15=dz
    n16=(n1*n15)
    n17=(n14+n16)
    n18=(n17*n17)
    n19=(n5+n12+n18)
    n20=0.5
    n21=(n19*n20)
    n22=-0.5
    n23=(n21+n22)
    n24=7.5625
    n25=(n23*n23*n24)
    n26=(n21+n20)
    n27=-13.0625
    n28=(n23*n26*n27)
    n29=22.5625
    n30=(n26*n26*n29)
    n31=(n7+n13+n25+n28+n28+n30)
    n32=(n26*n29)
    n33=-26.125
    n34=(n23*n33)
    n35=(n26*n33)
    n36=(n23*n24)
    n37=(n32+n32+n34+n35+n36+n36)
    n38=(n37*n20)
    n39=(n17*n38)
    n40=(n39+n39)
    n41=(n15*n40)
    n42=(n38+n6)
    n43=(n11*n42)
    n44=(n43+n43)
    n45=(n9*n44)
    n46=(n4*n42)
    n47=(n46+n46)
    n48=(n2*n47)
    n49=(n41+n45+n48)
    n50=(n42*n9)
    n51=(n4*n2)
    n52=(n51+n51)
    n53=(n11*n9)
    n54=(n53+n53)
    n55=(n17*n15)
    n56=(n55+n55)
    n57=(n52+n54+n56)
    n58=(n57*n20)
    n59=(n58*n24)
    n60=(n58*n33)
    n61=(n58*n29)
    n62=(n59+n59+n60+n60+n61+n61)
    n63=(n62*n20)
    n64=(n11*n63)
    n65=(n50+n50+n64+n64)
    n66=(n9*n65)
    n67=(n42*n2)
    n68=(n4*n63)
    n69=(n67+n67+n68+n68)
    n70=(n2*n69)
    n71=(n38*n15)
    n72=(n17*n63)
    n73=(n71+n72+n72+n71)
    n74=(n15*n73)
    n75=(n66+n70+n74)
    n76=(n63*n15)
    n77=(n15*n15)
    n78=(n77+n77)
    n79=(n2*n2)
    n80=(n79+n79)
    n81=(n9*n9)
    n82=(n81+n81)
    n83=(n78+n80+n82)
    n84=(n83*n20)
    n85=(n84*n24)
    n86=(n84*n33)
    n87=(n84*n29)
    n88=(n85+n85+n86+n86+n87+n87)
    n89=(n88*n20)
    n90=(n17*n89)
    n91=(n76+n76+n76+n76+n90+n90)
    n92=(n15*n91)
    n93=(n63*n9)
    n94=(n11*n89)
    n95=(n93+n93+n93+n93+n94+n94)
    n96=(n9*n95)
    n97=(n63*n2)
    n98=(n4*n89)
    n99=(n97+n97+n97+n97+n98+n98)
    n100=(n2*n99)
    n101=(n92+n96+n100)
    n102=(n89*n15)
    n103=(n102+n102+n102+n102+n102+n102)
    n104=(n15*n103)
    n105=(n89*n9)
    n106=(n105+n105+n105+n105+n105+n105)
    n107=(n9*n106)
    n108=(n89*n2)
    n109=(n108+n108+n108+n108+n108+n108)
    n110=(n2*n109)
    n111=(n104+n107+n110)
    return n31,n49,n75,n101,n111


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

poly=compilepolys(camera_pos,fn)
print(poly)

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
