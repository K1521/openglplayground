
import sympy as sy
import numpy as np




x,y,z,r,rp,rm=sy.symbols("x,y,z,r,rp,rm")
# # in the following pointdict gets calculated
# import sys
# sys.path.append('./')
# import algebra.dcga as dcga
# p=dcga.point(x,y,z)
# #r=x*x+y*x+z*z
# #rm=(r-1)/2
# #rp=(r+1)/2
# for b in p.blades:
#     expr=p.blades[b]
#     expr=expr.simplify().subs(1,1.0).subs(x**2 + y**2 + z**2 + 1.0,r+1).subs(x**2 + y**2 + z**2 - 1.0,r-1)
#     p.blades[b]=eval(str(expr).replace("r + 1","rp*2").replace("r - 1","rm*2"))
# pointdict=p.blades
# #alternatively
# pointdict={
#     33: x**2, 65: x*y, 129: x*z, 257: rm*x, 513: rp*x,
#     34: x*y, 66: y**2, 130: y*z, 258: rm*y, 514: rp*y,
#     36: x*z, 68: y*z, 132: z**2, 260: rm*z, 516: rp*z,
#     40: rm*x, 72: rm*y, 136: rm*z, 264: rm**2, 520: rm*rp,
#     48: rp*x, 80: rp*y, 144: rp*z, 272: rm*rp, 528: rp**2
# }

# list(pointdict.values())
# rows = np.array([
#     [33, 65, 129, 257, 513],
#     [34, 66, 130, 258, 514],
#     [36, 68, 132, 260, 516],
#     [40, 72, 136, 264, 520],
#     [48, 80, 144, 272, 528]
# ])
# print(np.bitwise_and.reduce(rows, axis=0))
# print(np.bitwise_and.reduce(rows, axis=1))

a={ 1:x, 2:y,  4:z,  8:rm, 16:rp}
b={32:x,64:y,128:z,256:rm,512:rp}
pointdict ={ai^bi:ab*bb for ai,ab in a.items() for bi,bb in b.items()}


#print(pointdict)
# print(len(set(pointdict.values())),len(pointdict))

eye=np.eye(len(set(pointdict.values())))

basisdict={}#basisnum->npvector
sytobasisnum=dict()#basisnum<-sympybasis #only first ocurrence
basisvector=[]#index->sympybasis

for basisnum,basissy in pointdict.items():
    if basissy in sytobasisnum:
        basisdict[basisnum]=basisdict[sytobasisnum[basissy]]
    else:
        indexcounter=len(basisvector)
        basisvector.append(basissy)
        basisdict[basisnum]=eye[indexcounter]
        sytobasisnum[basissy]=basisnum
# print(basisvector)
# print(basisdict)
# print(len(basisvector),len(basisdict))



# i need basisdict=basisnum->npvector and basisvector=index->sympybasis
# basisdict geht von der basisblade zu einer neuen basis.
# basisdict ist also eine sparse matrix zu einer neuen basis
# die neue basis ist basisvector
# pointdict=basisdict@basisvector
# 







#x,y,z,(r-1),(r+1)
