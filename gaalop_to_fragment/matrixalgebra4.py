import sympy as sy
import numpy as np






class TraceBasisDCGA:
    def __init__(self):


        self.point,self.poly_basis_vec =  TraceBasisDCGA._makedcgabasis()
    
    @staticmethod
    def _makedcgabasis():
        # i need basisdict=basisnum->npvector and basisvector=index->sympybasis
        # basisdict geht von der basisblade zu einer neuen basis.
        # basisdict ist also eine sparse matrix zu einer neuen basis
        # die neue basis ist basisvector
        # pointdict=basisdict@basisvector

        #r=x*x+y*x+z*z
        #rm=(r-1)/2
        #rp=(r+1)/2

        x,y,z,r,rp,rm=sy.symbols("x,y,z,r,rp,rm")
        # # in the following pointdict gets calculated
        import sys
        sys.path.append('./')
        import algebra.dcga as dcga
        # p=dcga.point(x,y,z)
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
        point=dcga.multivec.frombasis(basisdict)
        return point,basisvector
        

    
    # def polypowmat(self):
    #     array=np.zeros((len(self.poly_basis_monoms),3))
    #     for i,m in enumerate(self.poly_basis_monoms):
    #         array[i]=m.powers
    #     return array
    

    def trace_linear(self, func):
        fun_mat = np.stack(list(func(self.point).blades.values()))
        return fun_mat

    





        
    
"""
# The expansion of (a*x + b*y + c*z + t)^n is given by:
# Sum over all i, j, k, l such that i + j + k + l = n:
# (a*x + b*y + c*z + t)^n = Σ (from i+j+k+l=n) [binomial(n, i, j, k, l) * a^i * b^j * c^k * t^l * x^i * y^j * z^k]
# where binomial(n, i, j, k, l) = n! / (i! * j! * k! * l!)





"""

if __name__=="__main__":
    import sys
    sys.path.append('./')
    import algebra.dcga as dcga
    tracer=TraceBasis(dcga.point)

    def funtoviz(point):
        obj=dcga.toroid(2,0.5)
        #obj=dcga.Plane(1,1,1,1)
        #obj=dcga.Plane(1,1,1,0.01).outer(dcga.toroid(2,0.5))

        #obj=sanwich(t,obj)
        prod=point.inner(obj)
        return prod
    funmat=tracer.trace_linear(funtoviz)
    print(tracer.move_basis(1,2,3))
    
    rotation_matrix = np.array([[ 0.61216981 ,-0.42242272 , 0.66778751],
 [ 0.20681806 , 0.87242045 , 0.4435741 ],
 [-0.76466516 , 0.24842691 , 0.59553687]])
    rotation_matrix[0]/=np.linalg.norm(rotation_matrix[0])
    rotation_matrix[1]/=np.linalg.norm(rotation_matrix[1])
    rotation_matrix[2]=np.cross(rotation_matrix[0],rotation_matrix[1])
    rotation_matrix[1]=np.cross(rotation_matrix[2],rotation_matrix[0])
    print(rotation_matrix)
    # rotation_matrix=np.eye(3)
    print(tracer.transform_basis(rotation_matrix,np.array([1,2,3])))
    print()

    print(funmat)
    print(funmat.shape)
    # print(tracer.point_mat)
    # print(str((funmat@tracer.point_mat@shiftmat@tracer.poly_basis_vec)[0]).replace("**","^"))