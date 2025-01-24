import sys
sys.path.append('./glslcompile')
import tracing_helper
import numpy as np
import glslprog

sys.path.append('./')
from variable_listener import Server
import algebra.dcga as dcga
from matrixalgebra4 import TraceBasisDCGA
server=Server()




tracer=TraceBasisDCGA()
def normalizerows(matrix):
    return matrix / np.linalg.norm(matrix, axis=1, keepdims=True)

def remove_duplicate_rows(matrix, threshold=0.999):
    # Normalize each row (L2 normalization)
    normalized_matrix = normalizerows(matrix)
    
    # Compute the pairwise dot product of rows
    dot_product = normalized_matrix @ normalized_matrix.T
    
    # Create a mask for rows that are not duplicates (close to 1 or -1)
    mask = np.ones(matrix.shape[0], dtype=bool)
    
    for i in range(matrix.shape[0]):
        if mask[i]:
            for j in range(i + 1, matrix.shape[0]):
                # If the dot product is too close to 1 (indicating rows are multiples), mask the row
                if abs(dot_product[i, j]) > threshold:
                    mask[j] = False
    
    # Use the mask to select only the non-duplicate rows
    return matrix[mask]

def remove_rows_by_rank(matrix):
    m=matrix[None,0]
    rank=1
    for i in range(1,len(matrix)):
        ms=np.vstack([m,matrix[i]])
        rs= np.linalg.matrix_rank(ms)
        if rs>rank:
            rank=rs
            m=ms
    #print(m)
    return m
    
def remove_rows_by_rankQR(matrix):
    # QR decomposition gives us the row space and rank of the matrix
    Q, R = np.linalg.qr(matrix)  # Perform QR decomposition
    print(Q)
    print(R)
    rank = np.linalg.matrix_rank(R)  # The rank is the number of non-zero rows in R
    independent_rows = R[:rank, :]
    return independent_rows
    # Now, we'll keep the rows of matrix corresponding to the rank
    #return matrix[:rank]

def funtoviz(point):
    obj=dcga.toroid(2,0.5)
    #obj=dcga.Plane(1,1,1,0.01)
    obj=dcga.Plane(1,3,2,0.5).outer(dcga.toroid(2,0.5))
    #obj=dcga.Plane(1,1,1,0.01).inner(dcga.toroid(2,0.5))

    #obj=sanwich(t,obj)
    prod=point.inner(obj)
    return prod
funmat=tracer.trace_linear(funtoviz)


# m=funmat/np.linalg.norm(funmat,axis=1,keepdims=True)
# print(m@m.T)
print(funmat.shape,np.linalg.matrix_rank(funmat.T))
#funmat=remove_duplicate_rows(funmat)
funmat=remove_rows_by_rankQR(funmat)/5
print(funmat.shape,np.linalg.matrix_rank(funmat.T))
#print(funmat)
print(tracer.poly_basis_vec)
#funmat=normalizerows(funmat)*50

#print(tracer.poly_basis_monoms)
#scene=opgraphtofuncasadi.generate_glsl_code(funtovisualize)
prog=glslprog.glslprogramm(version="440")
prog.parts.append(glslprog.glslprogrammpart(bodypath="./gaalop_to_fragment/fragmentshader12.glsl"))
prog.parts.append(glslprog.glslprogrammpart(header=f"""
const int polybasislength={len(tracer.poly_basis_vec)};
const int numpolys={len(funmat)};
//const int MAXPOLYDEGREE=4;
"""))
print(prog.parts[-1].header)

fragment_src = str(prog)
#print(fragment_src)
with open("./gaalop_to_fragment/lastfragment.glsl","w")as f:
    f.write(fragment_src)


window=tracing_helper.mywindow(800,600,fragment_src)
#window.variables.additem(key,value,"glUniform1fv")
#window.variables.additem(key,value,"glUniform1iv")
tracing_helper.keymouseeventhandler(window)
mat=(funmat)
#print(mat)


#TODO REMOVE DOUBLE ROWS



#window.variables.additem("coefficientsxyz[0][0]",funmat,"glUniform1fv")
import OpenGL.GL as GL
GL.glUseProgram(window.shader)
for i,j in np.ndindex(funmat.shape):
    #print(i,j)
    location = GL.glGetUniformLocation(window.shader,f"coefficientsxyz[{i}][{j}]")
    GL.glUniform1f(location, float(funmat[i,j]))




import cProfile
import pstats

profiler=cProfile.Profile()
with profiler:
    for i in range(10):
        window.loopiter()
        window.variables["cameraPos"]+=np.array([0,0.03,0])

pstats.Stats(profiler).strip_dirs().sort_stats("tottime").print_stats(10)



#polypowmat=tracer.polypowmat()

window.loop()