import sys
sys.path.append('./glslcompile')
import tracing_helper
import numpy as np
import glslprog

sys.path.append('./')
from variable_listener import Server
import algebra.dcga as dcga
from matrixalgebra3 import TraceBasis
server=Server()




tracer=TraceBasis(dcga.point)

def funtoviz(point):
    obj=dcga.toroid(2,0.5)
    #obj=dcga.Plane(1,1,1,1)
    #obj=dcga.Plane(1,1,1,0.01).outer(dcga.toroid(2,0.5))

    #obj=sanwich(t,obj)
    prod=point.inner(obj)
    return prod
funmat=tracer.trace_linear(funtoviz)
shiftmat=tracer.move_basis(1,0,0)@tracer.move_basis(0,1,0)
print(funmat)
print(tracer.point_mat)
print(str((funmat@tracer.point_mat@shiftmat@tracer.poly_basis_vec)[0]).replace("**","^"))


