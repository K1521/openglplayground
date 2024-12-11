import tracing_helper
import numpy as np
import opgraphtofun
import glslprog
import sys
sys.path.append('./')
from variable_listener import Server
import algebra.dcga as dcga
server=Server()





def funtovisualize(x,y,z):


    t=dcga.Translator(3,4,5)
    #print((t*t.reverse()).blades)
    def sanwich(V,m):
        return V*m*dcga.inverse(V)

    #xyz=(opg.VarNode("x"),opg.VarNode("y"),opg.VarNode("z"))
    point=dcga.point(x,y,z)
    obj=dcga.toroid(2,0.5)
    #obj=dcga.Plane(1,1,1,1)

    #obj=sanwich(t,obj)
    iprod=point.inner(obj)
    return list(iprod.blades.values())
#import opgraphtofuncasadi
scene,_=opgraphtofun.makefuntailor(funtovisualize)
#scene=opgraphtofuncasadi.generate_glsl_code(funtovisualize)
prog=glslprog.glslprogramm(version="440")
prog.parts.append(glslprog.glslprogrammpart(bodypath="./glslcompile/fragmentshader.glsl"))
prog.parts.append(scene)


fragment_src = str(prog)
#print(fragment_src)
with open("./glslcompile/lastfragment.glsl","w")as f:
    f.write(fragment_src)

window=tracing_helper.mywindow(800,600,fragment_src)
tracing_helper.keymouseeventhandler(window)
window.loop()