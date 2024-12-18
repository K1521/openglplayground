import tracing_helper
import numpy as np
import opgraphtofun
import glslprog
import sys
sys.path.append('./')
from variable_listener import Server
import algebra.dcga as dcga
server=Server()

#print(fragment_src)
with open("./glslcompile/simpletorusshader.glsl","r")as f:
    fragment_src=f.read()

window=tracing_helper.mywindow(800,600,fragment_src)
tracing_helper.keymouseeventhandler(window)
window.loop()