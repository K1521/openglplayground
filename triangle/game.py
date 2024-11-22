import pygame as pg
import OpenGL.GL as gl
import OpenGL.GL.shaders 
import numpy as np
import ctypes
class App:
    def __init__(self):
        pg.init()
        pg.display.set_mode((640,480),pg.OPENGL|pg.DOUBLEBUF)
        self.clock =pg.time.Clock()

        gl.glClearColor(0.1,0.2,0.2,1)
        self.shader=self.createShader("./triangle/vertshader.txt","./triangle/fragshader.txt")
        gl.glUseProgram(self.shader)
        self.triangle=Triangles()
        self.mainloop()
    def mainloop(self):
        running=True
        while running:
            for event in pg.event.get():
                if event.type==pg.QUIT:
                    running = False
            gl.glClear(gl.GL_COLOR_BUFFER_BIT)
            gl.glUseProgram(self.shader)
            gl.glDrawArrays(gl.GL_TRIANGLES,0,self.triangle.vertexcount)
            pg.display.flip()
            self.clock.tick(60)
        pg.quit()
    def createShader(self,vertexshaderpath,fragmentshaderpath):
        with open(vertexshaderpath)as f:
            vertscr=f.read()
        with open(fragmentshaderpath)as f:
            fragscr=f.read()

        return gl.shaders.compileProgram(
            gl.shaders.compileShader(vertscr,gl.GL_VERTEX_SHADER),
            gl.shaders.compileShader(fragscr,gl.GL_FRAGMENT_SHADER)
        )


class Triangles:
    def __init__(self):
        self.vertices=(
            -0.5,-0.5,0.0,1,0,0,
            0.5,-0.5,0.0,0,1,0,
            0.0,0.5,0.0,0,0,1)
        self.vertices=np.array(self.vertices,dtype=np.float32)
        self.vertexcount=3

        self.vao=gl.glGenVertexArrays(1)
        gl.glBindVertexArray(self.vao)
        self.vbo=gl.glGenBuffers(1)

        gl.glBindBuffer(gl.GL_ARRAY_BUFFER,self.vbo)
        gl.glBufferData(gl.GL_ARRAY_BUFFER,self.vertices.nbytes,self.vertices,gl.GL_STATIC_DRAW)
        
        gl.glVertexAttribPointer(0,3,gl.GL_FLOAT,gl.GL_FALSE,24,ctypes.c_void_p(0))
        gl.glVertexAttribPointer(1,3,gl.GL_FLOAT,gl.GL_FALSE,24,ctypes.c_void_p(12))
        gl.glEnableVertexAttribArray(1)
        gl.glEnableVertexAttribArray(0)

    def destroy(self):
        gl.DeleteVertexArrays([self.vao])
        gl.DeleteBuffers([self.vbo])

App()