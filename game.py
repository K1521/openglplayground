import pygame as pg
import OpenGL.GL as gl

class App:
    def __init__(self):
        pg.init()
        pg.display.set_mode((640,480),pg.OPENGL|pg.DOUBLEBUF)
        self.clock =pg.time.Clock()

        gl.glClearColor(0.1,0.2,0.2,1)
        self.mainloop()
    def mainloop(self):
        running=True
        while running:
            for event in pg.event.get():
                if event.type==pg.QUIT:
                    running = False
            gl.glClear(gl.GL_COLOR_BUFFER_BIT)
            pg.display.flip()
            self.clock.tick(60)
        pg.quit()
App()