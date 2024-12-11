import glfw
from OpenGL.GL import *
from OpenGL.GL.shaders import compileShader, compileProgram
import numpy as np
import time
import sys
import glslprog
import opgraphtofun

vertex_src="""
#version 330
layout(location = 0) in vec3 position;
void main() {
    gl_Position = vec4(position, 1.0);
}
"""

def normalize(v):
    norm = np.linalg.norm(v)
    return v / norm if norm != 0 else v

class ShaderVariable:
    def __init__(self,value,vartype:str,lastupdate:int):
        #self.key=key
        self.value=value
        self.lastupdate=lastupdate
        if vartype is not None:
            pass
        elif isinstance(value, (int, float)):
            vartype= "float"
        elif isinstance(value, tuple) and len(value) == 2:
            vartype= "vec2"
        elif isinstance(value, tuple) and len(value) == 3:
            vartype= "vec3"
        elif isinstance(value, tuple) and len(value) == 4:
            vartype= "vec4"
        elif isinstance(value, np.ndarray):
            if value.shape == (2,):
                vartype= "vec2"
            elif value.shape == (3,):
                vartype= "vec3"
            elif value.shape == (4,):
                vartype= "vec4"
            elif value.shape == (3, 3):
                vartype= "mat3"
        self.vartype=vartype

class ShaderVariables:
    def __init__(self,shader):
        self.shader=shader
        self.variables = {}  # Stores ShaderVariable objects
        self.updates = 0  # Global counter for updates
        self.lastupload=-1
    
    def additem(self,key,value,vartype=None):
        self.variables[key]=ShaderVariable(np.copy(value),vartype,self.updates)
        self.updates += 1
    
    def __getitem__(self,key):
        return np.copy(self.variables[key].value)

    def __setitem__(self, key, value):
        """Sets or updates a shader variable."""
        if key not in self.variables:
            # New variable
            #raise ValueError("use additem")
            self.additem(key,value)
        else:
            # Update existing variable
            variable = self.variables[key]
            
            if not np.array_equal(variable.value, value):
                variable.value = np.copy(value)
                variable.lastupdate = self.updates
                self.updates += 1

    def upload_changed_to_shader(self):
        """Uploads changed variables to the shader."""
        anychanged=False
        glUseProgram(self.shader)
        for key,variable in self.variables.items():
            if variable.lastupdate >= self.lastupload:
                anychanged=True
                location = glGetUniformLocation(self.shader, key)
                value = variable.value
                vtype = variable.vartype

                if vtype == "float":
                    glUniform1f(location, value)
                elif vtype == "vec2":
                    glUniform2f(location, *value)
                elif vtype == "vec3":
                    glUniform3f(location, *value)
                elif vtype == "vec4":
                    glUniform4f(location, *value)
                elif vtype == "mat3":
                    glUniformMatrix3fv(location, 1, GL_TRUE, value)
                else:
                    raise ValueError(f"Unknown variable type: {vtype}")

        self.lastupload=self.updates
        return anychanged



class updatedict(dict):#TODO dont inherit from dict
    def __init__(self):
        self.updates=0
        self.freezekeys=False
        
    def __setitem__(self, key, value):
        if self.freezekeys and key not in self.d:
            raise ValueError("Cant set new keys")
        self.updates+=1
        super().__setitem__(key,value)



class mywindow:
    def __init__(self,width,height,fragmentshader):
        self.windowsize=(width,height)
        
        self.camera_up = np.array([0.0, 1.0, 0.0], dtype=np.float32) 
        self.initwindow()
        self.vao=self.makevao()
        self.shader=self.compileshaders(fragmentshader)
        self.loopcallbacks=[]

        self.variables=ShaderVariables(self.shader)

        self.variables["cameraPos"] = np.array([0.0, 0.0, -5.0], dtype=np.float32)  # Camera position
        self.variables["cameraMatrix"]=np.eye(3)#cam to world
    
    @property
    def camera_pos(self):
        return self.variables["cameraPos"]

    @camera_pos.setter
    def camera_pos(self, value):
        self.variables["cameraPos"] = value
    
    @property
    def camera_matrix(self):
        return self.variables["cameraMatrix"]

    @camera_matrix.setter
    def camera_matrix(self, value):
        self.variables["cameraMatrix"] = value

    def loop(self):
        glUseProgram(self.shader)
        
        glUniform2f(glGetUniformLocation(self.shader, "windowsize"), *self.windowsize)
        glUniform3f(glGetUniformLocation(self.shader, "lookat"), 0.0, 0.0, 0.0)
        glUniform3f(glGetUniformLocation(self.shader, "lightPos"), 5.0, 5.0, -5.0)
        while not glfw.window_should_close(self.window):
            self.loopiter()
        glfw.terminate()
        

    def loopiter(self):

        glfw.poll_events()
        for callback in self.loopcallbacks:
            callback()

        if not self.variables.upload_changed_to_shader():
            time.sleep(0.1)#dont do anything if the picture doesnt change

            return
        
        


        glClear(GL_COLOR_BUFFER_BIT)
        glUseProgram(self.shader)

        glUniform1f(glGetUniformLocation(self.shader, "time"), glfw.get_time())


        glBindVertexArray(self.vao)#draw quad
        glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, None)

        glfw.swap_buffers(self.window)

    def CamMatForward(self,forward):
        camF = normalize(forward)
        #camF = normalize(lookat - ro)  # forward
        camR = normalize(np.cross(self.camera_up, camF))  # right
        camU = np.cross(camF, camR)  # up
        
        # Construct the camera matrix (column-major)
        camera_matrix = np.array([camR, camU, camF]).T  # Transpose to match OpenGL column-major format
        return camera_matrix
    
    def CamMatAngle(self,yaw,pitch):
        front = np.array([
            np.cos(np.radians(yaw)) * np.cos(np.radians(pitch)),
            np.sin(np.radians(pitch)),
            np.sin(np.radians(yaw)) * np.cos(np.radians(pitch))
        ], dtype=np.float32)
        return self.CamMatForward(front)
    
    def compileshaders(self,fragment_src):
        
        return compileProgram(
            compileShader(vertex_src, GL_VERTEX_SHADER),
            compileShader(fragment_src, GL_FRAGMENT_SHADER)
        )
        
    def initwindow(self):
        
        # Initialize GLFW
        if not glfw.init():
            raise Exception("Failed to initialize GLFW")

        windowsize=(800, 600)
        # Create a window
        window=self.window = glfw.create_window(*windowsize, "Sphere Tracer", None, None)
        if not window:
            glfw.terminate()
            raise Exception("Failed to create GLFW window")

        glfw.make_context_current(window)

        
    def makevao(self):
        # Define vertices for a fullscreen quad
        indices = np.array([0, 1, 2, 2, 3, 0], dtype=np.uint32)
        vertices = np.array([
            -1.0, -1.0, 0.0,
            1.0, -1.0, 0.0,
            1.0,  1.0, 0.0,
            -1.0,  1.0, 0.0
        ], dtype=np.float32)#*np.array([])


        # Create VAO and VBO
        vao = glGenVertexArrays(1)
        vbo = glGenBuffers(1)
        ebo = glGenBuffers(1)

        glBindVertexArray(vao)

        glBindBuffer(GL_ARRAY_BUFFER, vbo)
        glBufferData(GL_ARRAY_BUFFER, vertices.nbytes, vertices, GL_STATIC_DRAW)

        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ebo)
        glBufferData(GL_ELEMENT_ARRAY_BUFFER, indices.nbytes, indices, GL_STATIC_DRAW)

        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * vertices.itemsize, ctypes.c_void_p(0))
        glEnableVertexAttribArray(0)

        glBindBuffer(GL_ARRAY_BUFFER, 0)
        glBindVertexArray(0)
        return vao



class keymouseeventhandler:
    def __init__(self,mywindow:mywindow):
        self.mywindow=mywindow
        self.camera_speed = 0.05  # Movement speed
        self.yaw, self.pitch = 90., 0.0  # Camera angles
        self.sensitivity = 0.1  # Mouse sensitivity

        self.last_x, self.last_y = 400, 300  # Center of the screen
        self.mouse_pressed = False

        glfw.set_mouse_button_callback(mywindow.window, self.mouse_button_callback)
        glfw.set_cursor_pos_callback(mywindow.window, self.mouse_callback)
        self.mywindow.loopcallbacks.append(self.handle_keys)
    def updatecammat(self):
        self.mywindow.camera_matrix=self.mywindow.CamMatAngle(self.yaw,self.pitch)
    def handle_keys(self):
        window=self.mywindow.window
        camera_pos=self.mywindow.camera_pos
        camera_speed=self.camera_speed
        camera_matrix=self.mywindow.camera_matrix
        yaw=self.yaw
        

        camera_front=camera_matrix@np.array([0,0,1])

        camera_speedlocal=camera_speed
        rotspeed=0.5
        if glfw.get_key(window, glfw.KEY_LEFT_SHIFT) == glfw.PRESS:
            camera_speedlocal*=4
            rotspeed*=4
        if glfw.get_key(window, glfw.KEY_SPACE) == glfw.PRESS:
            camera_speedlocal*=0.01
            rotspeed*=0.01

        if glfw.get_key(window, glfw.KEY_W) == glfw.PRESS:  # Move forward
            camera_pos += camera_speedlocal * camera_front
        if glfw.get_key(window, glfw.KEY_S) == glfw.PRESS:  # Move backward
            camera_pos -= camera_speedlocal * camera_front
        if glfw.get_key(window, glfw.KEY_A) == glfw.PRESS:  # Move left
            camera_pos += camera_matrix@np.array([-1,0,0]) * camera_speedlocal
        if glfw.get_key(window, glfw.KEY_D) == glfw.PRESS:  # Move right
            camera_pos += camera_matrix@np.array([1,0,0]) * camera_speedlocal
        if glfw.get_key(window, glfw.KEY_E) == glfw.PRESS:  # Move right
            yaw += rotspeed
        if glfw.get_key(window, glfw.KEY_Q) == glfw.PRESS:  # Move right
            yaw -= rotspeed
        self.yaw=yaw

        self.mywindow.camera_pos=camera_pos

        self.updatecammat()

    def mouse_callback(self,window, xpos, ypos):
        x_offset = (xpos - self.last_x) * self.sensitivity
        y_offset = (self.last_y - ypos) * self.sensitivity  # Reversed: y-coordinates go from bottom to top
        self.last_x, self.last_y = xpos, ypos
        if not self.mouse_pressed:
            return

        self.yaw += x_offset
        self.pitch -= y_offset
        self.pitch = max(-89.0, min(89.0, self.pitch))


        self.updatecammat()
    

    def mouse_button_callback(self,window, button, action, mods):
        if button == glfw.MOUSE_BUTTON_LEFT:  # Left mouse button
            if action == glfw.PRESS:
                self.mouse_pressed = True
            elif action == glfw.RELEASE:
                self.mouse_pressed = False


