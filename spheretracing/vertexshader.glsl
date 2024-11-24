#version 330
layout(location = 0) in vec3 position;
//out vec2 fragUV;

//uniform vec3 windowsize;
void main() {
    //fragUV = (position.xy ) * 0.5; // Map to UV coordinates
    //fragUV=gl_FragCoord/windowsize;
    gl_Position = vec4(position, 1.0);
}
