#version 300 es
precision mediump float;

const float FOV=120.;
const float FOVfactor=1.0/tan(radians(FOV) * 0.5);

out vec4 color;

uniform vec3 cameraPos;
uniform vec2 windowsize;
uniform mat3 c2w;

float sum(vec3 v) {
    return v.x+v.y+v.z;
}
int sum(ivec3 v) {
    return v.x+v.y+v.z;
}

//simple sphere tracer to test controlls
float scene(vec3 p){
    return min(
        length(p-vec3(1.,2.,4.))-1.,
        length(p-vec3(-1.,2.,5.))-2.
    );
}

float raymarch(inout vec3 p,vec3 raydir){
    float x=0.0;
    //float xbest=0.0;
    //float ebest=inf;


    for(int i=0;i<20;i++){
        float e=scene(p+x*raydir);
        
        x+=e;
    }
    p+=x*raydir;
    return 0.;

}


void main(void) {
    vec2 uv=(2.0*gl_FragCoord.xy-windowsize)/windowsize.x;
    vec3 rayOrigin = cameraPos;
    vec3 rayDir =c2w*normalize(vec3(uv, FOVfactor));//cam to view

    vec3 p=rayOrigin;
    raymarch(p,rayDir);

    float checker = 0.3 + 0.7 * mod(sum(floor(p * 4.0)), 2.0);



    color = checker*vec4(1.0, 0.0, 0.0, 1.0);  // Red color
}