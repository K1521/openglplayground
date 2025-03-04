#version 440
//in vec2 fragUV;
out vec4 color;

uniform vec3 cameraPos;
uniform vec3 lightPos;
uniform float time;
uniform vec2 windowsize;
uniform mat3 cameraMatrix;


uniform float raydyinit;
uniform float rayfactor;
uniform float rayalpha;
uniform float rayepsilon;
uniform float raypow;


const float FOV=120;
const float FOVfactor=1/tan(radians(FOV) * 0.5);
const float EPSILON_RAYMARCHING=0.001;
const float EPSILON_NORMALS=0.001;
const float EPSILON_DERIV=0.0001;
const int MAX_RAY_ITER=128;


float sum(vec3 v) {
    return v.x+v.y+v.z;
}

float calcpolys(float x);
void compilepolys(vec3 p,vec3 d);



/*
float raymarch(vec3 rayDir, inout vec3 rayOrigin) {
    compilepolys(rayOrigin,rayDir);
    float magnitude=1000000;
    float x=0;
    for (int i = 0; i < MAX_RAY_ITER*100; i++) {
        
        magnitude = calcpolys(x);
        if (magnitude < EPSILON_RAYMARCHING) {
            break;  
        }
        x+=((sqrt(sqrt(magnitude)))*0.01);
    }
    float xbest=0;
    for (int i = 0; i < MAX_RAY_ITER*100; i++) {
        
        float m = calcpolys(x);
        if (m<magnitude) {
            magnitude=m;  
            xbest=x;
        }
        if (m < EPSILON_RAYMARCHING)break;
        x+=0.001;
    }
    rayOrigin += rayDir * xbest;

    return magnitude;
}*/



float raymarch(vec3 rayDir, inout vec3 rayOrigin) {
    compilepolys(rayOrigin,rayDir);
    float magnitude=1000000;
    float x=0;
    for (int i = 0; i < MAX_RAY_ITER*100; i++) {
        
        magnitude = calcpolys(x);
        if (magnitude < EPSILON_RAYMARCHING) {
            break;  
        }
        x+=(magnitude)*0.0001;
    }
    rayOrigin += rayDir * x;

    return magnitude;
}



bool any(bvec3 b) {
    return b.x || b.y || b.z;
}

    
mat3 getCam(vec3 ro,vec3 lookat){
    vec3 camF=normalize(lookat-ro);
    vec3 camR=normalize(cross(vec3(0,1,0),camF));
    vec3 camU=cross(camF,camR);
    return mat3(camR,camU,camF);
}

void main() {
    vec2 uv=(2*gl_FragCoord.xy-windowsize)/windowsize.x;
    vec3 rayOrigin = cameraPos;
    vec3 rayDir =cameraMatrix* normalize(vec3(uv, FOVfactor));


    // Sphere tracing

    vec3 p=rayOrigin;
    float dist=raymarch(rayDir,p);


        
    
    // Checkerboard pattern
    float checker = 0.3 + 0.7 * mod(sum(floor(p * 4.0)), 2.0); // Alternates between 0.5 and 1.0
    vec3 col=vec3(checker);
    col=pow(col,vec3(0.4545));//gamma correction

    if ((dist < EPSILON_RAYMARCHING)) {

    } else {
        col*=vec3(1,0.5,0.5);
        
        //color = vec4(0.0, 0.0, 0.0, 1.0); // Background
    }
    
    if(any(isnan(col))){
        col=vec3(1,1,0);
    }
    if(any(isinf(vec3(p))) || abs(p.x)>10E10||abs(p.y)>10E10||abs(p.z)>10E10){
        col=vec3(0,0,0.5);
    }
    if(length(uv)<0.01){col*=vec3(0.7,1,0.7);}//dot in middle of screen

    color= vec4(col,1);
}


//cutoff
