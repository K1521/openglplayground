#version 300 es
precision mediump float;
uniform sampler2D roots01tex;
uniform sampler2D roots23tex;
uniform float mipLevel;
uniform float aspectratio;


uniform vec3 cameraPos;
uniform vec2 windowsize;
uniform mat3 cameraMatrix;

const float FOV=120.;
const float FOVfactor=1./tan(radians(FOV) * 0.5);
const float EPSILON_RAYMARCHING=0.001;
//const float EPSILON_NORMALS=0.001;
//const float EPSILON_DERIV=0.0001;
const float EPSILON_ROOTS=0.001;
//const int MAX_RAY_ITER=128;

const float nan=sqrt(-1.);
const float inf=pow(9.,999.);
const float pi=3.14159265359;
const float goldenangle = (3.0 - sqrt(5.0)) * pi;


out vec4 color;
#define Complex vec2


float sum(vec3 v) {
    return v.x+v.y+v.z;
}
int sum(ivec3 v) {
    return v.x+v.y+v.z;
}


float getroot(vec3 rayDir, inout vec3 rayOrigin,vec2 uv){
    vec4 roots01in=texture(roots01tex, uv);//textureLod(roots01tex, uv, mipLevel);
    vec4 roots23in=texture(roots23tex, uv);//textureLod(roots23tex, uv, mipLevel);
    Complex[4] roots;
    roots[0]=roots01in.xy;
    roots[1]=roots01in.zw;
    roots[2]=roots23in.xy;
    roots[3]=roots23in.zw;
    float beste=inf;
    float bestx=inf;
    for(int i = 0; i < 4; ++i){
        Complex r=roots[i];
        r.y=abs(r.y);
        if(r.x>=0. && r.y<beste){
            beste=r.y;
            bestx=r.x;
        }
    }
    rayOrigin += rayDir * bestx;
    return beste;
}

void main() {
    //vec2 uv=(2.*gl_FragCoord.xy-windowsize)/windowsize.x;
    vec2 uv=(gl_FragCoord.xy)/windowsize;
    vec2 uvray=(uv*2.-1.)/vec2(1.,aspectratio);
    
    vec3 rayOrigin = cameraPos;
    vec3 rayDir =cameraMatrix*normalize(vec3(uvray, FOVfactor));//cam to view
   
    
    vec3 p=rayOrigin;//vec3(0);//rayOrigin;
    float dist=getroot(rayDir,p,uv);
    //float x=dot(rayDir,p-rayOrigin);
    //debugcolor(vec3((x)/10.));
    //float dist=DualComplexRaymarch4(rayDir,p);
    //p=(cameraMatrix)*p+rayOrigin;


        
    
    // Checkerboard pattern
    float checker = 0.3 + 0.7 * mod(sum(floor(p * 4.0)), 2.0); // Alternates between 0.5 and 1.0
    //float checker = 0.3 + 0.7 * mod(sum(floor(p/(length(p-rayOrigin)/100+1) * 4.0)), 2.0);
    //checker=(abs(mod(p.x,0.3))<0.03 || abs(mod(p.y,0.3))<0.03 || abs(mod(p.z,0.3))<0.03) ?1.0:0.3;
    
    vec3 col=vec3(checker);
    //col=vec3(1);
    col*=1.;

    col*=mix(vec3(0.0, 1.0, 0.0), vec3(1.0, 0.0, 0.0), clamp(dist*20.,0.,1.));
    
    if(any(isnan(col))){
        col=vec3(1.,1.,0.);//nan is yellow
    }
    if(any(isinf(vec3(p))) || abs(p.x)>10E10||abs(p.y)>10E10||abs(p.z)>10E10){
        col=vec3(0.,0.,0.5);//blue
    }
    //if(length(uv)<0.01){col*=vec3(0.7,1,0.7);}//dot in middle of screen
    if(length(uv)<0.01 && length(uv)>0.005){col=vec3(0.5,1,0.5);}//circle in middle of screen



    color= vec4(col,1.);

    //color=vec4(texture(roots01tex, uv).xy,0.,1.);
    //color=vec4(uvray,0.,1.);
    
}


//cutoff
