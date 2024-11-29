#version 330
//in vec2 fragUV;
out vec4 color;

uniform vec3 cameraPos;
uniform vec3 lightPos;
uniform float time;
uniform vec2 windowsize;
uniform mat3 cameraMatrix;


const float FOV=120;
const float FOVfactor=1/tan(radians(FOV) * 0.5);
const float EPSILON_RAYMARCHING=0.001;
const float EPSILON_NORMALS=0.001;
const int MAX_RAY_ITER=128;

float sphere(vec3 p, float r) {
    float x=(length(p) - r);
    //return min(pow(abs(x),0.5),pow(abs(x),2));//sqrt(x*x);
    return x;
}

float sum(vec3 v) {
    return v.x+v.y+v.z;
}

vec4 scene(vec3 p);


vec3 normaltocol(vec3 normal){
    //return vec3(normal.x/2+0.5,normal.y/2+0.5,0.5-normal.z/2);
    return normal*vec2(1,-1).xxy/2+0.5;
    //return normal;
    //return normalize(pow((normal*vec2(1,-1).xxy+1)/2,vec3(1)));
}
/*
vec3 getNormal(vec3 p) {
    const float h = EPSILON_NORMALS;
    return normalize(vec3(
        scene(p + vec3(h, 0.0, 0.0)) - scene(p - vec3(h, 0.0, 0.0)),
        scene(p + vec3(0.0, h, 0.0)) - scene(p - vec3(0.0, h, 0.0)),
        scene(p + vec3(0.0, 0.0, h)) - scene(p - vec3(0.0, 0.0, h))
    ));
}*/
vec3 getNormal(vec3 p) {
    return normalize(scene(p).xyz);
}
vec3 getNormal2(vec3 p,vec3 rayDir){
    int iterations=2;
    vec3 normal=-rayDir*EPSILON_NORMALS;
    for(int i =0;i<iterations;i++){
        normal=getNormal(p+normal*EPSILON_NORMALS);
    }
    return normal;
}


float raymarch(vec3 rayDir,inout vec3 rayOrigin){
    //vec3 p=rayOrigin;
    float dy=1000;
    float magnitude;
    //float dx;
    for (int i = 0; i < MAX_RAY_ITER; i++) {
        vec4 s=scene(rayOrigin);
        magnitude =s.w ;
        //dy=0.5*dy+2*abs(dot(s.xyz,rayDir));
        dy=0.5*dy+2*length(s.xyz);
        

        if (magnitude < EPSILON_RAYMARCHING || any(isinf(rayOrigin))||isinf(magnitude))
            return magnitude; // Close enough
        rayOrigin += rayDir * (magnitude/(dy+0.001));
    }   
    return magnitude;
}

bool any(bvec3 b) {
    return b.x || b.y || b.z;
}

/*
float raymarch(vec3 rayDir,inout vec3 rayOrigin){
    //vec3 p=rayOrigin;
    float distance;
    for (int i = 0; i < MAX_RAY_ITER; i++) {
        distance = scene(rayOrigin);
        if (distance < EPSILON_RAYMARCHING)
            return distance; // Close enough
        rayOrigin += rayDir * distance;
    }   
    return distance;
}
*/
/*float raymarch (vec3 dir,inout vec3 start) {
    vec3 orig=start; 
    float lastd = 1000.0; 
    const int count=80;
    const float thresh=0.2;
    const float stepSize=0.00001;
    float diff;
    for (int i=0; i<count; i++) {
        float d = scene(start);
        diff = stepSize*(1.0+1000.0*d);
        if (d < thresh){
            start += dir*(lastd-thresh)/(lastd-d)*diff;
            return diff;
        }
        lastd = d;
        start += dir*diff;
    }
    return diff;
}*/

vec3 getlight(vec3 p,vec3 rayDir,vec3 color){

    // Normals and light direction
    vec3 normal = getNormal2(p, rayDir); // Adjusted normal to reduce artifacts
    vec3 lightDir = normalize(lightPos - p); // Direction to light source

    // Ambient light (constant base illumination)
    vec3 ambient = 0.1 * color;

    // Diffuse lighting
    float diffuseStrength = max(dot(normal, lightDir), 0.0);
    vec3 diffuse = diffuseStrength * vec3(1.0, 1.0, 1.0); // White light

    // Specular lighting (Phong reflection model)
    vec3 viewDir = normalize(-rayDir); // Direction towards the viewer
    vec3 reflectDir = reflect(-lightDir, normal); // Reflection direction
    float specularStrength = pow(max(dot(viewDir, reflectDir), 0.0), 16.0); // Shininess factor
    vec3 specular = specularStrength * vec3(1.0, 1.0, 1.0); // White specular highlights


    // Combine lighting components
    vec3 finalColor = ambient + diffuse * color + specular;

    return finalColor;
}
    
mat3 getCam(vec3 ro,vec3 lookat){
    vec3 camF=normalize(lookat-ro);
    vec3 camR=normalize(cross(vec3(0,1,0),camF));
    vec3 camU=cross(camF,camR);
    return mat3(camR,camU,camF);
}

void main() {
    //vec2 uv=(2*gl_FragCoord.xy/windowsize)-1;
    vec2 uv=(2*gl_FragCoord.xy-windowsize)/windowsize.x;
    //color = vec4(vec2(uv), 0.0, 1.0); // Background
    //return;
    vec3 rayOrigin = cameraPos;
    vec3 rayDir =cameraMatrix* normalize(vec3(uv, FOVfactor));


    // Sphere tracing

    vec3 p=rayOrigin;
    float dist=raymarch(rayDir,p);

    //if ( true||(dist < EPSILON_RAYMARCHING)) {
        
    //vec3 normal = getNormal2(p,rayDir);//usefull for reducing artefacts in unsigned distance functions
    // Checkerboard pattern
    float checker = 0.3 + 0.7 * mod(sum(floor(p * 4.0)), 2.0); // Alternates between 0.5 and 1.0
    //vec3 col=vec3(checker);
    vec3 col =getlight(p,rayDir,vec3(1)*checker);

    //col=normaltocol(getNormal2(p,rayDir));
    col=pow(col,vec3(0.4545));//gamma correction

    if ((dist < EPSILON_RAYMARCHING)) {

    } else {
        col*=vec3(1,0,0);
        
        //color = vec4(0.0, 0.0, 0.0, 1.0); // Background
    }
    
    if(any(isnan(col))){
        col=vec3(1,1,0);
    }
    if(any(isinf(vec3(p)))){
        col=vec3(0,0,0.5);
    }

    color= vec4(col,1);
}


//cutoff
