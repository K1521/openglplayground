#version 330
//in vec2 fragUV;
out vec4 color;

uniform vec3 cameraPos;
uniform vec3 lightPos;
uniform float time;
uniform vec2 windowsize;
uniform vec3 lookat;


const float FOV=1;
const float EPSILON_RAYMARCHING=0.001;
const float EPSILON_NORMALS=0.00001;
const int MAX_RAY_ITER=20;

float sphere(vec3 p, float r) {
    float x=(length(p) - r);
    //return min(pow(abs(x),0.5),pow(abs(x),2));//sqrt(x*x);
    return x;
}

float sum(vec3 v) {
    return v.x+v.y+v.z;
}

float scene(vec3 p) {
    return min(sphere(p, 1.0),sphere(p-vec3(0.6,0.7,0.8), 1.0));
    //return sphere(p, 1.0); // Animate the sphere
    //return sphere(p - vec3(sin(time), cos(time), 0.0), 1.0); // Animate the sphere
}


vec3 normaltocol(vec3 normal){
    //return vec3(normal.x/2+0.5,normal.y/2+0.5,0.5-normal.z/2);
    //return normal*vec2(1,-1).xxy/2+0.5;
    return normal;
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
    vec2 e = vec2(1.0, -1.0) * EPSILON_NORMALS;
    return normalize(
      e.xyy * scene(p + e.xyy) +
      e.yyx * scene(p + e.yyx) +
      e.yxy * scene(p + e.yxy) +
      e.xxx * scene(p + e.xxx));
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
    float distance;
    for (int i = 0; i < MAX_RAY_ITER; i++) {
        distance = scene(rayOrigin);
        if (distance < EPSILON_RAYMARCHING)
            return distance; // Close enough
        rayOrigin += rayDir * distance;
    }   
    return distance;
}

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
    vec3 rayDir =getCam(cameraPos,lookat)* normalize(vec3(uv, FOV));


    // Sphere tracing

    vec3 p=rayOrigin;
    float distance=raymarch(rayDir,p);

    if (true || (distance < EPSILON_RAYMARCHING)) {
        
        //vec3 normal = getNormal2(p,rayDir);//usefull for reducing artefacts in unsigned distance functions
    // Checkerboard pattern
        float checker = 0.3 + 0.7 * mod(sum(floor(p * 4.0)), 2.0); // Alternates between 0.5 and 1.0
        //vec3 col=vec3(checker);
        vec3 col =getlight(p,rayDir,vec3(1)*checker);
        col=pow(col,vec3(0.4545));//gamma correction

        color= vec4(col,1);
        //color=vec4(normaltocol(normal),1);
    } else {
        color = vec4(0.0, 0.0, 0.0, 1.0); // Background
    }
}
