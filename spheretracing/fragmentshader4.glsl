#version 330
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
    //return getNormal(p);
    int iterations=3;
    vec3 normal=-rayDir*EPSILON_NORMALS;
    for(int i =0;i<iterations;i++){
        normal=getNormal(p+normal*EPSILON_NORMALS);
    }
    normal*=sign(dot(normal,rayDir));
    return normal;
}



/*
float raymarch(vec3 rayDir,inout vec3 rayOrigin){
    //vec3 p=rayOrigin;
    float dy=raydyinit;
    float magnitude;
    //float dx;
    for (int i = 0; i < MAX_RAY_ITER; i++) {
        vec4 s=scene(rayOrigin);
        magnitude =s.w ;
        vec3 grad=s.xyz;
        //dy=0.5*dy+2*abs(dot(s.xyz,rayDir));
        dy=rayalpha*dy+rayfactor*length(grad);
        
        rayOrigin += rayDir * (magnitude/(pow(dy,raypow)+rayepsilon));
        if (magnitude < EPSILON_RAYMARCHING || any(isinf(rayOrigin))||isinf(magnitude)||abs(rayOrigin.x)>10E10||abs(rayOrigin.y)>10E10||abs(rayOrigin.z)>10E10)
            break; // Close enough
    }   

    //for (int i = 0; i < 10; i++) {
    //    vec4 s=scene(rayOrigin);
    //    magnitude =s.w ;
    //    dy=rayfactor*dot(s.xyz,rayDir);
    //    rayOrigin += rayDir * (magnitude/rayepsilon);
    //}   


    return magnitude;
}*/

#define EPSILON 0.01      // Kleine Verschiebung für benachbarte Punkte
#define MAX_RAY_ITER 100  // Maximale Anzahl der Raymarching-Iterationen
#define EPSILON_RAYMARCHING 0.001  // Toleranz für den Abbruch der Iteration

/*float raymarch(vec3 rayDir, inout vec3 rayOrigin) {
    float magnitude;
    vec3 p0, p1, p2;
    float f0, f1, f2;
    vec3 grad0, grad1, grad2;

    // Raymarching loop
    for (int i = 0; i < MAX_RAY_ITER; i++) {
        // Get distance and gradient at current position
        vec4 s = scene(rayOrigin);
        magnitude = s.w;
        grad0 = s.xyz;

        // Calculate neighboring points (parabolic interpolation)
        p1 = rayOrigin + EPSILON * rayDir;
        p2 = rayOrigin + 2.0 * EPSILON * rayDir;

        f0 = magnitude;
        f1 = scene(p1).w;
        f2 = scene(p2).w;
        grad1 = scene(p1).xyz;
        grad2 = scene(p2).xyz;

        // Calculate parabola coefficients a, b, c
        float a, b, c;
        float x0 = 0.0, x1 = EPSILON, x2 = 2.0 * EPSILON;
        float f_prime0 = dot(grad0, rayDir);
        float f_prime1 = dot(grad1, rayDir);
        float f_prime2 = dot(grad2, rayDir);

        // Simple parabola fitting for the step size
        float A = x1 * x1 - x0 * x0;
        float B = x2 * x2 - x0 * x0;
        float C = x2 * x1 - x0 * x1;

        a = (f0 * (x1 - x2) + f1 * (x2 - x0) + f2 * (x0 - x1)) / (A * C);
        b = (f0 * B + f1 * (x2 - x0) + f2 * (x0 - x1)) / (A * B);
        c = f0 * (B / (A * C)) + f1 * (C / (B * A)) + f2;

        // Compute next step along the ray
        float stepParabola = (-b + sqrt(b * b - 4.0 * a * c)) / (2.0 * a);

        // Move the ray origin by the computed step
        rayOrigin += rayDir * stepParabola;

        // Exit the loop if the ray is close enough to the surface
        if (magnitude < EPSILON_RAYMARCHING) {
            break; // Close enough
        }
    }

    return magnitude;
}*/
/*vec3 scene2(vec3 rayDir, vec3 p){
    vec3 w=vec3(scene(p-rayDir*EPSILON_DERIV).w,scene(p).w,scene(p+rayDir*EPSILON_DERIV).w);
    return vec3(w.y,(w.z-w.x)/(2*EPSILON_DERIV),(w.z-2*w.y+w.x)/(EPSILON_DERIV*EPSILON_DERIV));
}*/
vec3 scene2(vec3 rayDir, vec3 p){
    vec4 mid=scene(p);
    vec3 w=vec3(scene(p-rayDir*EPSILON_DERIV).w,mid.w,scene(p+rayDir*EPSILON_DERIV).w);
    return vec3(w.y,length(mid.xyz),(w.z-2*w.y+w.x)/(EPSILON_DERIV*EPSILON_DERIV));
}

/*vec3 scene2(vec3 rayDir, vec3 p){
    vec4 mid=scene(p);
    vec4 right=scene(p+rayDir*EPSILON_DERIV);
    //vec3 w=vec3(scene(p-rayDir*EPSILON_DERIV).w,mid.w,scene(p+rayDir*EPSILON_DERIV).w);
    return vec3(mid.w,length(mid.xyz),length(right.xyz)-length(mid.xyz)/(EPSILON_DERIV));
}*/
float calcstep(vec3 rayDir, vec3 p){
    vec4 s=scene(p);
    vec3 deriv=s.xyz;
    return s.w/max(length(deriv),abs(dot(deriv,rayDir)));



}


    



float raymarch(vec3 rayDir, inout vec3 rayOrigin) {
    float magnitude;
    
    for (int i = 0; i < MAX_RAY_ITER; i++) {
        vec4 s = scene(rayOrigin);
        magnitude = s.w;

        if (magnitude < EPSILON_RAYMARCHING) {
            break;  
        }

        rayOrigin += rayDir * calcstep(rayDir, rayOrigin)/rayfactor;
    }

    return magnitude;
}

/*float raymarch(vec3 rayDir, inout vec3 rayOrigin) {
    float magnitude;
    
    for (int i = 0; i < MAX_RAY_ITER; i++) {
        // Get distance and gradient at the current position
        vec4 s = scene(rayOrigin);
        magnitude = s.w;

        // If the magnitude is small enough, we've hit the surface
        if (magnitude < EPSILON_RAYMARCHING) {
            break;  // Close enough to the surface
        }

        // Step forward by the magnitude (this is linear raymarching)
        rayOrigin += rayDir * magnitude;
    }

    return magnitude;  // Return the distance to the surface
}*/

/*float raymarch(vec3 rayDir,inout vec3 rayOrigin){
    float fdx=0.01;
    float hlast=1;
    float dhlast=0;
    float dx=0;
    float x=0;
    float h;
    float dh;
    float dy;
    for (int i = 0; i < MAX_RAY_ITER; i++) {
        vec4 s=scene(rayOrigin+x*rayDir);
        h =s.w ;
        dh=dot(rayDir,s.xyz);
        float error=abs((dhlast*dx+hlast)/h);
    
        if(1.1>error && error>0.9 && fdx<1){
            fdx*=1.1;
        }else if(fdx>0.01){
            fdx/=2;
        }
        
        //dy=0*abs(dot(s.xyz,rayDir))+length(s.xyz);
        dy=rayalpha*dy+rayfactor*length(s.xyz);
        

        if (h < EPSILON_RAYMARCHING){

            rayOrigin=rayOrigin+x*rayDir;
            return h;
        }
             // Close enough
        dx=(h/(dy+rayepsilon))*fdx;
        x+=dx;
        hlast=h;
        dhlast=dh;
    }   
    rayOrigin=rayOrigin+x*rayDir;
    return h;
}*/








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
    vec3 col=vec3(checker);
    //vec3 col =getlight(p,rayDir,vec3(1)*checker);

    //col=normaltocol(getNormal2(p,rayDir));
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
    if(length(uv)<0.01){col*=vec3(0.7,1,0.7);}

    color= vec4(col,1);
}


//cutoff
