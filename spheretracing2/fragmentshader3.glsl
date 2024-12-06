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


vec3 normaltocol(vec3 normal){
    //return vec3(normal.x/2+0.5,normal.y/2+0.5,0.5-normal.z/2);
    return normal*vec2(1,-1).xxy/2+0.5;
    //return normal;
    //return normalize(pow((normal*vec2(1,-1).xxy+1)/2,vec3(1)));
}

vec3 getNormal(vec3 p){
    return normalize( -cross(dFdx(p), dFdy(p)) );
}

vec3 getlight(vec3 p,vec3 rayDir,vec3 color){

    // Normals and light direction
    vec3 normal = getNormal(p);
    vec3 lightDir = normalize(lightPos - p); // Direction to light source

    // Ambient light (constant base illumination)
    vec3 ambient = 0.1 * color;

    // Diffuse lighting
    float diffuseStrength = max(dot(normal, lightDir),0.0);
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

/*vec4 solveQuartic(float a,float b,float c,float d,float e){
    //https://en.wikipedia.org/wiki/Quartic_function
    float delta0=c*c-3*b*d+12*a*e;
    float delta1=2*c*c*c-9*b*c*d+27*b*b*e+27*a*d*d-72*a*c*e;
    float p=(8*a*c-3*b*b)/(8*a*a);
    float q=(b*b*b-4*a*b*c+8*a*a*d)/(8*a*a*a);
    float Q=pow(0.5*(delta1+sqrt(delta1*delta1-4*delta0*delta0*delta0)),1./3);
    float S=0.5*sqrt(-p*2/3+(Q+delta0/Q)/(3*a));

    float xpart1=-b/(4*a);
    float xpart2=-4*S*S-2*p;
    float xpart3=q/S;
    float xsqrt1=0.5*sqrt(xpart2+xpart3);
    float xsqrt2=0.5*sqrt(xpart2-xpart3);

    return vec4(xpart1-S+xsqrt1,xpart1-S-xsqrt1,xpart1+S+xsqrt2,xpart1+S-xsqrt2);
}*/
/*vec4 solveQuartic(float a, float b, float c, float d, float e) {
    float delta0 = c * c - 3.0 * b * d + 12.0 * a * e;
    float delta1 = 2.0 * c * c * c - 9.0 * b * c * d + 27.0 * b * b * e + 27.0 * a * d * d - 72.0 * a * c * e;
    float discriminant = delta1 * delta1 - 4.0 * delta0 * delta0 * delta0;

    float p = (8.0 * a * c - 3.0 * b * b) / (8.0 * a * a);
    float q = (b * b * b - 4.0 * a * b * c + 8.0 * a * a * d) / (8.0 * a * a * a);

    float xpart1 = -b / (4.0 * a);
    float S;

    if (discriminant > 0.0) {
        // General case
        float Q = pow(0.5 * (delta1 + sqrt(discriminant)), 1.0 / 3.0);
        S = 0.5 * sqrt(-2.0 * p / 3.0 + (Q + delta0 / Q) / (3.0 * a));
    } else if (discriminant == 0.0) {
        // Special case: repeated roots
        float Q = pow(delta1 / 2.0, 1.0 / 3.0); // Simplified Q
        S = 0.5 * sqrt(-2.0 * p / 3.0 + (Q + delta0 / Q) / (3.0 * a));
    } else {
        // Casus irreducibilis: use trigonometric solution
        float phi = acos(delta1 / (2.0 * sqrt(delta0 * delta0 * delta0)));
        S = 0.5 * sqrt(-2.0 * p / 3.0 + 2.0 / (3.0 * a) * sqrt(delta0) * cos(phi / 3.0));
    }

    float xpart2 = -4.0 * S * S - 2.0 * p;
    float xpart3 = q / S;

    float xsqrt1 = 0.5 * sqrt(xpart2 + xpart3);
    float xsqrt2 = 0.5 * sqrt(xpart2 - xpart3);

    return vec4(xpart1 - S + xsqrt1, xpart1 - S - xsqrt1, xpart1 + S + xsqrt2, xpart1 + S - xsqrt2);
}*/
const float nan=sqrt(-1);
float solveLinear(float a, float b){
    return -b/a;
}
vec2 solveQuadratic(float a, float b, float c) {
    if(abs(a)<1E-6)return vec2(solveLinear(b,c),nan);
    float m=-b/(2.0*a);
    float s=sqrt(m*m-c/a);
    return vec2(m+s,m-s);
}
vec3 solveCubic(float a, float b, float c, float d) {
    if(abs(a)<1E-6)return vec3(solveQuadratic(b,c,d),nan);
    // Normalize coefficients for a monic polynomial x^3 + px + q = 0
    float p = (3.0 * a * c - b * b) / (3.0 * a * a);
    float q = (2.0 * b * b * b - 9.0 * a * b * c + 27.0 * a * a * d) / (27.0 * a * a * a);

    // Calculate discriminant
    float discriminant = (q * q) / 4.0 + (p * p * p) / 27.0;

    vec3 roots;

    if (discriminant > 0.0) {
        // One real root
        float u = pow(-q / 2.0 + sqrt(discriminant), 1.0 / 3.0);
        float v = pow(-q / 2.0 - sqrt(discriminant), 1.0 / 3.0);
        roots[0] = u + v - b / (3.0 * a);
        roots[1] = roots[0]; // Duplicated roots
        roots[2] = roots[0];
    } else if (discriminant == 0.0) {
        // Two real roots (one is a double root)
        float u = pow(-q / 2.0, 1.0 / 3.0);
        roots[0] = 2.0 * u - b / (3.0 * a);
        roots[1] = -u - b / (3.0 * a);
        roots[2] = roots[1]; // Duplicated root
    } else {
        // Three real roots (casus irreducibilis)
        float r = sqrt(-p * p * p / 27.0);
        float phi = acos(-q / (2.0 * r));
        float s = pow(r, 1.0 / 3.0);
        roots[0] = 2.0 * s * cos(phi / 3.0) - b / (3.0 * a);
        roots[1] = 2.0 * s * cos((phi + 2.0 * 3.14159265359) / 3.0) - b / (3.0 * a);
        roots[2] = 2.0 * s * cos((phi + 4.0 * 3.14159265359) / 3.0) - b / (3.0 * a);
    }

    return roots;
}

// Solve quartic equation ax^4 + bx^3 + cx^2 + dx + e = 0
// Returns vec4 with real roots, unused roots set to INFINITY
vec4 solveQuartic(float a, float b, float c, float d, float e) {
    if(abs(a)<1E-6)return vec4(solveCubic(b,c,d,e),nan);

    float xoffset=0;
    // Compute intermediate values
    float p = (8.0 * a * c - 3.0 * b * b) / (8.0 * a * a);
    float q = (b * b * b - 4.0 * a * b * c + 8.0 * a * a * d) / (8.0 * a * a * a);

    // Compute Delta terms
    float delta0 = c * c - 3.0 * b * d + 12.0 * a * e;
    float delta1 = 2.0 * c * c * c - 9.0 * b * c * d + 27.0 * b * b * e + 27.0 * a * d * d - 72.0 * a * c * e;

    // Compute discriminant and Q
    float disc = delta1 * delta1 - 4.0 * delta0 * delta0 * delta0;
    
    // Q calculation
    float Q = pow(
        (delta1 + sqrt(abs(disc))) / 2.0, 
        1.0 / 3.0
    );

    // S calculation
    float S = 0.5 * sqrt(
        -2.0 / 3.0 * p + 
        (1.0 / (3.0 * a)) * (Q + (delta0 / Q))
    );

    // Root calculation
    vec4 roots;
    float offset = -b / (4.0 * a);

    roots.x = offset - S + 0.5 * sqrt(-4.0 * S * S - 2.0 * p + (q / S));
    roots.y = offset - S - 0.5 * sqrt(-4.0 * S * S - 2.0 * p + (q / S));
    roots.z = offset + S + 0.5 * sqrt(-4.0 * S * S - 2.0 * p - (q / S));
    roots.w = offset + S - 0.5 * sqrt(-4.0 * S * S - 2.0 * p - (q / S));

    return roots+xoffset;
}



float rootpolysquartic();
float raymarch2(vec3 rayDir, inout vec3 rayOrigin);
float raymarch(vec3 rayDir, inout vec3 rayOrigin) {
    
    float x=0;
    

    compilepolys(rayOrigin,rayDir);
    x=rootpolysquartic();
    if(isinf(x)){
        raymarch2(rayDir,rayOrigin);
    }
    rayOrigin += rayDir * x;
    return 0;
}
float raymarch2(vec3 rayDir, inout vec3 rayOrigin) {
    compilepolys(rayOrigin,rayDir);
    float magnitude=1000000;
    float x=0;
    for (int i = 0; i < MAX_RAY_ITER; i++) {
        
        magnitude = calcpolys(x);
        if (magnitude < EPSILON_RAYMARCHING) {
            break;  
        }
        x+=(magnitude)*0.01;
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
    vec3 rayDir =cameraMatrix* normalize(vec3(uv, FOVfactor));//cam to view


    // Sphere tracing

    vec3 p=rayOrigin;
    float dist=raymarch(rayDir,p);


        
    
    // Checkerboard pattern
    float checker = 0.3 + 0.7 * mod(sum(floor(p * 4.0)), 2.0); // Alternates between 0.5 and 1.0
    
    vec3 col=vec3(checker);
    //col=vec3(1);
    col*=normaltocol(transpose(cameraMatrix)*getNormal(p));
    //col=vec3(abs());
    col=getlight(p,rayDir,col);
    //col=pow(col,vec3(0.4545));//gamma correction
    //col=pow(col,vec3(2));

    if ((dist < EPSILON_RAYMARCHING)) {

    } else {
        col*=vec3(1,0.5,0.5);
        
        //color = vec4(0.0, 0.0, 0.0, 1.0); // Background
    }
    
    if(any(isnan(col))){
        col=vec3(0);//vec3(1,1,0);
    }
    if(any(isinf(vec3(p))) || abs(p.x)>10E10||abs(p.y)>10E10||abs(p.z)>10E10){
        col=vec3(0);//vec3(0,0,0.5);
    }
    if(length(uv)<0.01){col*=vec3(0.7,1,0.7);}//dot in middle of screen

    color= vec4(col,1);
}


//cutoff
