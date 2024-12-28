
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
const float EPSILON_DERIV=0.0001;
const float EPSILON_ROOTS=0.001;
const int MAX_RAY_ITER=128;

const float nan=sqrt(-1);
const float inf=pow(9,999);
const float pi=3.14159265359;
const float goldenangle = (3.0 - sqrt(5.0)) * pi;
#define dcomplex dvec2
#define dnumcomplex double

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
bool any(bvec3 b) {
    return b.x || b.y || b.z;
}

float square(float x){return x*x;}

float vmin(vec4 x){
    float minx=inf;

    for(int i=0;i<4;i++){
        float a=x[i];
        if(!isnan(a)){
            minx=min(minx,a);
        }
    }
    return minx;
}


dcomplex complex_multiply(dcomplex a, dcomplex b) {
    // Complex multiplication: (a.x + i*a.y) * (b.x + i*b.y)
    return dcomplex(
        a.x * b.x - a.y * b.y, // Real part
        a.x * b.y + a.y * b.x  // Imaginary part
    );
}
dcomplex complex_divide(dcomplex a, dcomplex b) {
    // Complex division: (a.x + i*a.y) / (b.x + i*b.y)
    dnumcomplex denominator = b.x * b.x + b.y * b.y;
    return dcomplex(
        (a.x * b.x + a.y * b.y) / denominator, // Real part
        (a.y * b.x - a.x * b.y) / denominator  // Imaginary part
    );
}


dnum evaluatePolynomial(dnum x, dpoly coefficients) {
    dnum result = coefficients[degree];
    for (int i = degree - 1; i >= 0; i--) {
        result = result * x + coefficients[i];
    }
    return result;
}
dnum evaluatePolynomialDerivative(dnum x, dpoly coefficients) {
    dnum result = coefficients[degree]*degree;
    for (int i = degree - 1; i >= 1; i--) {
        result = result * x + coefficients[i]*i;
    }
    return result;
}

dcomplex evaluatePolynomial(dcomplex x, dpoly coefficients) {
    dcomplex result = dcomplex(coefficients[degree],0);
    for (int i = degree - 1; i >= 0; i--) {
        result = complex_multiply(result , x) + dcomplex(coefficients[i],0);
    }
    return result;
}
dcomplex evaluatePolynomialDerivative(dcomplex x, dpoly coefficients) {
    dcomplex result = dcomplex(coefficients[degree]*degree,0);
    for (int i = degree - 1; i >= 1; i--) {
        result = complex_multiply(result , x) + dcomplex(coefficients[i]*i,0);
    }
    return result;
}
void initial_roots(out dcomplex[degree] roots) {
    const dcomplex r1 = dcomplex(cos(goldenangle), sin(goldenangle)); // Base complex number
    roots[0]=r1;
    for (int i = 1; i < degree; i++) {
        roots[i] = complex_multiply(r1, roots[i-1]);
    }
}
void initial_roots(out dcomplex[degree] roots,dcomplex center) {
    const dcomplex r1 = dcomplex(cos(goldenangle), sin(goldenangle)); // Base complex number
    roots[0]=dcomplex(1,0);
    for (int i = 1; i < degree; i++) {
        roots[i] = complex_multiply(r1, roots[i-1]);
    }
    for (int i = 0; i < degree; i++) {
        roots[i]+=center;
    }
}
void aberth_method(inout dcomplex[degree] roots, dpoly coefficients,int pdegree) {
    const float threshold = 1e-8; // Convergence threshold
    const int max_iterations = 40; // Maximum iterations to avoid infinite loop

    for (int iter = 0; iter < max_iterations; iter++) {
        float max_change = 0.0; // Track the largest change in roots

        for (int k = 0; k < pdegree; k++) {
            // Evaluate the polynomial and its derivative at the current root
            dcomplex a = complex_divide(
                evaluatePolynomial(roots[k], coefficients),
                evaluatePolynomialDerivative(roots[k], coefficients)
            );

            dcomplex s = dcomplex(0.0); // Summation term
            for (int j = 0; j < pdegree; j++) {
                if (j != k) { // Avoid self-interaction
                    dcomplex diff = roots[k] - roots[j];
                    //dnumcomplex denom = dot(diff, diff); // Squared magnitude
                    //if (denom > threshold) { // Check against threshold
                        s += complex_divide(dcomplex(1.0, 0.0), diff);
                    //}
                }
            }

            // Compute the correction term
            dcomplex w = complex_divide(a, dcomplex(1.0, 0.0) - complex_multiply(a, s));
            roots[k] -= w; // Update the root

            // Track the maximum change in root
            max_change = float(max(max_change, length(w)));
        }

        // If the maximum change is smaller than the threshold, stop early
        if (max_change < threshold) {
            break; // Converged, exit the loop
        }
    }
}


/*vec4 linearaprox(vec3 origin,int i){
    vec4 result;
    dpolys polys;
    compilepolys(origin,vec3(1,0,0),polys);
    result.x=polys[i][1];
    compilepolys(origin,vec3(0,1,0),polys);
    result.y=polys[i][1];
    compilepolys(origin,vec3(0,0,1),polys);
    result.z=polys[i][1];
    result.w=polys[i][0];
    return result;
}*/

vec2 newtonroot(int iter,dnum x,dpoly coefficients){
    dnum shift=x;
    dnum bestval=inf;
    for(int i=0;i<iter;i++){
        dnum f =evaluatePolynomial(x,coefficients);
        dnum df=evaluatePolynomialDerivative(x,coefficients);
        if(x>=0.0 && abs(f)<bestval){
            bestval=abs(f);
            shift=x;
        }
        if(bestval<EPSILON_ROOTS)break;
        x-=f/df;
    }
    return vec2(shift,bestval);
}



float raymarch(vec3 rayDir, inout vec3 rayOrigin) {
    dpolys polys;
    compilepolys(rayOrigin,rayDir,polys);
    dcomplex[degree] roots;
    
    float guessx=newtonroot(20,0,polys[0]).x;
    initial_roots(roots,dcomplex(guessx,0));
    int pdegree=degree;
    while(abs(polys[0][pdegree])<1E-12)pdegree--;
    aberth_method(roots,polys[0],pdegree);

    float x=inf;
    for(int i=0;i<pdegree;i++){
        vec2 r=vec2(roots[i]);
        if(square(r.y)<(1E-5) && r.x>0){
            x=min(r.x,x);
        }

    }
    

    rayOrigin += rayDir * x;
    return float(evaluatePolynomial(x,polys[0]));
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
    //float checker = 0.3 + 0.7 * mod(sum(floor(p/(length(p-rayOrigin)/100+1) * 4.0)), 2.0);
    
    vec3 col=vec3(checker);
    //col=vec3(1);
    col*=1;//normaltocol(transpose(cameraMatrix)*getNormal(p));
    //col=vec3(abs());
    //col=getlight(p,rayDir,col);
    //col=pow(col,vec3(0.4545));//gamma correction
    //col=pow(col,vec3(2));

    if ((abs(dist) < EPSILON_RAYMARCHING)) {
        col*=vec3(0.5,1,0.5);
    }else if ((abs(dist) < EPSILON_RAYMARCHING*10)){

    } 
    else {
        col*=vec3(1,0.5,0.5);//red tint
        
        //color = vec4(0.0, 0.0, 0.0, 1.0); // Background
    }
    
    if(any(isnan(col))){
        col=vec3(1,1,0);//nan is yellow
    }
    if(any(isinf(vec3(p))) || abs(p.x)>10E10||abs(p.y)>10E10||abs(p.z)>10E10){
        col=vec3(0,0,0.5);//blue
    }
    //if(length(uv)<0.01){col*=vec3(0.7,1,0.7);}//dot in middle of screen
    if(length(uv)<0.01 && length(uv)>0.005){col=vec3(0.5,1,0.5);}//circle in middle of screen

    color= vec4(col,1);
}


//cutoff
