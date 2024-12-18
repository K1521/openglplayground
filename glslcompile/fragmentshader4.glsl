
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



//https://www.shadertoy.com/view/fsB3Wt
float cbrt(in float x) { return sign(x) * pow(abs(x), 1.0 / 3.0); }
int solveQuartich(in float a, in float b, in float c, in float d, in float e, inout vec4 roots) {
    b /= a; c /= a; d /= a; e /= a; // Divide by leading coefficient to make it 1

    // Depress the quartic to x^4 + px^2 + qx + r by substituting x-b/4a
    // This can be found by substituting x+u and the solving for the value
    // of u that makes the t^3 term go away
    float bb = b * b;
    float p = (8.0 * c - 3.0 * bb) / 8.0;
    float q = (8.0 * d - 4.0 * c * b + bb * b) / 8.0;
    float r = (256.0 * e - 64.0 * d * b + 16.0 * c * bb - 3.0 * bb * bb) / 256.0;
    int n = 0; // Root counter

    // Solve for a root to (t^2)^3 + 2p(t^2)^2 + (p^2 - 4r)(t^2) - q^2 which resolves the
    // system of equations relating the product of two quadratics to the depressed quartic
    float ra =  2.0 * p;
    float rb =  p * p - 4.0 * r;
    float rc = -q * q;

    // Depress using the method above
    float ru = ra / 3.0;
    float rp = rb - ra * ru;
    float rq = rc - (rb - 2.0 * ra * ra / 9.0) * ru;

    float lambda;
    float rh = 0.25 * rq * rq + rp * rp * rp / 27.0;
    if (rh > 0.0) { // Use Cardano's formula in the case of one real root
        rh = sqrt(rh);
        float ro = -0.5 * rq;
        lambda = cbrt(ro - rh) + cbrt(ro + rh) - ru;
    }

    else { // Use complex arithmetic in the case of three real roots
        float rm = sqrt(-rp / 3.0);
        lambda = -2.0 * rm * sin(asin(1.5 * rq / (rp * rm)) / 3.0) - ru;
    }

    // Newton iteration to fix numerical problems (using Horners method)
    // Suggested by @NinjaKoala
    for(int i=0; i < 2; i++) {
        float a_2 = ra + lambda;
        float a_1 = rb + lambda * a_2;
        float b_2 = a_2 + lambda;

        float f = rc + lambda * a_1; // Evaluation of λ^3 + ra * λ^2 + rb * λ + rc
        float f1 = a_1 + lambda * b_2; // Derivative

        lambda -= f / f1; // Newton iteration step
    }

    // Solve two quadratics factored from the quartic using the cubic root
    if (lambda < 0.0) return n;
    float t = sqrt(lambda); // Because we solved for t^2 but want t
    float alpha = 2.0 * q / t, beta = lambda + ra;

    float u = 0.25 * b;
    t *= 0.5;

    float z = -alpha - beta;
    if (z > 0.0) {
        z = sqrt(z) * 0.5;
        float h = +t - u;
        roots.xy = vec2(h + z, h - z);
        n += 2;
    }

    float w = +alpha - beta;
    if (w > 0.0) {
        w = sqrt(w) * 0.5;
        float h = -t - u;
        roots.zw = vec2(h + w, h - w);
        if (n == 0) roots.xy = roots.zw;
        n += 2;
    }

    return n;
}


vec4 solveQuartic(float a, float b, float c, float d, float e) {
    if(abs(a)<1E-20)return vec4(solveCubic(b,c,d,e),nan);
    vec4 result=vec4(nan);


    /*if (stableness_score(a,b) > stableness_score(c,d)) {
        solveQuartich(e,d,c,b,a,result);
        return 1/result;
    }*/
    solveQuartich(a,b,c,d,e,result);
    return result;
}


double evaluatePolynomial(double x, double[numparams] polyparams) {
    double result = polyparams[degree];
    for (int i = degree - 1; i >= 0; i--) {
        result = result * x + polyparams[i];
    }
    return result;
}
double evaluatePolynomialDerivative(double x, double[numparams] polyparams) {
    double result = polyparams[degree]*degree;
    for (int i = degree - 1; i >= 1; i--) {
        result = result * x + polyparams[i]*i;
    }
    return result;
}
vec2 evaluatePolynomialAndDerivativef(float x, float[numparams] polyparams) {
    float resultd = polyparams[degree]*degree;
    float result = polyparams[degree];
    for (int i = degree - 1; i >= 1; i--) {
        resultd = resultd * x + polyparams[i]*i;
        result = result * x + polyparams[i];
    }
    result = result * x + polyparams[0];
    return vec2(result,resultd);
}

dvec2 newtonroot(int iter,double x,double[numparams] polyparams){
    double shift=x;
    double bestval=inf;
    for(int i=0;i<iter;i++){
        double f=evaluatePolynomial(x,polyparams);
        double df=evaluatePolynomialDerivative(x,polyparams);
        if(x>=0.0 && abs(f)<bestval){
            bestval=abs(f);
            shift=x;
        }
        if(bestval<EPSILON_ROOTS)break;
        x-=f/df;
    }
    return dvec2(shift,bestval);
}

vec2 newtonrootf(int iter,float x,double[numparams] polyparams){
    float[numparams] polyparamsf;
    for(int i=0;i<numparams;i++){polyparamsf[i]=float(polyparams[i]);}
    float shift=x;
    float bestval=inf;
    for(int i=0;i<iter;i++){
        vec2 f_df=evaluatePolynomialAndDerivativef(x,polyparamsf);
        float absf=abs(f_df.x);
        if(x>=0.0 && absf<bestval){
            bestval=absf;
            shift=x;
        }
        if(bestval<EPSILON_ROOTS)break;
        x-=f_df.x/f_df.y;
    }
    return vec2(shift,bestval);
}

int findroots2(double[numparams] polyparams,out double[degree] roots){
    //tries to find all roots>0 and refines them with newton method.
    //returns number of found roots
    dvec2 r=newtonrootf(100,0.0,polyparams);
    if(r.x<0)return 0;
    roots[0]=r.x;
    return 1;
}

int findroots(double[numparams] polyparams,out double[degree] roots){
    //tries to find all roots>0 and refines them with newton method.
    //returns number of found roots
    vec4 qroots= solveQuartic(float(polyparams[4]),float(polyparams[3]),float(polyparams[2]),float(polyparams[1]),float(polyparams[0]));
    int numroots=0;
    for(int i=0;i<4;i++){
        if(qroots[i]>0){
            //dvec2 result=dvec2(newtonrootf(10,qroots[i],polyparams));
            dvec2 result=newtonroot(10,qroots[i],polyparams);//changed
            if(result.y<EPSILON_ROOTS)
            roots[numroots++]=result.x;
        }
    }
    return numroots;
}

float findcommonroot(){
    double[degree] roots;
    //TODO Check if the polynom is 0 everywhere and if so chose the next polynom
    int numroots=findroots2(polys[0],roots);//find the roots of the first polynom

    for(int i=1;i<numpolys;i++){
        int numrootsneu=0;
        for(int j=0;j<numroots;j++){//check if it is a common root
            if(evaluatePolynomial(roots[j],polys[i])<EPSILON_ROOTS)
                roots[numrootsneu++]=roots[j];
        }
        numroots=numrootsneu;
        //TODO performance test Shift empty vs Transposed loop
    }

    double smallestroot=inf;
    for(int j=0;j<numroots;j++){
        smallestroot=min(smallestroot,roots[j]);
    }
    return float(smallestroot);
}



float raymarch(vec3 rayDir, inout vec3 rayOrigin) {
    compilepolys(rayOrigin,rayDir);
    float x=findcommonroot();
    

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

    if ((dist < EPSILON_RAYMARCHING)) {

    } else {
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
