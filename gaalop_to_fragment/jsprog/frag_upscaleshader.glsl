#version 300 es
precision mediump float;
const int polybasislength=15;
const int numpolys=1;



//const int polybasislength=...;
//const ivec3[polybasislength] polybasis=...;
//const int numpolys=...;
//const int MAXPOLYDEGREE=...;#line 11 1 //code before this is defined in other file
uniform float[numpolys*polybasislength] coefficientsxyz;
//uniform sampler2D roots01tex;
//uniform sampler2D roots23tex;
uniform float mipLevel;
uniform float aspectratio; 

uniform vec3 cameraPos;
//uniform vec3 lightPos;
//uniform float time;
uniform vec2 windowsize;
uniform mat3 cameraMatrix;

layout(location = 0) out vec4 roots01out;
layout(location = 1) out vec4 roots23out;
 
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
#define dcomplex vec2
#define dnumcomplex float
#define dnum float
#define dintervall vec2

vec3 overwritecol=vec3(0.);
bool overrideactive=false;

void debugcolor(vec3 c){
    overrideactive=true;
    overwritecol=c;
}//TODO complete the code in main

vec3 normaltocol(vec3 normal){
    //return vec3(normal.x/2+0.5,normal.y/2+0.5,0.5-normal.z/2);
    return normal*vec2(1,-1).xxy/.2+0.5;
    //return normal;
    //return normalize(pow((normal*vec2(1,-1).xxy+1)/2,vec3(1)));
}

vec3 getNormal(vec3 p){
    return normalize( -cross(dFdx(p), dFdy(p)) );
}


float sum(vec3 v) {
    return v.x+v.y+v.z;
}
int sum(ivec3 v) {
    return v.x+v.y+v.z;
}
float vmax(vec3 v) {
    return max(max(v.x, v.y), v.z);
}
/*bool any(bvec3 b) {
    return b.x || b.y || b.z;
}*/

#define DualComplex vec4
#define Complex vec2


Complex ComplexMul(Complex a, Complex b) {
    // Complex multiplication: (a.x + i*a.y) * (b.x + i*b.y)
    return Complex(
        a.x * b.x - a.y * b.y, // Real part
        a.x * b.y + a.y * b.x  // Imaginary part
    );
}
Complex ComplexSquare(Complex a) {
    // Complex multiplication: (a.x + i*a.y) * (b.x + i*b.y)
    return Complex(
        a.x * a.x - a.y * a.y, // Real part
        2.0 * a.x * a.y  // Imaginary part
    );
}
Complex ComplexDiv(Complex a, Complex b) {
    // Complex division: (a.x + i*a.y) / (b.x + i*b.y)
    return Complex(
        (a.x * b.x + a.y * b.y) ,
        (a.y * b.x - a.x * b.y)
    ) / dot(b,b);
}
Complex ComplexInv(Complex a) {
    // Complex division: 1 / (a.x + i*a.y)
    return a*Complex(1,-1) / dot(a,a);
}
Complex ComplexConjugate(Complex a) {
    // Complex division: 1 / (a.x + i*a.y)
    return a*Complex(1,-1);
}



DualComplex DualComplexMul(DualComplex a,DualComplex b){
    return DualComplex(ComplexMul(a.xy,b.xy),ComplexMul(a.xy,b.zw)+ComplexMul(a.zw,b.xy));
}
DualComplex DualComplexSqare(DualComplex a){
    return DualComplex(ComplexSquare(a.xy),2.0*ComplexMul(a.xy,a.zw));
}


//




DualComplex DualComplexSummofsquares(vec3 rayDir, vec3 rayOrigin,Complex a){
    DualComplex x=DualComplex(ComplexMul(Complex(rayDir.x,0.),a)+Complex(rayOrigin.x,0.),rayDir.x,0.);
    DualComplex y=DualComplex(ComplexMul(Complex(rayDir.y,0.),a)+Complex(rayOrigin.y,0.),rayDir.y,0.);;
    DualComplex z=DualComplex(ComplexMul(Complex(rayDir.z,0.),a)+Complex(rayOrigin.z,0.),rayDir.z,0.);;
    
    DualComplex xx=DualComplexSqare(x);
    DualComplex yy=DualComplexSqare(y);
    DualComplex zz=DualComplexSqare(z);

    DualComplex r=xx+yy+zz;
    DualComplex rp=(r+DualComplex(1.,0.,0.,0.))*0.5;
    DualComplex rm=(r-DualComplex(1.,0.,0.,0.))*0.5;

    
    if(numpolys==1){

        return DualComplexSqare(
            DualComplexMul(x,x*coefficientsxyz[0]+y*coefficientsxyz[1]+z*coefficientsxyz[2]+rm*coefficientsxyz[3]+rp*coefficientsxyz[4])+
            DualComplexMul(y,y*coefficientsxyz[5]+z*coefficientsxyz[6]+rm*coefficientsxyz[7]+rp*coefficientsxyz[8])+
            DualComplexMul(z,z*coefficientsxyz[9]+rm*coefficientsxyz[10]+rp*coefficientsxyz[11])+
            DualComplexMul(rm,rm*coefficientsxyz[12]+rp*coefficientsxyz[13])+
            DualComplexSqare(rp)*coefficientsxyz[14]
        );
    }

    DualComplex basis[15];
    basis[0] = xx;
    basis[1] = DualComplexMul(x,y);
    basis[2] = DualComplexMul(x,z);
    basis[3] = DualComplexMul(x,rm);
    basis[4] = DualComplexMul(x,rp);
    basis[5] = yy;
    basis[6] = DualComplexMul(y,z);
    basis[7] = DualComplexMul(y,rm);
    basis[8] = DualComplexMul(y,rp);
    basis[9] = zz;
    basis[10] = DualComplexMul(z,rm);
    basis[11] = DualComplexMul(z,rp);
    basis[12] = DualComplexSqare(rm);
    basis[13] = DualComplexMul(rp,rm);
    basis[14] = DualComplexSqare(rp);

    DualComplex sum=DualComplex(0.);
    for(int i=0;i<numpolys;i++){
        int index=polybasislength*i;
        DualComplex term = vec4(0.0);
        
        // Accumulate the weighted sum
        for (int j = 0; j < 15; j++) {
            term += coefficientsxyz[index + j] * basis[j];
        }
        
        // Square and add to sum
        sum += DualComplexSqare(term);
    }
    return (sum);


    //[x**2, x*y, x*z, rm*x, rp*x, y**2, y*z, rm*y, rp*y, z**2, rm*z, rp*z, rm**2, rm*rp, rp**2]

    

}
void aberth_method(inout Complex[4] roots, vec3 rayDir, vec3 rayOrigin,int pdegree) {
    const float threshold = 1e-4; // Convergence threshold
    const int max_iterations = 40; // Maximum iterations to avoid infinite loop

    for (int iter = 0; iter < max_iterations; iter++) {
        float max_change = 0.0; // Track the largest change in roots

        for (int k = 0; k < pdegree; k++) {
            // Evaluate the polynomial and its derivative at the current root
            DualComplex res=DualComplexSummofsquares(rayDir,rayOrigin,roots[k]);
            Complex a = ComplexDiv(
                res.xy,
                res.zw
            );

            /*Complex s = Complex(0.0); // Summation term
            for (int j = 0; j < pdegree; j++) {
                if (j != k) { // Avoid self-interaction
                    Complex diff = roots[k] - roots[j];
                    s += ComplexInv(diff);
                }
            }*/

            Complex rk=roots[k];

            Complex s = ComplexInv(rk-ComplexConjugate(rk)); // Summation term
            for (int j = 0; j < pdegree; j++) {
                if (j != k) { // Avoid self-interaction
                    s += ComplexInv(rk - roots[j])+ComplexInv(rk - ComplexConjugate(roots[j]));
                }
            }

            // Compute the correction term
            Complex w = ComplexDiv(a, Complex(1.0, 0.0) - ComplexMul(a, s));
            if(any(isnan(w)))continue;
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

void initial_roots(out Complex[4] roots,Complex center) {
    const Complex r1 = Complex(cos(goldenangle), sin(goldenangle)); // Base complex number
    roots[0]=r1;
    for (int i = 1; i < 4; i++) {
        roots[i] = ComplexMul(r1, roots[i-1]);
    }
    for (int i = 0; i < 4; i++) {
        roots[i]+=center;
    }
}






void main() {
    //vec2 uv=(2.*gl_FragCoord.xy-windowsize)/windowsize.x;
    vec2 uv=(gl_FragCoord.xy)/windowsize;
    vec2 uvray=(uv*2.-1.)/vec2(1.,aspectratio);
    
    vec3 rayOrigin = cameraPos;
    vec3 rayDir =cameraMatrix*normalize(vec3(uvray, FOVfactor));//cam to view
   
    //vec4 roots01in=//texture(roots01tex, uv);//, mipLevel);
    //vec4 roots23in=//texture(roots23tex, uv);//, mipLevel);
    Complex[4] roots;
    /*roots[0]=roots01in.xy;
    roots[1]=roots01in.zw;
    roots[2]=roots23in.xy;
    roots[3]=roots23in.zw;*/
initial_roots(roots,vec2(0.));
    


    aberth_method(roots,rayDir,rayOrigin,4);

    /*vec2 x=vec2(1.,1.);

    roots[0]=x;
    roots[1]=x;
    roots[2]=x;
    roots[3]=x;*/


    roots01out=vec4(roots[0],roots[1]);
    roots23out=vec4(roots[2],roots[3]);
}


//cutoff
