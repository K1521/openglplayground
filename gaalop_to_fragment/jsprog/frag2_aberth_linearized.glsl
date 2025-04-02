#version 300 es
precision highp float;
const int polybasislength=15;
const int numpolys=1;


//const int polybasislength=...;
//const ivec3[polybasislength] polybasis=...;
//const int numpolys=...;
//const int MAXPOLYDEGREE=...;#line 11 1 //code before this is defined in other file
uniform float[numpolys*polybasislength] coefficientsxyz;

//step divided by length of deriv in xyz

//in vec2 fragUV;
out vec4 color;

uniform vec3 cameraPos;
//uniform vec3 lightPos;
//uniform float time;
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
#define dnumcomplex float
#define dnum float
#define dintervall vec2

vec3 overwritecol=vec3(0.);
bool overrideactive=false;

void debugcolor(vec3 c){
    overrideactive=true;
    overwritecol=c;
}
void debugcoloronce(vec3 c){
    if(!overrideactive){
        overrideactive=true;
        overwritecol=c;
    }
}

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
/*bool any(bvec3 b) {
    return b.x || b.y || b.z;
}*/

#define complexDual vec4
#define complex vec2

complex complexMul(complex f, complex g) {
    return complex(f.x * g.x - f.y * g.y, f.x * g.y + f.y * g.x);
}

complex complexSqr(complex f) {
    return complex(f.x * f.x - f.y * f.y, 2.0 * f.x * f.y);
}

complex complexInverse(complex f) {
    return complex(f.x, -f.y)/dot(f,f);
}

complex complexDivide(complex f, complex g) {
    return vec2(f.x * g.x + f.y * g.y, f.y * g.x - f.x * g.y)/dot(g, g);
}

complex complexConjugate(complex f) {
    return vec2(f.x ,-f.y);
}

complexDual complexDualMul(complexDual a,complexDual b){//complex dual numbers
    return complexDual(complexMul(a.xy,b.xy),complexMul(a.xy,b.zw)+complexMul(a.zw,b.xy));
}

complexDual complexDualSqr(complexDual a){//complex dual numbers
    return complexDual(complexSqr(a.xy),2.*complexMul(a.xy,a.zw));
}

struct Poly4{
    float[5] coeffs;//coefficients of 1,x,x^2,x^3,x^4
};

Poly4 mkPoly4(float c0,float c1,float c2,float c3,float c4){
    return Poly4(float[5](c0,c1,c2,c3,c4));
}

Poly4 Poly4Mul(Poly4 a,Poly4 b){
    Poly4 res;
    for(int i=0;i<5;i++){
        float sum=0.;
        for(int j=0;j<=i;j++)sum+=a.coeffs[j]*b.coeffs[i-j];
        res.coeffs[i]=sum;
    }
    return res;
}

Poly4 Poly4Mul(Poly4 a,float b){
    Poly4 res;
    for(int i=0;i<5;i++)
        res.coeffs[i]=a.coeffs[i]*b;
    return res;
}

void Poly4SumMulIp(inout Poly4 sum,Poly4 a,float b){//sum+=a*b
    for(int i=0;i<5;i++)sum.coeffs[i]+=a.coeffs[i]*b;
}

Poly4 Poly4Add(Poly4 a,Poly4 b){
    Poly4 res;
    for(int i=0;i<5;i++)res.coeffs[i]=a.coeffs[i]+b.coeffs[i];
    return res;
}
Poly4 Poly4Sub(Poly4 a,Poly4 b){
    Poly4 res;
    for(int i=0;i<5;i++)res.coeffs[i]=a.coeffs[i]-b.coeffs[i];
    return res;
}

struct Poly8{
    float[9] coeffs;//coefficients of 1,x,x^2,x^3,x^4,x^5,x^6,x^7,x^8
};

Poly8 Poly4toPoly8Square(Poly4 a){
    Poly8 res;
    for(int i=0;i<9;i++)res.coeffs[i]=0.;
    for(int i=0;i<5;i++)
        for(int j=0;j<5;j++)
            res.coeffs[i+j]+=a.coeffs[i]*a.coeffs[j];
    return res;
}

void Poly8SumSqrIp(inout Poly8 sum,Poly4 b){//sum+=b*b
    Poly8 bb=Poly4toPoly8Square(b);
    for(int i=0;i<9;i++)
        sum.coeffs[i]+=bb.coeffs[i];
}

complexDual complexDualEvalPoly8(Poly8 poly,complex x){
    complexDual xDual=complexDual(x,1.,0.);
    complexDual res=poly.coeffs[8]*complexDual(1.,0.,0.,0.);//horners Method
    for(int i=7;i>=0;i--){
        res=poly.coeffs[i]*complexDual(1.,0.,0.,0.)+complexDualMul(res,xDual);
    }
    return res;
}


Poly8 polysummofsquares(vec3 rayDir, vec3 rayOrigin){

    
    //ro+rd*a  ,  rd
    Poly4 x=mkPoly4(rayOrigin.x, rayDir.x, 0.0, 0.0, 0.0);
    Poly4 y=mkPoly4(rayOrigin.y, rayDir.y, 0.0, 0.0, 0.0);
    Poly4 z=mkPoly4(rayOrigin.z, rayDir.z, 0.0, 0.0, 0.0);
    Poly4 one = mkPoly4(1.0, 0.0, 0.0, 0.0, 0.0);

    Poly4 xx=Poly4Mul(x,x);
    Poly4 yy=Poly4Mul(y,y);
    Poly4 zz=Poly4Mul(z,z);
    

    Poly4 r=Poly4Add(Poly4Add(xx,yy),zz);
    Poly4 rp=Poly4Mul(Poly4Add(r,one),0.5);
    Poly4 rm=Poly4Mul(Poly4Sub(r,one),0.5);

    Poly4 xy=Poly4Mul(x,y);
    Poly4 yz=Poly4Mul(y,z);
    Poly4 zx=Poly4Mul(z,x);

    Poly4 xrp=Poly4Mul(x,rp);
    Poly4 xrm=Poly4Mul(x,rm);
    Poly4 yrp=Poly4Mul(y,rp);
    Poly4 yrm=Poly4Mul(y,rm);
    Poly4 zrp=Poly4Mul(z,rp);
    Poly4 zrm=Poly4Mul(z,rm);

    Poly4 rprp=Poly4Mul(rp,rp);
    Poly4 rmrm=Poly4Mul(rm,rm);
    Poly4 rprm=Poly4Mul(rp,rm);

    Poly8 sumofsquares;
    for(int i=0;i<9;i++)sumofsquares.coeffs[i]=0.;

    for(int i=0;i<numpolys;i++){
        int index=polybasislength*i;
        Poly4 sum=mkPoly4(0.0,0.0,0.0,0.0,0.0);
        Poly4SumMulIp(sum,xx  ,coefficientsxyz[index+0 ]);
        Poly4SumMulIp(sum,xy  ,coefficientsxyz[index+1 ]);
        Poly4SumMulIp(sum,zx  ,coefficientsxyz[index+2 ]);
        Poly4SumMulIp(sum,xrm ,coefficientsxyz[index+3 ]);
        Poly4SumMulIp(sum,xrp ,coefficientsxyz[index+4 ]);
        Poly4SumMulIp(sum,yy  ,coefficientsxyz[index+5 ]);
        Poly4SumMulIp(sum,yz  ,coefficientsxyz[index+6 ]);
        Poly4SumMulIp(sum,yrm ,coefficientsxyz[index+7 ]);
        Poly4SumMulIp(sum,yrp ,coefficientsxyz[index+8 ]);
        Poly4SumMulIp(sum,zz  ,coefficientsxyz[index+9 ]);
        Poly4SumMulIp(sum,zrm ,coefficientsxyz[index+10]);
        Poly4SumMulIp(sum,zrp ,coefficientsxyz[index+11]);
        Poly4SumMulIp(sum,rmrm,coefficientsxyz[index+12]);
        Poly4SumMulIp(sum,rprm,coefficientsxyz[index+13]);
        Poly4SumMulIp(sum,rprp,coefficientsxyz[index+14]);

        Poly8SumSqrIp(sumofsquares,sum);
    }
    return sumofsquares;


    //[x**2, x*y, x*z, rm*x, rp*x, y**2, y*z, rm*y, rp*y, z**2, rm*z, rp*z, rm**2, rm*rp, rp**2]

    

}


complexDual summofsquares(vec3 rayDir, vec3 rayOrigin,vec2 root){

    
    //ro+rd*a  ,  rd
    complexDual x=complexDual(complex(rayOrigin.x,0.0)+rayDir.x*root,rayDir.x,0.0);
    complexDual y=complexDual(complex(rayOrigin.y,0.0)+rayDir.y*root,rayDir.y,0.0);
    complexDual z=complexDual(complex(rayOrigin.z,0.0)+rayDir.z*root,rayDir.z,0.0);
    const complexDual one = complexDual(1.0, 0.0, 0.0, 0.0);

    complexDual xx=complexDualSqr(x);
    complexDual yy=complexDualSqr(y);
    complexDual zz=complexDualSqr(z);
    

    complexDual r=xx+yy+zz;
    complexDual rp=(r+one)*0.5;
    complexDual rm=(r-one)*0.5;

    complexDual xy=complexDualMul(x,y);
    complexDual yz=complexDualMul(y,z);
    complexDual zx=complexDualMul(z,x);

    complexDual xrp=complexDualMul(x,rp);
    complexDual xrm=complexDualMul(x,rm);
    complexDual yrp=complexDualMul(y,rp);
    complexDual yrm=complexDualMul(y,rm);
    complexDual zrp=complexDualMul(z,rp);
    complexDual zrm=complexDualMul(z,rm);

    complexDual rprp=complexDualMul(rp,rp);
    complexDual rmrm=complexDualMul(rm,rm);
    complexDual rprm=complexDualMul(rp,rm);

    complexDual sum=complexDual(0.);
    for(int i=0;i<numpolys;i++){
        int index=polybasislength*i;
        sum+=complexDualSqr(
            coefficientsxyz[index+0]*xx+
            coefficientsxyz[index+1]*xy+
            coefficientsxyz[index+2]*zx+
            coefficientsxyz[index+3]*xrm+
            coefficientsxyz[index+4]*xrp+
            coefficientsxyz[index+5]*yy+
            coefficientsxyz[index+6]*yz+
            coefficientsxyz[index+7]*yrm+
            coefficientsxyz[index+8]*yrp+
            coefficientsxyz[index+9]*zz+
            coefficientsxyz[index+10]*zrm+
            coefficientsxyz[index+11]*zrp+
            coefficientsxyz[index+12]*rmrm+
            coefficientsxyz[index+13]*rprm+
            coefficientsxyz[index+14]*rprp
            );
    }
    return sum;


    //[x**2, x*y, x*z, rm*x, rp*x, y**2, y*z, rm*y, rp*y, z**2, rm*z, rp*z, rm**2, rm*rp, rp**2]

    

}
const int numroots=5;
void initialroots(out complex[numroots] roots) {
    const complex r1 = complex(cos(goldenangle), sin(goldenangle)); // Base complex number
    roots[0]=r1;
    for (int i = 1; i < numroots; i++) {
        roots[i] = complexMul(r1, roots[i-1]);
    }
}

void aberth_method(inout complex[numroots] roots,vec3 rd,vec3 ro) {
    const float threshold = 1e-8; // Convergence threshold
    const int max_iterations = 60; // Maximum iterations to avoid infinite loop
    const complex one = complex(1.0, 0.0);

    for (int iter = 0; iter < max_iterations; iter++) {
        float max_change = 0.0; // Track the largest change in roots

        for (int k = 0; k < numroots; k++) {

            complex rk=roots[k];
            // Evaluate the polynomial and its derivative at the current root
            complexDual ff_ = summofsquares(rd,ro,rk);

            //if(any(isnan(rk)))debugcoloronce(vec3(1.,1.,0.));
            //if(length(rk)>1E5)debugcoloronce(vec3(1.,1.,0.));

            if(dot(ff_.zw, ff_.zw)==0.)continue;//derivative is small


            complex a = complexDivide(
                ff_.xy,
                ff_.zw
            );

            complex s = complexInverse(rk-complexConjugate(rk));//complex(0.0); // Summation term
            for (int j = 0; j < numroots; j++) {
                if (j != k) { // Avoid self-interaction
                    //complex diff = rk - roots[j];
                    //dnumcomplex denom = dot(diff, diff); // Squared magnitude
                    //if (denom > threshold) { // Check against threshold
                        s += complexInverse(rk - roots[j])+complexInverse(rk - complexConjugate(roots[j]));
                    //}
                }
            }

            // Compute the correction term
            complex w = complexDivide(a, one - complexMul(a, s));
            if(any(isnan(w)))continue;
            roots[k] -= w; // Update the root

            // Track the maximum change in root
            max_change = max(max_change, length(w));
        }

        // If the maximum change is smaller than the threshold, stop early
        if (max_change < threshold) {
            break; // Converged, exit the loop
        }
    }
}

void aberth_method_linearized(inout complex[numroots] roots,vec3 rd,vec3 ro) {
    Poly8 sumofsquares=polysummofsquares(rd,ro);
    const float threshold = 1e-8; // Convergence threshold
    const int max_iterations = 60; // Maximum iterations to avoid infinite loop
    const complex one = complex(1.0, 0.0);

    for (int iter = 0; iter < max_iterations; iter++) {
        float max_change = 0.0; // Track the largest change in roots

        for (int k = 0; k < numroots; k++) {

            complex rk=roots[k];
            // Evaluate the polynomial and its derivative at the current root
            complexDual ff_ = complexDualEvalPoly8(sumofsquares,rk);

            //if(any(isnan(rk)))debugcoloronce(vec3(1.,1.,0.));
            //if(length(rk)>1E5)debugcoloronce(vec3(1.,1.,0.));

            if(dot(ff_.zw, ff_.zw)==0.)continue;//derivative is small


            complex a = complexDivide(
                ff_.xy,
                ff_.zw
            );

            complex s = complexInverse(rk-complexConjugate(rk));//complex(0.0); // Summation term
            for (int j = 0; j < numroots; j++) {
                if (j != k) { // Avoid self-interaction
                    //complex diff = rk - roots[j];
                    //dnumcomplex denom = dot(diff, diff); // Squared magnitude
                    //if (denom > threshold) { // Check against threshold
                        s += complexInverse(rk - roots[j])+complexInverse(rk - complexConjugate(roots[j]));
                    //}
                }
            }

            // Compute the correction term
            complex w = complexDivide(a, one - complexMul(a, s));
            if(any(isnan(w)))continue;
            roots[k] -= w; // Update the root

            // Track the maximum change in root
            max_change = max(max_change, length(w));
        }

        // If the maximum change is smaller than the threshold, stop early
        if (max_change < threshold) {
            break; // Converged, exit the loop
        }
    }
}



float raymarch(vec3 rayDir, inout vec3 rayOrigin) {
    
    float bestx=inf;
    float beste=inf;
    //float x=0.;

    complex[numroots] roots;
    initialroots(roots);
    aberth_method_linearized(roots,rayDir,rayOrigin);

    for(int i=0;i<numroots;i++){
        vec2 r=vec2(roots[i]);
        //if(any(isnan(r)))debugcolor(vec3(1.,0,1.));
        float e=r.y*r.y;
        /*if(e < beste && r.x>0.){
            beste=e;
            bestx=r.x;
        }*/
        if(e < (1E-2) && r.x>0. && r.x<bestx){
            beste=e;
            bestx=r.x;
        }
        /*if(r.x>0. && r.x<bestx){
            beste=e;
            bestx=r.x;
            //debugcolor(vec3(1.,1.,1.));
        }*/
        
    }
    if(isinf(bestx)){
        for(int i=0;i<numroots;i++){
            vec2 r=vec2(roots[i]);
            float e=r.y*r.y;
            if(e < beste && r.x>0.){
                beste=e;
                bestx=r.x;
            }
        }
    }
    //if(isinf(bestx))debugcolor(vec3(1.,0,1.));

    rayOrigin += rayDir * bestx;
    return beste;
}


void main() {
    vec2 uv=(2.*gl_FragCoord.xy-windowsize)/windowsize.x;
    vec3 rayOrigin = cameraPos;
    vec3 rayDir =cameraMatrix*normalize(vec3(uv, FOVfactor));//cam to view
   

    // Sphere tracing

    vec3 p=rayOrigin;//vec3(0);//rayOrigin;
    float dist=raymarch(rayDir,p);
    //float dist=raymarch4(rayDir,p);
    //p=(cameraMatrix)*p+rayOrigin;


        
    
    // Checkerboard pattern
    float checker = 0.3 + 0.7 * mod(sum(floor(p * 4.0)), 2.0); // Alternates between 0.5 and 1.0
    //float checker = 0.3 + 0.7 * mod(sum(floor(p/(length(p-rayOrigin)/100+1) * 4.0)), 2.0);
    //checker=(abs(mod(p.x,0.3))<0.03 || abs(mod(p.y,0.3))<0.03 || abs(mod(p.z,0.3))<0.03) ?1.0:0.3;
    
    vec3 col=vec3(checker);
    //col=vec3(1);
    col*=1.;//normaltocol(transpose(cameraMatrix)*getNormal(p));
    //col=vec3(abs());
    //col=getlight(p,rayDir,col);
    //col=pow(col,vec3(0.4545));//gamma correction
    //col=pow(col,vec3(2));

    /*if ((abs(dist) < EPSILON_RAYMARCHING)) {
        col*=vec3(0.5,1,0.5);
    }else if ((abs(dist) < EPSILON_RAYMARCHING*10)){

    } 
    else {
        col*=vec3(1,0.5,0.5);//red tint
        
        //color = vec4(0.0, 0.0, 0.0, 1.0); // Background
    }*/
    col*=mix(vec3(0.0, 1.0, 0.0), vec3(1.0, 0.0, 0.0), clamp((15.+log(dist))/15.,0.,1.));
    
    if(any(isnan(col))){
        col=vec3(1.,1.,0.);//nan is yellow
    }
    if(any(isinf(vec3(p))) || abs(p.x)>10E10||abs(p.y)>10E10||abs(p.z)>10E10){
        col=vec3(0.,0.,0.5);//blue
    }
    //if(length(uv)<0.01){col*=vec3(0.7,1,0.7);}//dot in middle of screen
    if(length(uv)<0.01 && length(uv)>0.005){col=vec3(0.5,1,0.5);}//circle in middle of screen


    if(overrideactive){
        col=overwritecol;
    }

    color= vec4(col,1.);
}


//cutoff
