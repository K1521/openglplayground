#version 300 es
precision mediump float;
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
#define dcomplex vec2
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

#define cDual vec4

dcomplex complexMul(dcomplex f, dcomplex g) {
    return dcomplex(f.x * g.x - f.y * g.y, f.x * g.y + f.y * g.x);
}

dcomplex complexSqr(dcomplex f) {
    return dcomplex(f.x * f.x - f.y * f.y, 2.0 * f.x * f.y);
}

dcomplex complexInverse(dcomplex f) {
    return dcomplex(f.x, -f.y)/dot(f,f);
}

dcomplex complexDivide(dcomplex f, dcomplex g) {
    return vec2(f.x * g.x + f.y * g.y, f.y * g.x - f.x * g.y)/dot(g, g);
}

dcomplex complexConjugate(dcomplex f) {
    return vec2(f.x ,-f.y);
}

cDual dMul(cDual a,cDual b){//complex dual numbers
    return cDual(complexMul(a.xy,b.xy),complexMul(a.xy,b.zw)+complexMul(a.zw,b.xy));
}

cDual dSqr(cDual a){//complex dual numbers
    return cDual(complexSqr(a.xy),2.*complexMul(a.xy,a.zw));
}






cDual summofsquares(vec3 rayDir, vec3 rayOrigin,vec2 root){

    
    //ro+rd*a  ,  rd
    cDual x=cDual(dcomplex(rayOrigin.x,0.0)+rayDir.x*root,rayDir.x,0.0);
    cDual y=cDual(dcomplex(rayOrigin.y,0.0)+rayDir.y*root,rayDir.y,0.0);
    cDual z=cDual(dcomplex(rayOrigin.z,0.0)+rayDir.z*root,rayDir.z,0.0);
    const cDual one = cDual(1.0, 0.0, 0.0, 0.0);

    cDual xx=dSqr(x);
    cDual yy=dSqr(y);
    cDual zz=dSqr(z);
    

    cDual r=xx+yy+zz;
    cDual rp=(r+one)*0.5;
    cDual rm=(r-one)*0.5;

    cDual xy=dMul(x,y);
    cDual yz=dMul(y,z);
    cDual zx=dMul(z,x);

    cDual xrp=dMul(x,rp);
    cDual xrm=dMul(x,rm);
    cDual yrp=dMul(y,rp);
    cDual yrm=dMul(y,rm);
    cDual zrp=dMul(z,rp);
    cDual zrm=dMul(z,rm);

    cDual rprp=dMul(rp,rp);
    cDual rmrm=dMul(rm,rm);
    cDual rprm=dMul(rp,rm);

    cDual sum=cDual(0.);
    for(int i=0;i<numpolys;i++){
        int index=polybasislength*i;
        sum+=dSqr(
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
void initialroots(out dcomplex[numroots] roots) {
    const dcomplex r1 = dcomplex(cos(goldenangle), sin(goldenangle)); // Base complex number
    roots[0]=r1;
    for (int i = 1; i < numroots; i++) {
        roots[i] = complexMul(r1, roots[i-1]);
    }
}

void aberth_method(inout dcomplex[numroots] roots,vec3 rd,vec3 ro) {
    const float threshold = 1e-8; // Convergence threshold
    const int max_iterations = 60; // Maximum iterations to avoid infinite loop
    const dcomplex one = dcomplex(1.0, 0.0);

    for (int iter = 0; iter < max_iterations; iter++) {
        float max_change = 0.0; // Track the largest change in roots

        for (int k = 0; k < numroots; k++) {

            dcomplex rk=roots[k];
            // Evaluate the polynomial and its derivative at the current root
            cDual ff_ = summofsquares(rd,ro,rk);

            //if(any(isnan(rk)))debugcoloronce(vec3(1.,1.,0.));
            //if(length(rk)>1E5)debugcoloronce(vec3(1.,1.,0.));

            if(dot(ff_.zw, ff_.zw)==0.)continue;//derivative is small


            dcomplex a = complexDivide(
                ff_.xy,
                ff_.zw
            );

            dcomplex s = complexInverse(rk-complexConjugate(rk));//dcomplex(0.0); // Summation term
            for (int j = 0; j < numroots; j++) {
                if (j != k) { // Avoid self-interaction
                    //dcomplex diff = rk - roots[j];
                    //dnumcomplex denom = dot(diff, diff); // Squared magnitude
                    //if (denom > threshold) { // Check against threshold
                        s += complexInverse(rk - roots[j])+complexInverse(rk - complexConjugate(roots[j]));
                    //}
                }
            }

            // Compute the correction term
            dcomplex w = complexDivide(a, one - complexMul(a, s));
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

    dcomplex[numroots] roots;
    initialroots(roots);
    aberth_method(roots,rayDir,rayOrigin);

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
