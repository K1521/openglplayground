#version 440
/*source=part1*/

const int polybasislength=15;
const int numpolys=5;
//const int MAXPOLYDEGREE=4;

/*source=part0*/
//const int polybasislength=...;
//const ivec3[polybasislength] polybasis=...;
//const int numpolys=...;
//const int MAXPOLYDEGREE=...;
#line 5 1 //code before this is defined in other file
uniform float[numpolys][polybasislength] coefficientsxyz;

//step divided by length of deriv in xyz

//in vec2 fragUV;
out vec4 color;

uniform vec3 cameraPos;
//uniform vec3 lightPos;
//uniform float time;
uniform vec2 windowsize;
uniform mat3 cameraMatrix;


const float FOV=120;
const float FOVfactor=1/tan(radians(FOV) * 0.5);
const float EPSILON_RAYMARCHING=0.001;
//const float EPSILON_NORMALS=0.001;
//const float EPSILON_DERIV=0.0001;
const float EPSILON_ROOTS=0.001;
//const int MAX_RAY_ITER=128;

const float nan=sqrt(-1);
const float inf=pow(9,999);
const float pi=3.14159265359;
const float goldenangle = (3.0 - sqrt(5.0)) * pi;
#define dcomplex vec2
#define dnumcomplex float
#define dnum float
#define dintervall vec2

vec3 overwritecol=vec3(0);
bool overrideactive=false;

void debugcolor(vec3 c){
    overrideactive=true;
    overwritecol=c;
}//TODO complete the code in main

vec3 normaltocol(vec3 normal){
    //return vec3(normal.x/2+0.5,normal.y/2+0.5,0.5-normal.z/2);
    return normal*vec2(1,-1).xxy/2+0.5;
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
bool any(bvec3 b) {
    return b.x || b.y || b.z;
}

#define Dual vec4

Dual dMul(Dual a,Dual b){
    return Dual(a.w*b.xyz+b.w*a.xyz,a.w*b.w);
}
Dual dSqr(Dual a){
    return Dual(2*a.w*a.xyz,a.w*a.w);
}

Dual dSqrt(Dual a){
    float sqrtf=sqrt(a.w);
    return Dual(a.xyz/(2.*sqrtf),sqrtf);
}

Dual dAbs(Dual a) {
    return Dual(a.xyz*sign(a.w),abs(a.w));
}



Dual summofsquares(vec3 rayDir, vec3 rayOrigin){
    Dual x=Dual(1.0,0.0,0.0,rayOrigin.x);
    Dual y=Dual(0.0,1.0,0.0,rayOrigin.y);
    Dual z=Dual(0.0,0.0,1.0,rayOrigin.z);
    
    Dual xx=dSqr(x);
    Dual yy=dSqr(y);
    Dual zz=dSqr(z);

    Dual r=xx+yy+zz;
    Dual rp=(r+Dual(0.,0.,0.,1.))*0.5;
    Dual rm=(r-Dual(0.,0.,0.,1.))*0.5;

    Dual xy=dMul(x,y);
    Dual yz=dMul(y,z);
    Dual zx=dMul(z,x);

    Dual xrp=dMul(x,rp);
    Dual xrm=dMul(x,rm);
    Dual yrp=dMul(y,rp);
    Dual yrm=dMul(y,rm);
    Dual zrp=dMul(z,rp);
    Dual zrm=dMul(z,rm);

    Dual rprp=dMul(rp,rp);
    Dual rmrm=dMul(rm,rm);
    Dual rprm=dMul(rp,rm);

    Dual sum=Dual(0);
    for(int i=0;i<numpolys;i++){
        sum+=dSqr(
            coefficientsxyz[i][0]*xx+
            coefficientsxyz[i][1]*xy+
            coefficientsxyz[i][2]*zx+
            coefficientsxyz[i][3]*xrm+
            coefficientsxyz[i][4]*xrp+
            coefficientsxyz[i][5]*yy+
            coefficientsxyz[i][6]*yz+
            coefficientsxyz[i][7]*yrm+
            coefficientsxyz[i][8]*yrp+
            coefficientsxyz[i][9]*zz+
            coefficientsxyz[i][10]*zrm+
            coefficientsxyz[i][11]*zrp+
            coefficientsxyz[i][12]*rmrm+
            coefficientsxyz[i][13]*rprm+
            coefficientsxyz[i][14]*rprp
            );
    }
    return sum;


    //[x**2, x*y, x*z, rm*x, rp*x, y**2, y*z, rm*y, rp*y, z**2, rm*z, rp*z, rm**2, rm*rp, rp**2]

    

}


float raymarch(vec3 rayDir, inout vec3 rayOrigin) {
    
    float bestx=0;
    float beste=inf;
    float x=0;

    for(int i = 0; i < 60; ++i){

        Dual res = summofsquares(rayDir,rayOrigin+rayDir*x);
        float f=abs(res.w);
        float e=f;///sqrt(x+1);
        if(e < beste){
            beste=e;
            bestx=x;
        }

        // Newton: x = x - f(x) / f'(x)
        //x += min(4, abs(res[0] / res[1]));

        // Halley: x = x - 2f(x)f'(x) / (2(f'(x)²) - f(x)f''(x))
        //x += min(4, abs(2. * res[0] * res[1] / (2. * res[1] * res[1] - res[0] * res[2])));
        
        
        x += min(4, abs(f / length(res.xyz)));

    }
    x=bestx;
    /*for(int i=0;i<10;i++){
        Dual res = summofsquares(rayDir,rayOrigin+rayDir*x);
        float f=res.w;
        float e=abs(f)/(x+1);
        if(e < beste && x>0){
            beste=e;
            bestx=x;
        }

        // Newton: x = x - f(x) / f'(x)
        //x += min(4, abs(res[0] / res[1]));

        // Halley: x = x - 2f(x)f'(x) / (2(f'(x)²) - f(x)f''(x))
        //x += -2. * res[0] * res[1] / (2. * res[1] * res[1] - res[0] * res[2]);

        //x += min(4, (f / length(res.xyz))*sign(dot(res.xyz,rayDir)));
        x -= res.w / dot(res.xyz,rayDir);

    }*/

    rayOrigin += rayDir * bestx;
    return beste;
}




void main() {
    vec2 uv=(2*gl_FragCoord.xy-windowsize)/windowsize.x;
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
    col*=1;//normaltocol(transpose(cameraMatrix)*getNormal(p));
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
    col*=mix(vec3(0.0, 1.0, 0.0), vec3(1.0, 0.0, 0.0), clamp(dist,0,1));
    
    if(any(isnan(col))){
        col=vec3(1,1,0);//nan is yellow
    }
    if(any(isinf(vec3(p))) || abs(p.x)>10E10||abs(p.y)>10E10||abs(p.z)>10E10){
        col=vec3(0,0,0.5);//blue
    }
    //if(length(uv)<0.01){col*=vec3(0.7,1,0.7);}//dot in middle of screen
    if(length(uv)<0.01 && length(uv)>0.005){col=vec3(0.5,1,0.5);}//circle in middle of screen


    if(overrideactive){
        col=overwritecol;
    }

    color= vec4(col,1);
}


//cutoff
