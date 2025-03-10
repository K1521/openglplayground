#version 300 es
precision mediump float;
const int polybasislength=15;
const int numpolys=1;


//const int polybasislength=...;
//const ivec3[polybasislength] polybasis=...;
//const int numpolys=...;
//const int MAXPOLYDEGREE=...;#line 11 1 //code before this is defined in other file
uniform float[numpolys*polybasislength] coefficientsxyz;
/*layout(std140) uniform MyUBO {
    float[numpolys*polybasislength] coefficientsxyz;
};*/

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

#define Dualxyz vec4

Dualxyz DualxyzMul(Dualxyz a,Dualxyz b){
  //  vec3 v = a.w * b.xyz + b.w * a.xyz;
//return Dualxyz(v.x, v.y, v.z, a.w * b.w);
//return Dualxyz(a.w*b.x+b.w*a.x,a.w*b.y+b.w*a.y,a.w*b.z+b.w*a.z,a.w*b.w);
    return Dualxyz(a.w*b.xyz+b.w*a.xyz,a.w*b.w);
}
Dualxyz DualxyzSqr(Dualxyz a){
    return Dualxyz(2.*a.w*a.xyz,a.w*a.w);
}

Dualxyz DualxyzSqrt(Dualxyz a){
    float sqrtf=sqrt(a.w);
    return Dualxyz(a.xyz/(2.*sqrtf),sqrtf);
}

Dualxyz DualxyzAbs(Dualxyz a) {
    return Dualxyz(a.xyz*sign(a.w),abs(a.w));
}

//




Dualxyz DualxyzSummofsquares(vec3 rayDir, vec3 rayOrigin){
    Dualxyz x=Dualxyz(1.0,0.0,0.0,rayOrigin.x);
    Dualxyz y=Dualxyz(0.0,1.0,0.0,rayOrigin.y);
    Dualxyz z=Dualxyz(0.0,0.0,1.0,rayOrigin.z);
    
    Dualxyz xx=DualxyzSqr(x);
    Dualxyz yy=DualxyzSqr(y);
    Dualxyz zz=DualxyzSqr(z);

    Dualxyz r=xx+yy+zz;
    Dualxyz rp=(r+Dualxyz(0.,0.,0.,1.))*0.5;
    Dualxyz rm=(r-Dualxyz(0.,0.,0.,1.))*0.5;

    
    if(numpolys==1){

        return DualxyzSqr(
            DualxyzMul(x,x*coefficientsxyz[0]+y*coefficientsxyz[1]+z*coefficientsxyz[2]+rm*coefficientsxyz[3]+rp*coefficientsxyz[4])+
            DualxyzMul(y,y*coefficientsxyz[5]+z*coefficientsxyz[6]+rm*coefficientsxyz[7]+rp*coefficientsxyz[8])+
            DualxyzMul(z,z*coefficientsxyz[9]+rm*coefficientsxyz[10]+rp*coefficientsxyz[11])+
            DualxyzMul(rm,rm*coefficientsxyz[12]+rp*coefficientsxyz[13])+
            DualxyzSqr(rp)*coefficientsxyz[14]
        );
    }

    Dualxyz basis[15];
    basis[0] = xx;
    basis[1] = DualxyzMul(x,y);
    basis[2] = DualxyzMul(x,z);
    basis[3] = DualxyzMul(x,rm);
    basis[4] = DualxyzMul(x,rp);
    basis[5] = yy;
    basis[6] = DualxyzMul(y,z);
    basis[7] = DualxyzMul(y,rm);
    basis[8] = DualxyzMul(y,rp);
    basis[9] = zz;
    basis[10] = DualxyzMul(z,rm);
    basis[11] = DualxyzMul(z,rp);
    basis[12] = DualxyzSqr(rm);
    basis[13] = DualxyzMul(rp,rm);
    basis[14] = DualxyzSqr(rp);

    Dualxyz sum=Dualxyz(0.);
    for(int i=0;i<numpolys;i++){
        int index=polybasislength*i;
        Dualxyz term = vec4(0.0);
        
        // Accumulate the weighted sum
        for (int j = 0; j < 15; j++) {
            term += coefficientsxyz[index + j] * basis[j];
        }
        
        // Square and add to sum
        sum += DualxyzSqr(term);
    }
    return (sum);


    //[x**2, x*y, x*z, rm*x, rp*x, y**2, y*z, rm*y, rp*y, z**2, rm*z, rp*z, rm**2, rm*rp, rp**2]

    

}


float DualxyzRaymarch(vec3 rayDir, inout vec3 rayOrigin) {
    
    float bestx=0.;
    float beste=inf;
    float x=0.;

    for(int i = 0; i < 60; ++i){

        Dualxyz res = DualxyzSummofsquares(rayDir,rayOrigin+rayDir*x);

        float e=abs(res.w);///sqrt(x+1);
        if(e < beste){
            beste=e;
            bestx=x;
        }

        float dx= abs(res.w / length(res.xyz))*2.;
        x+=dx;
        if(dx<0.0001)break;
        /*if(dx<0.0001){
            //debugcolor(vec3(0.,0.,float(i+1)/20.));
            //debugcolor(vec3(0.,0.,float(i+1)/10.));
            break;
        }*/
        

        // Newton: x = x - f(x) / f'(x)
        //x += min(4, abs(res[0] / res[1]));

        // Halley: x = x - 2f(x)f'(x) / (2(f'(x)²) - f(x)f''(x))
        //x += min(4, abs(2. * res[0] * res[1] / (2. * res[1] * res[1] - res[0] * res[2])));
        
        
        //x += min(4., abs(f / length(res.xyz)));
        
        /*if(x>100.){
            //debugcolor(vec2(float(i+1)/60.,0.).xxy);
            //debugcolor(vec3(0.,0.,float(i+1)/10.));
            break;
        }*/
        //if(i==55)debugcolor(vec3((dx)));
        /*if(dx==0.||i==59){
            //debugcolor(vec3(float(i+1)/60.));
            //debugcolor(vec3(float(i+1)/60.,log2(res.w)/256.+0.5,log2(dx)/256.+0.5));
            break;
        }*/

        //x += min(4., abs(max(f-00.01,0.) / (length(2.*res.xyz)+0.01)));

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
    vec2 uv=(2.*gl_FragCoord.xy-windowsize)/windowsize.x;
    vec3 rayOrigin = cameraPos;
    vec3 rayDir =cameraMatrix*normalize(vec3(uv, FOVfactor));//cam to view
   

    // Sphere tracing

    vec3 p=rayOrigin;//vec3(0);//rayOrigin;
    float dist=DualxyzRaymarch(rayDir,p);
    //float x=dot(rayDir,p-rayOrigin);
    //debugcolor(vec3((x)/10.));
    //float dist=DualxyzRaymarch4(rayDir,p);
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
    col*=mix(vec3(0.0, 1.0, 0.0), vec3(1.0, 0.0, 0.0), clamp(dist*20.,0.,1.));
    
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
