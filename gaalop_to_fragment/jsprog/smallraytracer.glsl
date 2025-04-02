#version 300 es
precision mediump float;

const float FOV=120.;
const float FOVfactor=1.0/tan(radians(FOV) * 0.5);

out vec4 color;

uniform vec3 cameraPos;
uniform vec2 windowsize;
uniform mat3 cameraMatrix;



const float nan=sqrt(-1.);
const float inf=pow(9.,999.);
const float pi=3.14159265359;
const float goldenangle = (3.0 - sqrt(5.0)) * pi;

float intersectplane(vec4 plane,vec3 rd,vec3 ro){
    //a,b,c,d=plane
    //ax+by+cz-d=0;
    //plane.xyz*(pos)-d=0
    //dot(plane.xyz,ro+rd*t)-plane.w=0;
    //dot(plane.xyz,ro)+dot(plane.xyz,rd)*t-plane.w=0;
    //dot(plane.xyz,ro)-plane.w=-dot(plane.xyz,rd)*t;
    //(dot(plane.xyz,ro)-plane.w)/-dot(plane.xyz,rd)=t;
    //(plane.w-dot(plane.xyz,ro))/dot(plane.xyz,rd)=t;
    return (plane.w-dot(plane.xyz,ro))/dot(plane.xyz,rd);
}
float intersectsphere(vec4 sphere,vec3 rd,vec3 ro){
    //a,b,c,r=sphere
    //(xyz-sphere.xyz)^2=sphere.w^2
    //((ro+rd*t)-sphere.xyz)^2=sphere.w^2
    //(rd*t+(ro-sphere.xyz))^2=sphere.w^2
    //(rd*t+(ro-sphere.xyz))^2=sphere.w^2
    //(rd*t+(ro-sphere.xyz)).x^2+(rd*t+(ro-sphere.xyz)).y^2+(rd*t+(ro-sphere.xyz)).z^2=sphere.w^2
    //(rd.x*t)^2+(ro.x-sphere.x)^2+2*(rd.x*t)*(ro.x-sphere.x)+(rd*t+(ro-sphere.xyz)).y^2+(rd*t+(ro-sphere.xyz)).z^2=sphere.w^2
    //dot(rd*t,rd*t)+dot((ro-sphere.xyz),(ro-sphere.xyz))+2*dot(rd*t,ro-sphere.xyz)=sphere.w^2
    //t*t*dot(rd,rd)+dot((ro-sphere.xyz),(ro-sphere.xyz))+2*t*dot(rd,ro-sphere.xyz)=sphere.w^2
    //t*t*dot(rd,rd)+dot((ro-sphere.xyz),(ro-sphere.xyz))+2*t*dot(rd,ro-sphere.xyz)-sphere.w^2=0;
    //t*t*dot(rd,rd)+t*2*dot(rd,ro-sphere.xyz)+(dot((ro-sphere.xyz),(ro-sphere.xyz))-sphere.w^2)=0;
    //x1,x2=-2*dot(rd,ro-sphere.xyz)/(2*dot(rd,rd))+-sqrt((dot(rd,ro-sphere.xyz)/(dot(rd,rd)))^2-(dot((ro-sphere.xyz),(ro-sphere.xyz))-sphere.w^2)/dot(rd,rd));


    //(ro+rd*t-sphere.xyz)^2=sphere.w^2
    //(rd*t+(ro-sphere.xyz))^2=sphere.w^2
    //t*t*(rd)**2+t*(2*rd*(ro-sphere.xyz))+(ro-sphere.xyz)^2-sphere.w^2=0;

    /*float a=dot(rd,rd);
    float b2a=-dot(rd,ro-sphere.xyz)/a;
    float c=dot((ro-sphere.xyz),(ro-sphere.xyz))-sphere.w*sphere.w;
    float det=b2a*b2a-c/a;
    //return 1.;
    if(det<0.)return -inf;
    float sqrtdet=sqrt(det);
    float x2=b2a-sqrtdet;
    if(x2>0.)return x2;
    float x1=b2a+sqrtdet;
    return x1;*/

    float a=dot(rd,rd);
    float b=2.*dot(rd,ro-sphere.xyz);
    float c=dot((ro-sphere.xyz),(ro-sphere.xyz))-sphere.w*sphere.w;

    float det = b * b - 4.0 * a * c; // Discriminant

    if (det < 0.0) {
        return -inf; // No intersection
    }

    float sqrtdet = sqrt(det);
    float t1 = (-b - sqrtdet) / (2.0 * a); // First root
    float t2 = (-b + sqrtdet) / (2.0 * a); // Second root

    // Return the nearest positive intersection distance
    if (t1 > 0.0 && t1 < t2) {
        return t1;
    } else if (t2 > 0.0) {
        return t2;
    }
    return -inf; // No valid intersection

}


#define SPHERE 1
#define PLANE 2
struct object{
    int type;
    vec4 params;
    vec3 ka;//ambient
    vec3 kd;//diffuse
    float ks;//specular
    vec3 kr;//reflection
    //vec3 kt;//transmission
};

struct light{
    vec4 sphereparams;
    vec3 Ii;//intensety
};

vec3 Ia=vec3(1.0)*0.5;//ambient



object[] objects=object[](
    object( SPHERE,vec4(0.1,0.2,0.6,0.1),vec3(0.1),vec3(0.5),10.,vec3(0.9) ),
    object( SPHERE,vec4(0.5,0.7,0.6,0.1),vec3(0.1),vec3(0.5),10.,vec3(0.9) ),
    object( PLANE,vec4(-1.,0.,0.,1.),vec3(1.,0.,0.),0.5*vec3(1.,0.,0.)*0.5,10.,vec3(0.1) ),
    object( PLANE,vec4(1.,0.,0.,1.),vec3(0.,0.,1.),0.5*vec3(0.,0.,1.)*0.5,10.,vec3(0.1) ),
    object( PLANE,vec4(0.,0.,1.,1.),vec3(0.5),0.5*vec3(0.5)*0.5,10.,vec3(0.1) )
);

light[1] lights=light[](
    light(vec4(0.,0.,0.,0.1),vec3(1.,1.,0.))
);

vec4 intersect(int i,vec3 rd,vec3 ro){
    vec4 nt=vec4(0.,0.,0.,-inf);//normal,t
    //return vec4(0.,0.,0.,1.);
    if(objects[i].type==SPHERE){
        nt.w=intersectsphere(objects[i].params,rd,ro);
        nt.xyz=normalize(rd*nt.w+ro-objects[i].params.xyz);
    }else if(objects[i].type==PLANE){
        nt.w=intersectplane(objects[i].params,rd,ro);
        nt.xyz=objects[i].params.xyz;
    }
    return nt;
}

vec4 lightray(vec3 rd,vec3 ro){//rgb,dist
    vec4 ct=vec4(0.,0.,1.,inf);//color,dist along ray
    int index=-1;
    for(int i=0;i<lights.length();i++){
        float t=intersectsphere(lights[i].sphereparams,rd,ro);
        if(t>0.01 && t<ct.w){
            ct.w=t;
            ct.rgb=lights[i].Ii;
        }
    }
    return ct;

}


vec3 raymarch0(vec3 rd,vec3 ro){
    vec4 nt=vec4(0.,0.,0.,inf);//normal,dist along ray
    int index=-1;
    for(int i=0;i<objects.length();i++){
        vec4 ntact=intersect(i,rd,ro);
        if(ntact.w>0.01 && ntact.w<nt.w){
            nt=ntact;
            index=i;
        }
    }
    vec4 ctlight=lightray(rd,ro);
    //return ctlight.rgb;

    if(ctlight.w<nt.w || index==-1)return ctlight.rgb;//hit light or void
    //if(index==-1)return vec3(1.,0.,1.);
    vec3 Ip=vec3(0.);
    Ip+=Ia*objects[index].ka;
    //Ip+=Ia*objects[index].ka;
    vec3 collisionpoint=ro+rd*nt.w;
    
    


    return Ip;


}

vec3 raymarch1(vec3 rd,vec3 ro){
    vec4 nt=vec4(0.,0.,0.,inf);//normal,dist along ray
    int index=-1;
    for(int i=0;i<objects.length();i++){
        vec4 ntact=intersect(i,rd,ro);
        if(ntact.w>0.01 && ntact.w<nt.w){
            nt=ntact;
            index=i;
        }
    }
    vec4 ctlight=lightray(rd,ro);
    //return ctlight.rgb;

    if(ctlight.w<nt.w || index==-1)return ctlight.rgb;//hit light or void
    //if(index==-1)return vec3(1.,0.,1.);
    vec3 Ip=vec3(0.);
    Ip+=Ia*objects[index].ka;
    //Ip+=Ia*objects[index].ka;
    vec3 collisionpoint=ro+rd*nt.w;
    
    vec3 Li;
    for(int i=0;i<lights.length();i++){
        Li=lights[i].sphereparams.xyz-collisionpoint;
        vec3 shadowraycolor=raymarch0(Li,collisionpoint);
        float fd=dot(Li,nt.xyz);
        Ip+=fd*objects[index].kd;
    }
    
    vec3 R=reflect(rd,nt.xyz);
    Ip+=raymarch0(R,collisionpoint)*objects[index].kr;

    return Ip;


}

vec3 raymarch2(vec3 rd,vec3 ro){
    vec4 nt=vec4(0.,0.,0.,inf);//normal,dist along ray
    int index=-1;
    for(int i=0;i<objects.length();i++){
        vec4 ntact=intersect(i,rd,ro);
        if(ntact.w>0.01 && ntact.w<nt.w){
            nt=ntact;
            index=i;
        }
    }
    vec4 ctlight=lightray(rd,ro);
    //return ctlight.rgb;

    if(ctlight.w<nt.w || index==-1)return ctlight.rgb;//hit light or void
    //if(index==-1)return vec3(1.,0.,1.);
    vec3 Ip=vec3(0.);
    Ip+=Ia*objects[index].ka;
    //Ip+=Ia*objects[index].ka;
    vec3 collisionpoint=ro+rd*nt.w;
    
    vec3 Li;
    for(int i=0;i<lights.length();i++){
        Li=lights[i].sphereparams.xyz-collisionpoint;
        vec3 shadowraycolor=raymarch0(Li,collisionpoint);
        float fd=dot(Li,nt.xyz);
        Ip+=fd*objects[index].kd;
    }
    
    vec3 R=reflect(rd,nt.xyz);
    Ip+=raymarch1(R,collisionpoint)*objects[index].kr;

    return Ip;


}


void main(void) {
    vec2 uv=(2.0*gl_FragCoord.xy-windowsize)/windowsize.x;
    vec3 ro = cameraPos;
    vec3 rd =cameraMatrix*normalize(vec3(uv, FOVfactor));//cam to view


    color=vec4(1.);
    color.rgb=raymarch2(rd,ro);
    //color.rgb=lightray(rd,ro).rgb;


}