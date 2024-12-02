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


float sum(vec3 v) {
    return v.x+v.y+v.z;
}

//float calcpolys(float x);
//void compilepolys(vec3 p,vec3 d);

const int numpolys=1;
const int numparams=5;
float[1][5] polys;
//float A;



float calcpolys(float x){
    //x-=A;
    float[5] pows;
    pows[0]=1;
    for(int j=1;j<numparams;j++){
        pows[j]=pows[j-1]*x/j;
    }

    float s=0;
    for(int i=0;i<numpolys;i++){
        float[5] poly=polys[i];
        float d=0;
        for (int j = 0; j < numparams; j++) {
            d += poly[j] * pows[j];
        }
        s+=abs(d);
    }
    return s;
}

//void compilepolys(vec3 p,vec3 d,float a){
void compilepolys(vec3 p,vec3 d){
//A=a;
float a=0;
float ox=p.x;
float oy=p.y;
float oz=p.z;
float dx=d.x;
float dy=d.y;
float dz=d.z;
float n0=ox;
float n1=a;
float n2=dx;
float n3=(n1*n2);
float n4=(n0+n3);
float n5=(n4*n4);
float n6=20.0;
float n7=(n5*n6);
float n8=oy;
float n9=dy;
float n10=(n1*n9);
float n11=(n8+n10);
float n12=48.0;
float n13=(n4*n11*n12);
float n14=oz;
float n15=dz;
float n16=(n1*n15);
float n17=(n14+n16);
float n18=60.0;
float n19=(n4*n17*n18);
float n20=(n11*n11);
float n21=(n17*n17);
float n22=(n5+n20+n21);
float n23=0.5;
float n24=(n22*n23);
float n25=-0.5;
float n26=(n24+n25);
float n27=268.5;
float n28=(n4*n26*n27);
float n29=(n24+n23);
float n30=-280.5;
float n31=(n4*n29*n30);
float n32=(n20*n12);
float n33=80.0;
float n34=(n11*n17*n33);
float n35=358.0;
float n36=(n11*n26*n35);
float n37=-374.0;
float n38=(n11*n29*n37);
float n39=100.0;
float n40=(n21*n39);
float n41=527.5;
float n42=(n17*n26*n41);
float n43=-547.5;
float n44=(n17*n29*n43);
float n45=2382.5625;
float n46=(n26*n26*n45);
float n47=-2488.0625;
float n48=(n26*n29*n47);
float n49=2597.5625;
float n50=(n29*n29*n49);
float n51=(n7+n13+n19+n28+n31+n13+n32+n34+n36+n38+n19+n34+n40+n42+n44+n28+n36+n42+n46+n48+n31+n38+n44+n48+n50);
float n52=-1095.0;
float n53=(n29*n52);
float n54=1055.0;
float n55=(n26*n54);
float n56=160.0;
float n57=(n11*n56);
float n58=(n29*n49);
float n59=-4976.125;
float n60=(n26*n59);
float n61=(n17*n52);
float n62=-748.0;
float n63=(n11*n62);
float n64=-561.0;
float n65=(n4*n64);
float n66=(n29*n59);
float n67=(n26*n45);
float n68=(n17*n54);
float n69=716.0;
float n70=(n11*n69);
float n71=537.0;
float n72=(n4*n71);
float n73=(n58+n58+n60+n61+n63+n65+n66+n67+n67+n68+n70+n72);
float n74=(n73*n23);
float n75=(n74+n39);
float n76=(n17*n75);
float n77=120.0;
float n78=(n4*n77);
float n79=(n53+n55+n57+n76+n76+n78);
float n80=(n15*n79);
float n81=(n29*n62);
float n82=(n26*n69);
float n83=(n17*n56);
float n84=(n74+n12);
float n85=(n11*n84);
float n86=96.0;
float n87=(n4*n86);
float n88=(n81+n82+n83+n85+n85+n87);
float n89=(n9*n88);
float n90=(n29*n64);
float n91=(n26*n71);
float n92=(n17*n77);
float n93=(n11*n86);
float n94=(n74+n6);
float n95=(n4*n94);
float n96=(n90+n91+n92+n93+n95+n95);
float n97=(n2*n96);
float n98=(n80+n89+n97);
float n99=2;
float n100=(n2*n99);
float n101=(n100*n18);
float n102=(n9*n99);
float n103=(n102*n33);
float n104=(n75*n15);
float n105=(n4*n2);
float n106=(n11*n9);
float n107=(n17*n15);
float n108=(n105+n105+n106+n106+n107+n107);
float n109=(n108*n23);
float n110=(n109*n99);
float n111=(n110*n41);
float n112=(n110*n43);
float n113=(n100*n27);
float n114=(n102*n35);
float n115=(n109*n45);
float n116=(n110*n47);
float n117=(n15*n99);
float n118=(n117*n41);
float n119=(n100*n30);
float n120=(n102*n37);
float n121=(n109*n49);
float n122=(n117*n43);
float n123=(n113+n114+n115+n115+n116+n118+n119+n120+n116+n121+n121+n122);
float n124=(n123*n23);
float n125=(n17*n124);
float n126=(n101+n103+n104+n111+n112+n104+n125+n125);
float n127=(n15*n126);
float n128=(n100*n12);
float n129=(n84*n9);
float n130=(n110*n35);
float n131=(n110*n37);
float n132=(n117*n33);
float n133=(n11*n124);
float n134=(n128+n129+n129+n130+n131+n132+n133+n133);
float n135=(n9*n134);
float n136=(n94*n2);
float n137=(n102*n12);
float n138=(n117*n18);
float n139=(n110*n27);
float n140=(n110*n30);
float n141=(n4*n124);
float n142=(n136+n136+n137+n138+n139+n140+n141+n141);
float n143=(n2*n142);
float n144=(n127+n135+n143);
float n145=(n124*n15);
float n146=(n2*n64);
float n147=(n9*n62);
float n148=(n109*n59);
float n149=(n15*n52);
float n150=(n2*n71);
float n151=(n9*n69);
float n152=(n15*n54);
float n153=(n146+n147+n121+n121+n148+n149+n150+n151+n148+n115+n115+n152);
float n154=(n153*n23);
float n155=(n15*n154);
float n156=(n2*n2);
float n157=(n9*n9);
float n158=(n15*n15);
float n159=(n156+n156+n157+n157+n158+n158);
float n160=(n159*n23);
float n161=(n160*n99);
float n162=(n161*n41);
float n163=(n161*n43);
float n164=(n160*n45);
float n165=(n161*n47);
float n166=(n160*n49);
float n167=(n164+n164+n165+n165+n166+n166);
float n168=(n167*n23);
float n169=(n17*n168);
float n170=(n145+n145+n155+n155+n162+n163+n169+n169);
float n171=(n15*n170);
float n172=(n124*n9);
float n173=(n9*n154);
float n174=(n161*n35);
float n175=(n161*n37);
float n176=(n11*n168);
float n177=(n172+n172+n173+n173+n174+n175+n176+n176);
float n178=(n9*n177);
float n179=(n124*n2);
float n180=(n2*n154);
float n181=(n161*n27);
float n182=(n161*n30);
float n183=(n4*n168);
float n184=(n179+n179+n180+n180+n181+n182+n183+n183);
float n185=(n2*n184);
float n186=(n171+n178+n185);
float n187=(n168*n15);
float n188=(n160*n59);
float n189=(n166+n166+n188+n188+n164+n164);
float n190=(n189*n23);
float n191=(n15*n190);
float n192=(n187+n187+n187+n187+n191+n191);
float n193=(n15*n192);
float n194=(n168*n9);
float n195=(n9*n190);
float n196=(n194+n194+n194+n194+n195+n195);
float n197=(n9*n196);
float n198=(n168*n2);
float n199=(n2*n190);
float n200=(n198+n198+n198+n198+n199+n199);
float n201=(n2*n200);
float n202=(n193+n197+n201);
polys[0][0]=51;
polys[0][1]=n98;
polys[0][2]=n144;
polys[0][3]=n186;
polys[0][4]=n202;
}






float raymarch(vec3 rayDir, inout vec3 rayOrigin) {
    
    float magnitude;
    float x=0;
    for (int i = 0; i < MAX_RAY_ITER*100; i++) {
        compilepolys(rayOrigin+x*rayDir,rayDir);
        magnitude = calcpolys(0);
        if (magnitude < EPSILON_RAYMARCHING) {
            break;  
        }
        x+=magnitude*0.0001;
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
    vec3 rayDir =cameraMatrix* normalize(vec3(uv, FOVfactor));


    // Sphere tracing

    vec3 p=rayOrigin;
    float dist=raymarch(rayDir,p);


        
    
    // Checkerboard pattern
    float checker = 0.3 + 0.7 * mod(sum(floor(p * 4.0)), 2.0); // Alternates between 0.5 and 1.0
    vec3 col=vec3(checker);
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
    if(length(uv)<0.01){col*=vec3(0.7,1,0.7);}//dot in middle of screen

    color= vec4(col,1);
}


//cutoff
