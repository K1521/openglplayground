#version 440
/*source=part1*/

const int numparams=5;
const int numpolys=25;
const int degree=numparams-1;
double[numpolys][numparams] polys;
void compilepolys(vec3 p,vec3 d);


/*source=part0*/

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

/*source=part1*/

void compilepolys(vec3 p,vec3 d){
double ox=p.x;
double oy=p.y;
double oz=p.z;
double dx=d.x;
double dy=d.y;
double dz=d.z;
double n0=ox;
double n1=(n0*n0);
double n2=22.5577;
double n3=(n1*n2);
double n4=oy;
double n5=(n0*n4);
double n6=22.5625;
double n7=(n5*n6);
double n8=oz;
double n9=(n0*n8);
double n10=(n9*n6);
double n11=(n4*n4);
double n12=(n8*n8);
double n13=(n1+n11+n12);
double n14=0.5;
double n15=(n13*n14);
double n16=-0.5;
double n17=(n15+n16);
double n18=(n0*n17);
double n19=0.16454482671904333;
double n20=(n18*n19);
double n21=(n11*n2);
double n22=(n4*n8);
double n23=(n22*n6);
double n24=(n4*n17);
double n25=(n24*n19);
double n26=(n12*n6);
double n27=(n8*n17);
double n28=(n27*n19);
double n29=(n17*n17);
double n30=0.0012000000000000001;
double n31=(n29*n30);
double n32=(n3+n7+n10+n20+n7+n21+n23+n25+n10+n23+n26+n28+n20+n25+n28+n31);
double n33=13.0577;
double n34=(n1*n33);
double n35=13.0625;
double n36=(n5*n35);
double n37=(n9*n35);
double n38=(n15+n14);
double n39=(n0*n38);
double n40=(n39*n19);
double n41=(n11*n33);
double n42=(n22*n35);
double n43=(n4*n38);
double n44=(n43*n19);
double n45=(n12*n35);
double n46=(n8*n38);
double n47=(n46*n19);
double n48=0.09526279441628827;
double n49=(n18*n48);
double n50=(n24*n48);
double n51=(n27*n48);
double n52=(n17*n38);
double n53=(n52*n30);
double n54=(n34+n36+n37+n40+n36+n41+n42+n44+n37+n42+n45+n47+n49+n50+n51+n53);
double n55=7.5577;
double n56=(n1*n55);
double n57=7.5625;
double n58=(n5*n57);
double n59=(n9*n57);
double n60=(n39*n48);
double n61=(n11*n55);
double n62=(n22*n57);
double n63=(n43*n48);
double n64=(n12*n57);
double n65=(n46*n48);
double n66=(n38*n38);
double n67=(n66*n30);
double n68=(n56+n58+n59+n60+n58+n61+n62+n63+n59+n62+n64+n65+n60+n63+n65+n67);
double n69=-32;
double n70=(n1*n69);
double n71=-16;
double n72=(n9*n71);
double n73=-0.27712812921102037;
double n74=(n18*n73);
double n75=0.27712812921102037;
double n76=(n39*n75);
double n77=(n12*n71);
double n78=(n27*n73);
double n79=(n46*n75);
double n80=(n29*n55);
double n81=-13.0577;
double n82=(n52*n81);
double n83=(n66*n2);
double n84=(n70+n72+n74+n76+n72+n77+n78+n79+n74+n78+n80+n82+n76+n79+n82+n83);
double n85=(n1*n71);
double n86=16;
double n87=(n5*n86);
double n88=(n22*n71);
double n89=-1;
double n90=(n88*n89);
double n91=(n24*n73);
double n92=(n91*n89);
double n93=(n29*n57);
double n94=-13.0625;
double n95=(n52*n94);
double n96=(n43*n73);
double n97=(n66*n6);
double n98=(n85+n87+n90+n92+n93+n95+n96+n95+n97);
double n99=(n1*n73);
double n100=(n5*n75);
double n101=(n22*n75);
double n102=(n99+n100+n101);
double n103=-7.5625;
double n104=(n18*n103);
double n105=-7.5577;
double n106=(n24*n105);
double n107=(n27*n103);
double n108=-0.09526279441628827;
double n109=(n52*n108);
double n110=(n39*n35);
double n111=(n43*n33);
double n112=(n46*n35);
double n113=(n66*n19);
double n114=(n102+n104+n106+n107+n109+n110+n111+n112+n113);
double n115=(n18*n94);
double n116=(n24*n81);
double n117=(n27*n94);
double n118=(n29*n108);
double n119=(n39*n6);
double n120=(n43*n2);
double n121=(n46*n6);
double n122=(n52*n19);
double n123=(n102+n115+n116+n117+n118+n119+n120+n121+n122);
double n124=(n11*n71);
double n125=(n85+n124+n93+n95+n95+n97);
double n126=(n11*n73);
double n127=(n99+n126);
double n128=(n24*n103);
double n129=(n43*n35);
double n130=(n127+n104+n128+n107+n109+n110+n129+n112+n113);
double n131=(n24*n94);
double n132=(n43*n6);
double n133=(n127+n115+n131+n117+n118+n119+n132+n121+n122);
double n134=(n99+n104+n110);
double n135=(n134+n100+n101+n106+n111+n107+n112+n109+n113);
double n136=(n134+n126+n128+n129+n107+n112+n109+n113);
double n137=(n99+n115+n119);
double n138=(n137+n100+n101+n116+n120+n117+n121+n118+n122);
double n139=(n137+n126+n131+n132+n117+n121+n118+n122);
double n140=32;
double n141=(n5*n140);
double n142=(n9*n86);
double n143=(n18*n75);
double n144=(n39*n73);
double n145=(n22*n86);
double n146=(n24*n75);
double n147=(n141+n142+n143+n144+n145+n146+n93+n95+n96+n95+n97);
double n148=(n87+n142+n143+n144+n124+n93+n95+n95+n97);
double n149=(n9*n75);
double n150=(n100+n149);
double n151=(n18*n105);
double n152=(n39*n33);
double n153=(n150+n151+n152+n126+n128+n129+n107+n112+n109+n113);
double n154=(n18*n81);
double n155=(n39*n2);
double n156=(n150+n154+n155+n126+n131+n132+n117+n121+n118+n122);
double n157=(n100+n126+n149);
double n158=(n157+n151+n128+n107+n109+n152+n129+n112+n113);
double n159=(n157+n154+n131+n117+n118+n155+n132+n121+n122);
double n160=(n11*n69);
double n161=(n96*n89);
double n162=(n160+n88+n91+n161+n88+n77+n78+n79+n91+n78+n80+n82+n161+n79+n82+n83);
double n163=dz;
double n164=0.32908965343808666;
double n165=(n17*n164);
double n166=45.125;
double n167=(n4*n166);
double n168=(n17*n30);
double n169=(n8*n164);
double n170=(n4*n164);
double n171=(n0*n164);
double n172=(n168+n168+n169+n170+n171);
double n173=(n172*n14);
double n174=(n173+n6);
double n175=(n8*n174);
double n176=(n0*n166);
double n177=(n165+n167+n175+n175+n176);
double n178=(n163*n177);
double n179=dy;
double n180=(n8*n166);
double n181=(n165+n180);
double n182=(n173+n2);
double n183=(n4*n182);
double n184=(n181+n183+n183+n176);
double n185=(n179*n184);
double n186=dx;
double n187=(n0*n182);
double n188=(n181+n167+n187+n187);
double n189=(n186*n188);
double n190=(n178+n185+n189);
double n191=(n17*n48);
double n192=(n38*n19);
double n193=(n191+n192);
double n194=26.125;
double n195=(n4*n194);
double n196=(n38*n30);
double n197=(n8*n48);
double n198=(n4*n48);
double n199=(n0*n48);
double n200=(n8*n19);
double n201=(n4*n19);
double n202=(n0*n19);
double n203=(n196+n197+n198+n199+n168+n200+n201+n202);
double n204=(n203*n14);
double n205=(n204+n35);
double n206=(n8*n205);
double n207=(n0*n194);
double n208=(n193+n195+n206+n206+n207);
double n209=(n163*n208);
double n210=(n8*n194);
double n211=(n193+n210);
double n212=(n204+n33);
double n213=(n4*n212);
double n214=(n211+n213+n213+n207);
double n215=(n179*n214);
double n216=(n0*n212);
double n217=(n211+n195+n216+n216);
double n218=(n186*n217);
double n219=(n209+n215+n218);
double n220=0.19052558883257653;
double n221=(n38*n220);
double n222=15.125;
double n223=(n4*n222);
double n224=(n8*n220);
double n225=(n4*n220);
double n226=(n0*n220);
double n227=(n196+n196+n224+n225+n226);
double n228=(n227*n14);
double n229=(n228+n57);
double n230=(n8*n229);
double n231=(n0*n222);
double n232=(n221+n223+n230+n230+n231);
double n233=(n163*n232);
double n234=(n8*n222);
double n235=(n221+n234);
double n236=(n228+n55);
double n237=(n4*n236);
double n238=(n235+n237+n237+n231);
double n239=(n179*n238);
double n240=(n0*n236);
double n241=(n235+n223+n240+n240);
double n242=(n186*n241);
double n243=(n233+n239+n242);
double n244=(n38*n2);
double n245=-26.1154;
double n246=(n17*n245);
double n247=0.5542562584220407;
double n248=(n8*n247);
double n249=(n244+n244+n246+n248);
double n250=(n0*n247);
double n251=(n38*n245);
double n252=(n17*n55);
double n253=-0.5542562584220407;
double n254=(n8*n253);
double n255=(n251+n252+n252+n254);
double n256=(n0*n253);
double n257=(n249+n250+n255+n256);
double n258=(n257*n14);
double n259=(n4*n258);
double n260=(n259+n259);
double n261=(n179*n260);
double n262=(n38*n247);
double n263=(n17*n253);
double n264=(n262+n263);
double n265=(n258+n71);
double n266=(n8*n265);
double n267=(n0*n69);
double n268=(n264+n266+n266+n267);
double n269=(n163*n268);
double n270=(n8*n69);
double n271=(n264+n270);
double n272=(n258+n69);
double n273=(n0*n272);
double n274=(n271+n273+n273);
double n275=(n186*n274);
double n276=(n261+n269+n275);
double n277=(n38*n6);
double n278=(n277+n277);
double n279=(n4*n73);
double n280=-26.125;
double n281=(n17*n280);
double n282=(n278+n279+n281);
double n283=(n38*n280);
double n284=(n17*n57);
double n285=(n283+n284+n284);
double n286=(n4*n75);
double n287=(n285+n286);
double n288=(n282+n287);
double n289=(n288*n14);
double n290=(n8*n289);
double n291=(n4*n86);
double n292=(n290+n290+n291);
double n293=(n163*n292);
double n294=(n38*n73);
double n295=(n17*n75);
double n296=(n294+n295);
double n297=(n4*n289);
double n298=(n8*n86);
double n299=(n0*n86);
double n300=(n296+n297+n297+n298+n299);
double n301=(n179*n300);
double n302=(n289+n71);
double n303=(n0*n302);
double n304=(n291+n303+n303);
double n305=(n186*n304);
double n306=(n293+n301+n305);
double n307=(n38*n35);
double n308=(n17*n103);
double n309=(n307+n308);
double n310=(n192+n192);
double n311=(n8*n35);
double n312=(n310+n311);
double n313=(n4*n33);
double n314=(n0*n35);
double n315=(n17*n108);
double n316=(n38*n108);
double n317=(n8*n103);
double n318=(n316+n317);
double n319=(n4*n105);
double n320=(n0*n103);
double n321=(n318+n319+n320);
double n322=(n312+n313+n314+n315+n321);
double n323=(n322*n14);
double n324=(n8*n323);
double n325=(n309+n324+n324+n286);
double n326=(n163*n325);
double n327=(n38*n33);
double n328=(n17*n105);
double n329=(n327+n328);
double n330=(n4*n323);
double n331=(n8*n75);
double n332=(n0*n75);
double n333=(n329+n330+n330+n331+n332);
double n334=(n179*n333);
double n335=(n309+n286);
double n336=(n323+n73);
double n337=(n0*n336);
double n338=(n335+n337+n337);
double n339=(n186*n338);
double n340=(n326+n334+n339);
double n341=(n17*n94);
double n342=(n277+n341);
double n343=(n17*n19);
double n344=(n8*n6);
double n345=(n343+n344);
double n346=(n4*n2);
double n347=(n0*n6);
double n348=(n8*n94);
double n349=(n192+n315+n315+n348);
double n350=(n4*n81);
double n351=(n0*n94);
double n352=(n345+n346+n347+n349+n350+n351);
double n353=(n352*n14);
double n354=(n8*n353);
double n355=(n342+n354+n354+n286);
double n356=(n163*n355);
double n357=(n17*n81);
double n358=(n244+n357);
double n359=(n4*n353);
double n360=(n358+n359+n359+n331+n332);
double n361=(n179*n360);
double n362=(n342+n286);
double n363=(n353+n73);
double n364=(n0*n363);
double n365=(n362+n364+n364);
double n366=(n186*n365);
double n367=(n356+n361+n366);
double n368=(n278+n281);
double n369=(n368+n285);
double n370=(n369*n14);
double n371=(n8*n370);
double n372=(n371+n371);
double n373=(n163*n372);
double n374=(n370+n71);
double n375=(n4*n374);
double n376=(n375+n375);
double n377=(n179*n376);
double n378=(n0*n374);
double n379=(n378+n378);
double n380=(n186*n379);
double n381=(n373+n377+n380);
double n382=(n4*n35);
double n383=(n312+n382);
double n384=(n4*n103);
double n385=(n318+n384);
double n386=(n385+n320);
double n387=(n383+n314+n315+n386);
double n388=(n387*n14);
double n389=(n8*n388);
double n390=(n309+n389+n389);
double n391=(n163*n390);
double n392=(n388+n73);
double n393=(n4*n392);
double n394=(n309+n393+n393);
double n395=(n179*n394);
double n396=(n0*n392);
double n397=(n309+n396+n396);
double n398=(n186*n397);
double n399=(n391+n395+n398);
double n400=(n4*n6);
double n401=(n345+n400);
double n402=(n4*n94);
double n403=(n349+n402);
double n404=(n401+n347+n403+n351);
double n405=(n404*n14);
double n406=(n8*n405);
double n407=(n342+n406+n406);
double n408=(n163*n407);
double n409=(n405+n73);
double n410=(n4*n409);
double n411=(n342+n410+n410);
double n412=(n179*n411);
double n413=(n0*n409);
double n414=(n342+n413+n413);
double n415=(n186*n414);
double n416=(n408+n412+n415);
double n417=(n310+n315+n311);
double n418=(n417+n313+n314+n321);
double n419=(n418*n14);
double n420=(n8*n419);
double n421=(n335+n420+n420);
double n422=(n163*n421);
double n423=(n4*n419);
double n424=(n329+n331+n332+n423+n423);
double n425=(n179*n424);
double n426=(n419+n73);
double n427=(n0*n426);
double n428=(n286+n307+n308+n427+n427);
double n429=(n186*n428);
double n430=(n422+n425+n429);
double n431=(n417+n382);
double n432=(n431+n314+n386);
double n433=(n432*n14);
double n434=(n8*n433);
double n435=(n309+n434+n434);
double n436=(n163*n435);
double n437=(n433+n73);
double n438=(n4*n437);
double n439=(n309+n438+n438);
double n440=(n179*n439);
double n441=(n0*n437);
double n442=(n309+n441+n441);
double n443=(n186*n442);
double n444=(n436+n440+n443);
double n445=(n362+n354+n354);
double n446=(n163*n445);
double n447=(n286+n277+n341+n364+n364);
double n448=(n186*n447);
double n449=(n446+n361+n448);
double n450=(n0*n73);
double n451=(n282+n450+n287+n332);
double n452=(n451*n14);
double n453=(n8*n452);
double n454=(n291+n453+n453+n299);
double n455=(n163*n454);
double n456=(n4*n452);
double n457=(n0*n140);
double n458=(n296+n298+n456+n456+n457);
double n459=(n179*n458);
double n460=(n0*n452);
double n461=(n4*n140);
double n462=(n296+n460+n460+n298+n461);
double n463=(n186*n462);
double n464=(n455+n459+n463);
double n465=(n368+n450+n285+n332);
double n466=(n465*n14);
double n467=(n8*n466);
double n468=(n467+n467+n299);
double n469=(n163*n468);
double n470=(n466+n71);
double n471=(n4*n470);
double n472=(n471+n471+n299);
double n473=(n179*n472);
double n474=(n0*n466);
double n475=(n296+n474+n474+n298+n291);
double n476=(n186*n475);
double n477=(n469+n473+n476);
double n478=(n0*n33);
double n479=(n0*n105);
double n480=(n385+n479);
double n481=(n431+n478+n480);
double n482=(n481*n14);
double n483=(n8*n482);
double n484=(n309+n483+n483+n332);
double n485=(n163*n484);
double n486=(n482+n73);
double n487=(n4*n486);
double n488=(n309+n487+n487+n332);
double n489=(n179*n488);
double n490=(n0*n482);
double n491=(n329+n490+n490+n331+n286);
double n492=(n186*n491);
double n493=(n485+n489+n492);
double n494=(n0*n2);
double n495=(n0*n81);
double n496=(n401+n494+n403+n495);
double n497=(n496*n14);
double n498=(n8*n497);
double n499=(n342+n498+n498+n332);
double n500=(n163*n499);
double n501=(n497+n73);
double n502=(n4*n501);
double n503=(n342+n502+n502+n332);
double n504=(n179*n503);
double n505=(n0*n497);
double n506=(n358+n505+n505+n331+n286);
double n507=(n186*n506);
double n508=(n500+n504+n507);
double n509=(n383+n478+n315+n480);
double n510=(n509*n14);
double n511=(n8*n510);
double n512=(n309+n511+n511+n332);
double n513=(n163*n512);
double n514=(n510+n73);
double n515=(n4*n514);
double n516=(n309+n515+n515+n332);
double n517=(n179*n516);
double n518=(n0*n510);
double n519=(n329+n518+n518+n331+n286);
double n520=(n186*n519);
double n521=(n513+n517+n520);
double n522=(n4*n247);
double n523=(n4*n253);
double n524=(n249+n522+n255+n523);
double n525=(n524*n14);
double n526=(n0*n525);
double n527=(n526+n526);
double n528=(n186*n527);
double n529=(n525+n71);
double n530=(n8*n529);
double n531=(n4*n69);
double n532=(n264+n530+n530+n531);
double n533=(n163*n532);
double n534=(n525+n69);
double n535=(n4*n534);
double n536=(n271+n535+n535);
double n537=(n179*n536);
double n538=(n528+n533+n537);
double n539=(n186*n166);
double n540=(n179*n166);
double n541=(n174*n163);
double n542=(n0*n186);
double n543=(n542+n542);
double n544=(n4*n179);
double n545=(n544+n544);
double n546=(n8*n163);
double n547=(n546+n546);
double n548=(n543+n545+n547);
double n549=(n548*n14);
double n550=(n549*n164);
double n551=(n186*n164);
double n552=(n179*n164);
double n553=(n549*n30);
double n554=(n163*n164);
double n555=(n551+n552+n553+n553+n554);
double n556=(n555*n14);
double n557=(n8*n556);
double n558=(n539+n540+n541+n550+n541+n557+n557);
double n559=(n163*n558);
double n560=(n182*n179);
double n561=(n163*n166);
double n562=(n4*n556);
double n563=(n539+n560+n560+n550+n561+n562+n562);
double n564=(n179*n563);
double n565=(n182*n186);
double n566=(n0*n556);
double n567=(n565+n565+n540+n561+n550+n566+n566);
double n568=(n186*n567);
double n569=(n559+n564+n568);
double n570=(n569*n14);
double n571=(n186*n194);
double n572=(n179*n194);
double n573=(n205*n163);
double n574=(n549*n19);
double n575=(n549*n48);
double n576=(n186*n19);
double n577=(n179*n19);
double n578=(n163*n19);
double n579=(n186*n48);
double n580=(n179*n48);
double n581=(n163*n48);
double n582=(n576+n577+n553+n578+n579+n580+n553+n581);
double n583=(n582*n14);
double n584=(n8*n583);
double n585=(n571+n572+n573+n574+n575+n573+n584+n584);
double n586=(n163*n585);
double n587=(n212*n179);
double n588=(n163*n194);
double n589=(n4*n583);
double n590=(n571+n587+n587+n574+n575+n588+n589+n589);
double n591=(n179*n590);
double n592=(n212*n186);
double n593=(n0*n583);
double n594=(n592+n592+n572+n588+n574+n575+n593+n593);
double n595=(n186*n594);
double n596=(n586+n591+n595);
double n597=(n596*n14);
double n598=(n186*n222);
double n599=(n179*n222);
double n600=(n229*n163);
double n601=(n549*n220);
double n602=(n186*n220);
double n603=(n179*n220);
double n604=(n163*n220);
double n605=(n602+n603+n553+n553+n604);
double n606=(n605*n14);
double n607=(n8*n606);
double n608=(n598+n599+n600+n601+n600+n607+n607);
double n609=(n163*n608);
double n610=(n236*n179);
double n611=(n163*n222);
double n612=(n4*n606);
double n613=(n598+n610+n610+n601+n611+n612+n612);
double n614=(n179*n613);
double n615=(n236*n186);
double n616=(n0*n606);
double n617=(n615+n615+n599+n611+n601+n616+n616);
double n618=(n186*n617);
double n619=(n609+n614+n618);
double n620=(n619*n14);
double n621=(n186*n69);
double n622=(n265*n163);
double n623=(n549*n253);
double n624=(n549*n247);
double n625=(n186*n253);
double n626=(n163*n253);
double n627=(n549*n55);
double n628=(n549*n245);
double n629=(n186*n247);
double n630=(n163*n247);
double n631=(n549*n2);
double n632=(n625+n626+n627+n627+n628+n629+n630+n628+n631+n631);
double n633=(n632*n14);
double n634=(n8*n633);
double n635=(n621+n622+n622+n623+n624+n634+n634);
double n636=(n163*n635);
double n637=(n272*n186);
double n638=(n163*n69);
double n639=(n0*n633);
double n640=(n637+n637+n638+n623+n624+n639+n639);
double n641=(n186*n640);
double n642=(n258*n179);
double n643=(n4*n633);
double n644=(n642+n643+n643+n642);
double n645=(n179*n644);
double n646=(n636+n641+n645);
double n647=(n646*n14);
double n648=(n186*n86);
double n649=(n289*n179);
double n650=(n163*n86);
double n651=(n549*n75);
double n652=(n549*n73);
double n653=(n179*n75);
double n654=(n549*n57);
double n655=(n549*n280);
double n656=(n179*n73);
double n657=(n549*n6);
double n658=(n653+n654+n654+n655+n656+n655+n657+n657);
double n659=(n658*n14);
double n660=(n4*n659);
double n661=(n648+n649+n649+n650+n651+n652+n660+n660);
double n662=(n179*n661);
double n663=(n302*n186);
double n664=(n179*n86);
double n665=(n0*n659);
double n666=(n663+n663+n664+n665+n665);
double n667=(n186*n666);
double n668=(n289*n163);
double n669=(n8*n659);
double n670=(n664+n668+n669+n669+n668);
double n671=(n163*n670);
double n672=(n662+n667+n671);
double n673=(n672*n14);
double n674=(n323*n163);
double n675=(n549*n103);
double n676=(n549*n35);
double n677=(n186*n103);
double n678=(n179*n105);
double n679=(n549*n108);
double n680=(n163*n103);
double n681=(n186*n35);
double n682=(n179*n33);
double n683=(n163*n35);
double n684=(n677+n678+n679+n680+n681+n682+n679+n574+n574+n683);
double n685=(n684*n14);
double n686=(n8*n685);
double n687=(n653+n674+n675+n676+n674+n686+n686);
double n688=(n163*n687);
double n689=(n186*n75);
double n690=(n323*n179);
double n691=(n163*n75);
double n692=(n549*n105);
double n693=(n549*n33);
double n694=(n4*n685);
double n695=(n689+n690+n690+n691+n692+n693+n694+n694);
double n696=(n179*n695);
double n697=(n336*n186);
double n698=(n0*n685);
double n699=(n697+n697+n653+n675+n676+n698+n698);
double n700=(n186*n699);
double n701=(n688+n696+n700);
double n702=(n701*n14);
double n703=(n353*n163);
double n704=(n549*n94);
double n705=(n186*n94);
double n706=(n179*n81);
double n707=(n163*n94);
double n708=(n186*n6);
double n709=(n179*n2);
double n710=(n163*n6);
double n711=(n705+n706+n679+n679+n574+n707+n708+n709+n574+n710);
double n712=(n711*n14);
double n713=(n8*n712);
double n714=(n653+n703+n704+n657+n703+n713+n713);
double n715=(n163*n714);
double n716=(n353*n179);
double n717=(n689+n716+n716);
double n718=(n549*n81);
double n719=(n4*n712);
double n720=(n717+n691+n718+n631+n719+n719);
double n721=(n179*n720);
double n722=(n363*n186);
double n723=(n0*n712);
double n724=(n722+n722+n653+n704+n657+n723+n723);
double n725=(n186*n724);
double n726=(n715+n721+n725);
double n727=(n726*n14);
double n728=(n374*n179);
double n729=(n654+n654+n655+n655+n657+n657);
double n730=(n729*n14);
double n731=(n4*n730);
double n732=(n728+n728+n731+n731);
double n733=(n179*n732);
double n734=(n374*n186);
double n735=(n0*n730);
double n736=(n734+n734+n735+n735);
double n737=(n186*n736);
double n738=(n370*n163);
double n739=(n8*n730);
double n740=(n738+n739+n739+n738);
double n741=(n163*n740);
double n742=(n733+n737+n741);
double n743=(n742*n14);
double n744=(n388*n163);
double n745=(n179*n103);
double n746=(n179*n35);
double n747=(n677+n745+n679+n680+n681+n746+n679+n574+n574+n683);
double n748=(n747*n14);
double n749=(n8*n748);
double n750=(n744+n675+n676+n744+n749+n749);
double n751=(n163*n750);
double n752=(n392*n179);
double n753=(n4*n748);
double n754=(n752+n752+n675+n676+n753+n753);
double n755=(n179*n754);
double n756=(n392*n186);
double n757=(n0*n748);
double n758=(n756+n756+n675+n676+n757+n757);
double n759=(n186*n758);
double n760=(n751+n755+n759);
double n761=(n760*n14);
double n762=(n405*n163);
double n763=(n179*n94);
double n764=(n179*n6);
double n765=(n705+n763+n679+n679+n574+n707+n708+n764+n574+n710);
double n766=(n765*n14);
double n767=(n8*n766);
double n768=(n762+n704+n657+n762+n767+n767);
double n769=(n163*n768);
double n770=(n409*n179);
double n771=(n4*n766);
double n772=(n770+n770+n704+n657+n771+n771);
double n773=(n179*n772);
double n774=(n409*n186);
double n775=(n0*n766);
double n776=(n774+n774+n704+n657+n775+n775);
double n777=(n186*n776);
double n778=(n769+n773+n777);
double n779=(n778*n14);
double n780=(n419*n163);
double n781=(n653+n780+n675+n676+n780+n686+n686);
double n782=(n163*n781);
double n783=(n419*n179);
double n784=(n689+n783+n783+n692+n693+n691+n694+n694);
double n785=(n179*n784);
double n786=(n426*n186);
double n787=(n786+n786+n653+n675+n676+n698+n698);
double n788=(n186*n787);
double n789=(n782+n785+n788);
double n790=(n789*n14);
double n791=(n433*n163);
double n792=(n791+n675+n676+n791+n749+n749);
double n793=(n163*n792);
double n794=(n437*n179);
double n795=(n794+n794+n675+n676+n753+n753);
double n796=(n179*n795);
double n797=(n437*n186);
double n798=(n797+n797+n675+n676+n757+n757);
double n799=(n186*n798);
double n800=(n793+n796+n799);
double n801=(n800*n14);
double n802=(n452*n186);
double n803=(n179*n140);
double n804=(n186*n73);
double n805=(n689+n653+n654+n654+n655+n804+n656+n655+n657+n657);
double n806=(n805*n14);
double n807=(n0*n806);
double n808=(n802+n802+n803+n650+n651+n652+n807+n807);
double n809=(n186*n808);
double n810=(n452*n163);
double n811=(n8*n806);
double n812=(n648+n664+n810+n811+n811+n810);
double n813=(n163*n812);
double n814=(n186*n140);
double n815=(n452*n179);
double n816=(n4*n806);
double n817=(n814+n815+n815+n651+n652+n816+n816+n650);
double n818=(n179*n817);
double n819=(n809+n813+n818);
double n820=(n819*n14);
double n821=(n470*n179);
double n822=(n689+n654+n654+n655+n804+n655+n657+n657);
double n823=(n822*n14);
double n824=(n4*n823);
double n825=(n648+n821+n821+n824+n824);
double n826=(n179*n825);
double n827=(n466*n186);
double n828=(n0*n823);
double n829=(n827+n827+n664+n650+n651+n652+n828+n828);
double n830=(n186*n829);
double n831=(n466*n163);
double n832=(n8*n823);
double n833=(n648+n831+n832+n832+n831);
double n834=(n163*n833);
double n835=(n826+n830+n834);
double n836=(n835*n14);
double n837=(n482*n163);
double n838=(n186*n105);
double n839=(n186*n33);
double n840=(n838+n745+n679+n680+n839+n746+n679+n574+n574+n683);
double n841=(n840*n14);
double n842=(n8*n841);
double n843=(n689+n837+n675+n676+n837+n842+n842);
double n844=(n163*n843);
double n845=(n486*n179);
double n846=(n4*n841);
double n847=(n689+n845+n845+n675+n676+n846+n846);
double n848=(n179*n847);
double n849=(n482*n186);
double n850=(n0*n841);
double n851=(n849+n849+n653+n691+n692+n693+n850+n850);
double n852=(n186*n851);
double n853=(n844+n848+n852);
double n854=(n853*n14);
double n855=(n497*n163);
double n856=(n186*n81);
double n857=(n186*n2);
double n858=(n856+n763+n679+n679+n574+n707+n857+n764+n574+n710);
double n859=(n858*n14);
double n860=(n8*n859);
double n861=(n689+n855+n704+n657+n855+n860+n860);
double n862=(n163*n861);
double n863=(n501*n179);
double n864=(n4*n859);
double n865=(n689+n863+n863+n704+n657+n864+n864);
double n866=(n179*n865);
double n867=(n497*n186);
double n868=(n0*n859);
double n869=(n867+n867+n653+n691+n718+n631+n868+n868);
double n870=(n186*n869);
double n871=(n862+n866+n870);
double n872=(n871*n14);
double n873=(n510*n163);
double n874=(n689+n873+n675+n676+n873+n842+n842);
double n875=(n163*n874);
double n876=(n514*n179);
double n877=(n689+n876+n876+n675+n676+n846+n846);
double n878=(n179*n877);
double n879=(n510*n186);
double n880=(n879+n879+n653+n691+n692+n693+n850+n850);
double n881=(n186*n880);
double n882=(n875+n878+n881);
double n883=(n882*n14);
double n884=(n179*n69);
double n885=(n529*n163);
double n886=(n179*n253);
double n887=(n179*n247);
double n888=(n886+n626+n627+n627+n628+n887+n630+n628+n631+n631);
double n889=(n888*n14);
double n890=(n8*n889);
double n891=(n884+n885+n885+n623+n624+n890+n890);
double n892=(n163*n891);
double n893=(n534*n179);
double n894=(n4*n889);
double n895=(n893+n893+n638+n623+n624+n894+n894);
double n896=(n179*n895);
double n897=(n525*n186);
double n898=(n0*n889);
double n899=(n897+n898+n898+n897);
double n900=(n186*n899);
double n901=(n892+n896+n900);
double n902=(n901*n14);
double n903=(n163*n14);
double n904=(n556*n903);
double n905=(n186*n14);
double n906=(n905*n164);
double n907=(n179*n14);
double n908=(n907*n164);
double n909=(n0*n905);
double n910=(n909+n909);
double n911=(n4*n907);
double n912=(n911+n911);
double n913=(n8*n903);
double n914=(n913+n913);
double n915=(n910+n912+n914);
double n916=(n915*n14);
double n917=(n916*n30);
double n918=(n917+n917);
double n919=(n903*n164);
double n920=(n906+n908+n918+n919);
double n921=(n920*n14);
double n922=(n163*n921);
double n923=(n186*n905);
double n924=(n923+n923);
double n925=(n179*n907);
double n926=(n925+n925);
double n927=(n163*n903);
double n928=(n927+n927);
double n929=(n924+n926+n928);
double n930=(n929*n14);
double n931=(n930*n164);
double n932=(n930*n30);
double n933=(n932+n932);
double n934=(n933*n14);
double n935=(n8*n934);
double n936=(n904+n904+n922+n922+n931+n935+n935);
double n937=(n163*n936);
double n938=(n556*n907);
double n939=(n179*n921);
double n940=(n4*n934);
double n941=(n938+n938+n939+n939+n931+n940+n940);
double n942=(n179*n941);
double n943=(n556*n905);
double n944=(n186*n921);
double n945=(n0*n934);
double n946=(n943+n943+n944+n944+n931+n945+n945);
double n947=(n186*n946);
double n948=(n937+n942+n947);
double n949=0.3333333333333333;
double n950=(n948*n949);
double n951=(n583*n903);
double n952=(n905*n48);
double n953=(n907*n48);
double n954=(n903*n48);
double n955=(n905*n19);
double n956=(n907*n19);
double n957=(n903*n19);
double n958=(n952+n953+n917+n954+n955+n956+n917+n957);
double n959=(n958*n14);
double n960=(n163*n959);
double n961=(n930*n19);
double n962=(n930*n48);
double n963=(n951+n951+n960+n960+n961+n962+n935+n935);
double n964=(n163*n963);
double n965=(n583*n907);
double n966=(n179*n959);
double n967=(n965+n965+n966+n966+n961+n962+n940+n940);
double n968=(n179*n967);
double n969=(n583*n905);
double n970=(n186*n959);
double n971=(n969+n969+n970+n970+n961+n962+n945+n945);
double n972=(n186*n971);
double n973=(n964+n968+n972);
double n974=(n973*n949);
double n975=(n606*n903);
double n976=(n905*n220);
double n977=(n907*n220);
double n978=(n903*n220);
double n979=(n976+n977+n918+n978);
double n980=(n979*n14);
double n981=(n163*n980);
double n982=(n930*n220);
double n983=(n975+n975+n981+n981+n982+n935+n935);
double n984=(n163*n983);
double n985=(n606*n907);
double n986=(n179*n980);
double n987=(n985+n985+n986+n986+n982+n940+n940);
double n988=(n179*n987);
double n989=(n606*n905);
double n990=(n186*n980);
double n991=(n989+n989+n990+n990+n982+n945+n945);
double n992=(n186*n991);
double n993=(n984+n988+n992);
double n994=(n993*n949);
double n995=(n633*n903);
double n996=(n905*n247);
double n997=(n916*n2);
double n998=(n916*n245);
double n999=(n903*n247);
double n1000=(n997+n997+n998+n999);
double n1001=(n905*n253);
double n1002=(n916*n55);
double n1003=(n903*n253);
double n1004=(n998+n1002+n1002+n1003);
double n1005=(n996+n1000+n1001+n1004);
double n1006=(n1005*n14);
double n1007=(n163*n1006);
double n1008=(n930*n253);
double n1009=(n930*n247);
double n1010=(n930*n55);
double n1011=(n930*n245);
double n1012=(n930*n2);
double n1013=(n1010+n1010+n1011+n1011+n1012+n1012);
double n1014=(n1013*n14);
double n1015=(n8*n1014);
double n1016=(n995+n995+n1007+n1007+n1008+n1009+n1015+n1015);
double n1017=(n163*n1016);
double n1018=(n633*n907);
double n1019=(n179*n1006);
double n1020=(n4*n1014);
double n1021=(n1018+n1018+n1019+n1019+n1020+n1020);
double n1022=(n179*n1021);
double n1023=(n633*n905);
double n1024=(n186*n1006);
double n1025=(n0*n1014);
double n1026=(n1023+n1023+n1024+n1024+n1008+n1009+n1025+n1025);
double n1027=(n186*n1026);
double n1028=(n1017+n1022+n1027);
double n1029=(n1028*n949);
double n1030=(n659*n903);
double n1031=(n916*n280);
double n1032=(n916*n6);
double n1033=(n1032+n1032);
double n1034=(n907*n73);
double n1035=(n916*n57);
double n1036=(n1031+n1035+n1035);
double n1037=(n907*n75);
double n1038=(n1031+n1033+n1034+n1036+n1037);
double n1039=(n1038*n14);
double n1040=(n163*n1039);
double n1041=(n930*n57);
double n1042=(n930*n280);
double n1043=(n930*n6);
double n1044=(n1041+n1041+n1042+n1042+n1043+n1043);
double n1045=(n1044*n14);
double n1046=(n8*n1045);
double n1047=(n1030+n1030+n1040+n1040+n1046+n1046);
double n1048=(n163*n1047);
double n1049=(n659*n907);
double n1050=(n179*n1039);
double n1051=(n930*n75);
double n1052=(n930*n73);
double n1053=(n4*n1045);
double n1054=(n1049+n1049+n1050+n1050+n1051+n1052+n1053+n1053);
double n1055=(n179*n1054);
double n1056=(n659*n905);
double n1057=(n186*n1039);
double n1058=(n0*n1045);
double n1059=(n1056+n1056+n1057+n1057+n1058+n1058);
double n1060=(n186*n1059);
double n1061=(n1048+n1055+n1060);
double n1062=(n1061*n949);
double n1063=(n685*n903);
double n1064=(n1063+n1063);
double n1065=(n916*n108);
double n1066=(n905*n35);
double n1067=(n907*n33);
double n1068=(n916*n19);
double n1069=(n1068+n1068);
double n1070=(n903*n35);
double n1071=(n1069+n1070);
double n1072=(n905*n103);
double n1073=(n907*n105);
double n1074=(n903*n103);
double n1075=(n1065+n1074);
double n1076=(n1072+n1073+n1075);
double n1077=(n1065+n1066+n1067+n1071+n1076);
double n1078=(n1077*n14);
double n1079=(n163*n1078);
double n1080=(n930*n103);
double n1081=(n930*n35);
double n1082=(n930*n108);
double n1083=(n1082+n1082+n961+n961);
double n1084=(n1083*n14);
double n1085=(n8*n1084);
double n1086=(n1064+n1079+n1079+n1080+n1081+n1085+n1085);
double n1087=(n163*n1086);
double n1088=(n685*n907);
double n1089=(n1088+n1088);
double n1090=(n179*n1078);
double n1091=(n930*n105);
double n1092=(n930*n33);
double n1093=(n4*n1084);
double n1094=(n1089+n1090+n1090+n1091+n1092+n1093+n1093);
double n1095=(n179*n1094);
double n1096=(n685*n905);
double n1097=(n1096+n1096);
double n1098=(n186*n1078);
double n1099=(n0*n1084);
double n1100=(n1097+n1098+n1098+n1080+n1081+n1099+n1099);
double n1101=(n186*n1100);
double n1102=(n1087+n1095+n1101);
double n1103=(n1102*n949);
double n1104=(n712*n903);
double n1105=(n905*n6);
double n1106=(n907*n2);
double n1107=(n903*n6);
double n1108=(n1068+n1107);
double n1109=(n905*n94);
double n1110=(n907*n81);
double n1111=(n903*n94);
double n1112=(n1068+n1065+n1065+n1111);
double n1113=(n1105+n1106+n1108+n1109+n1110+n1112);
double n1114=(n1113*n14);
double n1115=(n163*n1114);
double n1116=(n930*n94);
double n1117=(n1104+n1104+n1115+n1115+n1116+n1043+n1085+n1085);
double n1118=(n163*n1117);
double n1119=(n712*n907);
double n1120=(n179*n1114);
double n1121=(n930*n81);
double n1122=(n1119+n1119+n1120+n1120+n1121+n1012+n1093+n1093);
double n1123=(n179*n1122);
double n1124=(n712*n905);
double n1125=(n186*n1114);
double n1126=(n1124+n1124+n1125+n1125+n1116+n1043+n1099+n1099);
double n1127=(n186*n1126);
double n1128=(n1118+n1123+n1127);
double n1129=(n1128*n949);
double n1130=(n730*n903);
double n1131=(n1033+n1031);
double n1132=(n1131+n1036);
double n1133=(n1132*n14);
double n1134=(n163*n1133);
double n1135=(n1130+n1130+n1134+n1134+n1046+n1046);
double n1136=(n163*n1135);
double n1137=(n730*n907);
double n1138=(n179*n1133);
double n1139=(n1137+n1138+n1138+n1137+n1053+n1053);
double n1140=(n179*n1139);
double n1141=(n730*n905);
double n1142=(n186*n1133);
double n1143=(n1141+n1141+n1142+n1142+n1058+n1058);
double n1144=(n186*n1143);
double n1145=(n1136+n1140+n1144);
double n1146=(n1145*n949);
double n1147=(n748*n903);
double n1148=(n1147+n1147);
double n1149=(n907*n35);
double n1150=(n1149+n1071);
double n1151=(n907*n103);
double n1152=(n1151+n1075);
double n1153=(n1072+n1152);
double n1154=(n1065+n1066+n1150+n1153);
double n1155=(n1154*n14);
double n1156=(n163*n1155);
double n1157=(n1148+n1156+n1156+n1080+n1081+n1085+n1085);
double n1158=(n163*n1157);
double n1159=(n748*n907);
double n1160=(n1159+n1159);
double n1161=(n179*n1155);
double n1162=(n1160+n1161+n1161+n1080+n1081+n1093+n1093);
double n1163=(n179*n1162);
double n1164=(n748*n905);
double n1165=(n1164+n1164);
double n1166=(n186*n1155);
double n1167=(n1165+n1166+n1166+n1080+n1081+n1099+n1099);
double n1168=(n186*n1167);
double n1169=(n1158+n1163+n1168);
double n1170=(n1169*n949);
double n1171=(n766*n903);
double n1172=(n907*n6);
double n1173=(n1172+n1108);
double n1174=(n907*n94);
double n1175=(n1174+n1112);
double n1176=(n1105+n1173+n1109+n1175);
double n1177=(n1176*n14);
double n1178=(n163*n1177);
double n1179=(n1171+n1171+n1178+n1178+n1116+n1043+n1085+n1085);
double n1180=(n163*n1179);
double n1181=(n766*n907);
double n1182=(n179*n1177);
double n1183=(n1181+n1181+n1182+n1182+n1116+n1043+n1093+n1093);
double n1184=(n179*n1183);
double n1185=(n766*n905);
double n1186=(n186*n1177);
double n1187=(n1185+n1185+n1186+n1186+n1116+n1043+n1099+n1099);
double n1188=(n186*n1187);
double n1189=(n1180+n1184+n1188);
double n1190=(n1189*n949);
double n1191=(n1069+n1065+n1070);
double n1192=(n1066+n1067+n1191+n1076);
double n1193=(n1192*n14);
double n1194=(n163*n1193);
double n1195=(n1064+n1194+n1194+n1080+n1081+n1085+n1085);
double n1196=(n163*n1195);
double n1197=(n179*n1193);
double n1198=(n1089+n1197+n1197+n1091+n1092+n1093+n1093);
double n1199=(n179*n1198);
double n1200=(n186*n1193);
double n1201=(n1097+n1200+n1200+n1080+n1081+n1099+n1099);
double n1202=(n186*n1201);
double n1203=(n1196+n1199+n1202);
double n1204=(n1203*n949);
double n1205=(n1149+n1191);
double n1206=(n1066+n1205+n1153);
double n1207=(n1206*n14);
double n1208=(n163*n1207);
double n1209=(n1148+n1208+n1208+n1080+n1081+n1085+n1085);
double n1210=(n163*n1209);
double n1211=(n179*n1207);
double n1212=(n1160+n1211+n1211+n1080+n1081+n1093+n1093);
double n1213=(n179*n1212);
double n1214=(n186*n1207);
double n1215=(n1165+n1214+n1214+n1080+n1081+n1099+n1099);
double n1216=(n186*n1215);
double n1217=(n1210+n1213+n1216);
double n1218=(n1217*n949);
double n1219=(n806*n903);
double n1220=(n905*n73);
double n1221=(n905*n75);
double n1222=(n1031+n1033+n1034+n1220+n1036+n1037+n1221);
double n1223=(n1222*n14);
double n1224=(n163*n1223);
double n1225=(n1219+n1219+n1224+n1224+n1046+n1046);
double n1226=(n163*n1225);
double n1227=(n806*n907);
double n1228=(n179*n1223);
double n1229=(n1227+n1227+n1228+n1228+n1051+n1052+n1053+n1053);
double n1230=(n179*n1229);
double n1231=(n806*n905);
double n1232=(n186*n1223);
double n1233=(n1231+n1231+n1232+n1232+n1051+n1052+n1058+n1058);
double n1234=(n186*n1233);
double n1235=(n1226+n1230+n1234);
double n1236=(n1235*n949);
double n1237=(n823*n903);
double n1238=(n1220+n1131+n1221+n1036);
double n1239=(n1238*n14);
double n1240=(n163*n1239);
double n1241=(n1237+n1237+n1240+n1240+n1046+n1046);
double n1242=(n163*n1241);
double n1243=(n823*n907);
double n1244=(n179*n1239);
double n1245=(n1243+n1244+n1244+n1243+n1053+n1053);
double n1246=(n179*n1245);
double n1247=(n823*n905);
double n1248=(n186*n1239);
double n1249=(n1247+n1247+n1248+n1248+n1051+n1052+n1058+n1058);
double n1250=(n186*n1249);
double n1251=(n1242+n1246+n1250);
double n1252=(n1251*n949);
double n1253=(n841*n903);
double n1254=(n1253+n1253);
double n1255=(n905*n33);
double n1256=(n905*n105);
double n1257=(n1256+n1152);
double n1258=(n1255+n1205+n1257);
double n1259=(n1258*n14);
double n1260=(n163*n1259);
double n1261=(n1254+n1260+n1260+n1080+n1081+n1085+n1085);
double n1262=(n163*n1261);
double n1263=(n841*n907);
double n1264=(n1263+n1263);
double n1265=(n179*n1259);
double n1266=(n1264+n1265+n1265+n1080+n1081+n1093+n1093);
double n1267=(n179*n1266);
double n1268=(n841*n905);
double n1269=(n1268+n1268);
double n1270=(n186*n1259);
double n1271=(n1269+n1270+n1270+n1091+n1092+n1099+n1099);
double n1272=(n186*n1271);
double n1273=(n1262+n1267+n1272);
double n1274=(n1273*n949);
double n1275=(n859*n903);
double n1276=(n905*n2);
double n1277=(n905*n81);
double n1278=(n1276+n1173+n1277+n1175);
double n1279=(n1278*n14);
double n1280=(n163*n1279);
double n1281=(n1275+n1275+n1280+n1280+n1116+n1043+n1085+n1085);
double n1282=(n163*n1281);
double n1283=(n859*n907);
double n1284=(n179*n1279);
double n1285=(n1283+n1283+n1284+n1284+n1116+n1043+n1093+n1093);
double n1286=(n179*n1285);
double n1287=(n859*n905);
double n1288=(n186*n1279);
double n1289=(n1287+n1287+n1288+n1288+n1121+n1012+n1099+n1099);
double n1290=(n186*n1289);
double n1291=(n1282+n1286+n1290);
double n1292=(n1291*n949);
double n1293=(n1065+n1255+n1150+n1257);
double n1294=(n1293*n14);
double n1295=(n163*n1294);
double n1296=(n1254+n1295+n1295+n1080+n1081+n1085+n1085);
double n1297=(n163*n1296);
double n1298=(n179*n1294);
double n1299=(n1264+n1298+n1298+n1080+n1081+n1093+n1093);
double n1300=(n179*n1299);
double n1301=(n186*n1294);
double n1302=(n1269+n1301+n1301+n1091+n1092+n1099+n1099);
double n1303=(n186*n1302);
double n1304=(n1297+n1300+n1303);
double n1305=(n1304*n949);
double n1306=(n889*n903);
double n1307=(n907*n247);
double n1308=(n907*n253);
double n1309=(n1307+n1000+n1308+n1004);
double n1310=(n1309*n14);
double n1311=(n163*n1310);
double n1312=(n1306+n1306+n1311+n1311+n1008+n1009+n1015+n1015);
double n1313=(n163*n1312);
double n1314=(n889*n907);
double n1315=(n179*n1310);
double n1316=(n1314+n1314+n1315+n1315+n1008+n1009+n1020+n1020);
double n1317=(n179*n1316);
double n1318=(n889*n905);
double n1319=(n186*n1310);
double n1320=(n1318+n1318+n1319+n1319+n1025+n1025);
double n1321=(n186*n1320);
double n1322=(n1313+n1317+n1321);
double n1323=(n1322*n949);
double n1324=(n163*n949);
double n1325=(n934*n1324);
double n1326=(n186*n949);
double n1327=(n186*n1326);
double n1328=(n1327+n1327);
double n1329=(n179*n949);
double n1330=(n179*n1329);
double n1331=(n1330+n1330);
double n1332=(n163*n1324);
double n1333=(n1332+n1332);
double n1334=(n1328+n1331+n1333);
double n1335=(n1334*n14);
double n1336=(n1335*n30);
double n1337=(n1336+n1336);
double n1338=(n1337*n14);
double n1339=(n903*n1338);
double n1340=(n905*n1326);
double n1341=(n907*n1329);
double n1342=(n903*n1324);
double n1343=(n1340+n1340+n1341+n1341+n1342+n1342);
double n1344=(n1343*n14);
double n1345=(n1344*n30);
double n1346=(n1345+n1345);
double n1347=(n1346*n14);
double n1348=(n163*n1347);
double n1349=(n1325+n1325+n1339+n1339+n1348+n1348);
double n1350=(n163*n1349);
double n1351=(n934*n1329);
double n1352=(n907*n1338);
double n1353=(n179*n1347);
double n1354=(n1351+n1351+n1352+n1352+n1353+n1353);
double n1355=(n179*n1354);
double n1356=(n934*n1326);
double n1357=(n905*n1338);
double n1358=(n186*n1347);
double n1359=(n1356+n1356+n1357+n1357+n1358+n1358);
double n1360=(n186*n1359);
double n1361=(n1350+n1355+n1360);
double n1362=0.25;
double n1363=(n1361*n1362);
double n1364=(n1014*n1329);
double n1365=(n1364+n1364);
double n1366=(n1335*n55);
double n1367=(n1335*n245);
double n1368=(n1335*n2);
double n1369=(n1366+n1366+n1367+n1367+n1368+n1368);
double n1370=(n1369*n14);
double n1371=(n907*n1370);
double n1372=(n1344*n2);
double n1373=(n1344*n245);
double n1374=(n1344*n55);
double n1375=(n1372+n1372+n1373+n1373+n1374+n1374);
double n1376=(n1375*n14);
double n1377=(n179*n1376);
double n1378=(n1365+n1371+n1371+n1377+n1377);
double n1379=(n179*n1378);
double n1380=(n1014*n1324);
double n1381=(n1380+n1380);
double n1382=(n903*n1370);
double n1383=(n163*n1376);
double n1384=(n1381+n1382+n1382+n1383+n1383);
double n1385=(n163*n1384);
double n1386=(n1014*n1326);
double n1387=(n1386+n1386);
double n1388=(n905*n1370);
double n1389=(n186*n1376);
double n1390=(n1387+n1388+n1388+n1389+n1389);
double n1391=(n186*n1390);
double n1392=(n1379+n1385+n1391);
double n1393=(n1392*n1362);
double n1394=(n1045*n1324);
double n1395=(n1335*n280);
double n1396=(n1335*n6);
double n1397=(n1335*n57);
double n1398=(n1395+n1396+n1396+n1397+n1397+n1395);
double n1399=(n1398*n14);
double n1400=(n903*n1399);
double n1401=(n1344*n6);
double n1402=(n1344*n280);
double n1403=(n1344*n57);
double n1404=(n1401+n1401+n1402+n1402+n1403+n1403);
double n1405=(n1404*n14);
double n1406=(n163*n1405);
double n1407=(n1394+n1394+n1400+n1400+n1406+n1406);
double n1408=(n163*n1407);
double n1409=(n1045*n1329);
double n1410=(n907*n1399);
double n1411=(n179*n1405);
double n1412=(n1409+n1409+n1410+n1410+n1411+n1411);
double n1413=(n179*n1412);
double n1414=(n1045*n1326);
double n1415=(n905*n1399);
double n1416=(n186*n1405);
double n1417=(n1414+n1414+n1415+n1415+n1416+n1416);
double n1418=(n186*n1417);
double n1419=(n1408+n1413+n1418);
double n1420=(n1419*n1362);
double n1421=(n1084*n1324);
double n1422=(n1335*n108);
double n1423=(n1335*n19);
double n1424=(n1422+n1423+n1423+n1422);
double n1425=(n1424*n14);
double n1426=(n903*n1425);
double n1427=(n1344*n108);
double n1428=(n1344*n19);
double n1429=(n1427+n1428+n1428+n1427);
double n1430=(n1429*n14);
double n1431=(n163*n1430);
double n1432=(n1421+n1421+n1426+n1426+n1431+n1431);
double n1433=(n163*n1432);
double n1434=(n1084*n1329);
double n1435=(n907*n1425);
double n1436=(n179*n1430);
double n1437=(n1434+n1434+n1435+n1435+n1436+n1436);
double n1438=(n179*n1437);
double n1439=(n1084*n1326);
double n1440=(n905*n1425);
double n1441=(n186*n1430);
double n1442=(n1439+n1439+n1440+n1440+n1441+n1441);
double n1443=(n186*n1442);
double n1444=(n1433+n1438+n1443);
double n1445=(n1444*n1362);
polys[0][0]=n32;
polys[1][0]=n54;
polys[2][0]=n54;
polys[3][0]=n68;
polys[4][0]=n84;
polys[5][0]=n98;
polys[6][0]=n114;
polys[7][0]=n123;
polys[8][0]=n98;
polys[9][0]=n125;
polys[10][0]=n130;
polys[11][0]=n133;
polys[12][0]=n135;
polys[13][0]=n136;
polys[14][0]=n138;
polys[15][0]=n139;
polys[16][0]=n147;
polys[17][0]=n148;
polys[18][0]=n153;
polys[19][0]=n156;
polys[20][0]=n147;
polys[21][0]=n148;
polys[22][0]=n158;
polys[23][0]=n159;
polys[24][0]=n162;
polys[0][1]=n190;
polys[1][1]=n219;
polys[2][1]=n219;
polys[3][1]=n243;
polys[4][1]=n276;
polys[5][1]=n306;
polys[6][1]=n340;
polys[7][1]=n367;
polys[8][1]=n306;
polys[9][1]=n381;
polys[10][1]=n399;
polys[11][1]=n416;
polys[12][1]=n430;
polys[13][1]=n444;
polys[14][1]=n449;
polys[15][1]=n416;
polys[16][1]=n464;
polys[17][1]=n477;
polys[18][1]=n493;
polys[19][1]=n508;
polys[20][1]=n464;
polys[21][1]=n477;
polys[22][1]=n521;
polys[23][1]=n508;
polys[24][1]=n538;
polys[0][2]=n570;
polys[1][2]=n597;
polys[2][2]=n597;
polys[3][2]=n620;
polys[4][2]=n647;
polys[5][2]=n673;
polys[6][2]=n702;
polys[7][2]=n727;
polys[8][2]=n673;
polys[9][2]=n743;
polys[10][2]=n761;
polys[11][2]=n779;
polys[12][2]=n790;
polys[13][2]=n801;
polys[14][2]=n727;
polys[15][2]=n779;
polys[16][2]=n820;
polys[17][2]=n836;
polys[18][2]=n854;
polys[19][2]=n872;
polys[20][2]=n820;
polys[21][2]=n836;
polys[22][2]=n883;
polys[23][2]=n872;
polys[24][2]=n902;
polys[0][3]=n950;
polys[1][3]=n974;
polys[2][3]=n974;
polys[3][3]=n994;
polys[4][3]=n1029;
polys[5][3]=n1062;
polys[6][3]=n1103;
polys[7][3]=n1129;
polys[8][3]=n1062;
polys[9][3]=n1146;
polys[10][3]=n1170;
polys[11][3]=n1190;
polys[12][3]=n1204;
polys[13][3]=n1218;
polys[14][3]=n1129;
polys[15][3]=n1190;
polys[16][3]=n1236;
polys[17][3]=n1252;
polys[18][3]=n1274;
polys[19][3]=n1292;
polys[20][3]=n1236;
polys[21][3]=n1252;
polys[22][3]=n1305;
polys[23][3]=n1292;
polys[24][3]=n1323;
polys[0][4]=n1363;
polys[1][4]=n1363;
polys[2][4]=n1363;
polys[3][4]=n1363;
polys[4][4]=n1393;
polys[5][4]=n1420;
polys[6][4]=n1445;
polys[7][4]=n1445;
polys[8][4]=n1420;
polys[9][4]=n1420;
polys[10][4]=n1445;
polys[11][4]=n1445;
polys[12][4]=n1445;
polys[13][4]=n1445;
polys[14][4]=n1445;
polys[15][4]=n1445;
polys[16][4]=n1420;
polys[17][4]=n1420;
polys[18][4]=n1445;
polys[19][4]=n1445;
polys[20][4]=n1420;
polys[21][4]=n1420;
polys[22][4]=n1445;
polys[23][4]=n1445;
polys[24][4]=n1393;
}


