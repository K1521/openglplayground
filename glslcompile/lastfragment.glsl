#version 440
/*source=part1*/

const int numparams=5;
const int numpolys=1;
const int degree=numparams-1;
#define dnum double
#define dpoly dnum[numparams]
#define dpolys dnum[numpolys][numparams]


void compilepolys(vec3 p,vec3 d,out dpolys polys);


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
const float pi=3.14159265359;


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
        roots[1] = 2.0 * s * cos((phi + 2.0 * pi) / 3.0) - b / (3.0 * a);
        roots[2] = 2.0 * s * cos((phi + 4.0 * pi) / 3.0) - b / (3.0 * a);
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

float findcommonroot(dpolys polys){
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
    dpolys polys;
    compilepolys(rayOrigin,rayDir,polys);
    float x=findcommonroot(polys);
    

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

/*source=part1*/

void compilepolys(vec3 p,vec3 d,out dpolys polys){
dnum ox=p.x;
dnum oy=p.y;
dnum oz=p.z;
dnum dx=d.x;
dnum dy=d.y;
dnum dz=d.z;
dnum n0=ox;
dnum n1=(n0*n0);
dnum n2=-1;
dnum n3=(n1*n2);
dnum n4=oy;
dnum n5=(n0*n4*n2);
dnum n6=oz;
dnum n7=(n0*n6*n2);
dnum n8=(n4*n4);
dnum n9=(n6*n6);
dnum n10=(n1+n8+n9);
dnum n11=0.5;
dnum n12=(n10*n11);
dnum n13=-0.5;
dnum n14=(n12+n13);
dnum n15=-1.7320508075688772;
dnum n16=(n0*n14*n15);
dnum n17=(n12+n11);
dnum n18=1.7320508075688772;
dnum n19=(n0*n17*n18);
dnum n20=(n8*n2);
dnum n21=(n4*n6*n2);
dnum n22=(n4*n14*n15);
dnum n23=(n4*n17*n18);
dnum n24=(n9*n2);
dnum n25=(n6*n14*n15);
dnum n26=(n6*n17*n18);
dnum n27=-2.9999999999999996;
dnum n28=(n14*n14*n27);
dnum n29=2.9999999999999996;
dnum n30=(n14*n17*n29);
dnum n31=(n17*n17*n27);
dnum n32=(n3+n5+n7+n16+n19+n5+n20+n21+n22+n23+n7+n21+n24+n25+n26+n16+n22+n25+n28+n30+n19+n23+n26+n30+n31);
dnum n33=dz;
dnum n34=3.4641016151377544;
dnum n35=(n17*n34);
dnum n36=-3.4641016151377544;
dnum n37=(n14*n36);
dnum n38=(n35+n37);
dnum n39=-2;
dnum n40=(n4*n39);
dnum n41=(n17*n27);
dnum n42=5.999999999999999;
dnum n43=(n14*n42);
dnum n44=(n6*n34);
dnum n45=(n4*n34);
dnum n46=(n0*n34);
dnum n47=(n17*n42);
dnum n48=(n14*n27);
dnum n49=(n6*n36);
dnum n50=(n4*n36);
dnum n51=(n0*n36);
dnum n52=(n41+n41+n43+n44+n45+n46+n47+n48+n48+n49+n50+n51);
dnum n53=(n52*n11);
dnum n54=(n53+n2);
dnum n55=(n6*n54);
dnum n56=(n0*n39);
dnum n57=(n38+n40+n55+n55+n56);
dnum n58=(n33*n57);
dnum n59=dy;
dnum n60=(n6*n39);
dnum n61=(n38+n60);
dnum n62=(n4*n54);
dnum n63=(n61+n62+n62+n56);
dnum n64=(n59*n63);
dnum n65=dx;
dnum n66=(n0*n54);
dnum n67=(n61+n40+n66+n66);
dnum n68=(n65*n67);
dnum n69=(n58+n64+n68);
dnum n70=2;
dnum n71=(n65*n70);
dnum n72=(n71*n2);
dnum n73=(n59*n70);
dnum n74=(n73*n2);
dnum n75=(n54*n33);
dnum n76=(n0*n65);
dnum n77=(n4*n59);
dnum n78=(n6*n33);
dnum n79=(n76+n76+n77+n77+n78+n78);
dnum n80=(n79*n11);
dnum n81=(n80*n70);
dnum n82=(n81*n15);
dnum n83=(n81*n18);
dnum n84=(n71*n15);
dnum n85=(n73*n15);
dnum n86=(n80*n27);
dnum n87=(n81*n29);
dnum n88=(n33*n70);
dnum n89=(n88*n15);
dnum n90=(n71*n18);
dnum n91=(n73*n18);
dnum n92=(n88*n18);
dnum n93=(n84+n85+n86+n86+n87+n89+n90+n91+n87+n86+n86+n92);
dnum n94=(n93*n11);
dnum n95=(n6*n94);
dnum n96=(n72+n74+n75+n82+n83+n75+n95+n95);
dnum n97=(n33*n96);
dnum n98=(n54*n59);
dnum n99=(n88*n2);
dnum n100=(n4*n94);
dnum n101=(n72+n98+n98+n82+n83+n99+n100+n100);
dnum n102=(n59*n101);
dnum n103=(n54*n65);
dnum n104=(n0*n94);
dnum n105=(n103+n103+n74+n99+n82+n83+n104+n104);
dnum n106=(n65*n105);
dnum n107=(n97+n102+n106);
dnum n108=(n107*n11);
dnum n109=(n33*n11);
dnum n110=(n94*n109);
dnum n111=(n65*n11);
dnum n112=(n111*n34);
dnum n113=(n59*n11);
dnum n114=(n113*n34);
dnum n115=(n0*n111);
dnum n116=(n4*n113);
dnum n117=(n6*n109);
dnum n118=(n115+n115+n116+n116+n117+n117);
dnum n119=(n118*n11);
dnum n120=(n119*n27);
dnum n121=(n119*n42);
dnum n122=(n109*n34);
dnum n123=(n111*n36);
dnum n124=(n113*n36);
dnum n125=(n109*n36);
dnum n126=(n112+n114+n120+n120+n121+n122+n123+n124+n120+n120+n121+n125);
dnum n127=(n126*n11);
dnum n128=(n33*n127);
dnum n129=(n65*n111);
dnum n130=(n59*n113);
dnum n131=(n33*n109);
dnum n132=(n129+n129+n130+n130+n131+n131);
dnum n133=(n132*n11);
dnum n134=(n133*n70);
dnum n135=(n134*n15);
dnum n136=(n134*n18);
dnum n137=(n133*n27);
dnum n138=(n134*n29);
dnum n139=(n137+n137+n138+n137+n137+n138);
dnum n140=(n139*n11);
dnum n141=(n6*n140);
dnum n142=(n110+n110+n128+n128+n135+n136+n141+n141);
dnum n143=(n33*n142);
dnum n144=(n94*n113);
dnum n145=(n59*n127);
dnum n146=(n4*n140);
dnum n147=(n144+n144+n145+n145+n135+n136+n146+n146);
dnum n148=(n59*n147);
dnum n149=(n94*n111);
dnum n150=(n65*n127);
dnum n151=(n0*n140);
dnum n152=(n149+n149+n150+n150+n135+n136+n151+n151);
dnum n153=(n65*n152);
dnum n154=(n143+n148+n153);
dnum n155=0.3333333333333333;
dnum n156=(n154*n155);
dnum n157=(n33*n155);
dnum n158=(n140*n157);
dnum n159=(n65*n155);
dnum n160=(n65*n159);
dnum n161=(n59*n155);
dnum n162=(n59*n161);
dnum n163=(n33*n157);
dnum n164=(n160+n160+n162+n162+n163+n163);
dnum n165=(n164*n11);
dnum n166=(n165*n27);
dnum n167=(n165*n42);
dnum n168=(n166+n166+n167+n166+n166+n167);
dnum n169=(n168*n11);
dnum n170=(n109*n169);
dnum n171=(n111*n159);
dnum n172=(n113*n161);
dnum n173=(n109*n157);
dnum n174=(n171+n171+n172+n172+n173+n173);
dnum n175=(n174*n11);
dnum n176=(n175*n27);
dnum n177=(n175*n42);
dnum n178=(n176+n176+n177+n176+n176+n177);
dnum n179=(n178*n11);
dnum n180=(n33*n179);
dnum n181=(n158+n158+n170+n170+n180+n180);
dnum n182=(n33*n181);
dnum n183=(n140*n161);
dnum n184=(n113*n169);
dnum n185=(n59*n179);
dnum n186=(n183+n183+n184+n184+n185+n185);
dnum n187=(n59*n186);
dnum n188=(n140*n159);
dnum n189=(n111*n169);
dnum n190=(n65*n179);
dnum n191=(n188+n188+n189+n189+n190+n190);
dnum n192=(n65*n191);
dnum n193=(n182+n187+n192);
dnum n194=0.25;
dnum n195=(n193*n194);
polys[0][0]=n32;
polys[0][1]=n69;
polys[0][2]=n108;
polys[0][3]=n156;
polys[0][4]=n195;
}


