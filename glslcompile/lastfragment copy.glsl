#version 440
/*source=part1*/

const int numparams=5;
const int numpolys=1;
double[numpolys][numparams] polys;
double calcpolys(double x);
float rootpolysquartic();
void compilepolys(vec3 p,vec3 d);


/*source=part0*/

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

const float nan=sqrt(-1);
const float inf=pow(9,999);


vec3 normaltocol(vec3 normal){
    //return vec3(normal.x/2+0.5,normal.y/2+0.5,0.5-normal.z/2);
    return normal*vec2(1,-1).xxy/2+0.5;
    //return normal;
    //return normalize(pow((normal*vec2(1,-1).xxy+1)/2,vec3(1)));
}

vec3 getNormal(dvec3 p){
    return normalize( -cross(dFdx(p), dFdy(p)) );
}

vec3 getlight(dvec3 p,vec3 rayDir,vec3 color){

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


float newtonroot(double x,double a, double b,double c, double d, double e){
    //return -b/(4*a);
    //return -(4*d)/e;
    //double x=0;
    double shift=x;
    double bestval=x*(x*(x*(x*(a)+b)+c)+d)+e;
    for(int i=0;i<10;i++){
        double f=x*(x*(x*(x*(a)+b)+c)+d)+e;//e+(d+(c)*x)*x;
        double df=x*(x*(x*(4*a)+3*b)+2*c)+d;
        if(abs(f)<bestval){
            bestval=abs(f);
            shift=x;
        }
        x-=f/df;
    }
    return float(shift);

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

// Solve quartic equation ax^4 + bx^3 + cx^2 + dx + e = 0
// Returns vec4 with real roots, unused roots set to INFINITY
/*vec4 solveQuartic(float a, float b, float c, float d, float e) {
    if(abs(a)<1E-6)return vec4(solveCubic(b,c,d,e),nan);

    float xoffset=0;
    // Compute intermediate values
    float p = (8.0 * a * c - 3.0 * b * b) / (8.0 * a * a);
    float q = (b * b * b - 4.0 * a * b * c + 8.0 * a * a * d) / (8.0 * a * a * a);

    // Compute Delta terms
    float delta0 = c * c - 3.0 * b * d + 12.0 * a * e;
    float delta1 = 2.0 * c * c * c - 9.0 * b * c * d + 27.0 * b * b * e + 27.0 * a * d * d - 72.0 * a * c * e;

    // Compute discriminant and Q
    float disc = delta1 * delta1 - 4.0 * delta0 * delta0 * delta0;
    
    // Q calculation
    float Q = pow(
        (delta1 + sqrt(abs(disc))) / 2.0, 
        1.0 / 3.0
    );

    // S calculation
    float S = 0.5 * sqrt(
        -2.0 / 3.0 * p + 
        (1.0 / (3.0 * a)) * (Q + (delta0 / Q))
    );

    // Root calculation
    vec4 roots;
    float offset = -b / (4.0 * a);

    roots.x = offset - S + 0.5 * sqrt(-4.0 * S * S - 2.0 * p + (q / S));
    roots.y = offset - S - 0.5 * sqrt(-4.0 * S * S - 2.0 * p + (q / S));
    roots.z = offset + S + 0.5 * sqrt(-4.0 * S * S - 2.0 * p - (q / S));
    roots.w = offset + S - 0.5 * sqrt(-4.0 * S * S - 2.0 * p - (q / S));

    return roots+xoffset;
}*/

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


float stableness_score(float a, float b) {
    float t = abs(a / b);
    return t + 1.0 / t;
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

float rootpolysquartic();
float raymarch2(vec3 rayDir, inout vec3 rayOrigin);
float raymarch(vec3 rayDir, inout vec3 rayOrigin) {
    
    float x=0;
    

    compilepolys(rayOrigin,rayDir);
    x=rootpolysquartic();
    x=newtonroot(x,polys[0][4],polys[0][3],polys[0][2],polys[0][1],polys[0][0]);
    /*if(isinf(x)){
        raymarch2(rayDir,rayOrigin);
    }*/
    rayOrigin += rayDir * x;
    return 0;
}
float raymarch2(vec3 rayDir, inout vec3 rayOrigin) {
    compilepolys(rayOrigin,rayDir);
    float magnitude=1000000;
    float x=0;
    for (int i = 0; i < MAX_RAY_ITER; i++) {
        
        magnitude = float((x));
        if (magnitude < EPSILON_RAYMARCHING) {
            break;  
        }
        x+=(magnitude)*0.01;
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
    dvec3 rayDir =cameraMatrix* normalize(vec3(uv, FOVfactor));//cam to view


    // Sphere tracing

    dvec3 p=rayOrigin;
    float dist=raymarch(rayDir,p);


        
    
    // Checkerboard pattern
    float checker = 0.3 + 0.7 * mod(sum(floor(p * 4.0)), 2.0); // Alternates between 0.5 and 1.0
    
    vec3 col=vec3(checker);
    //col=vec3(1);
    col*=normaltocol(transpose(cameraMatrix)*getNormal(p));
    //col=vec3(abs());
    col=getlight(p,rayDir,col);
    //col=pow(col,vec3(0.4545));//gamma correction
    //col=pow(col,vec3(2));

    if ((dist < EPSILON_RAYMARCHING)) {

    } else {
        col*=vec3(1,0.5,0.5);
        
        //color = vec4(0.0, 0.0, 0.0, 1.0); // Background
    }
    
    if(any(isnan(col))){
        col=vec3(0);//vec3(1,1,0);
    }
    if(any(isinf(vec3(p))) || abs(p.x)>10E10||abs(p.y)>10E10||abs(p.z)>10E10){
        col=vec3(0);//vec3(0,0,0.5);
    }
    //if(length(uv)<0.01){col*=vec3(0.7,1,0.7);}//dot in middle of screen
    if(length(uv)<0.01 && length(uv)>0.005){col=vec3(0.5,1,0.5);}

    color= vec4(col,1);
}


//cutoff

/*source=part1*/


//float A;

double calcpolys(double x){
    //x-=A;
    double[numparams] pows;
    pows[0]=1;
    for(int j=1;j<numparams;j++){
        pows[j]=pows[j-1]*x;
    }

    double s=0;
    for(int i=0;i<numpolys;i++){
        double[numparams] poly=polys[i];
        double d=0;
        for (int j = 0; j < numparams; j++) {
            d += poly[j] * pows[j];
        }
        s+=abs(d);
    }
    return s;
}


float rootpolysquartic(){
    float minx=inf;
    vec4 roots=solveQuartic(float(polys[0][4]),float(polys[0][3]),float(polys[0][2]),float(polys[0][1]),float(polys[0][0]));
    for(int i=0;i<5;i++){
    float root=roots[i];
        if((!isnan(root)) && root>0){
            minx=min(minx,root);
        }
    }

    return minx;
}

//void compilepolys(vec3 p,vec3 d,float a){
void compilepolys(vec3 p,vec3 d){
double a=0;
double ox=p.x;
double oy=p.y;
double oz=p.z;
double dx=d.x;
double dy=d.y;
double dz=d.z;
double n0=ox;
double n1=(n0*n0);
double n2=-16;
double n3=(n1*n2);
double n4=oy;
double n5=(n4*n4);
double n6=(n5*n2);
double n7=oz;
double n8=(n7*n7);
double n9=(n1+n5+n8);
double n10=0.5;
double n11=(n9*n10);
double n12=-0.5;
double n13=(n11+n12);
double n14=7.5625;
double n15=(n13*n13*n14);
double n16=(n11+n10);
double n17=-13.0625;
double n18=(n13*n16*n17);
double n19=22.5625;
double n20=(n16*n16*n19);
double n21=(n3+n6+n15+n18+n18+n20);
double n22=dz;
double n23=(n16*n19);
double n24=-26.125;
double n25=(n13*n24);
double n26=(n16*n24);
double n27=(n13*n14);
double n28=(n23+n23+n25+n26+n27+n27);
double n29=(n28*n10);
double n30=(n7*n29);
double n31=(n30+n30);
double n32=(n22*n31);
double n33=dy;
double n34=(n29+n2);
double n35=(n4*n34);
double n36=(n35+n35);
double n37=(n33*n36);
double n38=dx;
double n39=(n0*n34);
double n40=(n39+n39);
double n41=(n38*n40);
double n42=(n32+n37+n41);
double n43=(n34*n33);
double n44=(n0*n38);
double n45=(n4*n33);
double n46=(n7*n22);
double n47=(n44+n44+n45+n45+n46+n46);
double n48=(n47*n10);
double n49=(n48*n14);
double n50=(n48*n24);
double n51=(n48*n19);
double n52=(n49+n49+n50+n50+n51+n51);
double n53=(n52*n10);
double n54=(n4*n53);
double n55=(n43+n43+n54+n54);
double n56=(n33*n55);
double n57=(n34*n38);
double n58=(n0*n53);
double n59=(n57+n57+n58+n58);
double n60=(n38*n59);
double n61=(n29*n22);
double n62=(n7*n53);
double n63=(n61+n62+n62+n61);
double n64=(n22*n63);
double n65=(n56+n60+n64);
double n66=(n65*n10);
double n67=(n22*n10);
double n68=(n53*n67);
double n69=(n7*n67);
double n70=(n38*n10);
double n71=(n0*n70);
double n72=(n33*n10);
double n73=(n4*n72);
double n74=(n69+n69+n71+n71+n73+n73);
double n75=(n74*n10);
double n76=(n75*n19);
double n77=(n75*n24);
double n78=(n75*n14);
double n79=(n76+n76+n77+n77+n78+n78);
double n80=(n79*n10);
double n81=(n22*n80);
double n82=(n22*n67);
double n83=(n38*n70);
double n84=(n33*n72);
double n85=(n82+n82+n83+n83+n84+n84);
double n86=(n85*n10);
double n87=(n86*n14);
double n88=(n86*n24);
double n89=(n86*n19);
double n90=(n87+n87+n88+n88+n89+n89);
double n91=(n90*n10);
double n92=(n7*n91);
double n93=(n68+n68+n81+n81+n92+n92);
double n94=(n22*n93);
double n95=(n53*n72);
double n96=(n33*n80);
double n97=(n4*n91);
double n98=(n95+n96+n96+n95+n97+n97);
double n99=(n33*n98);
double n100=(n53*n70);
double n101=(n38*n80);
double n102=(n0*n91);
double n103=(n100+n100+n101+n101+n102+n102);
double n104=(n38*n103);
double n105=(n94+n99+n104);
double n106=0.3333333333333333;
double n107=(n105*n106);
double n108=(n22*n106);
double n109=(n91*n108);
double n110=(n38*n106);
double n111=(n38*n110);
double n112=(n33*n106);
double n113=(n33*n112);
double n114=(n22*n108);
double n115=(n111+n111+n113+n113+n114+n114);
double n116=(n115*n10);
double n117=(n116*n14);
double n118=(n116*n24);
double n119=(n116*n19);
double n120=(n117+n117+n118+n118+n119+n119);
double n121=(n120*n10);
double n122=(n67*n121);
double n123=(n70*n110);
double n124=(n72*n112);
double n125=(n67*n108);
double n126=(n123+n123+n124+n124+n125+n125);
double n127=(n126*n10);
double n128=(n127*n19);
double n129=(n127*n24);
double n130=(n127*n14);
double n131=(n128+n128+n129+n129+n130+n130);
double n132=(n131*n10);
double n133=(n22*n132);
double n134=(n109+n109+n122+n122+n133+n133);
double n135=(n22*n134);
double n136=(n91*n112);
double n137=(n72*n121);
double n138=(n33*n132);
double n139=(n136+n136+n137+n137+n138+n138);
double n140=(n33*n139);
double n141=(n91*n110);
double n142=(n70*n121);
double n143=(n38*n132);
double n144=(n141+n141+n142+n142+n143+n143);
double n145=(n38*n144);
double n146=(n135+n140+n145);
double n147=0.25;
double n148=(n146*n147);
polys[0][0]=n21;
polys[0][1]=n42;
polys[0][2]=n66;
polys[0][3]=n107;
polys[0][4]=n148;
}


