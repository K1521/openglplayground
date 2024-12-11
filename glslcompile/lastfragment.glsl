#version 440
/*source=part1*/

const int numparams=5;
const int numpolys=1;
float[numpolys][numparams] polys;
float calcpolys(float x);
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

float calcpolys(float x);
void compilepolys(vec3 p,vec3 d);



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
    if(isinf(x)){
        raymarch2(rayDir,rayOrigin);
    }
    rayOrigin += rayDir * x;
    return 0;
}
float raymarch2(vec3 rayDir, inout vec3 rayOrigin) {
    compilepolys(rayOrigin,rayDir);
    float magnitude=1000000;
    float x=0;
    for (int i = 0; i < MAX_RAY_ITER; i++) {
        
        magnitude = calcpolys(x);
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
    vec3 rayDir =cameraMatrix* normalize(vec3(uv, FOVfactor));//cam to view


    // Sphere tracing

    vec3 p=rayOrigin;
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

float calcpolys(float x){
    //x-=A;
    float[numparams] pows;
    pows[0]=1;
    for(int j=1;j<numparams;j++){
        pows[j]=pows[j-1]*x;
    }

    float s=0;
    for(int i=0;i<numpolys;i++){
        float[numparams] poly=polys[i];
        float d=0;
        for (int j = 0; j < numparams; j++) {
            d += poly[j] * pows[j];
        }
        s+=abs(d);
    }
    return s;
}


float rootpolysquartic(){
    float minx=inf;
    vec4 roots=solveQuartic(polys[0][4],polys[0][3],polys[0][2],polys[0][1],polys[0][0]);
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
float n6=-16;
float n7=(n5*n6);
float n8=oy;
float n9=dy;
float n10=(n1*n9);
float n11=(n8+n10);
float n12=(n11*n11);
float n13=(n12*n6);
float n14=oz;
float n15=dz;
float n16=(n1*n15);
float n17=(n14+n16);
float n18=(n17*n17);
float n19=(n5+n12+n18);
float n20=0.5;
float n21=(n19*n20);
float n22=-0.5;
float n23=(n21+n22);
float n24=7.5625;
float n25=(n23*n23*n24);
float n26=(n21+n20);
float n27=-13.0625;
float n28=(n23*n26*n27);
float n29=22.5625;
float n30=(n26*n26*n29);
float n31=(n7+n13+n25+n28+n28+n30);
float n32=(n26*n29);
float n33=-26.125;
float n34=(n23*n33);
float n35=(n26*n33);
float n36=(n23*n24);
float n37=(n32+n32+n34+n35+n36+n36);
float n38=(n37*n20);
float n39=(n17*n38);
float n40=(n39+n39);
float n41=(n15*n40);
float n42=(n38+n6);
float n43=(n11*n42);
float n44=(n43+n43);
float n45=(n9*n44);
float n46=(n4*n42);
float n47=(n46+n46);
float n48=(n2*n47);
float n49=(n41+n45+n48);
float n50=(n42*n9);
float n51=(n4*n2);
float n52=(n11*n9);
float n53=(n17*n15);
float n54=(n51+n51+n52+n52+n53+n53);
float n55=(n54*n20);
float n56=(n55*n24);
float n57=(n55*n33);
float n58=(n55*n29);
float n59=(n56+n56+n57+n57+n58+n58);
float n60=(n59*n20);
float n61=(n11*n60);
float n62=(n50+n50+n61+n61);
float n63=(n9*n62);
float n64=(n42*n2);
float n65=(n4*n60);
float n66=(n64+n64+n65+n65);
float n67=(n2*n66);
float n68=(n38*n15);
float n69=(n17*n60);
float n70=(n68+n69+n69+n68);
float n71=(n15*n70);
float n72=(n63+n67+n71);
float n73=(n72*n20);
float n74=(n15*n20);
float n75=(n60*n74);
float n76=(n17*n74);
float n77=(n2*n20);
float n78=(n4*n77);
float n79=(n9*n20);
float n80=(n11*n79);
float n81=(n76+n76+n78+n78+n80+n80);
float n82=(n81*n20);
float n83=(n82*n29);
float n84=(n82*n33);
float n85=(n82*n24);
float n86=(n83+n83+n84+n84+n85+n85);
float n87=(n86*n20);
float n88=(n15*n87);
float n89=(n15*n74);
float n90=(n2*n77);
float n91=(n9*n79);
float n92=(n89+n89+n90+n90+n91+n91);
float n93=(n92*n20);
float n94=(n93*n24);
float n95=(n93*n33);
float n96=(n93*n29);
float n97=(n94+n94+n95+n95+n96+n96);
float n98=(n97*n20);
float n99=(n17*n98);
float n100=(n75+n75+n88+n88+n99+n99);
float n101=(n15*n100);
float n102=(n60*n79);
float n103=(n9*n87);
float n104=(n11*n98);
float n105=(n102+n103+n103+n102+n104+n104);
float n106=(n9*n105);
float n107=(n60*n77);
float n108=(n2*n87);
float n109=(n4*n98);
float n110=(n107+n107+n108+n108+n109+n109);
float n111=(n2*n110);
float n112=(n101+n106+n111);
float n113=0.3333333333333333;
float n114=(n112*n113);
float n115=(n15*n113);
float n116=(n98*n115);
float n117=(n2*n113);
float n118=(n2*n117);
float n119=(n9*n113);
float n120=(n9*n119);
float n121=(n15*n115);
float n122=(n118+n118+n120+n120+n121+n121);
float n123=(n122*n20);
float n124=(n123*n24);
float n125=(n123*n33);
float n126=(n123*n29);
float n127=(n124+n124+n125+n125+n126+n126);
float n128=(n127*n20);
float n129=(n74*n128);
float n130=(n77*n117);
float n131=(n79*n119);
float n132=(n74*n115);
float n133=(n130+n130+n131+n131+n132+n132);
float n134=(n133*n20);
float n135=(n134*n29);
float n136=(n134*n33);
float n137=(n134*n24);
float n138=(n135+n135+n136+n136+n137+n137);
float n139=(n138*n20);
float n140=(n15*n139);
float n141=(n116+n116+n129+n129+n140+n140);
float n142=(n15*n141);
float n143=(n98*n119);
float n144=(n79*n128);
float n145=(n9*n139);
float n146=(n143+n143+n144+n144+n145+n145);
float n147=(n9*n146);
float n148=(n98*n117);
float n149=(n77*n128);
float n150=(n2*n139);
float n151=(n148+n148+n149+n149+n150+n150);
float n152=(n2*n151);
float n153=(n142+n147+n152);
float n154=0.25;
float n155=(n153*n154);
polys[0][0]=n31;
polys[0][1]=n49;
polys[0][2]=n73;
polys[0][3]=n114;
polys[0][4]=n155;
}


