

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


