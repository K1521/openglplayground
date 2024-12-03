
const int numpolys=1;
const int numparams=5;
float[numpolys][numparams] polys;
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
const float inf=pow(2,1024);

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
float n6=oy;
float n7=dy;
float n8=(n1*n7);
float n9=(n6+n8);
float n10=(n9*n9);
float n11=-1;
float n12=(n5+n10+n11);
float n13=oz;
float n14=dz;
float n15=(n1*n14);
float n16=(n13+n15);
float n17=(n16*n16);
float n18=(n5+n17+n11);
float n19=(n12*n18);
float n20=(n16*n12);
float n21=(n20+n20);
float n22=(n14*n21);
float n23=(n9*n18);
float n24=(n23+n23);
float n25=(n7*n24);
float n26=(n12+n18);
float n27=(n4*n26);
float n28=(n27+n27);
float n29=(n2*n28);
float n30=(n22+n25+n29);
float n31=(n18*n7);
float n32=(n4*n2);
float n33=(n32+n32);
float n34=(n16*n14);
float n35=(n33+n34+n34);
float n36=(n9*n35);
float n37=(n31+n31+n36+n36);
float n38=(n7*n37);
float n39=(n26*n2);
float n40=(n9*n7);
float n41=(n33+n40+n40);
float n42=(n41+n35);
float n43=(n4*n42);
float n44=(n39+n39+n43+n43);
float n45=(n2*n44);
float n46=(n16*n41);
float n47=(n12*n14);
float n48=(n46+n46+n47+n47);
float n49=(n14*n48);
float n50=(n38+n45+n49);
float n51=0.5;
float n52=(n50*n51);
float n53=(n14*n51);
float n54=(n16*n53);
float n55=(n2*n51);
float n56=(n4*n55);
float n57=(n56+n56);
float n58=(n54+n54+n57);
float n59=(n7*n58);
float n60=(n14*n53);
float n61=(n2*n55);
float n62=(n61+n61);
float n63=(n60+n60+n62);
float n64=(n9*n63);
float n65=(n7*n51);
float n66=(n35*n65);
float n67=(n59+n59+n64+n64+n66+n66);
float n68=(n7*n67);
float n69=(n41*n53);
float n70=(n9*n65);
float n71=(n57+n70+n70);
float n72=(n14*n71);
float n73=(n7*n65);
float n74=(n62+n73+n73);
float n75=(n16*n74);
float n76=(n69+n69+n72+n72+n75+n75);
float n77=(n14*n76);
float n78=(n42*n55);
float n79=(n58+n71);
float n80=(n2*n79);
float n81=(n63+n74);
float n82=(n4*n81);
float n83=(n78+n78+n80+n80+n82+n82);
float n84=(n2*n83);
float n85=(n68+n77+n84);
float n86=0.3333333333333333;
float n87=(n85*n86);
float n88=(n2*n86);
float n89=(n2*n88);
float n90=(n89+n89);
float n91=(n14*n86);
float n92=(n14*n91);
float n93=(n90+n92+n92);
float n94=(n65*n93);
float n95=(n55*n88);
float n96=(n95+n95);
float n97=(n53*n91);
float n98=(n96+n97+n97);
float n99=(n7*n98);
float n100=(n7*n86);
float n101=(n63*n100);
float n102=(n94+n94+n99+n99+n101+n101);
float n103=(n7*n102);
float n104=(n81*n88);
float n105=(n65*n100);
float n106=(n96+n105+n105);
float n107=(n98+n106);
float n108=(n2*n107);
float n109=(n7*n100);
float n110=(n90+n109+n109);
float n111=(n93+n110);
float n112=(n55*n111);
float n113=(n104+n104+n108+n108+n112+n112);
float n114=(n2*n113);
float n115=(n74*n91);
float n116=(n14*n106);
float n117=(n53*n110);
float n118=(n115+n115+n116+n116+n117+n117);
float n119=(n14*n118);
float n120=(n103+n114+n119);
float n121=0.25;
float n122=(n120*n121);
polys[0][0]=n19;
polys[0][1]=n30;
polys[0][2]=n52;
polys[0][3]=n87;
polys[0][4]=n122;
}


