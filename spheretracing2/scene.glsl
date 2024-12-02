
const int numpolys=1;
const int numparams=5;
float[numpolys][numparams] polys;
//float A;



float calcpolys(float x){
    //x-=A;
    float[numparams] pows;
    pows[0]=1;
    for(int j=1;j<numparams;j++){
        pows[j]=pows[j-1]*x/j;
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
float n52=(n51+n51);
float n53=(n11*n9);
float n54=(n53+n53);
float n55=(n17*n15);
float n56=(n55+n55);
float n57=(n52+n54+n56);
float n58=(n57*n20);
float n59=(n58*n24);
float n60=(n58*n33);
float n61=(n58*n29);
float n62=(n59+n59+n60+n60+n61+n61);
float n63=(n62*n20);
float n64=(n11*n63);
float n65=(n50+n50+n64+n64);
float n66=(n9*n65);
float n67=(n42*n2);
float n68=(n4*n63);
float n69=(n67+n67+n68+n68);
float n70=(n2*n69);
float n71=(n38*n15);
float n72=(n17*n63);
float n73=(n71+n72+n72+n71);
float n74=(n15*n73);
float n75=(n66+n70+n74);
float n76=(n63*n15);
float n77=(n15*n15);
float n78=(n77+n77);
float n79=(n2*n2);
float n80=(n79+n79);
float n81=(n9*n9);
float n82=(n81+n81);
float n83=(n78+n80+n82);
float n84=(n83*n20);
float n85=(n84*n24);
float n86=(n84*n33);
float n87=(n84*n29);
float n88=(n85+n85+n86+n86+n87+n87);
float n89=(n88*n20);
float n90=(n17*n89);
float n91=(n76+n76+n76+n76+n90+n90);
float n92=(n15*n91);
float n93=(n63*n9);
float n94=(n11*n89);
float n95=(n93+n93+n93+n93+n94+n94);
float n96=(n9*n95);
float n97=(n63*n2);
float n98=(n4*n89);
float n99=(n97+n97+n97+n97+n98+n98);
float n100=(n2*n99);
float n101=(n92+n96+n100);
float n102=(n89*n15);
float n103=(n102+n102+n102+n102+n102+n102);
float n104=(n15*n103);
float n105=(n89*n9);
float n106=(n105+n105+n105+n105+n105+n105);
float n107=(n9*n106);
float n108=(n89*n2);
float n109=(n108+n108+n108+n108+n108+n108);
float n110=(n2*n109);
float n111=(n104+n107+n110);
polys[0][0]=n31;
polys[0][1]=n49;
polys[0][2]=n75;
polys[0][3]=n101;
polys[0][4]=n111;
}


