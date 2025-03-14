
def calcpoly(x):
    return poly[0]+poly[1]*x+poly[2]*x**2+poly[3]*x**3+poly[4]*x**3
    
def compilepolys(p,d):
    a=0
    ox=p[0]
    oy=p[1]
    oz=p[2]
    dx=d[0]
    dy=d[1]
    dz=d[2]
    n0=ox
    n1=a
    n2=dx
    n3=(n1*n2)
    n4=(n0+n3)
    n5=(n4*n4)
    n6=-16
    n7=(n5*n6)
    n8=oy
    n9=dy
    n10=(n1*n9)
    n11=(n8+n10)
    n12=(n11*n11)
    n13=(n12*n6)
    n14=oz
    n15=dz
    n16=(n1*n15)
    n17=(n14+n16)
    n18=(n17*n17)
    n19=(n5+n12+n18)
    n20=0.5
    n21=(n19*n20)
    n22=-0.5
    n23=(n21+n22)
    n24=7.5625
    n25=(n23*n23*n24)
    n26=(n21+n20)
    n27=-13.0625
    n28=(n23*n26*n27)
    n29=22.5625
    n30=(n26*n26*n29)
    n31=(n7+n13+n25+n28+n28+n30)
    n32=(n26*n29)
    n33=-26.125
    n34=(n23*n33)
    n35=(n26*n33)
    n36=(n23*n24)
    n37=(n32+n32+n34+n35+n36+n36)
    n38=(n37*n20)
    n39=(n17*n38)
    n40=(n39+n39)
    n41=(n15*n40)
    n42=(n38+n6)
    n43=(n11*n42)
    n44=(n43+n43)
    n45=(n9*n44)
    n46=(n4*n42)
    n47=(n46+n46)
    n48=(n2*n47)
    n49=(n41+n45+n48)
    n50=(n42*n9)
    n51=(n4*n2)
    n52=(n11*n9)
    n53=(n17*n15)
    n54=(n51+n51+n52+n52+n53+n53)
    n55=(n54*n20)
    n56=(n55*n24)
    n57=(n55*n33)
    n58=(n55*n29)
    n59=(n56+n56+n57+n57+n58+n58)
    n60=(n59*n20)
    n61=(n11*n60)
    n62=(n50+n50+n61+n61)
    n63=(n9*n62)
    n64=(n42*n2)
    n65=(n4*n60)
    n66=(n64+n64+n65+n65)
    n67=(n2*n66)
    n68=(n38*n15)
    n69=(n17*n60)
    n70=(n68+n69+n69+n68)
    n71=(n15*n70)
    n72=(n63+n67+n71)
    n73=(n72*n20)
    n74=(n15*n20)
    n75=(n60*n74)
    n76=(n17*n74)
    n77=(n2*n20)
    n78=(n4*n77)
    n79=(n9*n20)
    n80=(n11*n79)
    n81=(n76+n76+n78+n78+n80+n80)
    n82=(n81*n20)
    n83=(n82*n29)
    n84=(n82*n33)
    n85=(n82*n24)
    n86=(n83+n83+n84+n84+n85+n85)
    n87=(n86*n20)
    n88=(n15*n87)
    n89=(n15*n74)
    n90=(n2*n77)
    n91=(n9*n79)
    n92=(n89+n89+n90+n90+n91+n91)
    n93=(n92*n20)
    n94=(n93*n24)
    n95=(n93*n33)
    n96=(n93*n29)
    n97=(n94+n94+n95+n95+n96+n96)
    n98=(n97*n20)
    n99=(n17*n98)
    n100=(n75+n75+n88+n88+n99+n99)
    n101=(n15*n100)
    n102=(n60*n79)
    n103=(n9*n87)
    n104=(n11*n98)
    n105=(n102+n103+n103+n102+n104+n104)
    n106=(n9*n105)
    n107=(n60*n77)
    n108=(n2*n87)
    n109=(n4*n98)
    n110=(n107+n107+n108+n108+n109+n109)
    n111=(n2*n110)
    n112=(n101+n106+n111)
    n113=0.3333333333333333
    n114=(n112*n113)
    n115=(n15*n113)
    n116=(n98*n115)
    n117=(n2*n113)
    n118=(n2*n117)
    n119=(n9*n113)
    n120=(n9*n119)
    n121=(n15*n115)
    n122=(n118+n118+n120+n120+n121+n121)
    n123=(n122*n20)
    n124=(n123*n24)
    n125=(n123*n33)
    n126=(n123*n29)
    n127=(n124+n124+n125+n125+n126+n126)
    n128=(n127*n20)
    n129=(n74*n128)
    n130=(n77*n117)
    n131=(n79*n119)
    n132=(n74*n115)
    n133=(n130+n130+n131+n131+n132+n132)
    n134=(n133*n20)
    n135=(n134*n29)
    n136=(n134*n33)
    n137=(n134*n24)
    n138=(n135+n135+n136+n136+n137+n137)
    n139=(n138*n20)
    n140=(n15*n139)
    n141=(n116+n116+n129+n129+n140+n140)
    n142=(n15*n141)
    n143=(n98*n119)
    n144=(n79*n128)
    n145=(n9*n139)
    n146=(n143+n143+n144+n144+n145+n145)
    n147=(n9*n146)
    n148=(n98*n117)
    n149=(n77*n128)
    n150=(n2*n139)
    n151=(n148+n148+n149+n149+n150+n150)
    n152=(n2*n151)
    n153=(n142+n147+n152)
    n154=0.25
    n155=(n153*n154)
    return n31,n49,n73,n114,n155



