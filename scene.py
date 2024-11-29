
import numpy as np
sign=np.sign

def scene(p):
    x,y,z=p
    n0=x
    n1=(n0*n0)
    n2=y
    n3=(n2*n2)
    n4=z
    n5=(n4*n4)
    n6=(n1+n3+n5)
    n7=0.5
    n8=(n6*n7)
    n9=(n8+n7)
    n10=22.5625
    n11=-16
    n12=(n1*n11)
    n13=(n3*n11)
    n14=-0.5
    n15=(n8+n14)
    n16=7.5625
    n17=(n15*n15*n16)
    n18=(n8+n7)
    n19=-13.0625
    n20=(n15*n18*n19)
    n21=(n18*n18*n10)
    n22=(n12+n13+n17+n20+n20+n21)
    n23=sign(n22)
    n24=(n9*n10*n23)
    n25=(n8+n14)
    n26=(n23+n23)
    n27=(n25*n19*n26)
    n28=(n9*n19*n26)
    n29=(n25*n16*n23)
    n30=(n24+n24+n27+n28+n29+n29)
    n31=(n30*n7)
    n32=(n23*n11)
    n33=(n31+n32)
    n34=(n0*n33)
    n35=(n34+n34)
    n36=(n2*n33)
    n37=(n36+n36)
    n38=(n4*n31)
    n39=(n38+n38)
    n40=abs(n22)
    return (n35,n37,n39,n40)
