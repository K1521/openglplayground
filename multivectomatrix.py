#from algebra import dcga
from algebra import blademul
dcga=blademul.algebra(2,2)
multivec=blademul.sortgeo(dcga)
blades=multivec.allblades()
#blades=dcga.multivec.allblades()
import numpy as np
table=np.empty([len(blades)]*2,dtype=object)
for i,a in enumerate(blades):
    for j,b in enumerate(blades):
        #print(a.geo(b))
        blade=a.geo(b).lst[0]
        table[i,j]="+-"[blade.magnitude<0]+str(blade.basis)
print(table)