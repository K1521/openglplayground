from matplotlib import pyplot as plt
import numpy as np


x=np.linspace(0,1,256,endpoint=False)
print(x)
xt=np.tile(x, (4, 1))
xt *= 2**np.arange(4)[:,None]*np.pi
#xt= np.mod(xt * 2**np.arange(4)[:, None],1) 
#xt=np.vstack([[0.5]* xt.shape[1],1,xt,xt+0.5)])
xt=np.vstack([np.sin(xt),np.cos(xt)])
#print(xt[:,::32])
#xt=xt/xt.max(axis=1,keepdims=True)
xt=xt/np.linalg.norm(xt,axis=0)
for a in xt:
    plt.plot(a)
plt.show()


a=(xt[:,16][:,None]*xt).sum(axis=0)
plt.plot(a)
plt.show()