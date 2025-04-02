

from matplotlib import pyplot as plt
import numpy as np
import math
golden_ratio=(1+math.sqrt(5))/2
golden_angle=2*math.pi/golden_ratio
golden_complex=math.cos(golden_angle)+math.sin(golden_angle)*1j


def plothist(history):
    history=np.array(history)
    for i in range(len(history[0])):
        # i want to plot with points
        plt.plot(history[:,i].real,history[:,i].imag,marker='o')
    plt.show()

def Aberth2(initial,f,f_,iterations,addconjugate=False):
    #x=np.linspace(0,5,100)

    history=[]
    roots=initial[:]
    for it in range(iterations):
        for r in range(len(roots)):
            root=roots[r]
            p=f(root)
            if abs(p)<1e-15:
                continue
            p_=f_(root)
            s=0
            for j in range(len(roots)):
                if r!=j:
                    s+=1/(root-roots[j])+1/(root-roots[j].conjugate())
            if addconjugate:
                s+=1/(root-root.conjugate())
            n=p/p_
            roots[r]-=n/(1-n*s)
        history.append(roots[:])
        #print(roots)
    return roots,history
    

def f(x):
    return (-1 + 12.73333*x - 12.55*x**2 + 4.266667*x**3 - 0.45*x**4)**2+(15 - 20.68333*x + 8.741667*x**2 - 1.066667*x**3 + 0.008333333*x**4)**2
def randpoly(degree=4):
    x=np.linspace(-1,1,degree+1)
    y=np.random.uniform(-1,1,degree+1)
    return np.poly1d(np.polyfit(x,y,degree))

def randompoints(x):  

    return np.vstack([x,np.random.uniform(-1,1,len(x))])

def randpoly2(numcommonpoints=3):
    degree=4
    numpoints=degree+1-numcommonpoints
    x=np.linspace(-1,1,degree+1)
    #split x in x1 x2 with numpoints/numcommonpoints
    x1=x[:numpoints]
    x2=x[numpoints:]
    #print(numpoints,x1,x2)


    commonpoints=randompoints(x2)
    p=0
    for i in range(3):
        x,y=np.hstack([commonpoints,randompoints(x1)])
        #print(x,y)
        p+=np.poly1d(np.polyfit(x,y,degree))**2
    return p

# x=np.linspace(-1,1,100)
# for i in range(10):
#     plt.plot(x,randpoly2()(x))
# plt.show()
initial=[golden_complex**i for i in range(1,4+1)]

dists1=[]
dists2=[]
score1=0
score2=0
for i in range(500):
    p=randpoly2()
    #print(p)
    roots,_=Aberth2(initial,p,p.deriv(),3)
    #plothist(_)
    rootsconj,_=Aberth2(initial,p,p.deriv(),3,addconjugate=True)
    nproots=p.roots

   
    
    
    roots=sorted(roots,key=lambda x:x.real)
    rootsconj=sorted(rootsconj,key=lambda x:x.real)
    nproots=sorted(nproots,key=lambda x:x.real)

    # roots=np.array(roots)
    # rootsconj=np.array(rootsconj)
    # nproots=np.array(nproots)

    # plt.plot(roots.real,roots.imag,color="r")
    # plt.plot(rootsconj.real,rootsconj.imag,color="g")
    # plt.plot(nproots.real,nproots.imag,color="b")
    # plt.show()

    dist1=0
    dist2=0
    for i in range(4):
        dist1+=min(abs(roots[i]-nproots[2*i]),abs(roots[i]-nproots[2*i+1]))
        dist2+=min(abs(rootsconj[i]-nproots[2*i]),abs(rootsconj[i]-nproots[2*i+1]))
    dists1.append(dist1)
    dists2.append(dist2)
    score1+=dist1<dist2
    score2+=dist2<dist1
print(score1,score2)

bins = 50#np.linspace(0,4, 100)

plt.hist(dists1, bins, alpha=0.5, label='dist1')
plt.hist(dists2, bins, alpha=0.5, label='dist2')
plt.legend(loc='upper right')
plt.show()

exit()

initial=[golden_complex**i for i in range(1,8+1)]

roots,history=Aberth(initial,f,100)

def froots(x,roots):
    return math.prod(x-root for root in roots)

#plot trajectory of roots in the complex plane
history=np.array(history)
# for i in range(len(history[0])):
#     for j in range(len(history)-1):
#         start=history[j][i]
#         end=history[j+1][i]
#         plt.plot([start.real,end.real],[start.imag,end.imag],color='black')
for i in range(len(history[0])):
    # i want to plot with points
    plt.plot(history[:,i].real,history[:,i].imag,marker='o')
plt.show()

x=np.linspace(0,5,100)
poly=f(np.polynomial.Polynomial([0,1]))

plt.plot(x,f(x),color='green')
#plt.plot(x,f(Dual(x,1)).f_,color='blue')
plt.plot(x,f(x)/froots(x,poly.roots()),color='red')
#plt.plot(x,f(x)/froots(x,roots+[z.conjugate() for z in roots]),color='purple')

plt.plot(x,f(x)/froots(x,roots),color='purple')

#plt.plot(x,poly(x),color='blue')
print(poly.roots())
plt.show()