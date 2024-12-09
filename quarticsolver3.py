def ferrari(a,b,c,d,e):
    "resolution of P=ax^4+bx^3+cx^2+dx+e=0"
    "CN all coeffs real."
    "First shift : x= z-b/4/a  =>  P=z^4+pz^2+qz+r"
    z0=b/4/a
    a2,b2,c2,d2 = a*a,b*b,c*c,d*d 
    p = -3*b2/(8*a2)+c/a
    q = b*b2/8/a/a2 - 1/2*b*c/a2 + d/a
    r = -3/256*b2*b2/a2/a2 +c*b2/a2/a/16-b*d/a2/4+e/a
    "Second find X so P2=AX^3+BX^2+C^X+D=0"
    A=8
    B=-4*p
    C=-8*r
    D=4*r*p-q*q
    y0,y1,y2=cardan(A,B,C,D)
    if abs(y1.imag)<abs(y0.imag): y0=y1 
    if abs(y2.imag)<abs(y0.imag): y0=y2 
    a0=(-p+2*y0.real)**.5
    if a0==0 : b0=y0**2-r
    else : b0=-q/2/a0
    r0,r1=roots2(1,a0,y0+b0)
    r2,r3=roots2(1,-a0,y0-b0)
    return (r0-z0,r1-z0,r2-z0,r3-z0) 

def roots2(a,b,c):
    bp=b/2    
    delta=bp*bp-a*c
    u1=(-bp-delta**.5)/a
    u2=-u1-b/a
    return u1,u2  