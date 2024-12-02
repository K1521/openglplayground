

/*vec4 solveQuartic(float a,float b,float c,float d,float e){
    //https://en.wikipedia.org/wiki/Quartic_function
    float delta0=c*c-3*b*d+12*a*e;
    float delta1=2*c*c*c-9*b*c*d+27*b*b*e+27*a*d*d-72*a*c*e;
    float p=(8*a*c-3*b*b)/(8*a*a);
    float q=(b*b*b-4*a*b*c+8*a*a*d)/(8*a*a*a);
    float Q=pow((delta1+sqrt(delta1*delta1-4*delta0*delta0*delta0)),1/3);
    float S=0.5*sqrt(-p*2/3+(Q+delta0/Q)/(3*a));

    float xpart1=-b/(4*a);
    float xpart2=-4*S*S-2*p;
    float xpart3=q/S;
    float xsqrt1=0.5*sqrt(xpart2+xpart3);
    float xsqrt2=0.5*sqrt(xpart2-xpart3);

    return vec4(xpart-S+xsqrt1,xpart-S-xsqrt1,xpart+S+xsqrt2,xpart+S-xsqrt2);
}*/

