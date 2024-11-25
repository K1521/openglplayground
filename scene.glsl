float scene(vec3 p){
float x=p.x;float y=p.y;float z=p.z;
float n0=x;
float n1=(n0*n0);
float n2=y;
float n3=(n2*n2);
float n4=z;
float n5=(n4*n4);
float n6=-1;
float n7=sqrt(n1+n3+n5)+n6;
return n7;
}