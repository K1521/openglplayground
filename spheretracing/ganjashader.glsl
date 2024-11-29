#version 300 es
in vec4 position;
out vec4 Pos;
uniform mat4 mv;
uniform mat4 p;
void main() {
    Pos=mv*position; gl_Position = p*Pos;
}


#extension GL_EXT_frag_depth : enable
precision highp float;
uniform vec3 color; uniform vec3 color2;
uniform vec3 color3; uniform float b[${counts[grade]}];
uniform float ratio; 
out vec4 col;
in vec4 Pos;
float product_len (in float z, in float y, in float x, in float[${counts[grade]}] b) {
    ${this.nVector(options.up.length>tot?2:1,[])[options.IPNS?"IPNS_GLSL":"OPNS_GLSL"](this.nVector(grade,[]), options.up)}
    return sqrt(abs(sum));
}
vec3 find_root (in vec3 start, vec3 dir, in float thresh) {
    vec3 orig=start; 
    float lastd = 1000.0; 
    const int count=80;
    const thresh=0.2;
    const float stepSize=0.0001
    for (int i=0; i<count; i++) {
        float d = product_len(start[0],start[1],start[2],b);
        float diff = stepSize*(1.0+2000.0*d);
        if (d < thresh)
            return start + dir*(lastd-thresh)/(lastd-d)*diff;
        lastd = d;
        start += dir*diff;
    }
    return orig;
}
vec3 p = -5.0*normalize(color2) + dir+vec3(0.0,0.0,1.0); dir = normalize(dir);
void main() {
    vec3 dir = ((-Pos[0]/5.0)*color + color2 + vec3(0.0,Pos[1]/5.0*ratio,0.0)); 
    vec3 L = 5.0*normalize( -0.5*color + 0.85*color2 + vec3(0.0,-0.5,0.0) );
    vec3 d2 = find_root( p , dir, ${grade!=tot-1?(options.thresh||0.2):"0.0075"} );
    float dl2 = dot(d2-p,d2-p); const float h=0.0001;
    if (dl2>0.0) {
        vec3 n = normalize(vec3(
            product_len(d2[0]+h,d2[1],d2[2],b)-product_len(d2[0]-h,d2[1],d2[2],b),
            product_len(d2[0],d2[1]+h,d2[2],b)-product_len(d2[0],d2[1]-h,d2[2],b),
            product_len(d2[0],d2[1],d2[2]+h,b)-product_len(d2[0],d2[1],d2[2]-h,b)
            ));
        gl_FragDepth = dl2/50.0;
        col = vec4(max(0.2,abs(dot(n,normalize(L-d2))))*color3 + pow(abs(dot(n,normalize(normalize(L-d2)+dir))),100.0),1.0);
    } else discard;
}




             
#version 300 es
in vec4 position; out vec4 Pos; uniform mat4 mv; uniform mat4 p;
void main() {
    Pos=mv*position; gl_Position = p*Pos;
}


#extension GL_EXT_frag_depth : enable
precision highp float;
uniform vec3 color; uniform vec3 color2;
uniform vec3 color3; uniform float b[${counts[grade]}];
uniform float ratio; ${gl2?"out vec4 col;":""}
in vec4 Pos;
float product_len (in float z, in float y, in float x, in float[${counts[grade]}] b) {
    ${this.nVector(1,[])[options.IPNS?"IPNS_GLSL":"OPNS_GLSL"](this.nVector(grade,[]), options.up)}
    return sqrt(abs(sum));
}
void main() {
    vec3 p = -5.0*normalize(color2) -Pos[0]/5.0*color + color2 + vec3(0.0,Pos[1]/5.0*ratio,0.0); 
    float d2 = 1.0 - 150.0*pow(product_len( p[0]*5.0, p[1]*5.0, p[2]*5.0, b),2.0);
    if (d2>0.0) {
        col = vec4(color3,d2);
    } else discard;
}