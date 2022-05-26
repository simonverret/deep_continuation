settings.outformat = "pdf";

arrowbar HookArrow = ArcArrow(HookHead, 2);
pen cross_pen = linewidth(1.5)+red;
path[] Cross(real s){
	return ((-s,s)--(s,-s)^^(-s,-s)--(s,s));
}
void draw_cross(picture pic=currentpicture, pair pos, real angle=0, pen mypen=cross_pen){
	draw(pic, shift(pos)*rotate(angle)*Cross(2), mypen);
}

real L = 70;
draw(Label("Re$(\omega)$", E), (-1.2*L,0)--(1.2*L,0), HookArrow);
draw(Label("Im$(\omega)$", E), (0,-1.2*L)--(0,0.8*L), HookArrow);

//poles
real xx = 55;
real yy = 10;
pair p1 = ( 00, 3*yy);
pair p2 = (-xx, yy);
pair p3 = (-xx,-yy);
pen contourpen1 = blue+linewidth(1.5);
pen contourpen2 = blue+linewidth(1.5)+linetype("2 3");

draw_cross(p1);
draw_cross(p2);
draw_cross(p3);

pair offset = (2,0);
label("$\omega_n$", p1+offset, E);
label("$-\omega_{j}+i\eta$", p2+offset, NE);
label("$-\omega_{j}-i\eta$", p3+offset, SE);

real off = -0.1;
draw(reverse(arc((0,0),L,degrees(angle((-L,off))),degrees(angle((L,off))))), contourpen2, MidArrow(6));
draw(shift(0,off)*((-L,0)--(L,0)), contourpen1, MidArrow(6));
label("$C_{\infty}$",rotate(-60)*(L,0),SE);
label("($+$)", (-L,L));



real SHIFT = 220;

draw(Label("Re$(\omega)$", E), (SHIFT-1.2*L,0)--(SHIFT+1.2*L,0), HookArrow);
draw(Label("Im$(\omega)$", E), (SHIFT+0,-1.2*L)--(SHIFT+0,0.8*L), HookArrow);

//poles
p1 = (SHIFT+00, 3*yy);
p2 = (SHIFT+xx, yy);
p3 = (SHIFT+xx,-yy);

draw_cross(p1);
draw_cross(p2);
draw_cross(p3);

pair offset = (-2,0);
label("$\omega_n$", p1+offset, W);
label("$\omega_{j}+i\eta$", p2+offset, NW);
label("$\omega_{j}-i\eta$", p3+offset, SW);

real off = -0.1;
draw(reverse(arc((SHIFT+0,0),L,degrees(angle((-L,off))),degrees(angle((L,off))))), contourpen2, MidArrow(6));
draw(shift(SHIFT+0,off)*((-L,0)--(L,0)), contourpen1, MidArrow(6));
label("$C_{\infty}$",shift(SHIFT,0)*rotate(-60)*(L,0),SE);

label("($-$)", (SHIFT-L,L));

