use super::*;
pub struct Immiediate{}
use ::bsp::*;

type V2=(f32,f32);		pub type V3=(f32,f32,f32);	pub type V4=(f32,f32,f32,f32);
type M33=(V3,V3,V3);	pub type M43=(V3,V3,V3,V3);	pub type M44=(V4,V4,V4,V4);

pub type Color=u32;//todo-newtype, then we can do conversions with 'from'

pub fn begin(){
	unsafe{
	glUseProgram(0);
	glMatrixMode(GL_PROJECTION);
	glLoadMatrixf(&matrix::identity().ax.x);
	glMatrixMode(GL_MODELVIEW);
	glLoadMatrixf(&matrix::identity().ax.x);
	}
}

pub fn end(){
}

pub fn gl_vertex(a:Vec3){
	unsafe {glVertex3f(a.x,a.y,a.z);}
}

pub fn clear(color:u32){
    unsafe {
        glClearColor(0.5, 0.5, 0.5, 1.0);
        glClear(GL_COLOR_BUFFER_BIT);
        glSetMatrixIdentity(GL_PROJECTION);
        glSetMatrixIdentity(GL_MODELVIEW);
        set_matrix(0, &matrix::identity());
    }
}

pub fn set_matrix(index:i32,m:&Mat44f){
    unsafe{
    glMatrixMode(match index{_=>GL_MODELVIEW});
    glLoadMatrixf(&m.ax.x);
    }
}

pub fn line(a:Vec3,b:Vec3){
	unsafe{
    	glBegin(GL_LINE_STRIP);
	    gl_vertex(a);
	    gl_vertex(b);
    	glEnd();
    }
}
pub fn lines_xy(vts:&Vec<(f32,f32)>,z:f32,close:bool){
    unsafe {
        glBegin(GL_LINE_STRIP);
        for v in vts {
            gl_vertex(Vec3(v.0,v.1,z));
        }
        if close {gl_vertex(Vec3(vts[0].0,vts[0].1,z))}
        glEnd();
    }
}
pub fn line_strip2(a:Vec3,b:Vec3,c:Vec3){
    line(a,b);line(b,c);
}
pub fn line_strip3(a:Vec3,b:Vec3,c:Vec3,d:Vec3){
    line(a,b);line(b,c);line(c,d);
}
pub fn rect_vertices(a:Vec3,b:Vec3)->(Vec3,Vec3,Vec3,Vec3) {
    (Vec3(a.x, a.y, a.z),
    Vec3(b.x, a.y, a.z),
    Vec3(a.x, b.y, a.z),
    Vec3(b.x, b.y, a.z))
}

pub fn rect(a:Vec3,b:Vec3) {
    let (aa,ab,ba,bb)=rect_vertices(a,b);
    quad(aa,ab,ba,bb);
}
pub fn rect_outline(a:Vec3,b:Vec3) {
    let (aa,ab,ba,bb)=rect_vertices(a,b);
    line(aa,ab);line(ab,bb);line(bb,ba);line(ba,aa);

}
pub fn rect_corners_xy(a:Vec3,b:Vec3,corner_fraction:f32) {
    let dx=b.x-a.x;
    let dy=b.y-a.y;
    let cx=dx*corner_fraction;
    let cy=dy*corner_fraction;

    let aa=Vec3(a.x, a.y, a.z);
    let ab=Vec3(a.x, b.y, a.z);
    let ba=Vec3(b.x, a.y, a.z);
    let bb=Vec3(b.x, b.y, a.z);

    line_strip2( Vec3(a.x, a.y+cy,a.z), aa, Vec3(a.x+cx, a.y,a.z));
    line_strip2( Vec3(a.x, b.y-cy,a.z), ab, Vec3(a.x+cx, b.y,a.z));

    line_strip2( Vec3(b.x, a.y+cy,a.z), ba, Vec3(b.x-cx, a.y,a.z));
    line_strip2( Vec3(b.x, b.y-cy,a.z), bb, Vec3(b.x-cx, b.y,a.z));
}
pub fn circle_point_xy(a:f32, r:f32)->Vec3{
    Vec3(a.cos()*r,a.sin()*r,0.0)
}
pub fn circle_point_xz(a:f32, r:f32)->Vec3{
    Vec3(a.cos()*r,0.0,a.sin()*r)
}
pub fn circle_point_yz(a:f32, r:f32)->Vec3{
    Vec3(0.0,a.cos()*r,a.sin()*r)
}
pub fn curve_open<F:Fn(f32)->Vec3>(f:F, start:f32,end:f32,segs:i32){
    let mut i=0;
    let mut t=0.0f32;
    let dt=(end-start)/(segs as f32);
    let mut vs=f(t);
    while i<=segs{
        t+=dt;
        let ve=f(t);
        line(vs,ve);
        vs=ve;
    }
}
pub fn curve_closed<F:Fn(f32)->Vec3>(f:F, start:f32,end:f32,segs:i32){
    // todo,un-cutpaste.
    let mut i=0;
    let mut t=0.0f32;
    let dt=(end-start)/(segs as f32);
    let mut vs=f(t);
    let vstart=vs;
    while i<segs{
        t+=dt;
        let ve=f(t);
        line(vs,ve);
        vs=ve;
    }
    line(vs,vstart);//close it.
}
pub fn circle_xy(centre:Vec3, r:f32){
    curve_closed(|a|circle_point_xy(a,r), 0.0f32, 6.248f32, 32);
}
pub fn circle_yz(centre:Vec3, r:f32){
    curve_closed(|a|circle_point_yz(a,r), 0.0f32, 6.248f32, 32);
}
pub fn circle_xz(centre:Vec3, r:f32){
    curve_closed(|a|circle_point_xz(a,r), 0.0f32, 6.248f32, 32);
}
// wireframe sphereoid, aprox by axis circles.
pub fn sphere(centre:Vec3, r:f32){
    circle_xy(centre,r);
    circle_xz(centre,r);
    circle_yz(centre,r);
}


pub fn quad(a:Vec3,b:Vec3,c:Vec3,d:Vec3){
	unsafe{
		glBegin(GL_TRIANGLE_STRIP);
        gl_vertex(a);
        gl_vertex(b);
        gl_vertex(c);
        gl_vertex(d);
		glEnd();
	}
}
pub fn triangle(a:Vec3,b:Vec3,c:Vec3){
	unsafe{
		glBegin(GL_TRIANGLE_STRIP);
		gl_vertex(a);
		gl_vertex(b);
		gl_vertex(c);
		glEnd();
	}
}

pub fn sprite(a:Vec3,r:f32,color:Vec4) {
// todo - pointsprite buffer, camera facing, etc.
	unsafe{
		glBegin(GL_TRIANGLE_STRIP);
		glColor4f(color.x,color.y,color.z,color.w);
        glTexCoord2f(0.0,0.0);
		glVertex3f(a.x-r,a.y-r,a.z);
        glTexCoord2f(1.0,0.0);
		glVertex3f(a.x+r,a.y-r,a.z);
        glTexCoord2f(0.0,1.0);
		glVertex3f(a.x-r,a.y+r,a.z);
        glTexCoord2f(1.0,1.0);
		glVertex3f(a.x+r,a.y+r,a.z);
		glEnd();
	}
}

pub fn axes(a:Vec3, c:Color){
}








// thin wrapper around glut display functions,
// intended for simple debug graphics.


pub fn	image(size:(u32,u32),image:&Vec<u32>, pos:(f32,f32)) {
	unsafe {
//		let (tx,image)= self.get_texture_image(i);

		glRasterPos2f(pos.0,pos.1);
		glDrawPixels(size.0 as GLsizei,size.1 as GLsizei, GL_RGBA, GL_UNSIGNED_BYTE, image.as_ptr() as *const  c_void);
		glFlush();
	}
}

extern "C"{static mut glutStrokeMonoRoman:&'static u8;}
extern "C"{static mut glutBitmap8By13:&'static u8;}
extern "C" {
    fn glutBitmapCharacter(_:*const c_void, _:c_char)->();
    fn glRasterPos3f(_:f32,_:f32,_:f32);
}
pub fn char_at(&(x,y,z):&V3,color:u32, k:char) {
    unsafe {
        self::gl_color(color);
        glRasterPos3f(x, y, z);

        glutBitmapCharacter((&glutBitmap8By13)
                                as *const &u8 as *const *const u8 as *const c_void, k as c_char);
    }
}
pub fn string_at(&(x,y,z):&V3,color:u32, text:&str){
    unsafe {
        self::gl_color(color);
        glRasterPos3f(x, y, z);

        for c in text.chars() {
            glutBitmapCharacter((&glutBitmap8By13)
                                    as *const &u8 as *const *const u8 as *const c_void, c as c_char);
        }
    }
}

pub fn main_mode_text(s:&str){
    identity();
    string_at(&(-0.5f32,0.9f32,0.0f32),0xffc0c0c0,s);
}



pub fn identity(){
    unsafe {
        glMatrixMode(GL_PROJECTION);
        glLoadIdentity();
        glMatrixMode(GL_MODELVIEW);
        glLoadIdentity();
    }
}

pub fn get_format(bytes_per_pixel:u32, alpha_bits:u32)->(GLenum,GLenum) {
	match (bytes_per_pixel,alpha_bits) {
		(4,_) => (GL_RGBA,GL_UNSIGNED_BYTE),
		(3,0) => (GL_RGB,GL_UNSIGNED_BYTE),
		(2,4) => (GL_RGBA, GL_UNSIGNED_SHORT_4_4_4_4),
		(2,1) => (GL_RGBA, GL_UNSIGNED_SHORT_5_5_5_1),
		(2,0) => (GL_RGB, GL_UNSIGNED_SHORT_5_6_5),
		(1,8) => (GL_RGB, GL_UNSIGNED_BYTE_3_3_2),	// todo:should mean compressed.
		(1,_) => (GL_RGB, GL_UNSIGNED_BYTE_3_3_2),	// todo:should mean compressed.
		_ => (GL_RGBA, GL_UNSIGNED_BYTE)
	}
}

pub fn create_texture<Texel>((w,h):(u32,u32), raw_pixels:&Vec<Texel>, alpha_bits:u32)->GLuint {
	// todo: generic over format, u16->1555, u32->8888 u8->dxt5 and so on
	unsafe {
		let (fmt,fmt2)=get_format(size_of::<Texel>() as u32, alpha_bits);
		assert!(w*h==raw_pixels.len() as u32);
		let mut tx:[GLuint;1]=[0;1];
		glGenTextures(1,tx.as_mut_ptr());
		glBindTexture(GL_TEXTURE_2D,tx[0]);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER,GL_LINEAR as GLint);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER,GL_LINEAR as GLint);
		glTexImage2D(GL_TEXTURE_2D, 0, fmt as GLint, w as GLsizei,h as GLsizei, 0, fmt, fmt2, raw_pixels.as_ptr() as *const c_void); 
		tx[0]
	}
}
pub fn line_c(&(x0,y0,z0):&(f32,f32,f32),&(x1,y1,z1):&(f32,f32,f32), color:u32) {
	unsafe {
		glBegin(GL_LINES);
		gl_color(color);
		glVertex3f(x0,y0,z0); glVertex3f(x1,y1,z1);
		glEnd();
	}
}
pub fn cuboid(size:&V3,color:u32) {
	cuboid_aabb_at(&(0.0f32,0.0f32,0.0f32),size,color)
}

pub fn cuboid_aabb_at(centre:&V3, size:&V3, color:u32){
	let &(x,y,z)=size;
	let pts=(0..8).map(|n|(if n&1==0{-x}else{x},if n&2==0{-y}else{y},if n&4==0{-z}else{z})).collect::<Vec<_>>();
	let edges=[(0,1),(2,3),(0,2),(1,3), (0,4),(1,5),(2,6),(3,7), (4,5),(6,7),(4,6),(5,7)];
	unsafe {
		glBegin(GL_LINES);
		gl_color(color);
		for e in edges.iter(){
			gl_vertex_v3(&pts[e.0]);
			gl_vertex_v3(&pts[e.1]);
		}

		glEnd();
	}
}

pub fn v3isometric(&(x,y,z):&(f32,f32,f32))->(f32,f32,f32) {(x+y,z+(x-y)*0.5, z)}

pub fn line_iso(v0:&V3,v1:&V3,color:u32, scale:f32) {
	line_c(&v3isometric(&v3scale(v0,scale)),&v3isometric(&v3scale(v1,scale)), color)
}
pub unsafe fn gl_vertex_v3(&(x,y,z):&V3) {
	glVertex3f(x,y,z);
}
pub unsafe fn gl_tex0(&(u,v):&(f32,f32)) {
	glTexCoord2f(u,v);
}
pub fn unpack_sub(color:u32,org:u32,s:f32)->V4{
    let r=((color)-org)&255;
    let g=((color>>8)-org)&255;
    let b=((color>>16)-org)&255;
    let a=((color>>24)-org)&255;
    (r as f32*s,g as f32*s,b as f32*s,a as f32*s)
}
fn pack_sub((r,g,b,a):V4,org:u32,f:f32)->Color{
    let o=org as i32;
    let v0=clamp((r*f)as i32+o,(0,255))as u32;
    let v1=clamp((g*f)as i32+o,(0,255))as u32;
    let v2=clamp((b*f)as i32+o,(0,255))as u32;
    let v3=clamp((a*f)as i32+o,(0,255))as u32;
    v0|(v1<<8)|(v2<<16)|(v3<<24)
}
fn unpack(color:Color)->V4{unpack_sub(color,0, 1.0f32/255.0f32)}
fn pack(color:V4)->Color{pack_sub(color,0, 255.0f32)}

pub unsafe fn gl_color(color:Color) {
    let (r,g,b,a)=unpack(color);
	glColor3f(r,g,b);
}
pub fn tri_iso(v0:&V3,v1:&V3,v2:&V3,color:u32, scale:f32 ) {
	let tv0=v3isometric(&v3scale(v0,scale));
	let tv1=v3isometric(&v3scale(v1,scale));
	let tv2=v3isometric(&v3scale(v2,scale));
	unsafe {
		glBegin(GL_TRIANGLES);
		gl_color(color);
		gl_vertex_v3(&tv0);	
		gl_vertex_v3(&tv1);	
		gl_vertex_v3(&tv2);	
		glEnd();
	}
}
pub fn tri_iso_tex(
		(v0,uv0):(&V3,V2), 
		(v1,uv1):(&V3,V2),
		(v2,uv2):(&V3,V2),
		color:u32, scale:f32 ) {
	let tv0=v3isometric(&v3scale(v0,scale));
	let tv1=v3isometric(&v3scale(v1,scale));
	let tv2=v3isometric(&v3scale(v2,scale));
	unsafe {
		glBegin(GL_TRIANGLES);
		gl_color(color);
		gl_tex0(&uv0);
		gl_vertex_v3(&tv0);
		gl_tex0(&uv1);
		gl_vertex_v3(&tv1);	
		gl_tex0(&uv2);
		gl_vertex_v3(&tv2);	
		glEnd();
	}
}

static s_init:bool=false;
pub unsafe fn init() {
	//dump!();
	if !s_init{
		let mut argc:c_int=0;
		let argv:Vec<*const c_char> =Vec::new();
		glutInit((&mut argc) as *mut c_int,0 as *const *const c_char );

		glutInitDisplayMode(GLUT_RGBA|GLUT_DOUBLE);
		glutInitWindowSize(1024,1024);
		let win=glutCreateWindow(c_str("testbed"));
		glutDisplayFunc(draw_null as *const u8);
		glDrawBuffer(GL_BACK);
		glClear(GL_COLOR_BUFFER_BIT|GL_DEPTH_BUFFER_BIT);
	}
	//dump!();
}
pub unsafe fn draw_null(){
	glFlush();
}
pub unsafe fn show() {
	glFlush();
}

// todo: some malarchy for quit key.
// you could even buffer everything up and allow
// teh user to zoom in and out on a 2d image.
pub fn draw_win_loop() {
	unsafe {
//		while true {
		glutMainLoop();
//		}
	}
}

pub fn grid_xy_scaled_at(s:f32, subdiv:i32, color:u32, (cx,cy,cz):(f32,f32,f32)){

	let f:f32=s/(subdiv as f32);
	let z=0.0f32;
	for i in -subdiv..(subdiv+1){
		let x=i as f32 * f;
		line_c(&(cx-s,cy+x,cz+z),&(cx+s,cy+x,cz+z),color);
		line_c(&(cx+x,cy-s,cz+z),&(cx+x,cy+s,cz+z),color);
	}
}
pub fn grid_xz_scaled_at(s:f32, subdiv:i32, color:u32, (cx,cy,cz):(f32,f32,f32)){
	let f:f32=s/(subdiv as f32);
	let y=0.0f32;
	for i in -subdiv..subdiv+1{
		let x=i as f32 * f;
		line_c(&(cx-s,cy+y,cz+x),&(cx+s,cy+y,cz+x),color);
		line_c(&(cx+x,cy+y,cz-s),&(cx+x,cy+y,cz+s),color);
	}
}
pub fn grid_yz_scaled_at(s:f32, subdiv:i32, color:u32, (cx,cy,cz):(f32,f32,f32)){
	let f:f32=s/(subdiv as f32);
	let z=0.0f32;
	for i in -subdiv..subdiv+1{
		let f=i as f32 * f;
		line_c(&(cx,cy-s,cz+f), &(cx,cy+s,cz+f), color);
		line_c(&(cx,cy+f,cz-s), &(cx,cy+f,cz+s), color);
	}
}

pub fn grid_xy_at(centre:(f32,f32,f32)){
	// multiscale debug grids on the origin
	grid_xy_scaled_at(0.01f32,10,0x80a0a0a0,  centre);
	grid_xy_scaled_at(0.1f32,10,0x80a0a0a0, centre);
	grid_xy_scaled_at(1.0f32,10,0x80a0a0a0,  centre);
	grid_xy_scaled_at(100.0f32,10,0x80a0a0a0, centre);
}
pub fn grid_xz_at(centre:(f32,f32,f32)){
	// multiscale debug grids on the origin
	grid_xz_scaled_at(0.01f32,10,0x80a0a0a0, centre);
	grid_xz_scaled_at(0.1f32,10,0x80a0a0a0, centre);
	grid_xz_scaled_at(1.0f32,10,0x80a0a0a0, centre);
	grid_xz_scaled_at(100.0f32,10,0x80a0a0a0, centre);
}
pub fn grid_yz_at(centre:(f32,f32,f32)){
	// multiscale debug grids on the origin
	grid_yz_scaled_at(0.01f32,10,0x80a0a0a0, centre);
	grid_yz_scaled_at(0.1f32,10,0x80a0a0a0, centre);
	grid_yz_scaled_at(1.0f32,10,0x80a0a0a0, centre);
	grid_yz_scaled_at(100.0f32,10,0x80a0a0a0, centre);
}
pub fn grid_xy(){
	grid_xy_at((0.0f32,0.0f32,0.0f32));
}
pub fn grid_xz(){
	grid_xz_at((0.0f32,0.0f32,0.0f32));
}
pub fn grid_yz(){
	grid_yz_at((0.0f32,0.0f32,0.0f32));
}
pub fn grid(a:Axes){
	match a{
		Axes::XY=>grid_xy(),
		Axes::YZ=>grid_yz(),
		Axes::XZ=>grid_xz(),
	}
}

pub unsafe fn set_texture(tex_unit:i32, tex_id:GLuint) {
	assert!(tex_unit>=0 && tex_unit<16);
	glActiveTexture((GL_TEXTURE0 as int +tex_unit as int) as GLenum);
	if tex_id>0 {
		glEnable(GL_TEXTURE_2D);
		glBindTexture(GL_TEXTURE_2D, tex_id);
	} else {
		glDisable(GL_TEXTURE_2D);
	}
}


pub fn random_color3(a:uint,b:uint,c:uint)->u32 {
	(a*b*c ^(a<<3)^(b<<8)*(c<<2)^(a<<19)^(b<<22)*(c<<13) )as u32
}
pub fn random_color(a:uint)->u32 {
	(a^(a<<3)^(a<<8)*(a<<2)^(a<<19)^(a<<22)*(a<<13) )as u32
}

pub enum Prim {
	Points,
	Lines,
	Tris,
	TriStrips,
}

pub trait MatrixStack{
	fn push(&mut self,m:&Mat44f);
	fn get(&self)->Mat44f;
	fn pop(&mut self);

	fn push_mul(&mut self,m:&Mat44f) { let mt=self.get().mul(m); self.push(&mt);}
	fn push_set(&mut self,m:&Mat44f) { self.push(m);}
	fn replace_mul(&mut self,m:&Mat44f) { self.pop(); self.push_mul(m); }
	fn replace_set(&mut self, m:&Mat44f) { self.pop(); self.push_set(m);}

	fn depth(&self)->usize;
}

/// thin wrapper for opengl calls
pub trait Renderer :Sized {

	fn set_matrix(&mut self,index:i32,m:&Mat44f);
	fn begin(&mut self, Prim);
	fn vertex(&mut self, p:&V3);
	fn color(&mut self, c:u32);
	fn vertexc(&mut self, p:&V3, c:u32) {self.color(c);self.vertex(p);}
	fn end(&mut self);
	fn draw<T:Render>(&mut self, a:&T){a.draw_into(self);}
}

pub trait Render :Sized{
	fn draw_into<R:Renderer>(&self,&mut R);
}

struct GlRenderer {
}

struct SceneCapture{

    lines:Vec<[(V3,V3,Color);2]>,
}

//impl Renderer for GlRenderer {
//}





