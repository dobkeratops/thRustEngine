use super::*;
pub struct Immiediate{}
use ::bsp::*;

/// debug graphics.. GL immediate mode rendering utilities

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
pub fn gl_vertex_tc(pos:&Vec3,uv:&Vec2,col:u32){
	unsafe {
		gl_color(col);
		glTexCoord2f(uv.x,uv.y);
		pos.gl_vertex3();
	}
}
pub fn v3_gl_vertex(a:V3){
	unsafe {glVertex3f(a.0,a.1,a.2);}
}
pub fn gl_vertex2<V:HasXY<Elem=f32>>(a:&V){
	unsafe {glVertex3f(a.x(),a.y(),0.0f32);}
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

pub fn line<V:RenderVertex>(a:&V,b:&V){
	unsafe{
    	glBegin(GL_LINE_STRIP);
	    a.render_vertex();
	    b.render_vertex();
    	glEnd();
    }
}
pub fn lines_xy<V:HasXY<Elem=f32>>(vts:&Vec<V>,z:f32,close:bool){
    unsafe {
        glBegin(GL_LINE_STRIP);
        for v in vts {
            gl_vertex(Vec3(v.x(),v.y(),z));
        }
        if close {gl_vertex(Vec3(vts[0].x(),vts[0].y(),z))}
        glEnd();
    }
}
pub fn line_strip2<V:RenderVertex>(a:&V,b:&V,c:&V){
	unsafe {
	glBegin(GL_LINE_STRIP);
	a.render_vertex();
	b.render_vertex();
	c.render_vertex();
	glEnd();
	}	
}
pub fn line_strip2_c<V3:HasXYZ<Elem=f32>>(a:&V3,b:&V3,c:&V3,color:Color){
	line_strip2(&VertexCRef{pos:a,color:&color},&VertexCRef{pos:b,color:&color},&VertexCRef{pos:c,color:&color});
}
pub fn line_strip3<V3:HasXYZ<Elem=f32>>(a:&V3,b:&V3,c:&V3,d:&V3){
    line(a,b);line(b,c);line(c,d);
}
pub fn rect_vertices_xy<V3:HasXYZ<Elem=f32>>(a:&V3,b:&V3)->(V3,V3,V3,V3) {
    (V3::from_xyz(a.x(), a.y(), a.z()),
     V3::from_xyz(b.x(), a.y(), a.z()),
     V3::from_xyz(a.x(), b.y(), a.z()),
     V3::from_xyz(b.x(), b.y(), a.z()))
}
pub fn rect_vertices_v2<V2:HasXY<Elem=f32>>(a:&V2,b:&V2)->(V2,V2,V2,V2) {
    (V2::from_xy(a.x(), a.y()),
     V2::from_xy(b.x(), a.y()),
     V2::from_xy(a.x(), b.y()),
     V2::from_xy(b.x(), b.y()))
}

pub fn crosshair_xy(a:&Vec3, inner:f32, outer:f32, color:Color){
	line_c(&a.vadd_x(-inner), &a.vadd_x(-outer), color);
	line_c(&a.vadd_x(inner), &a.vadd_x(outer), color);
	line_c(&a.vadd_y(-inner), &a.vadd_y(-outer), color);
	line_c(&a.vadd_y(inner), &a.vadd_y(outer), color);
}

pub fn crosshair_z(a:&Vec3, inner:f32, outer:f32, color:Color){
	line_c( &a.vadd_z(-inner), &a.vadd_z(-outer), color);
	line_c( &a.vadd_z(inner), &a.vadd_z(outer), color);
}

pub fn crosshair_xyz(a:&Vec3, inner:f32, outer:f32, color:Color){
	crosshair_xy(a, inner,outer,color);
	crosshair_z(a,inner,outer,color);	
}

pub fn rect_outline<V3:HasXYZ<Elem=f32>>(a:&V3,b:&V3) {
    let (aa,ab,ba,bb)=rect_vertices_xy(a,b);
    quad_outline(&aa,&ab,&bb,&ba);
}

pub fn rect_outline_v2<V2:HasXY<Elem=f32>>(a:&V2,b:&V2,color:u32) {
    let (aa,ab,ba,bb)=rect_vertices_v2(a,b);
	unsafe{
	glBegin(GL_LINE_STRIP);
	gl_color(color);
	gl_vertex2(&aa);

	gl_color(color);
	gl_vertex2(&ab);

	gl_color(color);
	gl_vertex2(&bb);

	gl_color(color);
	gl_vertex2(&ba);

	gl_color(color);
	gl_vertex2(&aa);
	glEnd();
	}
}

pub fn box_corners(a:&Vec3,b:&Vec3,f:f32,c:Color){
	rect_corners_xy(a, &Vec3(b.x,b.y,a.z), f,c);
	rect_corners_xy(a, &Vec3(b.x,b.y,b.z), f,c);
	line_ends(&Vec3(a.x,a.y,a.z),&Vec3(a.x,a.y,b.z),f,c);
	line_ends(&Vec3(b.x,a.y,a.z),&Vec3(b.x,a.y,b.z),f,c);
	line_ends(&Vec3(a.x,b.y,a.z),&Vec3(a.x,b.y,b.z),f,c);
	line_ends(&Vec3(b.x,b.y,a.z),&Vec3(b.x,b.y,b.z),f,c);
}

fn line_ends(a:&Vec3,b:&Vec3,f:f32,color:Color){
	line_c(a,&a.vlerp(b,f),color);
	line_c(b,&b.vlerp(a,f),color);
}

pub fn rect_corners_xy(a:&Vec3,b:&Vec3,corner_fraction:f32,c:Color) {
    let dx=b.x-a.x;
    let dy=b.y-a.y;
    let cx=dx*corner_fraction;
    let cy=dy*corner_fraction;

    let aa=Vec3(a.x, a.y, a.z);
    let ab=Vec3(a.x, b.y, a.z);
    let ba=Vec3(b.x, a.y, a.z);
    let bb=Vec3(b.x, b.y, a.z);

    line_strip2_c( &Vec3(a.x, a.y+cy,a.z), &aa, &Vec3(a.x+cx, a.y,a.z),c);
    line_strip2_c( &Vec3(a.x, b.y-cy,a.z), &ab, &Vec3(a.x+cx, b.y,a.z),c);

    line_strip2_c( &Vec3(b.x, a.y+cy,a.z), &ba, &Vec3(b.x-cx, a.y,a.z),c);
    line_strip2_c( &Vec3(b.x, b.y-cy,a.z), &bb, &Vec3(b.x-cx, b.y,a.z),c);
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
        line(&vs,&ve);
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
        line(&vs,&ve);
        vs=ve;
    }
    line(&vs,&vstart);//close it.
}

pub fn circle_xy(centre:&Vec3, r:f32){
    curve_closed(|a|circle_point_xy(a,r), 0.0f32, PI, 32);
}
pub fn arc_xy<A:Angle>(centre:&Vec3,r:f32, a0:A,a1:A){
	curve_open(|a|circle_point_xy(a,r), a0.to_radians(),a1.to_radians(),32);
}
pub fn circle_yz(centre:&Vec3, r:f32){
    curve_closed(|a|circle_point_yz(a,r), 0.0f32, 6.248f32, 32);
}
pub fn arc_yz<A:Angle>(centre:&Vec3,r:f32, a0:A,a1:A){
	curve_open(|a|circle_point_yz(a,r), a0.to_radians(),a1.to_radians(),32);
}
pub fn circle_xz(centre:&Vec3, r:f32){
    curve_closed(|a|circle_point_xz(a,r), 0.0f32, 6.248f32, 32);
}
pub fn arc_xz<A:Angle>(centre:&Vec3,r:f32, a0:A,a1:A){
	curve_open(|a|circle_point_xz(a,r), a0.to_radians(),a1.to_radians(),32);
}

// wireframe sphereoid, aprox by axis circles.
pub fn sphere(centre:&Vec3, r:f32){
    circle_xy(centre,r);
    circle_xz(centre,r);
    circle_yz(centre,r);
}

// render to the global GL renderer
// TODO - passes, states..
trait GlRender {
	fn gl_render(&self);
}

impl<V:HasXYZ<Elem=f32>> GlRender for Line<V,Color> {
	fn gl_render(&self){
		line_c(&self.vertex[0],&self.vertex[1],self.attr)
	}
}


/*
pub fn quad<V:HasXYZ<Elem=f32>>(a:&V,b:&V,c:&V,d:&V){
	unsafe{
		glBegin(GL_TRIANGLE_STRIP);
        a.gl_vertex3();
        a.gl_vertex3();
        a.gl_vertex3();
        a.gl_vertex3();
		glEnd();
	}
}
*/
// todo - macro..
pub struct VertexCT{
	pos:Vec3,tex:Vec2,color:u32
}

pub struct VertexNCT{
	pos:Vec3,norm:Vec3,tex:Vec2,color:Color
}

pub struct VertexC{
	pos:Vec3, color:Color,
}
pub struct VertexCRef<'a,V3:HasXYZ+'a>{
	pos:&'a V3, color:&'a Color,
}
pub struct Vertex{
	pos:Vec3,
}

struct Line<V,A=()> {
	vertex:[V;2],
	attr:A
}
struct Triangle<V,A=()> {
	vertex:[V;3],
	attr:A
}
struct Quad<V,A=()> {
	vertex:[V;4],
	attr:A
}


pub trait RenderVertex{
	fn render_vertex(&self);
}
impl RenderVertex for VertexCT{
	fn render_vertex(&self){
		unsafe {
		gl_color(self.color);
		glTexCoord2f(self.tex.x,self.tex.y);
		self.pos.gl_vertex3();
		}
	}
}
impl RenderVertex for VertexC{
	fn render_vertex(&self){
		unsafe {
			gl_color(self.color);	
			self.pos.gl_vertex3();
		}
	}
}

impl<'a,V:HasXYZ<Elem=f32>> RenderVertex for VertexCRef<'a,V>{
	fn render_vertex(&self){
		unsafe {
			gl_color(*self.color);	
			self.pos.gl_vertex3();
		}
	}
}

impl RenderVertex for VertexNCT{
	fn render_vertex(&self){
		unsafe{
		gl_color(self.color);
		glTexCoord2f(self.tex.x,self.tex.y);
		glNormal3f(self.norm.x,self.norm.y,self.norm.z);
		self.pos.gl_vertex3();
		}
	}
}

impl<V:HasXYZ<Elem=f32>> RenderVertex for V {
	fn render_vertex(&self){
		unsafe {glVertex3f(self.x(),self.y(),self.z());}
	}
}

pub fn quad<V:RenderVertex>(v0:&V,v1:&V,v2:&V,v3:&V){
	unsafe {
	glBegin(GL_TRIANGLE_STRIP);
	v0.render_vertex();
	v1.render_vertex();
	v2.render_vertex();
	v3.render_vertex();
	glEnd();
	}
}

pub fn rect_tex_c_crop<V2:HasXY<Elem=f32>>(a:&V2,b:&V2,z:f32,col:u32,uv0:&V2,uv1:&V2){
	let x0=a.x();
	let y0=a.y();
	let x1=b.x();
	let y1=b.y();
	let u0=uv0.x();
	let v0=uv0.y();
	let u1=uv1.x();
	let v1=uv1.y();
	unsafe {
	glBegin(GL_TRIANGLE_STRIP);
	gl_vertex_tc(&Vec3(x0,y0,z), &Vec2(0.0,0.0),col);
	gl_vertex_tc(&Vec3(x1,y0,z), &Vec2(0.0,1.0),col);
	gl_vertex_tc(&Vec3(x0,y1,z), &Vec2(1.0,0.0),col);
	gl_vertex_tc(&Vec3(x1,y1,z), &Vec2(1.0,1.0),col);
	glEnd();
	}
}


pub fn rect_tex<V:HasXY<Elem=f32>>(a:&V,b:&V,z:f32){
	rect_tex_c_crop(a,b,z,0xffffffff,&V::from_xy(0.0f32,1.0f32),&V::from_xy(1.0f32,0.0f32));
}
pub fn rect_xy_tex(a:&Vec2,b:&Vec2,z:f32){
	rect_tex_c_crop(&Vec2(a.x,a.y), &Vec2(b.x,b.y),z, 0xffffffff, &Vec2(0.0f32,1.0f32), &Vec2(1.0f32,0.0f32) );
}

pub fn quad_outline<V:HasXYZ<Elem=f32>>(aa:&V,ab:&V,bb:&V,ba:&V) {
    line(aa, ab);
    line(ab, bb);
    line(bb, ba);
    line(ba, aa);
}

pub fn triangle<V:HasXYZ<Elem=f32>>(a:&V,b:&V,c:&V){
	unsafe{
		glBegin(GL_TRIANGLE_STRIP);
		a.gl_vertex3();
		b.gl_vertex3();
		c.gl_vertex3();
		glEnd();
	}
}


pub fn sprite_at(a:&Vec3,r:f32,color:Color) {
// todo - pointsprite buffer, camera facing, etc.
	rect_tex_c_crop(
		&Vec2(a.x, a.y),&Vec2(a.x, a.y), a.z,
		0xffffffff,
		&Vec2(0.0f32,1.0f32), &Vec2(1.0f32,0.0f32));
/*
	unsafe{
		glBegin(GL_TRIANGLE_STRIP);
		glColor4f(color.x,color.y,color.z,color.w);
        glTexCoord2f(0.0,0.0);
		glVertex3f(a.x-r,a.y-r,a.z);

		glColor4f(color.x,color.y,color.z,color.w);
        glTexCoord2f(1.0,0.0);
		glVertex3f(a.x+r,a.y-r,a.z);

		glColor4f(color.x,color.y,color.z,color.w);
        glTexCoord2f(0.0,1.0);
		glVertex3f(a.x-r,a.y+r,a.z);

		glColor4f(color.x,color.y,color.z,color.w);
        glTexCoord2f(1.0,1.0);
		glVertex3f(a.x+r,a.y+r,a.z);
		glEnd();
	}
*/
}

pub fn axes_color_coded(centre:&Vec3, len:f32,xc:Color,yc:Color,zc:Color){
	line_c(centre, &centre.vadd_x(len), xc);
	line_c(centre, &centre.vadd_y(len), yc);
	line_c(centre, &centre.vadd_z(len), zc);
}
pub fn axes(centre:&Vec3,len:f32){
	axes_color_coded(centre,len,0xff0000,0x00ff00,0x0000ff);
}
pub fn axes_c(centre:&Vec3,len:f32,c:Color){
	axes_color_coded(centre,len,c,c,c);
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


#[cfg(target_os = "emscripten")]
fn glutBitmapCharacter(_:*const c_void, _:c_char)->() {
}
#[cfg(target_os = "emscripten")]
fn glRasterPos3f(_:f32,_:f32,_:f32){
}

#[cfg(target_os = "emscripten")]
pub fn char_at(&(x,y,z):&V3,color:u32, k:char) {
}
#[cfg(target_os = "emscripten")]
pub fn string_at(&(x,y,z):&V3,color:u32, text:&str){
}
#[cfg(not(target_os = "emscripten"))]
pub fn char_at(&(x,y,z):&V3,color:u32, k:char) {
    unsafe {
        self::gl_color(color);
        glRasterPos3f(x, y, z);

        glutBitmapCharacter((&glutBitmap8By13)
                                as *const &u8 as *const *const u8 as *const c_void, k as c_char);
    }
}
#[cfg(not(target_os = "emscripten"))]
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
		glTexImage2D(GL_TEXTURE_2D, 0, fmt as GLint, w as GLsizei,h as GLsizei, 0, fmt, fmt2, &raw_pixels[0]  as *const _ as _); 
		tx[0]
	}
}
pub fn line_c<V:HasXYZ<Elem=f32>>(v0:&V,v1:&V, color:u32) {
	unsafe {
		glBegin(GL_LINES);
		VertexC{pos:v0.to_vec3(),color:color}.render_vertex();
		VertexC{pos:v1.to_vec3(),color:color}.render_vertex();
		glEnd();
	}
}
pub fn arrow(vs:&V3, ve:&V3, head:f32,color:Color){
	let axis=v3sub_norm(ve,vs);
	let base=v3madd(ve,&axis,-head);
	let ofs=v3scale(&(axis.1,-axis.0,axis.2),head*0.5f32);
	let b0=v3add(&base,&ofs);
	let b1=v3sub(&base,&ofs);
	unsafe{
		glBegin(GL_LINE_STRIP);
		gl_color(color);
		gl_vertex3(&base);
		gl_color(color);
		gl_vertex3(&b0);
		gl_color(color);
		gl_vertex3(ve);
		gl_color(color);
		gl_vertex3(&b1);
		gl_color(color);
		gl_vertex3(&base);
		gl_color(color);
		gl_vertex3(vs);
		glEnd();
	}
}

pub fn arrow_mid(vs:&Vec3,ve:&Vec3, head:f32,color:Color){
}

pub fn circle_fill_xy_c<V3:HasXYZ<Elem=f32>>(pos:&V3,r2:f32, color:u32){
    //actually a square first.
	let r=r2*0.5f32;
    let x=pos.x(); let y=pos.y(); let z=pos.z();
    unsafe{
        glBegin(GL_TRIANGLE_STRIP);
        gl_color(color);
        glVertex3f(x-r,y-r2,z);

        gl_color(color);
        glVertex3f(x+r,y-r2,z);

        gl_color(color);
        glVertex3f(x-r2,y-r,z);

        gl_color(color);
        glVertex3f(x+r2,y-r,z);

        gl_color(color);
        glVertex3f(x-r2,y+r,z);

        gl_color(color);
        glVertex3f(x+r2,y+r,z);

        gl_color(color);
        glVertex3f(x-r,y+r2,z);

        gl_color(color);
        glVertex3f(x+r,y+r2,z);

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
		for e in edges.iter(){
		gl_color(color);
			gl_vertex3(&pts[e.0]);
		gl_color(color);
			gl_vertex3(&pts[e.1]);
		}

		glEnd();
	}
}

pub fn v3isometric(&(x,y,z):&(f32,f32,f32))->(f32,f32,f32) {(x+y,z+(x-y)*0.5, z)}

pub fn line_iso(v0:&V3,v1:&V3,color:u32, scale:f32) {
	line_c(&v3isometric(&v3scale(v0,scale)),&v3isometric(&v3scale(v1,scale)), color)
}

pub fn gl_vertex3<V:GlVertex3>(v:&V){v.gl_vertex3()}

pub trait GlVertex3 {
    fn gl_vertex3(&self);
}
impl<V> GlVertex3 for V where V:HasXYZ<Elem=f32>{
    fn gl_vertex3(&self){
        unsafe{glVertex3f(self.x(),self.y(),self.z())}
    }
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
fn pack_sub<V:HasXYZW<Elem=f32>>(v:&V,org:u32,f:f32)->Color{
    let o=org as i32;
	let (r,g,b,a)=(v.x(),v.y(),v.z(),v.w());
    let v0=clamp((r*f)as i32+o,(0,255))as u32;
    let v1=clamp((g*f)as i32+o,(0,255))as u32;
    let v2=clamp((b*f)as i32+o,(0,255))as u32;
    let v3=clamp((a*f)as i32+o,(0,255))as u32;
    v0|(v1<<8)|(v2<<16)|(v3<<24)
}
pub fn unpack(color:Color)->V4{unpack_sub(color,0, 1.0f32/255.0f32)}
pub fn pack<V:HasXYZW<Elem=f32>>(color:&V)->Color{pack_sub(color,0, 255.0f32)}

pub unsafe fn gl_color(color:Color) {
    let (r,g,b,a)=unpack(color);
	glColor4f(r,g,b,a);
}
trait GlColor {
	fn gl_color(&self);
}
impl GlColor for Color {
	fn gl_color(&self){unsafe {gl_color(*self)}}
}
impl GlColor for Vec4 {
	fn gl_color(&self){
		unsafe {glColor4f(self.x,self.y,self.z,self.w)}
	}
}

pub fn tri_iso(v0:&V3,v1:&V3,v2:&V3,color:u32, scale:f32 ) {
	let tv0=v3isometric(&v3scale(v0,scale));
	let tv1=v3isometric(&v3scale(v1,scale));
	let tv2=v3isometric(&v3scale(v2,scale));
	unsafe {
		glBegin(GL_TRIANGLES);
		gl_color(color);
		gl_vertex3(&tv0);
		gl_color(color);
		gl_vertex3(&tv1);
		gl_color(color);
		gl_vertex3(&tv2);
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
		gl_vertex3(&tv0);

		gl_color(color);
		gl_tex0(&uv1);
		gl_vertex3(&tv1);

		gl_color(color);
		gl_tex0(&uv2);
		gl_vertex3(&tv2);
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
		glBindTexture(GL_TEXTURE_2D, 0);
		glDisable(GL_TEXTURE_2D);
	}
}
pub unsafe fn set_matrix_projection(m:&Mat44) {
            glSetMatrix(GL_PROJECTION, &m.ax.x);
}
pub unsafe fn set_matrix_modelview(m:&Mat44) {
            glSetMatrix(GL_MODELVIEW, &m.ax.x);
}
pub unsafe fn set_matrix_p_mv(p:&Mat44,m:&Mat44){
	set_matrix_projection(p);
	set_matrix_modelview(m);
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





