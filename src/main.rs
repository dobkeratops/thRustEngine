//#![feature(macro_rules)]
//#![feature(default_type_params)]
#![allow(unused_variables)]
#![allow(dead_code)]
#![allow(unreachable_code)]
#![allow(unused_unsafe)]
#![allow(unused_mut)]
#![allow(non_snake_case)]
#![allow(unused_imports)]
#![allow(non_camel_case_types)]
#![allow(non_upper_case_globals)] 
#![allow(unused_variables)]
#![allow(unreachable_patterns)]
//#[warn(unused_macros)]
#![allow(unused_macros)]
#![feature(link_args)]
//#![feature(drop_types_in_const)]
//#![overflow_checks(off)]
//#![feature(link_args)]


//#![allow(non_camel_cased_types)]

#[macro_use]
pub mod r3d;
pub mod window;
use window::Flow;
mod world;
pub use r3d::*;
pub mod editor;
pub mod texture;
pub mod test;
pub mod emscripten;
pub mod shaders;
pub use texture::*;
pub use shaders::*;
pub use window::sto;
pub use window::Window;
pub use emscripten::*;
pub use std::fs::File;

#[cfg(not(target_os="emscripten"))]
extern crate image;

#[cfg(target_os="macos")]
#[link_args="-framework OpenGL -framework glut -L/usr/local/lib -F/Library/Frameworks -framework SDL2 -framework Cocoa"]
extern{}

#[cfg(target_os = "android")]
extern { fn android_logw(s:*const c_char);}

#[cfg(not(target_os = "android"))]
unsafe fn android_logw(s:*const c_char) {
	println!("{:?}",CStr::from_ptr(s).to_str());
}

fn android_logw_str(s:&str){
	unsafe {android_logw(s.as_ptr() as *const c_char)}
}
fn android_logw_string(s:&String){
	unsafe {android_logw(s.as_ptr() as *const c_char)}

}

// why is this crap not working?!
/*
#[cfg(target_os = "macos")]
#[link(name = "OpenGL", kind = "framework")]

#[cfg(target_os = "macos")]
#[link(name = "glut", kind = "framework")]

#[cfg(target_os = "macos")]
#[link(name = "CoreFoundation", kind = "framework")]


#[cfg(target_os = "macos")]
#[link(name = "GL",kind="static")]
*/
/*
#[link(name = "GLU",kind="static")]
#[link(name = "Xext",kind="static")]
#[link(name = "glut",kind="static")]
#[link(name = "GL",kind="static")]
#[link(name = "stdc++",kind="static")]
#[link(name = "hello",kind="static")]
*/
//#[cfg(target_os = "macos")]
//#[link(name = "Cocoa", kind = "framework")]
//#[link(name = "SDL2", kind = "framework")]


//#[link(name = "GLEW")]
//#[link(name = "cstuff",kind="static")]



//#define fbx_printf printf
//#define ASSERT(X)
//#include "texture.h"

/*
enum VertexAttrIndex
{
	VAI_pos=0,
	VAI_color,
	VAI_normal,
	VAI_tex0,
	VAI_tex1,
	VAI_count
}
*/
//GFX gfx;



//typedef int IndexType;
//	enum {IndexSize = sizeof(IndexType) };
//typedef	::TestVertex Vertex;
#[derive(Clone,Debug)]
pub struct	Mesh 
{
	pub vertex_size:GLsizei,
	pub vbo:GLuint,
	pub ibo:GLuint,
	pub num_vertices:GLuint,num_indices:GLuint
}

// todo - enum looked like it needed horrid casts




pub fn generate_torus_vertex(ij:uint, (num_u,num_v):(uint,uint))->self::MyVertex {
	let pi=3.14159265f32;
	let tau=pi*2.0f32;
	let (i,j)=div_rem(ij, num_u);
	let fi=i as f32 * (1.0 / num_u as f32);
	let fj=j as f32 * (1.0 / num_v as f32);

	let rx=0.125f32;
	let ry=rx*0.33f32;
	let pi=3.14159265f32;
	let tau=pi*2.0f32;
	let (sx,cx)=sin_cos(fi*tau);
	let (sy,cy)=sin_cos(fj*tau);
	let norm=Vec3(sy*cx, sy*sx, cy).vnormalize().vscale(0.1);

	MyVertex{
		pos:[(rx+sy*ry)*cx, (rx+sy*ry)*sx, ry*cy],
		color:[1.0,1.0,1.0,fj],
		norm:[norm.x,norm.y,norm.z],
		tex0:[fi*16.0, fj*2.0],
	}	
}

unsafe fn create_buffer(size:GLsizei, data:*const c_void, buffer_type:GLenum)->GLuint {
	let mut id:GLuint=0;
	glGenBuffers(1,&mut id);
	glBindBuffer(buffer_type,id);
	
	glBufferData(buffer_type, size, data, GL_STATIC_DRAW);
	// error..
	glBindBuffer(buffer_type,0);
	id
}

unsafe fn create_vertex_buffer_from_ptr(size:GLsizei, data:*const c_void)->GLuint {
	create_buffer(size,data,GL_ARRAY_BUFFER)
}
unsafe fn create_index_buffer_from_ptr(size:GLsizei, data:*const c_void)->GLuint {
	create_buffer(size,data,GL_ELEMENT_ARRAY_BUFFER)
}
unsafe fn create_vertex_buffer<T>(data:&Vec<T>)->GLuint {
	create_buffer(data.len()as GLsizei *mem::size_of::<T>() as GLsizei, as_void_ptr(&data[0]), GL_ARRAY_BUFFER)
}
unsafe fn create_index_buffer<T>(data:&Vec<T>)->GLuint {
	create_buffer(data.len()as GLsizei *mem::size_of::<T>() as GLsizei, as_void_ptr(&data[0]), GL_ELEMENT_ARRAY_BUFFER)
}


impl Mesh {
	/// create a grid mesh , TODO - take a vertex generator
	fn new_torus(num:(uint,uint))->Mesh
	{
		// TODO: 2d fill array. from_fn_2f(numi,numj, &|..|->..)
		let strip_indices = (num.0+1)*2 +2;
		let num_indices=(num.1)*strip_indices;
		let indices=vec_from_fn(num_indices as usize,
			&|ij_u|->GLuint{
				let ij = ij_u as isize;
				let (j,i1)=div_rem(ij, strip_indices as isize);
				let i=i1 as isize;
				let num0 = num.0 as isize;
				let num1 = num.1 as isize;
				let i2=cmp::min(cmp::max(i-1,0),(num0*2+1)); // first,last value is repeated - degen tri.
				let (i,dj)=div_rem(i2,2);	// i hope that inlines to >> &
				let ai= i as isize; let adj = dj as isize;
				let aj = j as isize;
				(((aj+adj)%(num1 as isize))*(num0 as isize)+(i % (num0 as isize))) as GLuint
			}
		);
		
		let num_vertices=num.0*num.1;
		let vertices=vec_from_fn(num_vertices as usize,&|i|generate_torus_vertex(i as u32,num));

 		unsafe {
			Mesh{
				num_vertices:num_vertices as GLuint,
				num_indices:num_indices as GLuint,
				vertex_size: mem::size_of_val(&vertices[0]) as GLsizei,
				vbo: create_vertex_buffer(&vertices),
				ibo: create_index_buffer(&indices)
			}
		}

	}
}

static mut g_grid_mesh:Mesh=Mesh{
	num_vertices:0,
	num_indices:0,
	vbo:-1i32 as uint,
	ibo:-1i32 as uint,
	vertex_size:0
};

type UniformIndex=GLint;





impl Mesh {
	fn	render_mesh_from_buffer(&self)
	{
		unsafe {

			let	client_state:[GLenum;3]=[GL_VERTEX_ARRAY,GL_COLOR_ARRAY,GL_TEXTURE_COORD_ARRAY];
			for &x in client_state.iter() {glEnableClientState(x);};

			glBindBuffer(GL_ARRAY_BUFFER, self.vbo);
			glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, self.ibo);

			let baseVertex=0 as *const MyVertex;
			let	stride=mem::size_of_val(&*baseVertex) as GLsizei;

			glVertexPointer(3, GL_FLOAT, stride,  0 as *const c_void);//(&(*baseVertex).pos[0]) as *f32 as *c_void);
			glColorPointer(4,GL_FLOAT, stride, 12 as *const c_void);//(&(*baseVertex).color[0]) as *f32 as *c_void);
			glTexCoordPointer(2, GL_FLOAT, stride, (12+16) as *const c_void);//(&(*baseVertex).tex0[0]) as *f32 as *c_void);
			glDrawElements(GL_TRIANGLE_STRIP, self.num_indices as GLsizei, GL_UNSIGNED_INT,0 as *const c_void);

			for &x in client_state.iter() {glDisableClientState(x);};
		}
	}
}

fn safe_set_uniform1i(loc:GLint, value:GLint) {
	// to do - validate
	unsafe {	
//		glUniform1i(loc, value);
	}
}
fn safe_set_uniform(loc:GLint, pvalue:&Vec4) {
	// to do - validate
	unsafe {	
		glUniform4fv(loc, 1, &pvalue.x as *const GLfloat);
	}
}

//Vec4 g_FogColor=Vec4::<f32>::new(0.25,0.5,0.5,1.0);
static g_fog_color:Vec4 =Vec4{x:0.25,y:0.5,z:0.5,w:1.0};
type RenderMode_t=usize;
impl Mesh {
	unsafe fn render_mesh_shader(&self, matP:&Mat44,rot_trans:&Mat44,modei:RenderMode_t,tex0i:TextureIndex,tex1i:TextureIndex)  {

		let shu=&g_shader_uniforms[modei];
		glUseProgram(g_shader_program[modei]);
		glUniformMatrix4fvARB(shu.mat_proj, 1,  GL_FALSE, &matP.ax.x);
		glUniformMatrix4fvARB(shu.mat_model_view, 1, GL_FALSE, &rot_trans.ax.x);

		
		let clientState:[GLenum;3]=[GL_VERTEX_ARRAY,GL_COLOR_ARRAY,GL_TEXTURE_COORD_ARRAY];

		glBindBuffer(GL_ARRAY_BUFFER, self.vbo);
		glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, self.ibo);

		let vsa=&g_vertex_shader_attrib[modei];
		let shu=&g_shader_uniforms[modei];

		glEnableVertexAttribArray(VertexAttrIndex::VAI_pos.into());
		glEnableVertexAttribArray(VertexAttrIndex::VAI_color.into());
		glEnableVertexAttribArray(VertexAttrIndex::VAI_tex0.into());
		glEnableVertexAttribArray(VertexAttrIndex::VAI_norm.into());

		safe_set_uniform1i(shu.tex0, 0);
		safe_set_uniform1i(shu.tex1, 1);
		safe_set_uniform(shu.specular_dir, &Vec4(0.032,0.707f32,0.707f32,0.0f32));
		safe_set_uniform(shu.specular_color, &Vec4(1.0f32,0.75f32,0.5f32,0.0f32));
		safe_set_uniform(shu.ambient, &Vec4(0.25f32,0.25f32,0.25f32,1.0f32));
		safe_set_uniform(shu.diffuse_dx, &Vec4(0.1f32,0.0f32,0.25f32,0.0f32));
		safe_set_uniform(shu.diffuse_dy, &Vec4(0.3f32,0.25f32,0.5f32,0.0f32));
		safe_set_uniform(shu.diffuse_dz, &Vec4(0.25f32,0.0f32,0.1f32,0.0f32));
		safe_set_uniform(shu.fog_color, &g_fog_color);
		safe_set_uniform(shu.fog_falloff, &Vec4(0.5f32,0.25f32,0.0f32,0.0f32));

		glActiveTexture(GL_TEXTURE0+0);
		glBindTexture(GL_TEXTURE_2D, g_textures[tex0i]);
		glActiveTexture(GL_TEXTURE0+1);
		glBindTexture(GL_TEXTURE_2D, g_textures[tex1i]);

//		glVertexAttribPointer(VAI_pos as GLuint,	3,GL_FLOAT, GL_FALSE, stride, &((*baseVertex).pos[0]) as *f32 as *c_void);
		// to do: Rustic struct element offset macro
		let baseVertex=0 as *const MyVertex;
		glVertexAttribPointer(VertexAttrIndex::VAI_pos.into(),	3,GL_FLOAT, GL_FALSE, self.vertex_size, as_void_ptr(&(*baseVertex).pos));

		glVertexAttribPointer(VertexAttrIndex::VAI_color.into(),	4,GL_FLOAT, GL_FALSE, self.vertex_size, as_void_ptr(&(*baseVertex).color)); 

		glVertexAttribPointer(VertexAttrIndex::VAI_tex0.into(),	2,GL_FLOAT, GL_FALSE, self.vertex_size, as_void_ptr(&(*baseVertex).tex0));

		glVertexAttribPointer(VertexAttrIndex::VAI_norm.into(),	3,GL_FLOAT, GL_FALSE, self.vertex_size, as_void_ptr(&(*baseVertex).norm));
//		].iter().map(|&x|glVertexAttribPointer(x.0.into(), x.1, x.2, x.3, x.4, x.5));

		glDrawElements(GL_TRIANGLE_STRIP, self.num_indices as GLsizei, GL_UNSIGNED_INT,0 as *const c_void);
	}
}

static mut g_angle:f32=0.0f32;
static mut g_frame:int=0;

static g_num_torus:int = 256;
/// render a load of meshes in a lissajous

type StringMap<T,K=String> = HashMap<K,T>;

#[cfg(target_os = "emscripten")]
unsafe fn glUniformMatrix4fvARB(loc:i32, count:i32,  flags:u8, ptr:*const f32){
	glUniform4fv(
		loc,
		count*4,
		ptr);
}
pub unsafe  fn test_draw_2d(){
	draw::begin();

	let z=0.5f32;
	glVertex3f(0.0f32,0.0f32,z);
	glColor3f(1.0f32,0.0f32,1.0f32);
	glVertex3f(0.0f32,1.0f32,z);
	glColor3f(1.0f32,1.0f32,0.0f32);
	glVertex3f(1.0f32,1.0f32,z);
	glColor3f(0.0f32,1.0f32,1.0f32);
	glVertex3f(1.0f32,0.0f32,z);
	glColor3f(0.0f32,1.0f32,1.0f32);
	glVertex3f(0.0f32,0.0f32,z);
	glColor3f(0.0f32,1.0f32,1.0f32);
	glEnd();
	glEnable(GL_TEXTURE_2D);
	glBindTexture(GL_TEXTURE_2D,g_textures[1]);
	draw::rect_tex(&Vec2(-0.4,-0.4),&Vec2(0.2,0.2),z);
	draw::end();
	glBegin(GL_TRIANGLE_STRIP);

	glTexCoord2f(0.0,0.0);
	glColor3f(1.0f32,1.0f32,1.0f32);
	glVertex3f(0.0f32,0.0f32,z);

	glTexCoord2f(100.0,0.0);
	glColor3f(1.0f32,1.0f32,1.0f32);
	glVertex3f(1.0f32,0.0f32,z);

	glTexCoord2f(0.0,100.0);
	glColor3f(1.0f32,1.0f32,0.0f32);
	glVertex3f(0.0f32,1.0f32,z);

	glTexCoord2f(100.0,100.0);
	glColor3f(1.0f32,1.0f32,1.0f32);
	glVertex3f(1.0f32,1.0f32,z);

	glEnd();

	for x in 0..10 {
		for y in 0..10 {
			let fx=x as f32*0.1f32;
			let fy=y as f32*0.1f32;

			draw::sprite_at(&Vec3(fx,fy,z),0.5f32, draw::pack(&Vec4(fx,fy,0.5f32,1.0f32)));
		}
	}
	draw::end();

}

pub fn	render_no_swap(debug:u32) 
{
	//android_logw("render noswap");
	
	lazy_create_resources();	
	let x:StringMap<usize>;
	unsafe {
//		android_logw(c_str("render_no_swap"));
//		println!("{:?}",g_grid_mesh);
		g_angle+=0.0025f32;

//		glDrawBuffer(GL_BACK);
		//glClearColor(g_fog_color.x+sin(g_angle*2.0),g_fog_color.y,g_fog_color.z,g_fog_color.w);
		//glClear(GL_COLOR_BUFFER_BIT|GL_DEPTH_BUFFER_BIT);
		let matI=matrix::identity::<f32>();
		let matP = matrix::projection_frustum(-0.5f32,0.5f32,-0.5f32,0.5f32, 90.0f32, 1.0f32, 0.5f32,5.0f32);

		let pi=3.14159265f32;
		let tau=pi*2.0f32;

		let r0 = 1.0f32;
		let r1 = 0.5f32;
		let sda=0.25f32;
		let mut a0=g_angle*1.1f32+0.1f32;
		let mut a1=g_angle*1.09f32+1.5f32;
		let mut a2=g_angle*1.05f32+0.5f32;
		let mut a3=g_angle*1.11f32;
		let mut a4=g_angle*1.11f32+0.7f32;
		let mut a5=g_angle*1.105f32;
		let da0=tau*0.071f32*sda;
		let da1=tau*0.042f32*sda;
		let da2=tau*0.081f32*sda;
		let da3=tau*0.091f32*sda;
		let da4=tau*0.153f32*sda;
		let da5=tau*0.1621f32*sda;

		for i in 0..(g_num_torus) {

			let matT = matrix::translate_xyz(
				cos(a0)*r0+cos(a3)*r1, 
				cos(a1)*r0+cos(a4)*r1, 
				cos(a2)*r0+cos(a5)*r1 -2.0*r0);

			let rot_x = matrix::rotate_x(a0);
			let rot_y = matrix::rotate_x(a1*0.245f32);
			let rot_xy=rot_x.mul_matrix(&rot_y);
			let rot_trans = matT.mul_matrix(&rot_xy);
	
			let matMV = matT;	// toodo - combine rotation...
			//io::println(format!("{:?}", g_shader_program));

			{
				// draw every combo of 2 textures and the modes
				let ii=i as usize;
				let rmode=ii % RenderModeCount;
				let ig=ii/RenderModeCount;
				let ig2=ig/RenderModeCount;
				g_grid_mesh.render_mesh_shader(&matP,&rot_trans, rmode, 1+(ig%4), 1+(ig2%4));
			}


			a0+=da0;a1+=da1;a2+=da2;a3+=da3;a4+=da4;a5+=da5;
		}
		//test_draw_2d();
		glUseProgram(0);
		g_frame+=1;
	}
}

struct Actor {
	pos:Vec3, vec:Vec3, color:Vec4
}

struct WorldState {
	actors:Vec<Actor>
}


fn idle()
{
	unsafe {
		glutPostRedisplay();
	}
}



static mut g_lazy_init:bool=false;
pub fn lazy_create_resources() {
	
	unsafe {
		if g_lazy_init==false {
			println!("lazy init shadertest resources\n");
			android_logw(c_str("lazy init shadertest resources\0"));
			create_shaders();
			create_textures();
			g_lazy_init=true;
			g_grid_mesh = Mesh::new_torus((16,16)); //new GridMesh(16,16);
		} else {
		}
	}
}

struct Foo<'a>(&'a usize,&'a usize);

impl<'a> Foo<'a> {
	//fn compare<'x,'y,'z>(&'x self,a:&'y Foo, b:&'z Foo)->&'y Foo{
	//	a
	//}
	fn compare2(&self,a:&usize, b:&usize)->&usize{
		&self.0
	}
}

struct ShaderTest {
    time:i32
}
impl<A:'static> window::Window<A> for ShaderTest {
    fn update(&mut self, a:&mut A,_:f32)->Flow<A>{
		use window::Flow;
        self.time-=1;
        if self.time>0 {
            Flow::Continue()
        } else {
            Flow::Pop()//(world::new())
        }
    }
    fn render(&self,a:&A, _:&window::WinCursor){
        render_no_swap(if self.time &15>8{1}else{0});
    }
}

struct ShowBsp {
    time:i32
}

impl<A> window::Window<A> for ShowBsp {
}



const SDL_INIT_TIMER:u32=          0x00000001;
const SDL_INIT_AUDIO:u32=          0x00000010;
const SDL_INIT_VIDEO:u32=          0x00000020;  /**< SDL_INIT_VIDEO implies SDL_INIT_EVENTS */
const SDL_INIT_JOYSTICK:u32=       0x00000200;  /**< SDL_INIT_JOYSTICK implies SDL_INIT_EVENTS */
const SDL_INIT_HAPTIC:u32=         0x00001000;
const SDL_INIT_GAMECONTROLLER:u32= 0x00002000;  /**< SDL_INIT_GAMECONTROLLER implies SDL_INIT_JOYSTICK */
const SDL_INIT_EVENTS:u32=         0x00004000;
const SDL_INIT_NOPARACHUTE:u32=    0x00100000;  /**< compatibility; this flag is ignored. */
const SDL_INIT_EVERYTHING:u32=
SDL_INIT_TIMER | SDL_INIT_AUDIO | SDL_INIT_VIDEO | SDL_INIT_EVENTS |
SDL_INIT_JOYSTICK | SDL_INIT_HAPTIC | SDL_INIT_GAMECONTROLLER ;
/* @} */

/**
 *  This function initializes  the subsystems specified by \c flags
 */

const SDL_WINDOW_OPENGL:u32 = 0x00000002;             /**< window usable with OpenGL context */

#[repr(u32)]
enum  SDL_GLattr
{
SDL_GL_RED_SIZE,
SDL_GL_GREEN_SIZE,
SDL_GL_BLUE_SIZE,
SDL_GL_ALPHA_SIZE,
SDL_GL_BUFFER_SIZE,
SDL_GL_DOUBLEBUFFER,
SDL_GL_DEPTH_SIZE,
SDL_GL_STENCIL_SIZE,
SDL_GL_ACCUM_RED_SIZE,
SDL_GL_ACCUM_GREEN_SIZE,
SDL_GL_ACCUM_BLUE_SIZE,
SDL_GL_ACCUM_ALPHA_SIZE,
SDL_GL_STEREO,
SDL_GL_MULTISAMPLEBUFFERS,
SDL_GL_MULTISAMPLESAMPLES,
SDL_GL_ACCELERATED_VISUAL,
SDL_GL_RETAINED_BACKING,
SDL_GL_CONTEXT_MAJOR_VERSION,
SDL_GL_CONTEXT_MINOR_VERSION,
SDL_GL_CONTEXT_EGL,
SDL_GL_CONTEXT_FLAGS,
SDL_GL_CONTEXT_PROFILE_MASK,
SDL_GL_SHARE_WITH_CURRENT_CONTEXT,
SDL_GL_FRAMEBUFFER_SRGB_CAPABLE,
SDL_GL_CONTEXT_RELEASE_BEHAVIOR
}
use SDL_GLattr::*;
type SDL_WindowPtr=*const u8;
type SDL_WindowSurfacePtr=*const u8;
#[repr(C)]
struct SDL_Event{
    padding:[u8;56],
}
extern "C"{
    fn SDL_Init(_:u32)->isize;
    fn SDL_CreateWindow(_:*const u8,_:i32,_:i32,_:i32,_:i32,_:u32)->SDL_WindowPtr;
    fn SDL_GL_SwapWindow(_:SDL_WindowPtr);
    fn SDL_PollEvent(_:*mut SDL_Event)->bool;
    fn SDL_GL_SetAttribute(_:SDL_GLattr,_:isize);
    fn SDL_GetWindowSurface(_:SDL_WindowPtr)->SDL_WindowSurfacePtr;
    fn SDL_UpdateWindowSurface(_:SDL_WindowPtr);
}

#[cfg(target_os = "emscripten")]
fn test_file_download(){
	println!("test file download..");
	unsafe{
		extern{ fn printf(_:*const c_char);}
		emscripten::emscripten_wget(cstr!("https://upload.wikimedia.org/wikipedia/commons/1/17/Begur_Sa_Tuna_02_JMM.JPG\0"),cstr!("tmp_image.dat\0"));
	}
	let mut buffer=Vec::<u8>::new();
	let f=File::open("tmp_image.dat").unwrap().read_to_end(&mut buffer);
	
	println!("loaded {:?} bytes\n",buffer.len());
}

#[cfg(SDL)]
fn sdl_mainloop() {
    unsafe {
        trace!();
        SDL_Init(SDL_INIT_EVERYTHING);
        let win = SDL_CreateWindow(c_str("hello sdl\0"), 0, 0, 640, 480, SDL_WINDOW_OPENGL);
        SDL_GL_SetAttribute(SDL_GL_DOUBLEBUFFER, 1);
        let surf = SDL_GetWindowSurface(win);

        dump!(win);

        trace!();
        let mut n: i32 = 1000;
        while n > 0 {
            let mut e: SDL_Event = SDL_Event { padding: [0; 56] };
            while SDL_PollEvent(&mut e) {}
            trace!();
            n -= 1;
            glClearColor((n & 0xff) as f32 * 0.01f32, 0.0f32, 1.0f32, 0.0f32);
            glClear(GL_COLOR_BUFFER_BIT);
            SDL_GL_SwapWindow(win);
        }

        trace!();
    }
}


pub static mut g_test_texture:[GLuint;2]=[0;2];

fn test_browser_ui(){
	alert(prompt("input please").as_str());
	alert(if confirm("did that work?"){"yes"}else{"no"});
}
pub fn test_seq(){
   let a=window::ScrPos(0.1,0.2);let b=window::ScrPos(0.4,0.6);
    let c=v2avr(&a,&b);
    println!("checking derived maths for tuple malarchy a avr b= {:?} b-a={:?}",c,v2sub(&b,&a));
    println!("{:?}", seq![x; x*2 ;for 0..10]);
	println!("{:?}", seq![x; x*2 ;for 0..10 ;if 0!=x&1]);
	println!("{:?}", seq![1,2,3]);
	println!("{:?}", seq![1,2,3]);
	println!("{:?}", seq![1=>10,2=>3,3=>5]);
}

pub fn main(){

 	#[cfg(shadertest)]
	window::run_loop(vec![Box::new(ShaderTest{time:30000})],&mut ());

	#[cfg(not(target_os = "emscripten"))]
	window::run_loop(vec![world::new(),Box::new(ShaderTest{time:3000})],&mut ());

	#[cfg(any(target_os = "emscripten",editor))]
    window::run_loop(vec![editor::make_editor_window::<(),editor::Scene>()] , &mut ());
//  window::run_loop(test::new(),&mut ());
//  bsp::bsp::main();
//	shadertest();
//	world::main();
}

//~ShaderTest{time:100} as ~window::State
