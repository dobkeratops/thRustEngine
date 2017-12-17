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
pub mod test;
pub mod emscripten;

pub use window::sto;
pub use window::Window;
pub use emscripten::*;

//extern crate image;

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

static mut g_textures:[GLuint;5]=[0;5];
static mut g_shader_program_main:GLuint=-1i32 as uint;
static mut g_shader_program_debug:GLuint=-1i32 as uint;
static mut g_pixel_shader:GLuint=-1i32 as uint;
static mut g_vertex_shader:GLuint=-1i32 as uint;

#[derive(Copy,Clone,Debug)]
#[repr(u32)]
enum VertexAttrIndex  {
	VAI_pos		=0x0000,
	VAI_color,
	VAI_norm,
	VAI_tex0,
	VAI_tex1,
	VAI_joints,
	VAI_weights,
	VAI_tangent,
	VAI_binormal,
	VAI_count
}

impl Into<u32> for VertexAttrIndex{
	fn into(self)->u32{ self as u32}
}




unsafe fn get_attrib_location(shader_prog:GLuint, name:&str)->GLint {
	let r=glGetAttribLocation(shader_prog, c_str(name));
	println!("get attrib location {:?}={:?}", name, r);
	r
}
unsafe fn get_uniform_location(shader_prog:GLuint, name:&str)->GLint {
	let cs=CString::new(name);
	let r=glGetUniformLocation(shader_prog, cs.unwrap().as_ptr());
	println!("get uniform_location location {:?}={:?}", name, r);
	r
}

fn vec_from_fn<T:Sized+Clone,F:Fn(usize)->T>(num:usize , f:&F)->Vec<T>{
	// todo - this must be official ? generator, whatever
	let mut r=Vec::<T>::new();
	r.reserve(num);
	println!("init vector {:?}elems",num);
	for x in 0..num{
		r.push(f(x)) 
	}
	r
}


unsafe fn	create_and_compile_shader(shader_type:GLenum, source:&[&str]) ->GLuint
{
	android_logw_str("create_and_compile_shader");
	let	shader = glCreateShader(shader_type );
	android_logw_string(&format!("shader={:?}",shader));

	let sources_as_c_str=vec_from_fn(source.len(), &|x|c_str(source[x]) );
	let length = vec_from_fn(source.len() , &|x|source[x].len() as c_int );
	for i in 0..source.len() { println!("source adr={:?} source len={:?} ",sources_as_c_str[i],length[i])};
	
	android_logw_str("set shader source..");
	glShaderSource(shader, source.len() as GLsizei, &sources_as_c_str[0], 0 as *const c_int/*(&length[0])*/);
	android_logw_str("compile..");
	glCompileShader(shader);
	let	status:c_int=0;
	android_logw_str("get status..");
	glGetShaderiv(shader,GL_COMPILE_STATUS,&status);
	android_logw_str("got status");
	android_logw_string(&format!("status = {:?}",status));
	if status==GL_FALSE as GLint
	{
		android_logw_string(&format!("failed, getting log.."));
		let compile_log:[c_char;512]=[0 as c_char;512]; //int len;
	
		let log_len:c_int=0;
		glGetShaderInfoLog(shader, 512,&log_len as *const c_int, &compile_log[0]);
		android_logw_string(&format!("Compile Shader Failed: logsize={:?}",
				log_len));

		println!("compile shader {:?} failed: \n{:?}\n", shader, compile_log[0]);
//		println!("compile shader {:?} failed: \n{:?}\n", shader, 
//			CString::new((&compile_log[0]) as *const c_char).as_str());

		for s in source.iter() { android_logw(c_str(*s)) }
		println!("TODO here 168");
		println!("error {:?}", CStr::from_ptr(&compile_log[0]));
//		android_logw(
//			match c_str::CString::new(&compile_log[0],false).as_str() {
//				Some(s)=>c_str(s),
//				None=>c_str("couldn't unwrap error lol"),
//			}
//		);

		for i in 0..log_len {
//			android_logw_string(&format!("{:?}",compile_log[i]));
		}
		loop{}

	}	
	else {
		println!("create shader{:?} - compile suceeded\n",  shader);
		android_logw_string(&format!("create shader{:?} - compile suceeded\n",  shader));
	}
	android_logw_str("create shader-done");
	shader
}


#[derive(Clone,Debug)]
struct	VertexAttr {
	pos:GLint,color:GLint,norm:GLint,tex0:GLint,tex1:GLint,joints:GLint,weights:GLint,tangent:GLint,binormal:GLint,
}
static g_vertex_attr_empty:VertexAttr=VertexAttr{
	pos:-1,color:-1,norm:-1,tex0:-1,tex1:-1,joints:-1,weights:-1,tangent:-1,binormal:-1
};

static mut g_vertex_shader_attrib:VertexAttr=VertexAttr{
	pos:-1,color:-1,norm:-1,tex0:-1,tex1:-1,joints:-1,weights:-1,tangent:-1,binormal:-1
};

static mut g_vertex_shader_attrib_debug:VertexAttr=VertexAttr{
	pos:-1,color:-1,norm:-1,tex0:-1,tex1:-1,joints:-1,weights:-1,tangent:-1,binormal:-1
};

//g_vertex_attr_empty;

static mut g_shader_uniforms_main:UniformTable=UniformTable{
	mat_proj:-1,
	mat_model_view:-1,
	mat_model_view_proj:-1,
	mat_color:-1,
	mat_env_map:-1,
	tex0:-1,
	tex1:-1,
	cam_in_obj:-1,
	ambient:-1,
	diffuse_dx:-1,
	diffuse_dy:-1,
	diffuse_dz:-1,
	specular_color:-1,
	specular_dir:-1,
	sky_dir:-1,
	test_vec_4:-1,
	fog_color:-1,
	fog_falloff:-1,
	light0_pos_r:-1,
	light0_color:-1,
};

static mut g_shader_uniforms_debug:UniformTable=UniformTable{
	mat_proj:-1,
	mat_model_view:-1,
	mat_model_view_proj:-1,
	mat_color:-1,
	mat_env_map:-1,
	tex0:-1,
	tex1:-1,
	cam_in_obj:-1,
	ambient:-1,
	diffuse_dx:-1,
	diffuse_dy:-1,
	diffuse_dz:-1,
	specular_color:-1,
	specular_dir:-1,
	sky_dir:-1,
	test_vec_4:-1,
	fog_color:-1,
	fog_falloff:-1,
	light0_pos_r:-1,
	light0_color:-1,
};


//g_uniform_table_empty;

// Paired pixel and vertex shaders.

pub type VertexShader=GLuint;
pub type PixelShader=GLuint;
pub type ShaderProgram=GLuint;

unsafe fn create_texture(filename:String)->GLuint {
	return g_textures[0]
}

extern {pub fn bind_attrib_locations(prog:c_uint);}

unsafe fn	create_shader_program(
			pixelShaderSource:&[&str],
			vertexShaderSource:&[&str])->(PixelShader,VertexShader,ShaderProgram)
{

	android_logw(c_str("create_shader_program"));

	let pixelShaderOut = create_and_compile_shader(GL_FRAGMENT_SHADER, pixelShaderSource);
	let vertexShaderOut = create_and_compile_shader(GL_VERTEX_SHADER, vertexShaderSource);	let	prog = glCreateProgram();
	android_logw(c_str("bind attrib locations"));
	
	// assign attribute names before linking
	for &x in [
		(VertexAttrIndex::VAI_pos,"a_pos"),
		(VertexAttrIndex::VAI_color,"a_color"),
		(VertexAttrIndex::VAI_norm,"a_norm"),
		(VertexAttrIndex::VAI_tex0,"a_tex0"),
		(VertexAttrIndex::VAI_tex1,"a_tex1"),
		(VertexAttrIndex::VAI_joints,"a_joints"),
		(VertexAttrIndex::VAI_weights,"a_weights"),
		(VertexAttrIndex::VAI_tangent,"a_tangent"),
		(VertexAttrIndex::VAI_binormal,"a_binormal"),

	].iter() {glBindAttribLocation(prog, x.0 as GLuint, c_str(x.1));}

	glAttachShader(prog, pixelShaderOut);
	glAttachShader(prog, vertexShaderOut);

	println!("linking verteshader{:?}, pixelshader{:?} to program{:?}\n", vertexShaderOut, pixelShaderOut, prog);
	glLinkProgram(prog);
	let mut err:GLint=0;
	glGetProgramiv(prog,GL_LINK_STATUS,(&err) as *const GLint);
	
	let x=glGetAttribLocation(prog,c_str("a_color"));
	println!("write,read attrib location in prog {:?} a_color={:?}", prog, x);

	
	if err as GLenum==GL_INVALID_VALUE || err as GLenum==GL_INVALID_OPERATION {
		let mut buffer=[0 as GLchar;1024];
		let mut len:GLint=0;
		glGetProgramInfoLog(prog,1024,&len,&buffer[0]);
		println!("link program failed: {:?}",err);
		println!("todo\n");
//		println!("{:?}",CString::new(&buffer[0]));
	} else {
		println!("link program status {:?}", err);
	}

	(pixelShaderOut,vertexShaderOut,prog)
}

//TODO: split into default uniforms, default vertex, default vertex-shader-out

// TODO [cfg OPENGL_ES ..]
static shader_prefix_desktop:&'static str=&"\
#version 120	\n\
#define highp	\n\
#define mediump	\n\
#define lowp	\n\
";


static vertex_shader_prefix_gles:&'static str=&"\
#version 100			\n\
precision highp float;	\n\
";

//#version 100			\n\
static pixel_shader_prefix_gles:&'static str=&"\
precision mediump float;	\n\
";


static ps_vs_interface0:&'static str=&
"varying	highp vec4 v_pos;\n\
varying	highp vec4 v_color;\n\
varying	highp vec3 v_norm;\n\
varying	highp vec2 v_tex0;\n\
varying	highp vec3 v_tex1;\n\
varying	highp vec4 v_tangent;\n\
varying	highp vec4 v_binormal;\n";

static ps_vertex_format0:&'static str=&
"attribute vec3 a_pos;\n\
attribute vec2 a_tex0;\n\
attribute vec4 a_color;\n\
attribute vec3 a_norm;\n";

static vs_uniforms:&'static str=
&"\
uniform mat4 uMatProj;\n\
uniform mat4 uMatModelView;\n\
";

static g_VS_Default:&'static str=
&"\
uniform mat4 uMatProj;\n\
uniform mat4 uMatModelView;\n\
void main() {\n\
	vec4 posw = vec4(a_pos.xyz,1.0);\n\
	vec4 epos = uMatModelView * pos4;\n\
	vec3 enorm = (uMatModelView * vec4(a_norm.xyz,0.0)).xyz;\n\
	vec4 spos=uMatProj * epos;\n\
	gl_Position = spos;\n\
	v_pos = posw;\n\
	v_color = a_color;\n\
	v_tex0 = a_tex0;\n\
	v_tex1 = a_pos.xyz;\n\
	v_norm = enorm;\n\
}";

/// replacement debug vertex shader - dont apply transformations, just view vertices..
static g_VS_PassThru:&'static str=
&"
uniform mat4 uMatProj;\n\
uniform mat4 uMatModelView;\n\
void main() {\n\
	vec4 posw = vec4(a_pos.xyz,1.0);\n\
	vec4 epos = uMatModelView * posw;\n\
	vec3 enorm = (uMatModelView * vec4(a_norm.xyz,0.0)).xyz;\n\
	vec4 spos=uMatProj * epos;\n\
	gl_Position = vec4(posw.xyz,1.0);\n\
	v_pos = epos;\n\
	v_color = a_color;\n\
	v_tex0 = a_tex0;\n\
	v_tex1 = a_pos.xyz;\n\
	v_norm = enorm;\n\
}";
static g_VS_PassThruTweak:&'static str=
&"\
uniform mat4 uMatProj;\n\
uniform mat4 uMatModelView;\n\
void main() {\n\
	vec4 posw = vec4(a_pos.xyz,1.0);\n\
	vec4 epos = uMatModelView * posw;\n\
	vec3 enorm = (uMatModelView * vec4(a_norm.xyz,0.0)).xyz;\n\
	vec4 spos=uMatProj * epos;\n\
	gl_Position = spos;\n\
	v_pos = posw;\n\
	v_color = a_color;\n\
	v_tex0 = a_tex0;\n\
	v_tex1 = a_pos.xyz;\n\
	v_norm = enorm;\n\
}";

/// replacement debug vertex shader - dont apply perspective, just view transformed models
static g_VS_RotTransPers:&'static str=
&"\
uniform mat4 uMatProj;\n\
uniform mat4 uMatModelView;\n\
void main() {\n\
	vec4 posw = vec4(a_pos.xyz,1.0);\n\
	vec4 eye_pos = uMatModelView * posw;\n\
	vec3 eye_norm = (uMatModelView * vec4(a_norm.xyz,0.0)).xyz;\n\
	vec4 screen_pos=uMatProj * eye_pos;\n\
	gl_Position = screen_pos;\n\
	v_pos = vec4(eye_pos.xyz,1.0);\n\
	v_color = a_color;\n\
	v_tex0 = a_tex0;\n\
	v_tex1 = a_pos.xyz;\n\
	v_norm = eye_norm;\n\
}";

/// replacement debug vertex shader - dont apply perspective, just view transformed models
static g_VS_Translate2d:&'static str=
&"uniform mat4 uMatProj;\n\
uniform mat4 uMatModelView;\n\
void main() {\n\
	vec4 posw = vec4(a_pos.xyz,1.0);\n\
	vec4 epos = uMatModelView * posw;\n\
	vec3 enorm = (uMatModelView * vec4(a_norm.xyz,0.0)).xyz;\n\
	vec4 spos=uMatProj * epos;\n\
	gl_Position = vec4(posw.xyz,1.0)+uMatModelView[3].xyzw;\n\
	v_pos = posw;\n\
	v_color = a_color;\n\
	v_tex0 = a_tex0;\n\
	v_tex1 = a_pos.xyz;\n\
	v_norm = enorm;\n\
}";

static g_VS_Persp:&'static str=
&"uniform mat4 uMatProj;\n\
uniform mat4 uMatModelView;\n\
void main() {\n\
	vec4 posw = vec4(a_pos.xyz,1.0);\n\
	vec4 epos = uMatModelView * posw;\n\
	vec3 enorm = (uMatModelView * vec4(a_norm.xyz,0.0)).xyz;\n\
	vec4 spos=uMatProj * epos;\n\
	gl_Position = spos;\n\
	v_pos = posw;\n\
	v_color = a_color;\n\
	v_tex0 = a_tex0;\n\
	v_tex1 = a_pos.xyz;\n\
	v_norm = enorm;\n\
}";


/*
cases:
VSO:
	static scene
	animation,3bone
PS:
	2textures
	3textures
 */

// sanity check debug, checking that the andoir build does this ok..
static g_PS_ConcatForAndroid:&'static str= &"
precision mediump float; \n\
varying	highp vec4 v_pos;\n\
varying	highp vec4 v_color;\n\
varying	highp vec3 v_norm;\n\
varying	highp vec2 v_tex0;\n\
varying	highp vec3 v_tex1;\n\
varying	highp vec4 v_tangent;\n\
varying	highp vec4 v_binormal;\n\
uniform sampler2D uTex0;\n\
uniform sampler2D uTex1;\n\
uniform vec4 uSpecularDir;\n\
uniform float uSpecularPower;\n\
uniform vec4 uSpecularColor;\n\
uniform vec4 uFogColor;\n\
uniform vec4 uFogFalloff;\n\
uniform vec4 uAmbient;\n\
uniform vec4 uDiffuseDX;\n\
uniform vec4 uDiffuseDY;\n\
uniform vec4 uDiffuseDZ;\n\
    \n\
uniform vec4 uLightPos;\n\
uniform vec4 uLightColor;\n\
uniform vec4 uLightFalloff;\n\
vec4 applyFog(vec3 pos, vec4 color){\n\
	return mix(color,uFogColor,  clamp(-uFogFalloff.x-pos.z*uFogFalloff.y,0.0,1.0));\n\
}\n\
vec4 pointlight(vec3 pos, vec3 norm,vec3 lpos, vec4 color, vec4 falloff) {\n\
	vec3 dv=lpos-pos;\n\
	float d2=sqrt(dot(dv,dv));\n\
	float f=clamp( 1.0-(d2/falloff.x),0.0,1.0);\n\
	vec3 lv=normalize(dv);\n\
	return clamp(dot(lv,norm),0.0,1.0) * f*color;\n\
}\n\
void main() { \n\
	float inva=(v_color.w),a=(1.0-v_color.w);\n\
	vec4 t0=texture2D(uTex0, v_tex0);\n\
	vec4 t1=texture2D(uTex1, v_tex0);\n\
	float a0=t0.x*0.4+t0.y*0.6+t0.z*0.25;\n\
	float a1=t1.x*0.4+t1.y*0.6+t1.z*0.25;\n\
	float highlight=max(0.0,dot(v_norm,uSpecularDir.xyz));\n\
		highlight=(highlight*highlight);highlight=highlight*highlight;\n\
	vec4 surfaceColor=mix(t0,t1,v_color.w);\n\
	vec4 surfaceSpec=clamp(4.0*(surfaceColor-vec4(0.5,0.5,0.5,0.0)), vec4(0.0,0.0,0.0,0.0),vec4(1.0,1.0,1.0,1.0));\n\
	vec4 spec=highlight*uSpecularColor*surfaceSpec;\n\
	vec4 diff=uAmbient+v_norm.x*uDiffuseDX+v_norm.y*uDiffuseDY+v_norm.z*uDiffuseDZ;\n\
	float lx=0.5,ly=0.5;\n\
	diff+=pointlight(v_pos.xyz,v_norm.xyz, vec3(lx,ly,-1.0),		vec4(1.0,0.0,0.0,0.0),vec4(1.0,0.0,0.0,0.0));\n\
	diff+=pointlight(v_pos.xyz,v_norm.xyz, vec3(lx,-ly,-1.0), 	vec4(0.0,1.0,0.0,0.0),vec4(1.0,0.0,0.0,0.0));\n\
	diff+=pointlight(v_pos.xyz,v_norm.xyz, vec3(-lx,-ly,-1.0),	vec4(0.0,0.0,1.0,0.0),vec4(1.0,0.0,0.0,0.0));\n\
	diff+=pointlight(v_pos.xyz,v_norm.xyz, vec3(-lx,ly,-1.0), 	vec4(0.5,0.0,0.5,0.0),vec4(1.0,0.0,0.0,0.0));\n\
//	gl_FragColor =applyFog(v_pos.xyz,surfaceColor*diff*vec4(v_color.xyz,0.0)*2.0+spec);\n\
	gl_FragColor =vec4(v_norm.xyz,0.0)*0.5+vec4(0.5,0.5,0.5,1.0);\n\
}";


static g_PS_Alpha:&'static str= &
"uniform sampler2D uTex0;\n\
uniform sampler2D uTex1;\n\
uniform vec4 uSpecularDir;\n\
uniform float uSpecularPower;\n\
uniform vec4 uSpecularColor;\n\
uniform vec4 uFogColor;\n\
uniform vec4 uFogFalloff;\n\
uniform vec4 uAmbient;\n\
uniform vec4 uDiffuseDX;\n\
uniform vec4 uDiffuseDY;\n\
uniform vec4 uDiffuseDZ;\n\
    \n\
uniform vec4 uLightPos;\n\
uniform vec4 uLightColor;\n\
uniform vec4 uLightFalloff;\n\
vec4 applyFog(vec3 pos, vec4 color){\n\
	return mix(color,uFogColor,  clamp(-uFogFalloff.x-pos.z*uFogFalloff.y,0.0,1.0));\n\
}\n\
vec4 pointlight(vec3 pos, vec3 norm,vec3 lpos, vec4 color, vec4 falloff) {\n\
	vec3 dv=lpos-pos;\n\
	float d2=sqrt(dot(dv,dv));\n\
	float f=clamp( 1.0-(d2/falloff.x),0.0,1.0);\n\
	vec3 lv=normalize(dv);\n\
	return clamp(dot(lv,norm),0.0,1.0) * f*color;\n\
}\n\
void main() { \n\
	float inva=(v_color.w),a=(1.0-v_color.w);\n\
	vec4 t0=texture2D(uTex0, v_tex0);\n\
	vec4 t1=texture2D(uTex1, v_tex0);\n\
	float a0=t0.x*0.4+t0.y*0.6+t0.z*0.25;\n\
	float a1=t1.x*0.4+t1.y*0.6+t1.z*0.25;\n\
	float highlight=max(0.0,dot(v_norm,uSpecularDir.xyz));\n\
		highlight=(highlight*highlight);highlight=highlight*highlight;\n\
	vec4 surfaceColor=mix(t0,t1,v_color.w);\n\
	vec4 surfaceSpec=clamp(4.0*(surfaceColor-vec4(0.5,0.5,0.5,0.0)), vec4(0.0,0.0,0.0,0.0),vec4(1.0,1.0,1.0,1.0));\n\
	vec4 spec=highlight*uSpecularColor*surfaceSpec;\n\
	vec4 diff=uAmbient+v_norm.x*uDiffuseDX+v_norm.y*uDiffuseDY+v_norm.z*uDiffuseDZ;\n\
	float lx=0.5,ly=0.5;\n\
	diff+=pointlight(v_pos.xyz,v_norm.xyz, vec3(lx,ly,-1.0),		vec4(1.0,0.0,0.0,0.0),vec4(1.0,0.0,0.0,0.0));\n\
	diff+=pointlight(v_pos.xyz,v_norm.xyz, vec3(lx,-ly,-1.0), 	vec4(0.0,1.0,0.0,0.0),vec4(1.0,0.0,0.0,0.0));\n\
	diff+=pointlight(v_pos.xyz,v_norm.xyz, vec3(-lx,-ly,-1.0),	vec4(0.0,0.0,1.0,0.0),vec4(1.0,0.0,0.0,0.0));\n\
	diff+=pointlight(v_pos.xyz,v_norm.xyz, vec3(-lx,ly,-1.0), 	vec4(0.5,0.0,0.5,0.0),vec4(1.0,0.0,0.0,0.0));\n\
	gl_FragColor =applyFog(v_pos.xyz,surfaceColor*diff*vec4(v_color.xyz,0.0)*2.0+spec);\n\
//	gl_FragColor =vec4(v_norm.xyz,0.0)*0.5+vec4(0.5,0.5,0.5,1.0)+vec4(v_tex0,0.0,0.0);\n\
}";

// debug shader
static g_PS_Add:&'static str= &
"uniform sampler2D s_Tex0;\n\
uniform sampler2D s_Tex1;\n\
uniform vec4 uSpecularDir;\n\
uniform float uSpecularPower;\n\
uniform vec4 uSpecularColor;\n\
uniform vec4 uAmbient;\n\
uniform vec4 uDiffuseDX;\n\
uniform vec4 uDiffuseDY;\n\
uniform vec4 uDiffuseDZ;\n\
void main() { \n\
	float inva=(v_color.w),a=(1.0-v_color.w);\n\
	vec4 t0=texture2D(s_Tex0, v_tex0);\n\
	vec4 t1=texture2D(s_Tex1, v_tex0);\n\
	float a0=t0.x*0.4+t0.y*0.6+t0.z*0.25;\n\
	float a1=t1.x*0.4+t1.y*0.6+t1.z*0.25;\n\
	float highlight=max(0.0,dot(v_norm,uSpecularDir.xyz));\n\
	highlight=(highlight*highlight);highlight=highlight*highlight;\n\
	vec4 surfaceColor=t0+(t1-vec4(0.5f,0.5f,0.5f,0.0f))*v_color.w;\n\
	vec4 surfaceSpec=clamp(4.0*(surfaceColor-Vec4(0.5,0.5,0.5,0.0)), vec4(0.0,0.0,0.0,0.0),vec4(1.0,1.0,1.0,1.0));\n\
	vec4 spec=highlight*uSpecularColor*surfaceSpec;\n\
	vec4 diff=uAmbient+vso_norm.x*uDiffuseDX+vso_norm.y*uDiffuseDY+vso_norm.z*uDiffuseDZ;\n\
	gl_FragColor =surfaceColor*diff*vec4(v_color.xyz,0.0)*2.0+spec;\n\
}";

static g_PS_Flat:&'static str=&"\
void main {\n\
	gl_FragColor= mediump vec4(0.0, 1.0, 0.0, 1.0);\n\
}\n\
";

static g_PS_MinimumDebugAndroidCompiler:&'static str= &"\
precision mediump float; \n\
void main() \n\
{ \n\
 gl_FragColor = vec4(1.0, 0.0, 0.0, 1.0); \n\
} \n\
";

static g_PS_Tex3_AlphaMul:&'static str=&"\
uniform sampler2D s_tex0;\n\
uniform sampler2D s_tex1;\n\
uniform sampler2D s_tex2;\n\
uniform vec4 uSpecularDir;\n\
uniform float uSpecularPower;\n\
uniform vec4 uSpecularColor;\n\
uniform vec4 uAmbient;\n\
uniform vec4 uDiffuseDX;\n\
uniform vec4 uDiffuseDY;\n\
uniform vec4 uDiffuseDZ;\n\
void main() { \n\
	float inva=(v_color.w),a=(1.0-v_color.w);\n\
	vec4 t0=texture2D(s_Tex0, v_tex0);\n\
	vec4 t1=texture2D(s_Tex1, v_tex0);\n\
	float a0=t0.x*0.4+t0.y*0.6+t0.z*0.25;\n\
	float a1=t1.x*0.4+t1.y*0.6+t1.z*0.25;\n\
	float highlight=max(0.0,dot(v_norm,uSpecularDir.xyz));\n\
	highlight=(highlight*highlight);highlight=highlight*highlight;\n\
	vec4 surfaceColor=mix(t0,t1, v_color.w*t1.a);\n\
	vec4 surfaceSpec=clamp(4.0*(surfaceColor-Vec4(0.5,0.5,0.5,0.0)), vec4(0.0,0.0,0.0,0.0),vec4(1.0,1.0,1.0,1.0));\n\
	vec4 spec=highlight*uSpecularColor*surfaceSpec;\n\
	vec4 diff=uAmbient+vso_norm.x*uDiffuseDX+vso_norm.y*uDiffuseDY+vso_norm.z*uDiffuseDZ;\n\
	gl_FragColor =surfaceColor*diff*Vec4(v_color.xyz,0.0)*2.0+spec;\n\
}\
";


#[derive(Clone,Debug)]
struct UniformTable {
	mat_proj:UniformIndex,
	mat_model_view:UniformIndex,
	mat_model_view_proj:UniformIndex,
	mat_color:UniformIndex,
	mat_env_map:UniformIndex,
	tex0:UniformIndex,
	tex1:UniformIndex,
	cam_in_obj:UniformIndex,
	ambient:UniformIndex,
	diffuse_dx:UniformIndex,
	diffuse_dy:UniformIndex,
	diffuse_dz:UniformIndex,
	specular_color:UniformIndex,
	specular_dir:UniformIndex,
	sky_dir:UniformIndex,
	test_vec_4:UniformIndex,
	fog_color:UniformIndex,
	fog_falloff:UniformIndex,
	light0_pos_r:UniformIndex,
	light0_color:UniformIndex,
}
static g_uniform_table_empty:UniformTable= UniformTable{
	mat_proj:-1,
	mat_model_view:-1,
	mat_model_view_proj:-1,
	mat_color:-1,
	mat_env_map:-1,
	tex0:-1,
	tex1:-1,
	cam_in_obj:-1,
	ambient:-1,
	diffuse_dx:-1,
	diffuse_dy:-1,
	diffuse_dz:-1,
	specular_color:-1,
	specular_dir:-1,
	sky_dir:-1,
	test_vec_4:-1,
	fog_color:-1,
	fog_falloff:-1,
	light0_pos_r:-1,
	light0_color:-1,
};

//map_shader_params(VertexAttr* vsa,UniformTable* su,int prog)
fn map_shader_params(prog:GLuint)->(VertexAttr,UniformTable)
{
	// read attrib back from shader
	// at the minute we've preset these from VAI indices,
	// but leave this path for datadriven approch later

	// todo: rustic macro
	unsafe {
		let mut num_uniforms:GLint=0;
		glGetProgramiv( prog, GL_ACTIVE_UNIFORMS, &num_uniforms ); 
		for i in 0..num_uniforms {
			let mut uname:[c_char;256]=[0;256];
			let mut name_len:GLint=0;
			let mut unisize:GLint=0;
			let mut utype:GLuint=0;
			glGetActiveUniform(prog,i as GLuint, 255, &name_len, &unisize,&utype,&uname as *const c_char);
			let mut uindexr=glGetUniformLocation(prog,&uname[0]);
			println!("uniform {:?}index={:?}\tsize={:?}\ttype={:?}", uindexr,CStr::from_ptr(&uname[0]), unisize, utype);
			
		}
		(
			VertexAttr{
				pos: get_attrib_location(prog, &"a_pos"),
				color: get_attrib_location(prog, &"a_color"),
				norm: get_attrib_location(prog, &"a_norm"),
				tex0: get_attrib_location(prog, &"a_tex0"),
				tex1: get_attrib_location(prog, &"a_tex1"),
				joints: get_attrib_location(prog, &"a_joints"),
				weights: get_attrib_location(prog, &"a_weights"),
				tangent: get_attrib_location(prog, &"a_binormal"),
				binormal: get_attrib_location(prog, &"a_tangent")

			},
			UniformTable{
				mat_proj:get_uniform_location(prog,&"uMatProj"),
				mat_model_view:get_uniform_location(prog,&"uMatModelView"),
				specular_color:get_uniform_location(prog,&"uSpecularColor"),
				specular_dir:get_uniform_location(prog,&"uSpecularDir"),
				ambient:get_uniform_location(prog,&"uAmbient"),
				diffuse_dx:get_uniform_location(prog,&"uDiffuseDX"),
				diffuse_dy:get_uniform_location(prog,&"uDiffuseDY"),
				diffuse_dz:get_uniform_location(prog,&"uDiffuseDZ"),
				fog_color:get_uniform_location(prog,&"uFogColor"),
				fog_falloff:get_uniform_location(prog,&"uFogFalloff"),
				light0_pos_r:get_uniform_location(prog,&"uLight0PosR"),
				light0_color:get_uniform_location(prog,&"uLight0Color"),
				..g_uniform_table_empty
			}
		)
	}	
}

#[cfg(target_os = "android")]
fn get_shader_prefix(is_ps:int)->&'static str {
	if is_ps==0 {vertex_shader_prefix_gles} else {pixel_shader_prefix_gles}
}

#[cfg(not(target_os = "android"))]
fn get_shader_prefix(is_ps:int)->&'static str {
	shader_prefix_desktop
}

fn	create_shaders()
{
//android malarky	
//						&~[g_PS_ConcatForAndroid], // this works!
//						&~[g_PS_MinimumDebugAndroidCompiler],
//						&~[get_shader_prefix(1),ps_vs_interface0,g_PS_Flat], //PS_Alpha
	unsafe {
		android_logw(c_str("create shaders"));
		let (vsh,psh,prg)=create_shader_program( 
						&[get_shader_prefix(1),ps_vs_interface0,g_PS_Alpha],
						&[get_shader_prefix(0), ps_vertex_format0, ps_vs_interface0,  g_VS_PassThruTweak]);
		g_vertex_shader=vsh;
		g_pixel_shader=psh;	
		g_shader_program_main=prg;
		let (vs, su)=map_shader_params(g_shader_program_main);
		println!("vs={:?}",vs);
		println!("su={:?}",su);
		g_vertex_shader_attrib=vs;
		g_shader_uniforms_main=su;
		g_shader_uniforms_main.mat_proj=0;
		g_shader_uniforms_main.mat_model_view=4;
		g_shader_uniforms_main.mat_model_view_proj=8;

		let (vsh1,psh1,prg1)=create_shader_program( 
						&[get_shader_prefix(1),ps_vs_interface0,g_PS_Alpha],
						&[get_shader_prefix(0), ps_vertex_format0, ps_vs_interface0,  g_VS_RotTransPers]);
		g_shader_program_debug=prg1;
		

		{
			let (vs, su)=map_shader_params(g_shader_program_debug);
			g_vertex_shader_attrib_debug=vs;
			g_shader_uniforms_debug=su;
		}

	}
}


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

	MyVertex{
		pos:[(rx+sy*ry)*cx, (rx+sy*ry)*sx, ry*cy],
		color:[1.0,1.0,1.0,1.0],
		norm:[sy*cx, sy*sx, cy],
		tex0:[fi*8.0, fj*2.0],
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
				println!("indexx{:?} i={:?} j={:?}", ij, i, j);
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


//extern void	TestGl_Idle();

//float	angle=0.f;
//GridMesh*	g_pGridMesh;
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

impl Mesh {
	unsafe fn	render_mesh_shader(&self)  {

		
		let clientState:[GLenum;3]=[GL_VERTEX_ARRAY,GL_COLOR_ARRAY,GL_TEXTURE_COORD_ARRAY];

		glBindBuffer(GL_ARRAY_BUFFER, self.vbo);
		glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, self.ibo);

		let vsa=&g_vertex_shader_attrib;
		let shu=&g_shader_uniforms_main;

		glEnableVertexAttribArray(VertexAttrIndex::VAI_pos.into());
		glEnableVertexAttribArray(VertexAttrIndex::VAI_color.into());
		glEnableVertexAttribArray(VertexAttrIndex::VAI_tex0.into());
		glEnableVertexAttribArray(VertexAttrIndex::VAI_norm.into());

		safe_set_uniform1i(shu.tex0, 0);
		safe_set_uniform1i(shu.tex1, 1);
		safe_set_uniform(shu.specular_dir, &Vec4(0.032,0.707f32,0.707f32,0.0f32));
		safe_set_uniform(shu.specular_color, &Vec4(1.0f32,0.75f32,0.5f32,0.0f32));
		safe_set_uniform(shu.ambient, &Vec4(0.25f32,0.25f32,0.25f32,1.0f32));
		safe_set_uniform(shu.diffuse_dx, &Vec4(0.0f32,0.0f32,0.25f32,1.0f32));
		safe_set_uniform(shu.diffuse_dy, &Vec4(0.5f32,0.5f32,0.5f32,1.0f32));
		safe_set_uniform(shu.diffuse_dz, &Vec4(0.25f32,0.0f32,0.0f32,1.0f32));
		safe_set_uniform(shu.fog_color, &g_fog_color);
		safe_set_uniform(shu.fog_falloff, &Vec4(0.5f32,0.25f32,0.0f32,0.0f32));

		glActiveTexture(GL_TEXTURE0+0);
		glBindTexture(GL_TEXTURE_2D, g_textures[0]);
		glActiveTexture(GL_TEXTURE0+1);
		glBindTexture(GL_TEXTURE_2D, g_textures[0]);

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

#[cfg(ASMJS)]
unsafe fn glUniformMatrix4fvARB(loc:i32, count:i32,  flags:u8, ptr:*const f32){
	glUniform4fv(
		loc,
		count*4,
		ptr);
}


pub fn	render_no_swap() 
{
	//android_logw("render noswap");
	
	println!("shadetest::render_no_swap");
	lazy_create_resources();	
	let x:StringMap<usize>;
	unsafe {
		android_logw(c_str("render_no_swap"));
		println!("{:?}",g_grid_mesh);
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
//			println!("[{:?}]{:?} {:?}",i, matP,rot_trans);
			glUseProgram(g_shader_program_main);
			let shu=&g_shader_uniforms_main;
//			println!("matrix={:?} into {:?}",rot_trans,shu.mat_model_view);
/*
			glUniformMatrix4fvARB(shu.mat_proj, 1,  GL_FALSE, &matP.ax.x);
			glUniformMatrix4fvARB(shu.mat_model_view, 1, GL_FALSE, &rot_trans.ax.x);
			g_grid_mesh.render_mesh_shader();

			let mat_ident=matrix::identity();
			glUniformMatrix4fvARB(shu.mat_proj, 1,  GL_FALSE, &mat_ident.ax.x);
			glUniformMatrix4fvARB(shu.mat_model_view, 1, GL_FALSE, &rot_trans.ax.x);
			g_grid_mesh.render_mesh_shader();
*/

			{
			//	let mat_ident=matrix::identity();
				let shu=&g_shader_uniforms_debug;
				glUseProgram(g_shader_program_debug);
				glUniformMatrix4fvARB(shu.mat_proj, 1,  GL_FALSE, &matP.ax.x);
				glUniformMatrix4fvARB(shu.mat_model_view, 1, GL_FALSE, &rot_trans.ax.x);
				g_grid_mesh.render_mesh_shader();
			}


			a0+=da0;a1+=da1;a2+=da2;a3+=da3;a4+=da4;a5+=da5;
		}
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
		draw::rect_tex(&Vec2(-0.4,-0.4),&Vec2(0.2,0.1),z);
		draw::end();

		for x in 0..10 {
			for y in 0..10 {
				let fx=x as f32*0.1f32;
				let fy=y as f32*0.1f32;
				 
				draw::sprite_at(&Vec3(fx,fy,z),0.5f32, draw::pack(&Vec4(fx,fy,0.5f32,1.0f32)));
			}
		}

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

fn	create_textures() {
//	static_assert(sizeof(GLuint)==sizeof(int));
	unsafe {
		glGenTextures(1,&mut g_textures[0]);
		glBindTexture(GL_TEXTURE_2D,g_textures[0]);
		glTexEnvi( GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_MODULATE as GLint);
		glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR as GLint);
		glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR as GLint);
		glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT as GLint);
		glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT as GLint);

		let	(usize,vsize)=(256,256);
		let buffer:Vec<u32> = vec_from_fn(usize*vsize,&|index|{
			let (i,j)=div_rem(index,usize);
			(i+j*256+255*256*256) as u32
		});
		for i in 0 as GLint..8 as GLint {
			glTexImage2D(GL_TEXTURE_2D, i, GL_RGB as GLint, usize as GLint,vsize as GLint, 0, GL_RGB, GL_UNSIGNED_BYTE, as_void_ptr(&buffer[0]));
		}
		glBindTexture(GL_TEXTURE_2D,0);
	
//		g_textures[1] = get_texture(&"data/rocktile.tga");
//		g_textures[4] = get_texture(&"data/pebbles_texture.tga");
//		g_textures[3] = get_texture(&"data/grass.tga");
//		g_textures[2] = get_texture(&"data/cliffs.tga");
	}
}


static mut g_lazy_init:bool=false;
pub fn lazy_create_resources() {
	
	unsafe {
		if g_lazy_init==false {
			println!("lazy init shadertest resources\n");
			android_logw(c_str("lazy init shadertest resources \n"));
			create_shaders();
			create_textures();
			g_lazy_init=true;
			g_grid_mesh = Mesh::new_torus((16,16)); //new GridMesh(16,16);
		} else {
			println!("lazy init alredy done ok\n");
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
        render_no_swap();
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

#[cfg(ASMJS)]
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
fn load_file(fname:&str)->Vec<u8>{
	let mut buffer=Vec::<u8>::new();
	let f=File::open(fname).unwrap().read_to_end(&mut buffer);
	buffer
}
// todo - async..
fn create_texture_from_url(url:&str,waiting_color:u32)->GLuint{
	// todo - make a tmp filename hash
	unsafe {
		emscripten::emscripten_wget(c_str(url),c_str("tmp_image1.dat\0"));
		let buffer=load_file("tmp_image1.dat\0");
		let mut texname:GLuint=0;
		glGenTextures(1,&mut texname);
		glBindTexture(GL_TEXTURE_2D,texname);
		dump!(texname);

		glTexEnvi( GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_MODULATE as GLint);
		glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR as GLint);
		glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR as GLint);
		glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT as GLint);
		glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT as GLint);

		let	(usize,vsize)=(16,16);
		let buffer:Vec<u32> = vec_from_fn(usize*vsize,&|index|{
			waiting_color
		});
		for i in 0 as GLint..8 as GLint {
			glTexImage2D(GL_TEXTURE_2D, i, GL_RGB as GLint, usize as GLint,vsize as GLint, 0, GL_RGB, GL_UNSIGNED_BYTE, as_void_ptr(&buffer[0]));
		}
		texname
	}
}

#[cfg(SDL)]
fn sdl_mainloop() {
    unsafe {
        trace!();
        SDL_Init(SDL_INIT_EVERYTHING);
        let win = SDL_CreateWindow(c_str("hello sdl"), 0, 0, 640, 480, SDL_WINDOW_OPENGL);
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

pub fn main(){
	#[cfg(not(any(ASMJS,DESKTOP)))]
	compile_error!("must pass --cfg ASMJS or --cfg DESKTOP");


    let a=window::ScrPos(0.1,0.2);let b=window::ScrPos(0.4,0.6);
    let c=v2avr(&a,&b);
    println!("checking derived maths for tuple malarchy a avr b= {:?} b-a={:?}",c,v2sub(&b,&a));
    println!("{:?}", seq![x; x*2 ;for 0..10]);
	println!("{:?}", seq![x; x*2 ;for 0..10 ;if 0!=x&1]);
	println!("{:?}", seq![1,2,3]);
	println!("{:?}", seq![1,2,3]);
	println!("{:?}", seq![1=>10,2=>3,3=>5]);
#[cfg(ASMJS)]
unsafe {
	//test_browser_ui();
		//emscripten_run_script(cstr!("test_js_link('hello');"));
	//window::emscripten_set_main_loop(render_no_swap as *mut u8,60,1);
	//window::run_loop(vec![world::new()],&mut ());
//return;
	}
//	#[cfg(DESKTOP)]
//	window::run_loop(vec![Box::new(ShaderTest{time:30})],&mut ());
	#[cfg(ASMJS)]
//	test_file_download();
	#[cfg(DESKTOP)]
	window::run_loop(vec![world::new()],&mut ());
	#[cfg(ASMJS)]
    window::run_loop(vec![editor::make_editor_window::<(),editor::Scene>()] , &mut ());
    //window::run_loop(test::new(),&mut ());
//    bsp::bsp::main();
//	shadertest();
//	world::main();
}

//~ShaderTest{time:100} as ~window::State
