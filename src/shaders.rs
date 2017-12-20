use super::*;

#[derive(Clone,Debug,Copy)]
pub enum RenderMode{
	Default=0,
	Color,
	Tex0,
	Tex1,
	Tex0MulTex1,
	Tex0BlendTex1,
	Normal,
	TexCoord0,
	Light,
	SphericalHarmonicLight,
	FogTex0,
	Count
//	pub const Count:usize=6;
}
pub const RenderModeCount:usize=RenderMode::Count as usize;

pub static mut g_shader_program:[GLuint;RenderModeCount]=[-1i32 as uint;RenderModeCount];
pub static mut g_shader_uniforms:[UniformTable;RenderModeCount]=[
UniformTable{
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
};RenderModeCount
];
pub static mut g_pixel_shader:[GLuint;RenderModeCount]=[-1i32 as uint;RenderModeCount];
pub static mut g_vertex_shader:[GLuint;RenderModeCount]=[-1i32 as uint;RenderModeCount];

#[derive(Copy,Clone,Debug)]
#[repr(u32)]
pub enum VertexAttrIndex  {
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




pub unsafe fn get_attrib_location(shader_prog:GLuint, name:&str)->GLint {
	let r=glGetAttribLocation(shader_prog, c_str(name));
	println!("get attrib location {:?}={:?}", name, r);
	r
}
pub unsafe fn get_uniform_location(shader_prog:GLuint, name:&str)->GLint {
	let r=glGetUniformLocation(shader_prog, c_str(name));
	println!("get uniform_location location {:?}={:?}", name, r);
	r
}

pub unsafe fn	create_and_compile_shader(shader_type:GLenum, source:&str) ->GLuint
{
	android_logw_str("create_and_compile_shader\0");
	let	shader = glCreateShader(shader_type );
	android_logw_string(&format!("shader={:?}\0",shader));

	let sources_as_c_str:[*const c_char;1]=[c_str(source)];
	
	android_logw_str("set shader source..\0");
	glShaderSource(shader, 1 as GLsizei, &sources_as_c_str as *const *const c_char, 0 as *const c_int/*(&length[0])*/);
	android_logw_str("compile..\0");
	glCompileShader(shader);
	let	status:c_int=0;
	android_logw_str("get status..\0");
	glGetShaderiv(shader,GL_COMPILE_STATUS,&status);
	android_logw_str("got status\0");
	android_logw_string(&format!("status = {:?}\0",status));
	if status==GL_FALSE as GLint
	{
		android_logw_string(&format!("failed, getting log..\0"));
		let compile_log:[c_char;512]=[0 as c_char;512]; //int len;
	
		let log_len:c_int=0;
		glGetShaderInfoLog(shader, 512,&log_len as *const c_int, &compile_log[0]);
		android_logw_string(&format!("Compile Shader Failed: logsize={:?}\0",
				log_len));

		println!("compile shader {:?} failed: \n{:?}\n", shader, compile_log[0]);
//		println!("compile shader {:?} failed: \n{:?}\n", shader, 
//			CString::new((&compile_log[0]) as *const c_char).as_str());

		println!("TODO here 168");
		println!("{}",source);
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
		emscripten::alert(&format!("failed to build {}\0",source));
		panic!();


	}	
	else {
		println!("create shader{:?} - compile suceeded\n",  shader);
		android_logw_string(&format!("create shader{:?} - compile suceeded\n\0",  shader));
	}
	android_logw_str("create shader-done\0");
	shader
}


#[derive(Clone,Debug,Copy)]
pub struct	VertexAttr {
	pub pos:GLint,pub color:GLint,pub norm:GLint,pub tex0:GLint,pub tex1:GLint,pub joints:GLint,pub weights:GLint,pub tangent:GLint,pub binormal:GLint,
}
pub static g_vertex_attr_empty:VertexAttr=VertexAttr{
	pos:-1,color:-1,norm:-1,tex0:-1,tex1:-1,joints:-1,weights:-1,tangent:-1,binormal:-1
};

pub static mut g_vertex_shader_attrib:[VertexAttr;RenderModeCount]=[VertexAttr{
	pos:-1,color:-1,norm:-1,tex0:-1,tex1:-1,joints:-1,weights:-1,tangent:-1,binormal:-1};RenderModeCount];

//g_vertex_attr_empty;


pub static mut g_shader_uniforms_main:UniformTable=UniformTable{
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

pub static mut g_shader_uniforms_debug:UniformTable=UniformTable{
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

extern {pub fn bind_attrib_locations(prog:c_uint);}

pub unsafe fn	create_shader_program(
			pixelShaderSource:&str,
			vertexShaderSource:&str)->(PixelShader,VertexShader,ShaderProgram)
{

	android_logw(c_str("create_shader_program\0"));

	let pixelShaderOut = create_and_compile_shader(GL_FRAGMENT_SHADER, pixelShaderSource);
	let vertexShaderOut = create_and_compile_shader(GL_VERTEX_SHADER, vertexShaderSource);	
	let	prog = glCreateProgram();
	assert!(prog>=0);
	android_logw(c_str("bind attrib locations\0"));
	
	// assign attribute names before linking
	for &x in [
		(VertexAttrIndex::VAI_pos, "a_pos\0"),
		(VertexAttrIndex::VAI_color, "a_color\0"),
		(VertexAttrIndex::VAI_norm, "a_norm\0"),
		(VertexAttrIndex::VAI_tex0, "a_tex0\0"),
		(VertexAttrIndex::VAI_tex1, "a_tex1\0"),
		(VertexAttrIndex::VAI_joints, "a_joints\0"),
		(VertexAttrIndex::VAI_weights, "a_weights\0"),
		(VertexAttrIndex::VAI_tangent, "a_tangent\0"),
		(VertexAttrIndex::VAI_binormal, "a_binormal\0"),

	].iter() {glBindAttribLocation(prog, x.0 as GLuint, c_str(x.1));}

	glAttachShader(prog, pixelShaderOut);
	glAttachShader(prog, vertexShaderOut);

	println!("linking verteshader{:?}, pixelshader{:?} to program{:?}\n", vertexShaderOut, pixelShaderOut, prog);
	glLinkProgram(prog);
	let mut lstatus:GLint=0;
	glGetProgramiv(prog,GL_LINK_STATUS,(&lstatus) as *const GLint);
	

	
	if lstatus==(GL_FALSE as i32)
	{
		let mut buffer=[0 as GLchar;1024];
		let mut len:GLint=0;
		glGetProgramInfoLog(prog,1024,&len,&buffer[0]);
		println!("link program failed: {:?}",lstatus);
		println!("todo\n");
//		println!("{:?}",CString::new(&buffer[0]));
		panic!();
	} else {
		assert!(lstatus==(GL_TRUE as i32));
		println!("link ok");
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


static ps_vs_interface0:&'static str=&"
varying highp vec4 v_pos;
varying	highp vec4 v_color;
varying	highp vec3 v_norm;
varying	highp vec2 v_tex0;
varying	highp vec3 v_tex1uvw;
varying	highp vec4 v_tangent;
varying	highp vec4 v_binormal;
";

static ps_vertex_format0:&'static str=&
"attribute vec3 a_pos;
attribute vec2 a_tex0;
attribute vec4 a_color;
attribute vec3 a_norm;
";


static g_VS_Default:&'static str="
void main() {
	vec4 posw = vec4(a_pos.xyz,1.0);
	vec4 epos = uMatModelView * pos4;
	vec3 enorm = (uMatModelView * vec4(a_norm.xyz,0.0)).xyz;
	vec4 spos=uMatProj * epos;
	gl_Position = spos;
	v_pos = posw;
	v_color = a_color;
	v_tex0 = a_tex0;
	v_tex1uvw = a_pos.xyz;
	v_norm = enorm;
}
";

/// replacement debug vertex shader - dont apply transformations, just view vertices..
static g_VS_PassThru:&'static str="
void main() {
	vec4 posw = vec4(a_pos.xyz,1.0);
	vec4 epos = uMatModelView * posw;
	vec3 enorm = (uMatModelView * vec4(a_norm.xyz,0.0)).xyz;
	vec4 spos=uMatProj * epos;
	gl_Position = vec4(posw.xyz,1.0);
	v_pos = epos;
	v_color = a_color;
	v_tex0 = a_tex0;
	v_tex1uvw = a_pos.xyz;
	v_norm = enorm;
}
";
static g_vs_uniforms:&'static str="
uniform mat4 uMatProj;
uniform mat4 uMatModelView;
";

static g_VS_PassThruTweak:&'static str="
void main() {
	vec4 posw = vec4(a_pos.xyz,1.0);
	vec4 epos = uMatModelView * posw;
	vec3 enorm = (uMatModelView * vec4(a_norm.xyz,0.0)).xyz;
	vec4 spos=uMatProj * epos;
	gl_Position = spos;
	v_pos = posw;
	v_color = a_color;
	v_tex0 = a_tex0;
	v_tex1uvw = a_pos.xyz;
	v_norm = enorm;
}
";

/// replacement debug vertex shader - dont apply perspective, just view transformed models
static g_VS_RotTransPers:&'static str="
void main() {
	vec4 posw = vec4(a_pos.xyz,1.0);
	vec4 eye_pos = uMatModelView * posw;
	vec3 eye_norm = normalize((uMatModelView * vec4(a_norm.xyz,0.0)).xyz);
	vec4 screen_pos=uMatProj * eye_pos;
	gl_Position = screen_pos;
	v_pos = vec4(eye_pos.xyz,1.0);
	v_color = a_color;
	v_tex0 = a_tex0;
	v_tex1uvw = a_pos.xyz;
	v_norm = eye_norm;
}
";

/// replacement debug vertex shader - dont apply perspective, just view transformed models
static g_VS_Translate2d:&'static str="
void main() {
	vec4 posw = vec4(a_pos.xyz,1.0);
	vec4 epos = uMatModelView * posw;
	vec3 enorm = (uMatModelView * vec4(a_norm.xyz,0.0)).xyz;
	vec4 spos=uMatProj * epos;
	gl_Position = vec4(posw.xyz,1.0)+uMatModelView[3].xyzw;
	v_pos = posw;
	v_color = a_color;
	v_tex0 = a_tex0;
	v_tex1uvw = a_pos.xyz;
	v_norm = enorm;
}
";

static g_VS_Persp:&'static str="
void main() {
	vec4 posw = vec4(a_pos.xyz,1.0);
	vec4 epos = uMatModelView * posw;
	vec3 enorm = (uMatModelView * vec4(a_norm.xyz,0.0)).xyz;
	vec4 spos=uMatProj * epos;
	gl_Position = spos;
	v_pos = posw;
	v_color = a_color;
	v_tex0 = a_tex0;
	v_tex1uvw = a_pos.xyz;
	v_norm = enorm;
}
";


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
varying	highp vec3 v_tex1uvw;\n\
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

static g_PS_Alpha:&'static str= "
uniform sampler2D uTex0;
uniform sampler2D uTex1;
uniform vec4 uSpecularDir;
uniform float uSpecularPower;
uniform vec4 uSpecularColor;
uniform vec4 uFogColor;
uniform vec4 uFogFalloff;
uniform vec4 uAmbient;
uniform vec4 uDiffuseDX;
uniform vec4 uDiffuseDY;
uniform vec4 uDiffuseDZ;
    
uniform vec4 uLightPos;
uniform vec4 uLightColor;
uniform vec4 uLightFalloff;
void main() { 
	float inva=(v_color.w),a=(1.0-v_color.w);
	vec4 t0=texture2D(uTex0, v_tex0);
	vec4 t1=texture2D(uTex1, v_tex0);
	float a0=t0.x*0.4+t0.y*0.6+t0.z*0.25;
	float a1=t1.x*0.4+t1.y*0.6+t1.z*0.25;
	float highlight=max(0.0,dot(v_norm,uSpecularDir.xyz));
		highlight=(highlight*highlight);highlight=highlight*highlight;
	vec4 surfaceColor=mix(t0,t1,v_color.w);
	vec4 surfaceSpec=clamp(4.0*(surfaceColor-vec4(0.5,0.5,0.5,0.0)), vec4(0.0,0.0,0.0,0.0),vec4(1.0,1.0,1.0,1.0));
	vec4 spec=highlight*uSpecularColor*surfaceSpec;
	vec4 diff=uAmbient+v_norm.x*uDiffuseDX+v_norm.y*uDiffuseDY+v_norm.z*uDiffuseDZ;
	float lx=0.5,ly=0.5;
	gl_FragColor =applyFog(v_pos.xyz,surfaceColor*evalAllLight()*vec4(v_color.xyz,0.0)*2.0+spec);
//	gl_FragColor =vec4(v_norm.xyz,0.0)*0.5+vec4(0.5,0.5,0.5,1.0)+vec4(v_tex0,0.0,0.0);
}";

// debug shader
static g_PS_Add:&'static str= "
uniform sampler2D s_tex0;
uniform sampler2D s_tex1;
uniform vec4 uSpecularDir;
uniform float uSpecularPower;
uniform vec4 uSpecularColor;
uniform vec4 uAmbient;
uniform vec4 uDiffuseDX;
uniform vec4 uDiffuseDY;
uniform vec4 uDiffuseDZ;
void main() { 
	float inva=(v_color.w),a=(1.0-v_color.w);
	vec4 t0=texture2D(s_tex0, v_tex0);
	vec4 t1=texture2D(s_tex1, v_tex0);
	float a0=t0.x*0.4+t0.y*0.6+t0.z*0.25;
	float a1=t1.x*0.4+t1.y*0.6+t1.z*0.25;
	float highlight=max(0.0,dot(v_norm,uSpecularDir.xyz));
	highlight=(highlight*highlight);highlight=highlight*highlight;
	vec4 surfaceColor=t0+(t1-vec4(0.5f,0.5f,0.5f,0.0f))*v_color.w;
	vec4 surfaceSpec=clamp(4.0*(surfaceColor-Vec4(0.5,0.5,0.5,0.0)), vec4(0.0,0.0,0.0,0.0),vec4(1.0,1.0,1.0,1.0));
	vec4 spec=highlight*uSpecularColor*surfaceSpec;
	vec4 diff=uAmbient+vso_norm.x*uDiffuseDX+vso_norm.y*uDiffuseDY+vso_norm.z*uDiffuseDZ;
	gl_FragColor =surfaceColor*diff*vec4(v_color.xyz,0.0)*2.0+spec;
}";

static g_PS_Flat:&'static str="
void main() {
	gl_FragColor= mediump vec4(0.0, 1.0, 0.0, 1.0);
}
";

static g_PS_FogTex0:&'static str="
void main() {
	gl_FragColor= evalFog(getTex0());
}
";

static g_PS_Light:&'static str="
void main() {
	gl_FragColor= evalAllLight();
}
";
static g_PS_SphericalHarmonicLight:&'static str="
void main() {
	gl_FragColor= evalAllLight();
}
";


static g_PS_MinimumDebugAndroidCompiler:&'static str= &"
precision mediump float; 
void main() 
{ 
 gl_FragColor = vec4(1.0, 0.0, 0.0, 1.0); 
} 
";

static g_PS_Tex3_AlphaMul:&'static str=&"
uniform sampler2D s_tex0;	
uniform sampler2D s_tex1;	
uniform sampler2D s_tex2;	
uniform vec4 uSpecularDir;	
uniform float uSpecularPower;	
uniform vec4 uSpecularColor;	
uniform vec4 uAmbient;			
uniform vec4 uDiffuseDX;		
uniform vec4 uDiffuseDY;		
uniform vec4 uDiffuseDZ;		
void main() { 
	float inva=(v_color.w),a=(1.0-v_color.w);		
	vec4 t0=texture2D(s_tex0, v_tex0);				
	vec4 t1=texture2D(s_tex1, v_tex0);				
	float a0=t0.x*0.4+t0.y*0.6+t0.z*0.25;			
	float a1=t1.x*0.4+t1.y*0.6+t1.z*0.25;			
	float highlight=max(0.0,dot(v_norm,uSpecularDir.xyz));	
	highlight=(highlight*highlight);highlight=highlight*highlight; 
	vec4 surfaceColor=mix(t0,t1, v_color.w*t1.a);				 
	vec4 surfaceSpec=clamp(4.0*(surfaceColor-Vec4(0.5,0.5,0.5,0.0)), vec4(0.0,0.0,0.0,0.0),vec4(1.0,1.0,1.0,1.0));                            
	vec4 spec=highlight*uSpecularColor*surfaceSpec;	
	vec4 diff=uAmbient+vso_norm.x*uDiffuseDX+vso_norm.y*uDiffuseDY+vso_norm.z*uDiffuseDZ;
	gl_FragColor =surfaceColor*diff*Vec4(v_color.xyz,0.0)*2.0+spec;
}
";
static g_PS_DeclUniforms:&'static str="
uniform sampler2D s_tex0;			
uniform sampler2D s_tex1;			
uniform sampler2D s_tex2;			
uniform vec4 uSpecularDir;			
uniform float uSpecularPower;		
uniform vec4 uSpecularColor;		
uniform vec4 uAmbient;				
uniform vec4 uDiffuseDX;			
uniform vec4 uDiffuseDY;			
uniform vec4 uDiffuseDZ;
uniform vec4 uFogFalloff;
uniform vec4 uFogColor;

";

// passthrough various minimal versions
static g_PS_TexCoord0:&'static str="
void main() {
	gl_FragColor =vec4(mod(v_tex0.x,1.0),mod(v_tex0.y,1.0),0.0,1.0);
}
";

static g_PS_TexCoord1:&'static str="
void main() {
	gl_FragColor =vec4(mod(v_tex1uvw.x,1.0),mod(v_tex1uvw.y,1.0),mod(v_tex1uvw.z,1.0),1.0);
}
";

static g_PS_Normal:&'static str="
void main() {
	gl_FragColor =vec4(v_norm*0.5+vec3(0.5,0.5,0.5),1.0);
}
";
static g_PS_Color:&'static str="
void main() {
	gl_FragColor =v_color; 
}
";

static g_PS_Tex0:&'static str=&"
void main() {

	gl_FragColor=texture2D(s_tex0, v_tex0);
}
";

static  g_PS_common:&'static str="
vec4 applyFogAt(vec3 pos,vec4 unfogged_color){
	return mix(unfogged_color,uFogColor,  clamp(-uFogFalloff.x-pos.z*uFogFalloff.y,0.0,1.0));
}
vec4 evalFog(vec4 surface_color){
	return applyFogAt(v_pos.xyz, surface_color);
}

vec4 pointlight(vec3 pos, vec3 norm,vec3 lpos, vec4 color, vec4 falloff) {
	vec3 dv=lpos-pos;
	float d2=sqrt(dot(dv,dv));
	float f=clamp( 1.0-(d2/falloff.x),0.0,1.0);
	vec3 lv=normalize(dv);
	return clamp(dot(lv,norm),0.0,1.0) * f*color;
}
// hardcoded point lights;
// todo: feed aproximation of N lights
// through SH centre and 'most-significant-nearby-pointlight'

vec4 evalPointLights(){
	float lx=0.5,ly=0.5;
	vec4 acc=pointlight(v_pos.xyz,v_norm.xyz, vec3(lx,ly,-1.0),		vec4(1.0,0.0,0.0,0.0),vec4(1.0,0.0,0.0,0.0));
	acc+=pointlight(v_pos.xyz,v_norm.xyz, vec3(lx,-ly,-1.0), 	vec4(0.0,1.0,0.0,0.0),vec4(1.0,0.0,0.0,0.0));
	acc+=pointlight(v_pos.xyz,v_norm.xyz, vec3(-lx,-ly,-1.0),	vec4(0.0,0.0,1.0,0.0),vec4(1.0,0.0,0.0,0.0));
	acc+=pointlight(v_pos.xyz,v_norm.xyz, vec3(-lx,ly,-1.0), 	vec4(0.5,0.0,0.5,0.0),vec4(1.0,0.0,0.0,0.0));
	return acc;
}
vec4 evalSphericalHarmonic(){
	return uAmbient+v_norm.x*uDiffuseDX+v_norm.y*uDiffuseDY+v_norm.z*uDiffuseDZ;
}
vec4 evalAllLight(){
	return evalSphericalHarmonic()+evalPointLights();
}

vec4 getTex1(){
	vec3 factors=v_norm*v_norm;
	vec3 fnorm=normalize(factors*factors);	
	return 
		texture2D(s_tex1, v_tex1uvw.xy)*fnorm.x+
		texture2D(s_tex1, v_tex1uvw.xz)*fnorm.y+
		texture2D(s_tex1, v_tex1uvw.yz)*fnorm.z;
}

vec4 getTex0(){
	return texture2D(s_tex0,v_tex0);
}

";
static g_PS_Tex1Triplanar:&'static str=&"
void main() {
	gl_FragColor =getTex1();
}
";
static g_PS_Tex0MulTex1:&'static str=&"
// todo: we need to know the normalization factor,
// not all textures are mid-grey.
void main() {
	gl_FragColor =getTex1() * getTex0()*vec4(2.0,2.0,2.0,1.0);
}
";
static g_PS_Tex0AddTex1:&'static str=&"
// todo: we need to know the normalization factor,
// not all textures are mid-grey.
void main() {
	gl_FragColor =getTex1() + getTex0()-vec4(0.5, 0.5, 0.5, 0.5);
}
";
// blend the textures with their alpha; vertex alpha controls worldblend
static g_PS_Tex0BlendTex1:&'static str=&"
void main() {
	gl_FragColor =mix(getTex0(),getTex1,v_color.w);
}
";


// tex 1 blend added by vertex alpha
static g_PS_Tex0VertexBlendTex1:&'static str=&"
void main() {
	vec4 tex0=getTex0;
	vec4 tex1=getTex1;
	gl_FragColor =mix(getTex1() , getTex0(), clamp(tex1.w+(v_color.w-0.5)*2.0 ,0.0,1.0) );
}
";

#[derive(Clone,Debug,Default,Copy)]
pub struct UniformTable {
	pub mat_proj:UniformIndex,
	pub mat_model_view:UniformIndex,
	pub mat_model_view_proj:UniformIndex,
	pub mat_color:UniformIndex,
	pub mat_env_map:UniformIndex,
	pub tex0:UniformIndex,
	pub tex1:UniformIndex,
	pub cam_in_obj:UniformIndex,
	pub ambient:UniformIndex,
	pub diffuse_dx:UniformIndex,
	pub diffuse_dy:UniformIndex,
	pub diffuse_dz:UniformIndex,
	pub specular_color:UniformIndex,
	pub specular_dir:UniformIndex,
	pub sky_dir:UniformIndex,
	pub test_vec_4:UniformIndex,
	pub fog_color:UniformIndex,
	pub fog_falloff:UniformIndex,
	pub light0_pos_r:UniformIndex,
	pub light0_color:UniformIndex,
}
pub static g_uniform_table_empty:UniformTable= UniformTable{
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
pub fn map_shader_params(prog:GLuint)->(VertexAttr,UniformTable)
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
				pos: get_attrib_location(prog, &"a_pos\0"),
				color: get_attrib_location(prog, &"a_color\0"),
				norm: get_attrib_location(prog, &"a_norm\0"),
				tex0: get_attrib_location(prog, &"a_tex0\0"),
				tex1: get_attrib_location(prog, &"a_tex1\0"),
				joints: get_attrib_location(prog, &"a_joints\0"),
				weights: get_attrib_location(prog, &"a_weights\0"),
				tangent: get_attrib_location(prog, &"a_binormal\0"),
				binormal: get_attrib_location(prog, &"a_tangent\0")

			},
			UniformTable{
				mat_proj:get_uniform_location(prog,&"uMatProj\0"),
				mat_model_view:get_uniform_location(prog,&"uMatModelView\0"),
				specular_color:get_uniform_location(prog,&"uSpecularColor\0"),
				specular_dir:get_uniform_location(prog,&"uSpecularDir\0"),
				ambient:get_uniform_location(prog,&"uAmbient\0"),
				diffuse_dx:get_uniform_location(prog,&"uDiffuseDX\0"),
				diffuse_dy:get_uniform_location(prog,&"uDiffuseDY\0"),
				diffuse_dz:get_uniform_location(prog,&"uDiffuseDZ\0"),
				fog_color:get_uniform_location(prog,&"uFogColor\0"),
				fog_falloff:get_uniform_location(prog,&"uFogFalloff\0"),
				light0_pos_r:get_uniform_location(prog,&"uLight0PosR\0"),
				light0_color:get_uniform_location(prog,&"uLight0Color\0"),
				..g_uniform_table_empty
			}
		)
	}	
}
#[cfg(any(target_os = "android",target_os="emscripten"))]
pub fn get_shader_prefix(st:ShaderType)->&'static str {
	use ShaderType::*;
	match st{
		ShaderType::Pixel=>pixel_shader_prefix_gles,
		ShaderType::Vertex=>vertex_shader_prefix_gles,
		_=>panic!("can't deal with this case")
	}
}

/*
#[cfg(not(target_os = "android"))]
fn get_shader_prefix(is_ps:int)->&'static str {
	if is_ps==0 {vertex_shader_prefix_gles} else {pixel_shader_prefix_gles}
}
*/

#[cfg(any(target_os = "macos",target_os="linux"))]
pub fn get_shader_prefix(st:ShaderType)->&'static str {
	shader_prefix_desktop
}
pub fn concat_shader(src:&[&str])->String{
	let mut ret=String::new();
	// todo- insert \n after each statement?
	for (i,x) in src.iter().enumerate() {
		ret.push_str(format!("//from part {}\n",i).as_str());// todo - pass name thru macro
		ret.push_str(x);
		ret.push_str("\n");
	}
	ret.push_str("\0");	// null terminatin
	ret
}
pub fn create_shader_sub(mode:RenderMode, ps:&[&str], vs:&[&str]){
	println!("CREATE SHADER MODE {:?} \n",mode);
	let psconcat=concat_shader(ps);
	let vsconcat=concat_shader(vs);
	let (vsh,psh,prg)=unsafe{
		create_shader_program(&psconcat,&vsconcat)
	};
	let (vsa,su)=map_shader_params(prg);
	println!("vs={:?}",vs);
	println!("su={:?}",su);
	let modei=mode as usize;
	unsafe{
		g_shader_program[modei]=prg;
		g_vertex_shader[modei]=vsh;
		g_pixel_shader[modei]=psh;	
		g_shader_uniforms[modei]=su;
		g_vertex_shader_attrib[modei]=vsa;
	}
	println!("CREATE SHADER MODE DONE{:?} \n",mode);
}
pub fn create_shader_sub2(mode:RenderMode, ps_main:&str){
	create_shader_sub(
		mode, 
		&[	get_shader_prefix(ShaderType::Pixel),
			g_PS_DeclUniforms,
			ps_vs_interface0,
			g_PS_common,
			ps_main],
		&[get_shader_prefix(ShaderType::Vertex),
			g_vs_uniforms,
			ps_vertex_format0,
			ps_vs_interface0,
			g_VS_RotTransPers]);
}

pub fn	create_shaders()
{
	println!("create shaders");
//todo: vs, ps, interface, permute all with same interface
//or allow shader(vertexmode,texmode,lightmode)

	create_shader_sub(
		RenderMode::Default,
		&[	get_shader_prefix(ShaderType::Pixel),
			g_PS_DeclUniforms,
			ps_vs_interface0,
			g_PS_common,
			g_PS_Tex0/*g_PS_Alpha*/
		],
		&[	get_shader_prefix(ShaderType::Vertex),
			g_vs_uniforms,
			ps_vertex_format0,
			ps_vs_interface0, 
			g_VS_PassThruTweak
		]);

	create_shader_sub2(
		RenderMode::Tex0, 
		g_PS_Tex1Triplanar);

	create_shader_sub2(
		RenderMode::Tex1, 
		g_PS_Tex1Triplanar);

	create_shader_sub2(
		RenderMode::Tex0MulTex1, 
		g_PS_Tex0MulTex1);
	create_shader_sub2(
		RenderMode::Tex0BlendTex1, 
		g_PS_Tex0MulTex1);
	create_shader_sub2(
		RenderMode::Light, 
		g_PS_Light);
	create_shader_sub2(
		RenderMode::SphericalHarmonicLight, 
		g_PS_SphericalHarmonicLight);
	create_shader_sub2(
		RenderMode::FogTex0, 
		g_PS_FogTex0);
	create_shader_sub2(
		RenderMode::Color, 
		g_PS_Color);
	create_shader_sub2(
		RenderMode::Normal, 
		g_PS_Normal);
	create_shader_sub2(
		RenderMode::TexCoord0, 
		g_PS_TexCoord0);
}
