pub mod vectypes;
pub mod vector;
pub mod matrix;
pub mod quaternion;
pub mod geom;
pub mod mesh;
pub mod sdl;
pub mod landscape;
pub mod rawglbinding;
pub mod gl_constants;
pub mod gl_h_consts;
pub mod glut_h_consts;
pub mod tuplevecmath;
pub mod cstuff;
pub mod draw;
pub mod dimensionval;
pub mod macros;
pub mod scene;
pub mod classes;
pub mod bsp;
pub use sdl::*;
// Common standard library imports.. file, memory, operators..
pub use ::std::{io,fs,ops,path,mem,ffi,os,num,cmp,vec,collections,fmt,marker,convert};
pub use io::Read;
pub use fmt::Debug;

use ::std::f32::consts::PI;

pub use ffi::{CString,CStr};
pub use ffi::CString as c_str;

pub use os::raw;
pub use raw::{c_int,c_uint,c_void,c_char};

pub use collections::{HashSet,HashMap};


pub use fs::File;
pub use path::Path;
pub use mem::size_of;

pub use num::*;
pub use cmp::{Ord,PartialOrd,PartialEq};
pub use ops::{Add,Sub,Mul,Div,Rem,Neg,BitOr,BitAnd,Not,BitXor,Deref};

pub type int=i32;
pub type uint=u32;
type Scalar=f32;

pub use vector::*;

pub use rawglbinding::*;
pub use geom::*;
pub use gl_constants::*;
pub use gl_h_consts::*;
pub use glut_h_consts::*;
pub use vector::{Vec3f,Vec2f,Vec4f, Vec1,Vec3,Vec4,Vec2};
pub use matrix::{Mat33f,Mat34f,Mat43f,Mat44f,Mat34,Mat44,Mat33,Mat43};
//pub use dimensionval;


pub enum ShaderType{
	Pixel,Vertex,Geometry,Compute
}


//pub static GL_TEXTURE_2D:rawglbinding::uint=0x0DE1;

pub unsafe fn c_str(s:&str)->*const c_char {
	// unfortunately thats not how it works. these need to be manually null terminated!
	//assert!(s.len()>0);
	let len=s.len() as isize;
	
	let r=s.as_ptr() as *const c_char;
	let last_char=*r.offset(len-1);
	if last_char!=0{
		let mut i=0 as isize;
		while i<len{ 
			println!("s[{}]={:?}",i,*r.offset(i));
			i+=1;
		}
		panic!("non null terminated string passed to C '{}' length={} last_char='{}'",s,len, last_char);
	}
	r
}

pub fn vec_from_fn<T:Sized+Clone,F:Fn(usize)->T>(num:usize , f:&F)->Vec<T>{
	// todo - this must be official ? generator, whatever
	let mut r=Vec::<T>::new();
	r.reserve(num);
	println!("init vector {:?}elems",num);
	for x in 0..num{
		r.push(f(x)) 
	}
	r
}

// a few helpers
macro_rules! gl_verify{
	([$dbg:stmt] $($e:stmt;)+  )=>{
		$($e;
		{	let err=glGetError();
			if err!=GL_NO_ERROR && false{
				println!("{}:{} failed{}:{}",file!(),line!(),gl_error_str(err),stringify!($e));
				$dbg
			}
		})*
	};
	($($e:stmt;)+)=>{
		$($e;
		{	let err=glGetError();
			if err!=GL_NO_ERROR && false{
				println!("{}:{} failed{}:{}",file!(),line!(),gl_error_str(err),stringify!($e));
			}
		})*
	}
}


macro_rules! map_err_str{
	($e:expr=>$($x:ident,)*)=> {match $e{$($x=>stringify!($x),)*}}
}
pub fn gl_error_str(err:GLuint)->&'static str{
//	map_err_str!{err as uint=>
	match err{
		GL_NO_ERROR=>"GL_NO_ERROR",
		GL_INVALID_ENUM=>"GL_INVALID_ENUM",
		GL_INVALID_VALUE=>"GL_INVALID_VALUE",
		GL_INVALID_OPERATION=>"GL_INVALID_OPERATION",
		GL_INVALID_FRAMEBUFFER_OPERATION=>"GL_INVALID_FRAMEBUFFER_OPERATION",
		GL_OUT_OF_MEMORY=>"GL_OUT_OF_MEMORY",
		_=>"Unparsed Error",
	}
//	}
}
macro_rules! cstr{
	($txt:expr)=>{
		c_str(concat!($txt,"\0"))
	}
}

pub unsafe fn as_void_ptr<T>(ptr:&T)->*const c_void {
	ptr as *const T as *const c_void
}
#[derive(Clone,Debug)]
pub struct	MyVertex 
{
	pub pos:[f32;3],
	pub color:[f32;4],
	pub norm:[f32;3],
	pub tex0:[f32;2]
}


pub struct PackedARGB(pub u32);	// 'Color', most commonly packed 8888
pub struct PackedS8x4(pub u32);	// packed vector in -1 to 1 range
pub struct PackedARGB1555(pub u16);	// 'Color', most commonly packed 8888
pub struct PackedRGB565(pub u16);	// 'Color', most commonly packed 8888
type Color = PackedARGB;        // ånother synonym
type PackedARGB8888=PackedARGB;    // explicit
							// other representations can be more specific.
type Point3= vector::Vec3<f32>;
// todo: f16
type Normal= vector::Vec3<f32>;
struct PackedXYZ(pub u32);

fn to_u32(f:f32)->u32{
	(f*255.0f32) as u32
}

type Radians=f32;	// the most natural mathematical angle gets the raw type.
// anything which can be an angle..

pub trait Angle :Sized {
	fn to_radians(&self)->Radians{unimplemented!()}
	fn to_degrees(&self)->Degrees{unimplemented!()}
	fn from_fraction(&self,num:isize,denom:isize )->Self{unimplemented!()}
	fn to_fraction_num(&self,denom:isize)->isize{unimplemented!()}
	fn sin(&self)->f32{unimplemented!()}
	fn cos(&self)->f32{unimplemented!()}
	fn tan(&self)->f32{unimplemented!()}
	fn sin_cos(&self)->(f32,f32){(self.sin(),self.cos())}
}
impl Angle for Radians{
	fn to_fraction_num(&self,denom:isize)->isize{
		((*self) * (denom as f32 / (PI*2.0f32))) as isize
	}
	fn to_degrees(&self)->Degrees{ Degrees(*self * 360.0f32 / (2.0f32*PI)) }
}

pub struct Degrees(f32);

impl Angle for Degrees{
	fn to_radians(&self)->Radians{self.0*(PI*2.0f32/360.0f32)}
}

/// means of throwing ints into the typesys
pub struct TN7{}
/// means of throwing ints into the typesys
pub struct TN15{}
/// means of throwing ints into the typesys
pub struct TN16{}
/// means of throwing ints into the typesys
pub struct TN12{}
/// means of throwing ints into the typesys
pub struct TN8{}
/// means of throwing ints into the typesys
pub struct TN5{}
/// means of throwing ints into the typesys
pub struct TN6{}
/// means of throwing ints into the typesys
pub struct TN4{}
/// means of throwing ints into the typesys
pub struct TN3{}
/// means of throwing ints into the typesys
pub struct TN2{}

#[derive(Debug,Copy,Clone)]
pub struct FixedPt<T,Fraction>(T,Fraction);

/// getting a number from
pub trait TNum {
	type Output;
	fn value()->usize;
	fn get_phantom()->Self::Output;
}

impl TNum for TN16{ fn value()->usize{16} type Output=TN16; fn get_phantom()->TN16{TN16{}} }
impl TNum for TN15{ fn value()->usize{15} type Output=TN15; fn get_phantom()->TN15{TN15{}} }
impl TNum for TN12{ fn value()->usize{12} type Output=TN12; fn get_phantom()->TN12{TN12{}} }
impl TNum for TN8{ fn value()->usize{8} type Output=TN8; fn get_phantom()->TN8{TN8{}} }
impl TNum for TN7{ fn value()->usize{7} type Output=TN7; fn get_phantom()->TN7{TN7{}} }
impl TNum for TN5{ fn value()->usize{5} type Output=TN5; fn get_phantom()->TN5{TN5{}} }
impl TNum for TN4{ fn value()->usize{4} type Output=TN4; fn get_phantom()->TN4{TN4{}} }
impl TNum for TN3{ fn value()->usize{3} type Output=TN3; fn get_phantom()->TN3{TN3{}} }
impl TNum for TN2{ fn value()->usize{2} type Output=TN2; fn get_phantom()->TN2{TN2{}} }

impl<T,X:TNum<Output=X>> From<f32> for FixedPt<T,X> where f32:Into<T>, T: ::std::ops::Mul<T,Output=T>, T:From<i32>{
	fn from(s:f32)->Self{
		let x:f32 = (1<< (<X as TNum>::value() ) ) as f32;
		FixedPt::<T,X>((s * x).into() ,<X as TNum>::get_phantom())
	}
}
impl<T,X:TNum> From<FixedPt<T,X>> for f32 where T:Into<f32>{
	fn from(s: FixedPt<T,X> )->Self{
		s.0.into() * (1.0f32 / ((1<< <X as TNum>::value()) as f32))
	}
}


impl From<PackedARGB8888> for vector::Vec4<f32> {
	fn from(src:PackedARGB)->Self{
		let scale=1.0f32/255.0f32;		
		vector::Vec4(
			(src.0 & 0xff)as f32 * scale,
			((src.0>>8) & 0xff)as f32 * scale,
			((src.0>>16) & 0xff)as f32 * scale,
			((src.0>>24) & 0xff)as f32 * scale
		)
	}
}
impl From<vector::Vec4<f32>> for PackedARGB {
	fn from(src: vector::Vec4<f32>) ->Self {
		PackedARGB(to_u32(src.x)|(to_u32(src.y)<<8)|(to_u32(src.z)<<16)|(to_u32(src.w)<<24))
	}
}
impl From<PackedXYZ> for vector::Vec3<f32> {
	fn from(src:PackedXYZ)->Self{
		let centre = (1<<9) as i32;
		let scale=1.0f32/511.0f32;		
		let mask=(1i32<<10)-1;
		let val=src.0 as i32;
		vector::Vec3(
			(((val) & mask)-centre)as f32 * scale,
			(((val>>10) & mask)-centre)as f32 * scale,
			(((val>>20) & mask)-centre)as f32 * scale
		)
	}
}

impl From<vector::Vec3<f32>> for PackedXYZ {
	fn from(src: vector::Vec3<f32>) ->Self {
		let ix=((src.x*511.0f32)+512.0f32) as u32;
		let iy=((src.y*511.0f32)+512.0f32) as u32;
		let iz=((src.z*511.0f32)+512.0f32) as u32;
		PackedXYZ(ix|(iy<<10)|(iz<<20))
	}
}

mod vertex {
	use super::*;
	use super::vector::*;
	//various vertex types..
	struct PT(Vec3,Vec2);
	struct PC(Vec3,PackedARGB);
	struct PNCT(Vec3,Vec3,PackedARGB,Vec2);
}

/// simple wrapper around GL, aimed at debug graphics
pub trait DrawingWrapper {
	// todo - propper..textures/VBOs, bone matrices, shaders..
	fn push_matrix(&mut self, matrix::Mat44f);
	fn pop_matrix(&mut self);
	fn begin(&mut self,v_per_prim: i32); //tristrips=-1, 1=points,2=lines,3=tris,4=quads
	fn vertex(&mut self, a: Vec3f, u32);
	fn end(&mut self);
}

pub trait Draw : DrawingWrapper {
	fn print(&mut self, s:&str, cc:u32) {
		println!("{:?}",s);
	}
	fn point(&mut self,a:Vec3f, cc:u32) {
		self.begin(1); self.vertex(a,cc); self.end();
	}
	fn line(&mut self,a:Vec3f, b:Vec3f,cc:u32)	{
		self.begin(2); self.vertex(a,cc); self.vertex(b,cc); self.end();
	}
	fn triangle(&mut self,a:Vec3f, b:Vec3f, c:Vec3f, cc:u32)	{
		self.begin(3); self.vertex(a,cc); self.vertex(b,cc); self.vertex(c,cc); self.end();
	}
	fn quad(&mut self, a:Vec3f,b:Vec3f,c:Vec3f,d:Vec3f,cc:u32)	{
		self.begin(4); self.vertex(a,cc); self.vertex(b,cc); self.vertex(c,cc); self.vertex(d,cc); self.end();
	}
	fn cuboid_v(&mut self, a:&[Vec3f], cc:u32) {
		unimplemented!();
	}
	fn cuboid(&mut self, size:Vec3f, cc:u32) {
		unimplemented!();
	}
	fn grid(&mut self, vtc:&[Vec3f],colors:&[u32], sz:(i32,i32)){unimplemented!()}
	fn spheroid(&mut self, pos:Vec3f, sz:Vec3f, num:(i32,i32), cc:u32);
}


pub trait Object {

}

// completion of plain numeric types.. one/zero, trigmath

pub trait Half {
	fn half()->Self;
}
pub trait Two {
	fn two()->Self;
}

pub trait One {
	fn one()->Self;
	fn is_one(&self)->bool;
}
impl One for f32{
	fn one()->f32{1.0f32}
	fn is_one(&self)->bool{*self == Self::one()}
}
impl Two for f32{
	fn two()->f32{1.0f32}
}

impl One for f64{
	fn one()->f64{1.0f64}
	fn is_one(&self)->bool{*self == Self::one()}
}

pub fn two<F:Two>()->F{ <F as Two>::two() }
pub fn half<F:Half>()->F{ <F as Half>::half() }
pub fn one<F:One>()->F{ <F as One>::one() }
pub fn zero<F:Zero>()->F{ <F as Zero>::zero() }

pub trait Zero {
	fn zero()->Self;
	fn is_zero(self)->bool;
}
impl Zero for f32 {
	fn zero()->f32{return 0.0f32}
	fn is_zero(self)->bool { if self==Self::zero(){true}else{false}}
}

impl Zero for f64 {
	fn zero()->f64{return 0.0f64}
	fn is_zero(self)->bool { if self==Self::zero(){true}else{false}}
}
/*
pub trait RefNum {
	type Output;
}

impl<T> RefNum for T where
		for<'a,'b> &'a T : Sub<&'b T,Output=T>,
		for<'a,'b> &'a T : Add<&'b T,Output=T>,
		for<'a,'b> &'a T : Mul<&'b T,Output=T>,
		for<'a,'b> &'a T : Div<&'b T,Output=T>,
		for<'a> &'a Self : Neg<Output=Self>
//		for<'a,'b> &'a Self : Sub<&'b Self,Output=Self>,
{
	type Output = T;
}
*/



pub trait Num :
Copy+Neg<Output=Self> +
Add<Self,Output=Self>+
Sub<Self,Output=Self>+
Mul<Self,Output=Self>+
Div<Self,Output=Self> +
PartialOrd +
PartialEq +
Sqrt<Output=Self>+
RSqrt<Output=Self>+
Sized +
One+
Zero+
//where for <'a, 'b> &'a Self:Sub<&'b Self, Output=Self>
//where for<'a,'b> &'a Self : Sub<&'b Self,Output=Self>+

{
}


impl<T> Num for T where T:
Copy+
Neg<Output=T>+
Add<T,Output=T>+
Sub<T,Output=T>+
Mul<T,Output=T>+
Div<T,Output=T>+
Sqrt<Output=T>+
RSqrt<Output=T>+
PartialOrd+PartialEq+
Clone+
One+
Zero,
//for<'a,'b> &'a T : Sub<&'b T,Output=T>
{
}
/*
impl<'t, T> Num for &'t T where for<'a,'b> &'a T:
Copy+
Neg<Output=T>+
Add<&'b T,Output=T>+
Sub<&'b T,Output=T>+
Mul<&'b T,Output=T>+
Div<&'b T,Output=T>+
Sqrt<Output=T>+
RSqrt<Output=T>+
PartialOrd+PartialEq+
Clone+
One+
Zero,
{
}
*/

/*
impl<'a> One for &'a f32{
	fn one()->f32{1.0f32}
	fn is_one(&'a self)->bool{*self == Self::one()}
}

impl<'a> One for &'a f64{
	fn one()->f64{1.0f64}
	fn is_one(&'a self)->bool{*self == Self::one()}
}

impl<'a > Zero for &'a f32 {
	fn zero()->f32{return 0.0f32}
	fn is_zero(&'a self)->bool { if self==Self::zero(){true}else{false}}
}

impl<'a> Zero for &'a f64 {
	fn zero()->f64{return 0.0f64}
	fn is_zero(&'a self)->bool { if self==Self::zero(){true}else{false}}
}
*/

/// TODO - figure out which parts of this live in 'vector
/// we have better traits for this with dimensionable output
pub trait Float : Num+Half+Clone+Default {
	fn sin(self)->Self;
	fn cos(self)->Self;
	fn sin_cos(self)->(Self,Self){(self.sin(),self.cos())}
	fn tan(self)->Self;
    fn exp(self)->Self;
    fn ln(self)->Self;
    fn log(self,Self)->Self;
    fn powf(self,s:Self)->Self;
    fn acos(self)->Self;
    fn asin(self)->Self;
    fn atan(self)->Self;
    fn recip(self)->Self;
}
impl Float for f32{
	fn tan(self)->Self { self.tan()}
    fn atan(self)->Self { self.atan()}
    fn powf(self,s:Self)->Self{self.powf(s)}
    fn exp(self)->Self{self.exp()}
    fn ln(self)->Self{self.ln()}
    fn log(self,base:Self)->Self{self.log(base)}
    fn asin(self)->Self { self.asin()}
    fn acos(self)->Self { self.acos()}
	fn sin(self)->Self { self.sin()}
	fn cos(self)->Self { self.cos()}
	fn sin_cos(self)->(Self,Self){self.sin_cos()}
    fn recip(self)->Self { self.recip()}
}
impl Float for f64{
	fn tan(self)->Self {self.tan()}
    fn atan(self)->Self {self.atan()}
    fn powf(self,s:Self)->Self{self.powf(s)}
    fn exp(self)->Self{self.exp()}
    fn ln(self)->Self{self.ln()}
    fn log(self,base:Self)->Self{self.log(base)}
    fn acos(self)->Self {self.acos()}
    fn asin(self)->Self {self.asin()}
	fn sin(self)->Self {self.sin()}
	fn cos(self)->Self {self.cos()}
	fn sin_cos(self)->(Self,Self){self.sin_cos()}
    fn recip(self)->Self { self.recip()}
}
impl Half for f32{
	fn half()->f32{0.5f32}
}
impl Half for f64{
	fn half()->f64{0.5f64}
}

pub fn rand(seed:i32)->i32{
	(seed ^ (seed>>13) + (seed*9182) +01938)^(seed>>19)+(seed>>3)
}
/// random seed update and return float in 0-1 range
pub fn frand(seed:i32)->(i32,f32){
	// todo - generic over float types..
	(rand(seed), ((seed &0xffff)as f32)*(1.0/(0x10000 as f32)))
}
pub fn frands(seed:i32)->(i32,f32){
	let (seed,f)=frand(seed);
	(seed,f*2.0f32-1.0f32)
}

/// TODO - might need Num,Float to have ::Output, so &f32:Num<Output=f32>
/*
impl<'a> Float for &'a f32{
	fn tan(self)->Self{ self.tan()}
	fn sin(self)->Self{ self.sin()}
	fn cos(self)->Self{ self.cos()}
	fn sin_cos(self)->(Self,Self){self.sin_cos()}
}
impl<'a> Float for &'a f64{
	fn tan(self)->Self{ self.tan()}
	fn sin(self)->Self{ self.sin()}
	fn cos(self)->Self{ self.cos()}
	fn sin_cos(self)->(Self,Self){self.sin_cos()}
}
*/


pub fn sqrt(f:f32)->f32 { f.sqrt()}
pub fn sqr<A,R>(f:A)->R where A:Copy+Mul<A,Output=R> {f*f}
pub fn rcp<A,R>(f:A)->R where A:Copy+One+Div<A,Output=R> {one::<A>()/f}

pub fn sin<T:Float>(x:T)->T{x.sin()}
pub fn cos<T:Float>(x:T)->T{x.cos()}
pub fn sin_cos<T:Float>(x:T)->(T,T){x.sin_cos()}

pub fn min<T:PartialOrd>(a:T,b:T)->T{
	if a<b {a}else{b}
}
pub fn max<T:PartialOrd>(a:T,b:T)->T{
	if a>b {a}else{b}
}

pub fn div_rem<T:Div<T,Output=T>+Rem<T,Output=T>+Copy>(a:T,b:T)->(T,T){(a/b,a%b)}

// todo - math UT, ask if they can go in the stdlib.

// TODO - this is ordered as per hlsl,
// but wouldn't curry-friendly order also be nice?
pub fn clamp<T:Ord>(x:T, (lo,hi):(T,T))->T {
	max(min(x,hi),lo)
}
pub fn inrange<T:PartialOrd>(x:T, (lo,hi):(T,T))->bool {
	x>=lo && x<=hi
}

pub fn clamp_s<T:Ord+Neg<Output=T>+Clone>(value:T, limit:T)->T {
	clamp(value,(-limit.clone(),limit))
}
pub fn deadzone<T:Num>(value:T, deadzone:T)->T {
	if value<deadzone || value>deadzone { value }
		else {Zero::zero()}
}


/// composition of common types to reduce anglebracket hell. 'array of owned pointers'
pub type vecbox<T>=Vec<Box<T>>;
/// composition of common types to reduce anglebracket hell. 'optional owned pointer'
pub type optbox<T>=Option<Box<T>>;
/// composition of common types to reduce anglebracket hell. 'optional reference'
pub type optref<'l,T>=Option<&'l T>;

#[derive(Clone,Copy,Debug)]
pub struct Frustum{
	pub fov:f32,
	pub aspect:f32,
	pub znear:f32,
	pub zfar:f32,
}
pub fn Frustum(fov:f32,aspect:f32,(near,far):(f32,f32))->Frustum{
	Frustum{
		fov:fov,aspect:aspect,znear:near,zfar:far
	}
}
#[derive(Clone,Debug)]
pub struct Camera{
	pub frustum:Frustum,
	pub projection:Mat44f,
	pub view:Mat44f,			//
	pub camera_object:Mat44f,	// inverse of 'view' - where to draw a camera.
}
impl Camera{
	pub fn look_along(f:&Frustum, pos:&Vec3,dir:&Vec3,up:&Vec3)->Camera{
		let cam_obj=matrix::Matrix4::look_along(pos,dir,up).to_mat44();

		Camera{
			frustum:f.clone(),
			projection:{
				let mut mp = matrix::projection(f.fov,f.aspect, f.znear,f.zfar);
				mp.az= mp.az.vneg();
				mp
			},
			camera_object:cam_obj.clone(),
			view:cam_obj.inv_orthonormal_matrix(),

		}
	}
}

// types representing one & zero which can be applied to overloaded operators,
// e.g. for making a homogeneous vec4 point x,y,z,0
struct ScalarOne{}
struct ScalarZero{}

pub fn lerp_ref<'la,'lf,T,Ratio,Diff,Prod>(start:&'la T, end:&'la T, fraction:&'lf Ratio)->T
	where
		&'la T:Sub<&'la T,Output=Diff>,
		Diff:Mul<&'lf Ratio,Output=Prod>,
		Diff:'la,
		&'la T:Add<&'la Diff,Output=T>,
		Prod:Add<&'la T,Output=T>
{
	let diff=end-start;
	diff*fraction+start
}

pub fn inv_lerp_ref<'lx,T,F, Diff,Ratio>(x0:&'lx T, x1:&'lx T, x:&'lx T)->Ratio
	where
		&'lx T:Sub<&'lx T,Output=Diff>,
		Diff:Div<Diff,Output=Ratio>,
		Diff:'lx,
{
	//	let diff=x1-x0;
	//	let offset=x-x0;
	//	offset/&diff
	(x-x0)/(x1-x0)
}


/// e.g. implement NumericInteracton<Point,Fraction>


trait HasMulResult<B,R> : Mul<B,Output=R>{
}
trait HasSubResult<B,R> : Sub<B,Output=R>{
}
trait HasAddResult<B,R> : Add<B,Output=R>{
}

/// multiply-add, Multiply-Accumulate
/// component of linear interpolation, extrapolation etc
pub trait Madd<OFS,FACTOR> :
Copy+
Add< <OFS as Mul<FACTOR>>::Output, Output=Self >
	where OFS:Mul<FACTOR>
{
	type Offset:Mul<FACTOR,Output= <Self as Madd<OFS,FACTOR>>::OfsScaled >;
	type OfsScaled: Add<Self, Output=Self>;
	fn madd(self,d:OFS,f:FACTOR)-> Self{
		self+d*f
	}
}


impl<A,B,Factor,Prod> Madd<B,Factor> for A where
	B:Mul<Factor,Output=Prod>+Copy,A:Copy,
	Prod:Add<A,Output=A>,
	A:Add<Prod,Output=A>+Copy
{
	type Offset = B;
	type OfsScaled=Prod;
}

pub fn lerp<F,T:Lerpable<F>>((start,end):(T,T), fraction:F)->T
{
	start.madd(end-start,fraction)
}


// INTENT: output of operators contain no references to inputs
// the final output contains no borrows

pub trait MulRef<B> {
	type Output;
	fn mul_ref(&self,&B)->Self::Output;
}
pub trait AddRef<B> {
	type Output;
	fn add_ref(&self,&B)->Self::Output;
}

pub fn madd_r1<A,B,C,P>(a:&A,b:&B,c:&C)->A where
	B:MulRef<C,Output=P>,
	A:AddRef<P,Output=A>,
{
	let p=b.mul_ref(c);
	a.add_ref(&p)
}

/*
pub fn lerp_r1<'a,'b,'f,'d,'p,F,T,D,P>((a,b):(&'a T,&'b T), f:&'f F)->T where
	&'b T:Sub<&'a T,Output=D>,
	&'f F:Mul<&'d D,Output=P>,
	&'a T:Add<&'p P, Output=T>,
	D:'d,
	P:'p
{
	a + &(f * &(b-a))
}
*/

pub fn lerp_r<F,T,D,P>((a,b):(&T,&T), f:&F)->T where
		for<'x,'y> &'x T:Sub<&'y T,Output=D>,
		for<'x,'y> &'x F:Mul<&'y D,Output=P>,
		for<'x,'y> &'x T:Add<&'y P, Output=T>,
{
	a + &(f * &(b-a))
}



pub fn inv_lerp<F,T:InvLerp<F>>((x0,x1):(T,T),x:T)->F {
	(x-x0)/(x1-x0)
}
pub fn interp<X,Y,F>(x:X,(x0,y0):(X,Y), (x1,y1):(X,Y))->Y
	where Y:Lerpable<F>,X:InvLerp<F>
{
	lerp((y1,y0),inv_lerp((x1,x0),x))
}

impl Lerpable<f32> for f32{
	type Diff=f32;
	type Prod=f32;
}
impl InvLerp<f32> for f32{
	type Diff=f32;
	type Prod=f32;
}
impl Lerpable<f32> for Vec3<f32>{
	type Diff=Vec3<f32>;
	type Prod=Vec3<f32>;
}
pub enum Axes {
	XY,XZ,YZ
}
macro_rules! new{
    ($e:expr=>$traitname:path)=>{
        sto::new($e) as sto<$traitname>
    }
}

macro_rules! dump{ ($($a:expr),*)=>
    (   {   let mut txt=String::new(); txt.push_str(format!("{:?}:{:?}",file!(),line!()).as_str());
            $( {
                let s=format!("{}={:?};",stringify!($a),$a);
                txt.push_str(s.as_str());
                }
            );*;
            println!("{:?}",txt);
        }
   )
}

macro_rules! trace{
    ()=>{println!("{:?}:{:?}:  ",file!(),line!())};
}
macro_rules! warn{
    ()=>{println!("warning {:?}:{:?}: ",file!(),line!())};

}

// multipurpose initializer/generator
// arrays,vec,hashmap,listcomp [x=>x*2;for 0..10;if..] §§§§
macro_rules! seq{
    [$($e:expr),*]=>{ vec![$($e),*] };
    [$($k:expr => $v:expr),*]=>{  {let mut hm=std::collections::HashMap::new(); $(hm.insert($k,$v));*; hm} };
	[ $i:ident;$e:expr ;for $r:expr]=>{{
		let mut v=Vec::new();
		for $i in $r {
			v.push($e);
		}
		v	
	}};
	[ $i:ident;$e:expr ;for $rng:expr ;if $pred:expr ]=>{{
		let mut v=Vec::new();
		for $i in $rng {
			if $pred{
				v.push($e)}
		}
		v	
	}};

}
// impl 'hasXY' etc to get these simple vector maths helpers
// TODO rework main vector maths to work this way
// these are mutually exclusive, e.g. HasXY !=HasXYZ
// named x() y() etc to avoid conflict with existing vx vx vy HasX etc
pub trait HasXY :Sized+Clone {
    type Elem : PartialOrd+Num+Zero+One+Default+Clone;
    fn x(&self)->Self::Elem;
    fn y(&self)->Self::Elem;
    fn from_xy(x:Self::Elem,y:Self::Elem)->Self;

	fn vadd_x(&self,f:Self::Elem)->Self{Self::from_xy(self.x()+f,self.y())}
	fn vadd_y(&self,f:Self::Elem)->Self{Self::from_xy(self.x(),self.y()+f)}

}
pub trait HasXYZ:Sized+Clone {
    type Elem : PartialOrd+Num+Num+Zero+One+Default+Clone;
	type Appended : HasXYZW<Elem=Self::Elem>;
    fn x(&self)->Self::Elem;
    fn y(&self)->Self::Elem;
    fn z(&self)->Self::Elem;
    fn from_xyz(x:Self::Elem,y:Self::Elem,z:Self::Elem)->Self;
	fn vadd_x(&self,f:Self::Elem)->Self{Self::from_xyz(self.x()+f,self.y(),self.z())}
	fn vadd_y(&self,f:Self::Elem)->Self{Self::from_xyz(self.x(),self.y()+f,self.z())}
	fn vadd_z(&self,f:Self::Elem)->Self{Self::from_xyz(self.x(),self.y(),self.z()+f)}
	fn append_w(&self,f:Self::Elem)->Self::Appended{Self::Appended::from_xyzw(self.x(),self.y(),self.z(),f)}
//	fn to_vec3(&self)->Vec3<Self::Elem>{Vec3(self.x(),self.y(),self.z())}
}
pub trait HasXYZW :Sized+Clone{
    type Elem : PartialOrd+Num+Num+Zero+One+Default+Clone;
    fn x(&self)->Self::Elem;
    fn y(&self)->Self::Elem;
    fn z(&self)->Self::Elem;
    fn w(&self)->Self::Elem;
    fn from_xyzw(x:Self::Elem,y:Self::Elem,z:Self::Elem,w:Self::Elem)->Self;
	fn vadd_x(&self,f:Self::Elem)->Self{Self::from_xyzw(self.x()+f,self.y(),self.z(),self.w())}
	fn vadd_y(&self,f:Self::Elem)->Self{Self::from_xyzw(self.x(),self.y()+f,self.z(),self.w())}
	fn vadd_z(&self,f:Self::Elem)->Self{Self::from_xyzw(self.x(),self.y(),self.z()+f,self.w())}
	fn vadd_w(&self,f:Self::Elem)->Self{Self::from_xyzw(self.x(),self.y(),self.z(),self.w()+f)}
}
impl<T:Float+Default> HasXY for (T,T){
    type Elem=T;
    fn x(&self)->T{self.0}
    fn y(&self)->T{self.1}
    fn from_xy(x:T,y:T)->Self{(x,y)}
}
impl<T:Float+Default> HasXYZ for (T,T,T){
    type Elem=T;
	type Appended=(T,T,T,T);
    fn x(&self)->T{self.0}
    fn y(&self)->T{self.1}
    fn z(&self)->T{self.2}
    fn from_xyz(x:T,y:T,z:T)->Self{(x,y,z)}
}
impl<T:Float+Default> HasXYZW for (T,T,T,T){
    type Elem=T;
    fn x(&self)->T{self.0}
    fn y(&self)->T{self.1}
    fn z(&self)->T{self.2}
    fn w(&self)->T{self.3}
    fn from_xyzw(x:T,y:T,z:T,w:T)->Self{(x,y,z,w)}
}

type VF3<F>=(F,F,F);
type VF2<F>=(F,F,F);
/// minimal tuple vmath.. vecmath capability without dependancies.
pub type V2=(f32,f32);		pub type V3=(f32,f32,f32);	pub type V4=(f32,f32,f32,f32);
pub type M33<V=V3>=(V,V,V);	pub type M43<V/*:HasXYZ*/=V3>=(V,V,V,V);	pub type M44<V/*:HasXYZW*/=V4>=(V,V,V,V);
pub fn v3neg<F:Float>(&(x,y,z):&VF3<F>)->VF3<F>			{ (-x,-y,-z)}
pub fn v3scale<V:HasXYZ>(v:&V,s:V::Elem)->V		{ V::from_xyz(v.x()*s,v.y()*s,v.z()*s)}
pub fn v3sub<V:HasXYZ>(a:&V,b:&V)->V	{	V::from_xyz(a.x()-b.x(),a.y()-b.y(),a.z()-b.z())}
pub fn v3add<V:HasXYZ>(a:&V,b:&V)->V	{	V::from_xyz(a.x()+b.x(),a.y()+b.y(),a.z()+b.z())}
pub fn v3min<V:HasXYZ>(a:&V,b:&V)->V{V::from_xyz(min(a.x(),b.x()),min(a.y(),b.y()),min(a.z(),b.z()))}
pub fn v3max<V:HasXYZ>(a:&V,b:&V)->V{V::from_xyz(max(a.x(),b.x()),max(a.y(),b.y()),max(a.z(),b.z()))}
pub fn v3zero<V:HasXYZ>()->V{V::from_xyz(zero(),zero(),zero())}
pub fn v3add4<V:HasXYZ>(a:&V,b:&V,c:&V,d:&V)->V	{v3add(&v3add(a,b),&v3add(c,d))}
pub fn v3add3<V:HasXYZ>(a:&V,b:&V,c:&V)->V		{v3add(&v3add(a,b),c)}
pub fn v3mad<V:HasXYZ>(v0:&V,v1:&V,f:V::Elem)->V 	{ v3add(v0,&v3scale(v1,f))}
pub fn v3lerp<V:HasXYZ>(v0:&V,v1:&V,f:V::Elem)->V	{ v3add(v0,&v3scale(&v3sub(v1,v0),f))}
pub fn v3dot<V:HasXYZ>(a:&V,b:&V)->V::Elem			{	a.x()*b.x()+a.y()*b.y()+a.z()*b.z() }
pub fn v3cross<V:HasXYZ>(a:&V,b:&V)->V			{ panic!("cross product untested");V::from_xyz((a.y()*b.z()-a.z()*b.y()),(a.z()*b.x()-a.x()*b.z()),(a.x()*b.y()-a.y()*b.x())) }
pub fn v3norm<T:Float+Default,V:HasXYZ<Elem=T>>(v0:&V)->V { v3scale(v0,one::<T>()/(v3dot(v0,v0).sqrt())) }
pub fn v3sub_norm<V:HasXYZ>(v0:&V,v1:&V)->V	 where V::Elem:Float{ v3norm(&v3sub(v0,v1))}
pub fn v3perp<V:HasXYZ>(v0:&V,axis:&V)->V		{ v3mad(v0, axis, -v3dot(v0,axis))}
pub fn v3para_perp<V:HasXYZ>(v0:&V,axis:&V)->(V,V){ let para=v3scale(axis, v3dot(v0,axis));let perp=v3sub(v0,&para); (para, perp) }
pub fn v3mat_mul<V:HasXYZ>(m:&M43<V>, p:&V)->V { v3add4(&v3scale(&m.0,p.x()), &v3scale(&m.1,p.y()), &v3scale(&m.2,p.z()),&m.3 ) }
// inv only if orthonormal
pub fn v3mat_invmul<V:HasXYZ>(m:&M43<V>,src:&V)->V { let ofs=v3sub(src,&m.3); V::from_xyz(v3dot(src,&m.0),v3dot(src,&m.1),v3dot(src,&m.2)) }
pub fn v3mat_lookat<V:HasXYZ>(pos:&V, at:&V,up:&V)->M43<V>	where V::Elem : Float{ let az=v3sub_norm(at,pos); let ax=v3norm(&v3cross(&az,up)); let ay=v3cross(&ax,&az); (ax,ay,az,pos.clone()) }
pub fn v3mat_identity()->M43 						{((1.0,0.0,0.0),(0.0,1.0,0.0),(0.0,0.0,1.0),(0.0,0.0,0.0))}
pub fn v3triangle_norm<V:HasXYZ>(v0:&V,v1:&V,v2:&V)->V where V::Elem:Float	{ let v01=v3sub(v1,v0); let v02=v3sub(v2,v0); v3norm(&v3cross(&v02,&v01))}
// inv only if orthonormal
pub fn v3dist_squared<V:HasXYZ>(v0:&V,v1:&V)->V::Elem where V::Elem : Float {let ofs=v3sub(v0,v1); return v3dot(&ofs,&ofs);}
pub fn v3length<V:HasXYZ>(v0:&V)->V::Elem where V::Elem:Float{return v3dot(v0,v0).sqrt();}
pub fn v3addto<V:HasXYZ>(dst:&mut V, src:&V){ let v=v3add(dst,src); *dst=v;}
pub fn v3madto<V:HasXYZ>(dst:&mut V, src:&V, f:V::Elem){ let v=v3mad(dst,src,f); *dst=v;}
pub fn v3fromv2<V2:HasXY,V3:HasXYZ<Elem=V2::Elem>>(xy:&V2,z:V2::Elem)->V3{ return V3::from_xyz(xy.x(), xy.y(), z)}
pub fn v3mat_inv(&(ref mx,ref my,ref mz,ref pos):&M43 )->M43{
    let (ax,ay,az)=((mx.0,my.0,mz.0),(mx.1,my.1,mz.1),(mx.2,my.2,mz.2));
    let invpos= (-v3dot(&ax,pos), -v3dot(&ay,pos), -v3dot(&az,pos));
    (ax,ay,az, invpos) }
pub fn v2zero<V:HasXY>()->V{V::from_xy(zero(),zero())}
pub fn v2sub<V:HasXY>(a:&V,b:&V)->V{ V::from_xy(a.x()-b.x(), a.y()-b.y()) }
pub fn v2add<V:HasXY>(a:&V,b:&V)->V{
    V::from_xy(a.x()+b.x(), a.y()+b.y())
}
pub fn v2lerp<V:HasXY>(a:&V,b:&V,f:V::Elem)->V{
    v2mad(&a, &v2sub(&b,&a), f)
}
pub fn v2mad<V:HasXY>(a:&V,b:&V,f:V::Elem)->V{
    V::from_xy(a.x()+b.x()*f, a.y()+b.y()*f)
}
pub fn v2scale<V:HasXY>(a:&V,f:V::Elem)->V{
    V::from_xy(a.x()*f, a.y()*f)
}
pub fn v2min<V:HasXY>(a:&V,b:&V)->V { V::from_xy(min(a.x(),b.x()),min(a.y(),b.y())) }
pub fn v2max<V:HasXY>(a:&V,b:&V)->V{ V::from_xy(max(a.x(),b.x()),max(a.y(),b.y())) }
pub fn v2avr<T:Float+Default,V:HasXY<Elem=T>>(a:&V,b:&V)->V { V::from_xy((a.x()+b.x())*half(), (a.y()+b.y())*half()) }
pub fn v2is_inside<V:HasXY>(v:&V, (minv,maxv):(&V,&V))->bool{
	inrange(v.x(), (minv.x(), maxv.x()))&&
	inrange(v.y(), (minv.y(), maxv.y()))
}
/// intersection returns the shape
trait Intersect<T>{
    type Output;
    fn intersect(&self,other:&T)->Self::Output;
}
/// collision test determines if shapes overlap,
/// and returns the closest point and so on
/// seperation distance is negative if in contact
struct Collision {
    distance:f32,
    normal:Vec3,
    point:Vec3,
}

// lift common

trait Collide<T>{
    fn collide(&self,other:&T)->Collision;
}



