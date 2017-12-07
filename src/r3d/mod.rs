pub mod vectypes;
pub mod vector;
pub mod matrix;
pub mod quaternion;
pub mod geom;
pub mod mesh;
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

// Common standard library imports.. file, memory, operators..
pub use ::std::{io,fs,ops,path,mem,ffi,os,num,cmp,vec,collections,fmt,marker,convert};
pub use io::Read;
pub use fmt::Debug;


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

//pub static GL_TEXTURE_2D:rawglbinding::uint=0x0DE1;

pub unsafe fn c_str(s:&str)->*const c_char {
	s.as_ptr() as *const c_char
}
pub unsafe fn as_void_ptr<T>(ptr:&T)->*const c_void {
	ptr as *const T as *const c_void
}


pub struct PackedARGB(pub u32);	// 'Color', most commonly packed 8888
pub struct PackedARGB1555(pub u16);	// 'Color', most commonly packed 8888
type Color = PackedARGB;
							// other representations can be more specific.
type Point3= vector::Vec3<f32>;
// todo: f16
type Normal= vector::Vec3<f32>;
struct PackedXYZ(pub u32);

fn to_u32(f:f32)->u32{
	(f*255.0f32) as u32
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


impl From<PackedARGB> for vector::Vec4<f32> {
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


pub trait Float : Num+Half {
	fn sin(self)->Self;
	fn cos(self)->Self;
	fn sin_cos(self)->(Self,Self){(self.sin(),self.cos())}
	fn tan(self)->Self;
}
impl Float for f32{
	fn tan(self)->Self { self.tan()}
	fn sin(self)->Self { self.sin()}
	fn cos(self)->Self { self.cos()}
	fn sin_cos(self)->(Self,Self){self.sin_cos()}
}
impl Float for f64{
	fn tan(self)->Self {self.tan()}
	fn sin(self)->Self {self.sin()}
	fn cos(self)->Self {self.cos()}
	fn sin_cos(self)->(Self,Self){self.sin_cos()}
}
impl Half for f32{
	fn half()->f32{0.5f32}
}
impl Half for f64{
	fn half()->f64{0.5f64}
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

pub fn clamp<T:Ord>(x:T, (lo,hi):(T,T))->T {
	max(min(x,hi),lo)
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
                let s=format!("\t{:?}={:?};",stringify!($a),$a);
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



/// minimal tuple vmath.. vecmath capability without dependancies.
pub type V2=(f32,f32);		pub type V3=(f32,f32,f32);	pub type V4=(f32,f32,f32,f32);
pub type M33=(V3,V3,V3);	pub type M43=(V3,V3,V3,V3);	pub type M44=(V4,V4,V4,V4);
pub fn v3neg(&(x,y,z):&V3)->V3			{ (-x,-y,-z)}
pub fn v3scale(&(x,y,z):&V3,s:f32)->V3		{ (x*s,y*s,z*s)}
pub fn v3sub(&(x0,y0,z0):&V3,&(x1,y1,z1):&V3)->V3	{	(x0-x1,y0-y1,z0-z1)}
pub fn v3add(&(x0,y0,z0):&V3,&(x1,y1,z1):&V3)->V3	{	(x0+x1,y0+y1,z0+z1)}
pub fn v3add4(a:&V3,b:&V3,c:&V3,d:&V3)->V3	{v3add(&v3add(a,b),&v3add(c,d))}
pub fn v3add3(a:&V3,b:&V3,c:&V3)->V3		{v3add(&v3add(a,b),c)}
pub fn v3mad(v0:&V3,v1:&V3,f:f32)->V3 	{ v3add(v0,&v3scale(v1,f))}
pub fn v3lerp(v0:&V3,v1:&V3,f:f32)->V3	{ v3add(v0,&v3scale(&v3sub(v1,v0),f))}
pub fn v3dot(a:&V3,b:&V3)->f32			{	a.0*b.0+a.1*b.1+a.2*b.2 }
pub fn v3cross(a:&V3,b:&V3)->V3			{ ((a.1*b.2-a.2*b.1),(a.2*b.0-b.2*a.0),(a.0*b.1-b.0*a.1)) }
pub fn v3norm(v0:&V3)->V3				{ v3scale(v0,1.0/(v3dot(v0,v0).sqrt())) }
pub fn v3sub_norm(v0:&V3,v1:&V3)->V3	{ v3norm(&v3sub(v0,v1))}
pub fn v3perp(v0:&V3,axis:&V3)->V3		{ v3mad(v0, axis, -v3dot(v0,axis))}
pub fn v3para_perp(v0:&V3,axis:&V3)->(V3,V3){ let para=v3scale(axis, v3dot(v0,axis)); (para, v3sub(v0,&para)) }
pub fn v3mat_mul(m:&M43, p:&V3)->V3 { v3add4(&v3scale(&m.0,p.0), &v3scale(&m.1,p.1), &v3scale(&m.2,p.2),&m.3 ) }
// inv only if orthonormal
pub fn v3mat_invmul(m:&M43,src:&V3)->V3 { let ofs=v3sub(src,&m.3); (v3dot(src,&m.0),v3dot(src,&m.1),v3dot(src,&m.2)) }
pub fn v3mat_lookat(pos:&V3, at:&V3,up:&V3)->M43	{ let az=v3sub_norm(at,pos); let ax=v3norm(&v3cross(&az,up)); let ay=v3cross(&ax,&az); (ax,ay,az,pos.clone()) }
pub fn v3mat_identity()->M43 						{((1.0,0.0,0.0),(0.0,1.0,0.0),(0.0,0.0,1.0),(0.0,0.0,0.0))}
pub fn v3triangle_norm(v0:&V3,v1:&V3,v2:&V3)->V3			{ let v01=v3sub(v1,v0); let v02=v3sub(v2,v0); v3norm(&v3cross(&v02,&v01))}
// inv only if orthonormal

pub fn v3mat_inv(&(ref mx,ref my,ref mz,ref pos):&M43 )->M43{
    let (ax,ay,az)=((mx.0,my.0,mz.0),(mx.1,my.1,mz.1),(mx.2,my.2,mz.2));
    let invpos= (-v3dot(&ax,pos), -v3dot(&ay,pos), -v3dot(&az,pos));
    (ax,ay,az, invpos) }

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



