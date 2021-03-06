use super::*;
use super::Clamp;
use r3d::*;
pub use ::std::f32::consts::PI;

/*
// Vec3*scalar
impl<A,B,C> Mul<B> for Vec3<A> 
	where
		A:Mul<B,Output=C>,B:Float
{
	type Output=Vec3<C>;
	fn mul(self,rhs:B)->Vec3<C> { 
		Vec3(self.x*rhs   , self.y*rhs, self.z*rhs)
	}
}
*/

pub trait NumElements {
	fn num_elements(&self)->usize;
}
impl<T:VElem> NumElements for Vec2<T>{ fn num_elements(&self)->usize{2}}
impl<T:VElem> NumElements for Vec3<T>{ fn num_elements(&self)->usize{3}}
impl<T:VElem> NumElements for Vec4<T>{ fn num_elements(&self)->usize{4}}

/// a
pub trait ArrayDimensions {
	fn array_dimensions(&self)->usize;
}
impl<T> ArrayDimensions for Vec<T>{
	fn array_dimensions(&self)->usize{1}
}
impl ArrayDimensions for f32 {
	fn array_dimensions(&self)->usize{0}
}
impl ArrayDimensions for f64{
	fn array_dimensions(&self)->usize{0}
}
// TODO.. should this recurse inward,
// e.g. 1+ T::array_dimensions..
impl<T:VElem> ArrayDimensions for Vec3<T> {
	fn array_dimensions(&self)->usize{1}
}


// Generic maths classes
// member functions prefixed with 'v' for easier life without code-completion, and to distinguish from operator overloads (official langauge level "add") etc

// using vec_xyzw Vec1,Vec2,Vec3,Vec4 types
/*
/// 1D vector - sounds dumb, but symetrical with declaring channels with n-dimensions . Symetry for 'vector transpose' as Matrix1<Scalar>
#[derive(Clone,Copy,Debug)]
#[repr(C)]
pub struct Vec1<X:Sized=f32> {pub x:X}

/// 2D vector type; seperate types for X,Y,Z. not yet utilized, goal is to allow pluging in 'Zero/One' types to make scalars as axis vectors, etc.
/// for texcoords, 2d rendering
#[derive(Clone,Debug,Copy,Default)]
#[repr(C)]
pub struct Vec2<X:Sized=f32,Y:Sized=X> {pub x:X, pub y:Y}

/// 3D vector type; seperate types for X,Y,Z. not yet utilized, goal is to allow pluging in 'Zero/One' types to make scalars as axis vectors, etc.
/// for 3d points/vectors, rgb
#[derive(Clone,Debug,Copy,Default)]
#[repr(C)]
pub struct Vec3<X:Sized=f32,Y:Sized=X,Z:Sized=Y> {pub x:X, pub y:Y, pub z:Z}



/// 4D vector type; seperate types for X,Y,Z. not yet utilized, goal is to allow pluging in 'Zero/One' types to make scalars as axis vectors, etc.
/// for homogeneous points, rgba, quaternions
#[derive(Clone,Debug,Copy,Default)]
#[repr(C)]
pub struct Vec4<X:Sized=f32,Y:Sized=X,Z:Sized=Y,W:Sized=Z> {pub x:X, pub y:Y,pub  z:Z, pub w:W}
*/
/// '8 element vector' written for completeness e.g. if we allow using these vectors to represent in-register SIMD operations with component-wise operations
/// todo.. at this level are we better off saying [T;8] etc.
#[derive(Clone,Debug,Copy)]
#[repr(C)]
pub struct Vec8<T=f32>(pub T,pub T,pub T,pub T, pub T,pub T,pub T,pub T);


/// '16 element vector' written for completeness e.g. if we allow using these vectors for componentwise SIMD

#[derive(Clone,Debug,Copy)]
#[repr(C)]
pub struct Vec16<T=f32>(pub T,pub T,pub T,pub T, pub T,pub T,pub T,pub T, pub T,pub T,pub T,pub T, pub T,pub T,pub T,pub T);


// constructors
pub fn vec4<X:VElem,Y:VElem,Z:VElem,W:VElem>(x:X,y:Y,z:Z,w:W)->Vec4<X,Y,Z,W> { Vec4{x:x,y:y,z:z,w:w}}
pub fn vec3<X:VElem,Y:VElem,Z:VElem>(x:X,y:Y,z:Z)->Vec3<X,Y,Z> { Vec3{x:x,y:y,z:z}}
pub fn vec2<X:VElem,Y:VElem>(x:X,y:Y)->Vec2<X,Y> { Vec2{x:x,y:y}}
pub fn vec1<X:VElem>(x:X)->Vec1<X> { Vec1{x:x}}

impl<T:VElem> HasElem for Vec1<T>{
    type Elem=T;
	fn vget(&self,i:i32)->T{
		match i{
			0=>self.x,
			_=>panic!()
		}
	}
}

impl Vec3<f32>{
    pub fn to_vec3i(&self)->Vec3<i32>{
        vec3(self.x as i32, self.y as i32, self.z as i32)
    }
}
/// map integers to vector with given 0-1 range
impl Vec3<i32>{
    pub fn to_vec3f(&self, zero_val:i32,one_val:i32)->Vec3<f32> {
        let diff=(one_val-zero_val) as f32;
        let conv = |x|((x-zero_val) as f32)/diff;
        vec3(conv(self.x), conv(self.y), conv(self.z))
    }
}

impl Vec4<f32>{
    pub fn to_vec4i(&self)->Vec4<i32>{
        vec4(self.x as i32, self.y as i32, self.z as i32,self.w as i32)
    }
}
/// map integers to vector with given 0-1 range
impl Vec4<i32>{
    pub fn to_vec4f(&self, zero_val:i32,one_val:i32)->Vec4<f32> {
        let diff=(one_val-zero_val) as f32;
        let conv = |x|((x-zero_val) as f32)/diff;
        vec4(conv(self.x), conv(self.y), conv(self.z), conv(self.w))
    }
}

impl<T:VElem> HasElem for Vec2<T>{
	type Elem=T;
	fn vget(&self,i:i32)->T{
		match i{
			0=>self.x,
			1=>self.y,
			_=>panic!()
		}
	}
}

impl<T:VElem> HasElem for Vec3<T>{
	type Elem=T;
	fn vget(&self,i:i32)->T{
		match i{
			0=>self.x,
			1=>self.y,
			2=>self.z,
			_=>panic!()
		}
	}
}
impl<T:VElem> HasElem for Vec4<T>{
	type Elem=T;
	fn vget(&self,i:i32)->T{
		match i{
			0=>self.x,
			1=>self.y,
			2=>self.z,
			3=>self.w,
			_=>panic!()
		}
	}
}





// TODO half-precision type for GL..
// TODO: Packed normal 10:10:10
// TODO: 565 colors

/*
pub struct Vec3f {x:float,y:float,z:float}
impl Vec3f {
	pub fn new2(x:float,y:float)->Vec3f	{ Vec3f{x:x,y:y,z:0.0} }
}
*/


pub trait VSplat<T>{
	fn vsplat(v:T)->Self;
	unsafe fn raw_ptr(&self)->*const T;
}

impl<T:VElem> VSplat<T> for Vec2<T> {
//	pub fn new(x:T,y:T)->Vec2<T>	{vec2(x,y)}
	 fn vsplat(v:T)->Vec2<T> { vec2(v.clone(),v)}
	 unsafe fn raw_ptr(&self)->*const T{&self.x as *const T}
}
pub trait CrossZ<T>{
	fn vcross_z(&self,other:&Self)->T;
}
impl<T:Num+VElem> CrossZ<T> for Vec2<T> {
	// 'cross_z' computes the z component of a 3d cross product, i.e. uses x/y
	 fn vcross_z(&self,other:&Vec2<T>)->T {self.x*other.y-self.y*other.x}
}
impl<T:VElem> VSplat<T> for Vec3<T> {
	 fn vsplat(v:T)->Vec3<T> { vec3(v,v,v)}
	 unsafe fn raw_ptr(&self)->*const T{&self.x as *const T}
}
impl<T:VElem> VSplat<T> for Vec4<T> {
	 fn vsplat(v:T)->Vec4<T> { vec4(v.clone(),v.clone(),v.clone(),v.clone())}	// todo -move to elsewhere
	 unsafe fn raw_ptr(&self)->*const T{&self.x as *const T}
}
/*
impl<T:Clone> Vec4<T> {
	pub fn new(x:T,y:T,z:T,w:T)->Vec4<T> {vec4(x.clone(),y.clone(),z.clone(),w.clone())}

	pub fn vfromake_vec3(xyz:Vec3<T>,w:T)->Vec4<T> {vec4(xyz.x.clone(),xyz.y.clone(),xyz.z.clone(),w.clone())}
	pub fn vfromake_vec2(xy:Vec2<T>,z:T,w:T)->Vec4<T> {vec4(xy.x.clone(),xy.y.clone(),z.clone(),w.clone())}
	pub fn vfromake_vec2vec2(xy:Vec2<T>,zw:Vec2<T>)->Vec4<T> {vec4(xy.x.clone(),xy.y.clone(),zw.x.clone(),zw.y.clone())}
}
*/

impl<T:Clone> Vec8<T> {
	pub fn clone_ref(a:&T,b:&T,c:&T,d:&T,e:&T,f:&T,g:&T,h:&T)->Vec8<T>		{Vec8(a.clone(),b.clone(),c.clone(),d.clone(),e.clone(),f.clone(),g.clone(),h.clone())}
}
// this is getting silly.. needs macro..
impl<T:Clone> Vec16<T> {
	pub fn clone_ref(a:&T,b:&T,c:&T,d:&T,e:&T,f:&T,g:&T,h:&T,
					i:&T,j:&T,k:&T,l:&T,m:&T,n:&T,o:&T,p:&T)
				->Vec16<T>		
	{Vec16(a.clone(),b.clone(),c.clone(),d.clone(),e.clone(),f.clone(),g.clone(),h.clone(),
	i.clone(),j.clone(),k.clone(),l.clone(),m.clone(),n.clone(),o.clone(),p.clone())}
}

// vector constants should include 'zero','one' functions.
// 'origin' can mean something other than 'zero'.
pub trait VecConsts : Zero
{
	fn origin()->Self;
	fn vaxis(i:int)->Self;
	fn one()->Self;
}
/// vectorized bitwise operations for SIMD
/// todo .. componentwise shifts, aswell..
pub trait VecBitOps :Sized{
	fn vand(&self,b:&Self)->Self;
	fn vor(&self,b:&Self)->Self;
	fn vxor(&self,b:&Self)->Self;
	fn vnot(&self)->Self;
	fn vnor(&self,b:&Self)->Self{ self.vor(b).vnot() }
	fn vnand(&self,b:&Self)->Self{ self.vand(b).vnot() }
}
pub trait VSelect<X>{
	fn vselect(&self,a:&X,b:&X)->X;
}


impl<B:VElem,T:BitSel<B>+VElem> VSelect<Vec4<B>> for Vec4<T> {
	fn vselect(&self,a:&Vec4<B>,b:&Vec4<B>)->Vec4<B>{ vec4(self.x.bitsel(&a.x,&b.x),self.y.bitsel(&a.y,&b.y), self.z.bitsel(&a.z,&b.z), self.w.bitsel(&a.w,&b.w)) }
}
macro_rules! impl_vec_bit_ops{

	(struct $VecN:ident<T>{$($elem:ident :T),*})=>{
		impl<T:VElem+Sized+BitAnd<T,Output=T>+BitOr<T,Output=T>+Not<Output=T> +BitXor<T,Output=T>> VecBitOps for $VecN<T> {
			fn vand(&self,b:&Self)->Self{ 
				$VecN{ $( $elem:(self.$elem & b.$elem) ),* }
			}
			fn vnand(&self,b:&Self)->Self{ 
				$VecN{ $( $elem:!(self.$elem & b.$elem) ),* }
			}
			fn vor(&self,b:&Self)->Self{ 
				$VecN{ $( $elem:(self.$elem | b.$elem) ),* }
			}
			fn vnor(&self,b:&Self)->Self{ 
				$VecN{ $( $elem:!(self.$elem | b.$elem) ),* }
			}
			fn vxor(&self,b:&Self)->Self{
				$VecN{ $( $elem:(self.$elem ^ b.$elem) ),* }
			}
			fn vnot(&self)->Self{
				$VecN{ $( $elem:!self.$elem ),* }
			}
		}
	}
}
impl_vec_bit_ops!(struct Vec2<T>{x:T,y:T});
impl_vec_bit_ops!(struct Vec3<T>{x:T,y:T,z:T});
impl_vec_bit_ops!(struct Vec4<T>{x:T,y:T,z:T,w:T});

impl<X:VElem,T:BitSel<X>+VElem> VSelect<Vec3<X>> for Vec3<T> {
	fn vselect(&self,a:&Vec3<X>,b:&Vec3<X>)->Vec3<X>{ vec3(self.x.bitsel(&a.x,&b.x),self.y.bitsel(&a.y,&b.y), self.z.bitsel(&a.z,&b.z)) }
}
trait AllBitOps :ops::BitAnd+ops::BitOr+ops::Not+ops::BitXor+Sized{type Output;}
impl<T:ops::BitAnd<Output=T>+ops::BitOr<Output=T>+ops::BitXor<Output=T>+ops::Not<Output=T>+ops::Not<Output=T>+Sized> AllBitOps for T{
	type Output=T;
}

pub trait ToVec3z<T:VElem>{
	fn to_vec3_z(&self,z:T)->Vec3<T>;
}
pub trait ToVec4<T:VElem>{
	fn to_vec4(&self)->Vec4<T>;
}
pub trait ToVec4w<T:VElem>{
	fn to_vec4_w(&self,w:T)->Vec4<T>;
}
pub trait ToVec4zw<T:VElem>{
	fn to_vec4_zw(&self,z:T,w:T)->Vec4<T>;
}
pub trait ToVec2w<T:VElem>{
	fn to_vec2(&self)->Vec2<T>;
}

impl<T:VElem> ToVec4w<T> for  Vec3<T>{
     fn to_vec4_w(&self, w:T)->Vec4<T>{vec4(self.x.clone(),self.y.clone(),self.z.clone(),w)}
}
impl<T:VElem> ToVec2<T> for Vec3<T>{

     fn to_vec2(&self)->Vec2<T>{vec2(self.x.clone(),self.y.clone())}
}
impl<T:VElem> ToVec3<T> for Vec4<T>{
     fn to_vec3(&self)->Vec3<T>{vec3(self.x.clone(),self.y.clone(),self.z.clone())}
}
impl<T:VElem> ToVec2<T> for Vec4<T>{
     fn to_vec2(&self)->Vec2<T>{vec2(self.x.clone(),self.y.clone())}
}
impl<T:VElem> ToVec3z<T> for Vec2<T>{
     fn to_vec3_z(&self, z:T)->Vec3<T>{vec3(self.x.clone(),self.y.clone(),z)}
}
impl<T:VElem> ToVec4zw<T> for Vec2<T>{
     fn to_vec4_zw(&self, z:T,w:T)->Vec4<T>{vec4(self.x.clone(),self.y.clone(),z,w)}
}

// float/int conversions 
pub trait VecConv {
	type OUT_I32;
	type OUT_U32;
	type OUT_USIZE;
	type OUT_ISIZE;
	type OUT_F32;
	type OUT_F64;

	fn vto_i32(&self)->Self::OUT_I32;
	fn vto_u32(&self)->Self::OUT_U32;
	fn vto_isize(&self)->Self::OUT_ISIZE;
	fn vto_usize(&self)->Self::OUT_USIZE;
	fn vto_f32(&self)->Self::OUT_F32;
	fn vto_f64(&self)->Self::OUT_F64;
}
macro_rules! impl_vec_conv{
	($t:ty)=>{
		impl VecConv for Vec3<$t>{
			type OUT_I32=Vec3<i32>;
			type OUT_U32=Vec3<u32>;
			type OUT_F32=Vec3<f32>;
			type OUT_F64=Vec3<f64>;
			type OUT_ISIZE=Vec3<isize>;
			type OUT_USIZE=Vec3<usize>;
			fn vto_i32(&self)->Vec3<i32>{
				vec3(self.x as i32,self.y as i32,self.z as i32)
			}
			fn vto_u32(&self)->Vec3<u32>{
				vec3(self.x as u32,self.y as u32,self.z as u32)
			}
			fn vto_isize(&self)->Vec3<isize>{
				vec3(self.x as isize,self.y as isize,self.z as isize)
			}
			fn vto_usize(&self)->Vec3<usize>{
				vec3(self.x as usize,self.y as usize,self.z as usize)
			}
			fn vto_f32(&self)->Vec3<f32>{
				vec3(self.x as f32,self.y as f32,self.z as f32)
			}
			fn vto_f64(&self)->Vec3<f64>{
				vec3(self.x as f64,self.y as f64,self.z as f64)
			}
		}
	}
}
impl_vec_conv!(f32);
impl_vec_conv!(i32);
impl_vec_conv!(u32);
impl_vec_conv!(usize);
impl_vec_conv!(f64);
impl_vec_conv!(isize);

impl<X:VElem,T:BitSel<X>+VElem> VSelect<Vec2<X>> for Vec2<T> {
	fn vselect(&self,a:&Vec2<X>,b:&Vec2<X>)->Vec2<X>{ vec2(self.x.bitsel(&a.x,&b.x),self.y.bitsel(&a.y,&b.y)) }
}

pub trait BitSel<X>{
	fn bitsel(&self,a:&X,b:&X)->X;
}
impl<X:VElem> BitSel<X> for bool {
	fn bitsel(&self,a:&X,b:&X)->X{ if *self{*a}else{*b} }
}
impl BitSel<f32> for u32 {
	fn bitsel(&self,a:&f32,b:&f32)->f32{ unimplemented!() /*mask malarky*/ }
}

/// concatenation/interleave/split operations,
/// 'treating vectors as fixed size collections' rather than just maths types,
/// Could be used to represent certain SIMD operations?
pub trait Concat {
	type Elem;
	type Output;
	type Append;
	type Pop;
	type Split;
	fn concat(&self,&Self)->Self::Output;
	fn interleave(&self,&Self)->Self::Output;
	fn split(&self)->(Self::Split,Self::Split);
	fn append(&self,&Self::Elem)->Self::Append;		// eg generic 'to homogeneous'
	fn pop(&self)->(Self::Pop,Self::Elem);			// eg generic 'from homogeneous'
}
impl<T:VElem> Concat for Vec1<T>{
	type Elem=T;
	type Output=Vec2<T>;
	type Append=Vec2<T>;
	type Pop=();			//'Vec0<T' to propogate typeinfo? might be silly.
	type Split=();
	fn concat(&self,rhs:&Self)->Self::Output{
		unimplemented!()
//		Vec2::clone_ref(&self.x,&rhs.x)
	}
	fn interleave(&self,rhs:&Self)->Self::Output{
		unimplemented!()
//		Vec2::clone_ref(&self.x,&rhs.x)
	}
	fn append(&self,v:&T)->Self::Append{
		unimplemented!()
//		Vec2::clone_ref(&self.x, v)
	}
	fn pop(&self)->(Self::Pop,T){
		((), self.x.clone())
	}
	fn split(&self)->(Self::Split,Self::Split){
		unimplemented!()
	}
}

impl<T:VElem> Concat for Vec2<T>{
	type Elem=T;
	type Output=Vec4<T>;
	type Append=Vec3<T>;
	type Pop=Vec1<T>;
	type Split=Vec1<T>;
	fn concat(&self,rhs:&Self)->Self::Output{
		unimplemented!()
//		Vec4::clone_ref(&self.x,&self.y, &rhs.x,&rhs.y)
	}
	fn interleave(&self,rhs:&Self)->Self::Output{
		unimplemented!()
//		Vec4::clone_ref(&self.x,&rhs.x, &self.y,&rhs.y)
	}
	fn append(&self,v:&T)->Self::Append{
		unimplemented!()
//		Vec3::clone_ref(&self.x,&self.y, v)
	}
	fn pop(&self)->(Self::Pop,T){
		unimplemented!()
//		(Vec1::clone_ref(&self.x), self.y.clone())
	}
	fn split(&self)->(Self::Split,Self::Split){
		unimplemented!()
//		(Vec1::clone_ref(&self.x),Vec1::clone_ref(&self.y))
	}
}
impl<T:VElem> Concat for Vec4<T>{
	type Elem=T;
	type Output=Vec8<T>;
	type Append=Vec4<T>;
	type Pop=Vec3<T>;
	type Split=Vec2<T>;
	fn concat(&self,rhs:&Self)->Self::Output{
		Vec8::clone_ref(&self.x,&self.y,&self.z,&self.w, &rhs.x,&rhs.y,&rhs.z,&rhs.w)
	}
	fn interleave(&self,rhs:&Self)->Self::Output{
		Vec8::clone_ref(&self.x,&rhs.x, &self.y,&rhs.y , &self.z,&rhs.z , &self.w,&rhs.w)
	}
	fn append(&self,v:&T)->Self::Append{
		unimplemented!()
	}
	fn pop(&self)->(Self::Pop,T){
		unimplemented!()
//		(Vec3::clone_ref(&self.x,&self.y,&self.z),self.w.clone())
	}
	fn split(&self)->(Self::Split,Self::Split){
		unimplemented!()
//		(Vec2::clone_ref(&self.x,&self.y),Vec2::clone_ref(&self.z,&self.w))
	}
}
impl<T:VElem> Concat for Vec8<T>{
	type Elem=T;
	type Output=Vec16<T>;
	type Append=();
	type Pop=();
	type Split=Vec4<T>;
	fn concat(&self,rhs:&Self)->Self::Output{
		Vec16::clone_ref(
			&self.0,&self.1,&self.2,&self.3,
			&self.4,&self.5,&self.6,&self.7,
			&rhs.0,&rhs.1,&rhs.2,&rhs.3,
			&rhs.4,&rhs.5,&rhs.6,&rhs.7
		)
	}
	fn interleave(&self,rhs:&Self)->Self::Output{
		Vec16::clone_ref(
			&self.0,&rhs.0, 
			&self.1,&rhs.1, 
			&self.2,&rhs.2, 
			&self.3,&rhs.3,
			&self.4,&rhs.4, 
			&self.5,&rhs.5, 
			&self.6,&rhs.6, 
			&self.7,&rhs.7,
		)
	}
	fn append(&self,v:&T)->Self::Append{
		unimplemented!()
	}
	fn pop(&self)->(Self::Pop,T){
		unimplemented!()
	}
	fn split(&self)->(Self::Split,Self::Split){
		unimplemented!()
//		(Vec4::clone_ref(
//			&self.0,&self.1,&self.2,&self.3),
//		Vec4::clone_ref(
//			&self.4,&self.5,&self.6,&self.7))
	}
}

impl<T:Clone> Concat for Vec16<T>{
	type Elem=T;
	type Output=();
	type Append=();
	type Pop=();
	type Split=Vec8<T>;
	fn interleave(&self,rhs:&Self)->Self::Output{unimplemented!()}
	fn concat(&self,rhs:&Self)->Self::Output{unimplemented!()}
	fn append(&self,v:&T)->Self::Append{
		unimplemented!()
	}
	fn pop(&self)->(Self::Pop,T){
		unimplemented!()
	}
	fn split(&self)->(Self::Split,Self::Split){
		(Vec8::clone_ref(
			&self.0,&self.1,&self.2,&self.3,
			&self.4,&self.5,&self.6,&self.7),
		Vec8::clone_ref(
			&self.8,&self.9,&self.10,&self.11,
			&self.12,&self.13,&self.14,&self.15))
	}
}
impl<T:VElem> Concat for Vec3<T>{
	type Elem=T;
	type Output=Vec8<T>;
	type Append=Vec4<T>;
	type Pop=Vec2<T>;
	type Split=();
	fn concat(&self,rhs:&Self)->Self::Output{
		unimplemented!()
	}
	fn interleave(&self,rhs:&Self)->Self::Output{
		unimplemented!()
	}
	fn append(&self,v:&T)->Self::Append{
		unimplemented!()
//		Vec4::clone_ref(&self.x, &self.y, &self.z, v)
	}
	fn pop(&self)->(Self::Pop,T){
		unimplemented!()
//		(Vec2::clone_ref(&self.x, &self.y), self.z.clone())
	}
	fn split(&self)->(Self::Split,Self::Split){
		unimplemented!()
	}
}

//	fn rsub(&self,&b:Self)->Self { b.sub(self)}
/// splat operations aka vector broadcast.
/// May be simpler than full permutes,
/// does not require acess to the float type

pub trait PermuteXYZ {
}

pub trait Permute :Siblings {
	// default implementation of permutes,
	// can be over-ridden with platform-specific SIMD..

	fn permute_x(&self)->Self::V1;
	fn permute_y(&self)->Self::V1;
	fn permute_z(&self)->Self::V1;
	fn permute_w(&self)->Self::V1;
	fn permute_xy(&self)->Self::V2;
	fn permute_yx(&self)->Self::V2;
	fn permute_xz(&self)->Self::V2;//Vec2<Self::ElemF>;
	fn permute_yz(&self)->Self::V2;//Vec2<Self::ElemF>;
	fn permute_zw(&self)->Self::V2;//Vec2<Self::ElemF>;
	fn permute_xyz(&self)->Self::V3;//Vec3<Self::ElemF>;
	fn permute_xyz0(&self)->Self::V4;//Vec4<Self::ElemF>;
	fn permute_xyz1(&self)->Self::V4;//Vec4<Self::ElemF>;

	fn permute_xyzw(&self)->Self::V4;//Vec4<Self::ElemF>;
	fn permute_wzyx(&self)->Self::V4;//Vec4<Self::ElemF>;

	fn permute_zyx(&self)->Self::V3;//Vec3<Self::ElemF>;
	fn permute_xzy(&self)->Self::V3;//Vec3<Self::ElemF>;

	fn to_vec4_pad0000(&self)->Self::V4;//Vec4<Self::ElemF>;
	fn to_vec4_pad0001(&self)->Self::V4;//Vec4<Self::ElemF>;
	// permutes for cross-product eval
	// i  j  k
	// ax ay az
	// bx by bz
	//
	// plus
	// x'=ay*bz-az*by
	// y'=az*bx-ax*bz
	// z'=ax*by-ay*bx
	fn permute_yzx(&self)->Self::V3;//Vec3<Self::ElemF>;
	fn permute_zxy(&self)->Self::V3;//Vec3<Self::ElemF>;
	// cros product =
	// a.yzx* b.zxy - a.zxy * b.yzx
	fn permute_yzxw(&self)->Self::V4;//Vec4<Self::ElemF>;
	fn permute_zxyw(&self)->Self::V4;//Vec4<Self::ElemF>;
	fn permute_yzx0(&self)->Self::V4;//Vec4<Self::ElemF>;
	fn permute_zxy0(&self)->Self::V4;//Vec4<Self::ElemF>;

	fn permute_xx(&self)->Self::V2;//Vec2<Self::ElemF>;
	fn permute_yy(&self)->Self::V2;//Vec2<Self::ElemF>;

	fn permute_xxx(&self)->Self::V3;//Vec3<Self::ElemF>;
	fn permute_yyy(&self)->Self::V3;//Vec3<Self::ElemF>;
	fn permute_zzz(&self)->Self::V3;//Vec3<Self::ElemF>;

	fn permute_xxxx(&self)->Self::V4;//Vec4<Self::ElemF>;
	fn permute_yyyy(&self)->Self::V4;//Vec4<Self::ElemF>;
	fn permute_zzzz(&self)->Self::V4;//Vec4<Self::ElemF>;
	fn permute_wwww(&self)->Self::V4;//Vec4<Self::ElemF>;
	fn permute_0(&self)->Self::V1;//Vec2<Self::ElemF>;
	fn permute_00(&self)->Self::V2;//Vec2<Self::ElemF>;
	fn permute_000(&self)->Self::V3;//Vec3<Self::ElemF>;
	fn permute_0000(&self)->Self::V4;//Vec4<Self::ElemF>;
	fn permute_1000(&self)->Self::V4;//Vec4<Self::ElemF>;
	fn permute_0100(&self)->Self::V4;//Vec4<Self::ElemF>;
	fn permute_0010(&self)->Self::V4;//Vec4<Self::ElemF>;
	fn permute_0001(&self)->Self::V4;//Vec4<Self::ElemF>;

	fn permute_1(&self)->Self::V1;//Vec2<Self::ElemF>;
	fn permute_11(&self)->Self::V2;//Vec2<Self::ElemF>;
	fn permute_111(&self)->Self::V3;//Vec3<Self::ElemF>;
	fn permute_1111(&self)->Self::V4;//;Vec4<Self::ElemF>;

	// transpose, pad with zeros most useful for 4 elements.
	// verious ways to do this efficiently on SIMD machines..
	// TODO , this should be a freefunction?
	fn transpose4x4(ax:&Self,ay:&Self,az:&Self,aw:&Self)->(Self::V4,Self::V4,Self::V4,Self::V4);
}

/// interface for constructing siblings from components
/// note some vector types may deliberately hide per-component access.
pub trait ConstructSiblings: Siblings+HasElem{
	fn make_vec1(x:Self::Elem)-> Self::V1;
	fn make_vec2(x:Self::Elem,y:Self::Elem)->Self::V2;
	fn make_vec3(x:Self::Elem,y:Self::Elem,z:Self::Elem)->Self::V3;
	fn make_vec4(x:Self::Elem,y:Self::Elem,z:Self::Elem,w:Self::Elem)->Self::V4;

	fn make_vec1_splat(x:Self::Elem)->Self::V1 { Self::make_vec1(x) }
	fn make_vec2_splat(x:Self::Elem)->Self::V2 { Self::make_vec2(x.clone(),x) }
	fn make_vec3_splat(x:Self::Elem)->Self::V3 { Self::make_vec3(x.clone(),x.clone(),x) }
	fn make_vec4_splat(x:Self::Elem)->Self::V4 { Self::make_vec4(x.clone(),x.clone(),x.clone(),x) }
}

pub trait VecN : Siblings {}	//'VecN family'
impl<T:VElem> VecN for Vec1<T>{}
impl<T:VElem> VecN for Vec2<T>{}
impl<T:VElem> VecN for Vec3<T>{}
impl<T:VElem> VecN for Vec4<T>{}

trait TupledVector {}	//'VecN family'
impl<T:VElem> TupledVector for (T,){}
impl<T:VElem> TupledVector for (T,T){}
impl<T:VElem> TupledVector for (T,T,T){}
impl<T:VElem> TupledVector for (T,T,T,T){}

impl<V:VElem,T:VElem> ConstructSiblings for V
where
	V:HasElem<Elem=T>,
	V:	HasElem+VecN
		+Siblings<
				V1=Vec1<T>,
				V2=Vec2<T>,
				V3=Vec3<T>,
				V4=Vec4<T>
					>
{
	fn make_vec1(x:Self::Elem)->Vec1<V::Elem> { vec1(x) }
	fn make_vec2(x:Self::Elem,y:Self::Elem)->Self::V2 {vec2(x,y)}
	fn make_vec3(x:Self::Elem,y:Self::Elem,z:Self::Elem)->Self::V3{vec3(x,y,z)}
	fn make_vec4(x:Self::Elem,y:Self::Elem,z:Self::Elem,w:Self::Elem)->Self::V4{vec4(x,y,z,w)}

}
/*
impl<V:VecSiblings+HasFloatElem+TupledVector> VecConstructSiblings for V{
	fn make_vec1(x:Self::ElemF)->Self::V1{ (x,) }
	fn make_vec2(x:Self::ElemF,y:Self::ElemF)->Self::V2{(x,y,)}
	fn make_vec3(x:Self::ElemF,y:Self::ElemF,z:Self::ElemF)->Self::V3{(x,y,z,)}
	fn make_vec4(x:Self::ElemF,y:Self::ElemF,z:Self::ElemF,w:Self::ElemF)->Self::V4{(x,y,z,w,)}
}
*/
/// Implement VecPermute for default case with 'VecFLoatAccessors'
/// note that permute interface should allow an impl WITHOUT use of a scalar type.
/// TODO decouple from specifc 'V2,V3 ,V4 versions' to allow impl on tuples,[T;N]
impl<T:Zero+One+Clone,V:VecAccessors+ConstructSiblings+HasElem<Elem=T>> Permute for V {
	// default implementation of permutes,
	// can be over-ridden with platform-specific SIMD..
	fn permute_x(&self)->Self::V1		{ Self::make_vec1(self.vx())}
	fn permute_y(&self)->Self::V1		{ Self::make_vec1(self.vy())}
	fn permute_z(&self)->Self::V1		{ Self::make_vec1(self.vz())}
	fn permute_w(&self)->Self::V1		{ Self::make_vec1(self.vw())}
	fn permute_xy(&self)->Self::V2		{ Self::make_vec2(self.vx(),self.vy())}
	fn permute_yx(&self)->Self::V2		{ Self::make_vec2(self.vy(),self.vx())}
	fn permute_xz(&self)->Self::V2	{ Self::make_vec2(self.vx(),self.vz())}
	fn permute_yz(&self)->Self::V2	{ Self::make_vec2(self.vy(),self.vz())}
	fn permute_zw(&self)->Self::V2	{ Self::make_vec2(self.vz(),self.vw())}
	fn permute_xyz(&self)->Self::V3	{ Self::make_vec3(self.vx(),self.vy(),self.vz())}
	fn permute_xyz0(&self)->Self::V4	{ Self::make_vec4(self.vx(),self.vy(),self.vz(), Zero::zero())}	// vec3 to homogeneous offset
	fn permute_xyz1(&self)->Self::V4	{ Self::make_vec4(self.vx(),self.vy(),self.vz(), One::one())}	// vec3 to homogeneous point

	fn permute_xyzw(&self)->Self::V4	{ Self::make_vec4(self.vx(),self.vy(),self.vz(),self.vw())}	// vec3 to homogeneous point
	fn permute_wzyx(&self)->Self::V4	{ Self::make_vec4(self.vw(),self.vz(),self.vy(),self.vx())}	// vec3 to homogeneous point

	fn permute_zyx(&self)->Self::V3	{ Self::make_vec3(self.vz(),self.vy(),self.vx())}	// when using as color components
	fn permute_xzy(&self)->Self::V3	{ Self::make_vec3(self.vx(),self.vz(),self.vy())}	// changing which is up, y or z

	fn to_vec4_pad0000(&self)->Self::V4 {Self::make_vec4(self.vx(),self.vy(),self.vz(),Zero::zero())}
	fn to_vec4_pad0001(&self)->Self::V4 {Self::make_vec4(self.vx(),self.vy(),self.vz(),One::one())}
	// permutes for cross-product eval
	// i  j  k
	// ax ay az
	// bx by bz
	//
    // plus
	// x'=ay*bz-az*by
	// y'=az*bx-ax*bz
	// z'=ax*by-ay*bx
	fn permute_yzx(&self)->Self::V3 {Self::make_vec3(self.vy(),self.vz(),self.vx())}
	fn permute_zxy(&self)->Self::V3 {Self::make_vec3(self.vz(),self.vx(),self.vy())}
	// cros product = 
	// a.yzx* b.zxy - a.zxy * b.yzx
	fn permute_yzxw(&self)->Self::V4 {Self::make_vec4(self.vy(),self.vz(),self.vx(), self.vw())}
	fn permute_zxyw(&self)->Self::V4 {Self::make_vec4(self.vz(),self.vx(),self.vy(), self.vw())}
	fn permute_yzx0(&self)->Self::V4 {Self::make_vec4(self.vy(),self.vz(),self.vx(), Zero::zero())}
	fn permute_zxy0(&self)->Self::V4 {Self::make_vec4(self.vz(),self.vx(),self.vy(), Zero::zero())}

	fn permute_xx(&self)->Self::V2 { Self::make_vec2(self.vx(),self.vx()) }
	fn permute_yy(&self)->Self::V2 { Self::make_vec2(self.vy(),self.vy()) }

	fn permute_xxx(&self)->Self::V3 { Self::make_vec3(self.vx(),self.vx(),self.vx()) }
	fn permute_yyy(&self)->Self::V3 { Self::make_vec3(self.vy(),self.vy(),self.vy()) }
	fn permute_zzz(&self)->Self::V3 { Self::make_vec3(self.vz(),self.vz(),self.vz()) }

	fn permute_xxxx(&self)->Self::V4 { Self::make_vec4(self.vx(),self.vx(),self.vx(),self.vx()) }
	fn permute_yyyy(&self)->Self::V4 { Self::make_vec4(self.vy(),self.vy(),self.vy(),self.vy()) }
	fn permute_zzzz(&self)->Self::V4 { Self::make_vec4(self.vz(),self.vz(),self.vz(),self.vz()) }
	fn permute_wwww(&self)->Self::V4 { Self::make_vec4(self.vw(),self.vw(),self.vw(),self.vw()) }
	fn permute_0(&self)->Self::V1	{ Self::make_vec1_splat(Zero::zero())}
	fn permute_00(&self)->Self::V2	{ Self::make_vec2_splat(Zero::zero())}
	fn permute_000(&self)->Self::V3	{ Self::make_vec3_splat(Zero::zero())}
	fn permute_0000(&self)->Self::V4	{ Self::make_vec4_splat(Zero::zero())}
	fn permute_0001(&self)->Self::V4	{ Self::make_vec4(Zero::zero(),Zero::zero(),Zero::zero(),One::one())}
	fn permute_1000(&self)->Self::V4	{ Self::make_vec4(One::one(), Zero::zero(),Zero::zero(),Zero::zero())}
	fn permute_0100(&self)->Self::V4	{ Self::make_vec4(Zero::zero(),One::one(), Zero::zero(),Zero::zero())}
	fn permute_0010(&self)->Self::V4	{ Self::make_vec4(Zero::zero(),Zero::zero(),One::one(), Zero::zero())}

	fn permute_1(&self)->Self::V1	{ Self::make_vec1_splat(One::one())}
	fn permute_11(&self)->Self::V2	{ Self::make_vec2_splat(One::one())}
	fn permute_111(&self)->Self::V3	{ Self::make_vec3_splat(One::one())}
	fn permute_1111(&self)->Self::V4	{ Self::make_vec4_splat(One::one())}

	// transpose, pad with zeros most useful for 4 elements.
	// verious ways to do this efficiently on SIMD machines..
	fn transpose4x4(ax:&Self,ay:&Self,az:&Self,aw:&Self)->(
		Self::V4,
		Self::V4,
		Self::V4,
		Self::V4
		)
	{
		(	Self::make_vec4(ax.vx(),ay.vx(),az.vx(),aw.vx()),
			 Self::make_vec4(ax.vy(),ay.vy(),az.vy(),aw.vy()),
			 Self::make_vec4(ax.vz(),ay.vz(),az.vz(),aw.vz()),
			 Self::make_vec4(ax.vw(),ay.vw(),az.vw(),aw.vw())
		)

	}
}

// free function interface.
pub fn transpose4x4<V:Permute+VecAccessors>(a:&V,b:&V,c:&V,d:&V)->(
		<V as Siblings>::V4,
		<V as Siblings>::V4,
		<V as Siblings>::V4,
		<V as Siblings>::V4,
		)
{	Permute::transpose4x4(a,b,c,d)
}

// assumptions - might not even have accessible component.
pub trait VecNumOps :Sized
{
    fn vassign_add(&mut self, b:&Self){ *self=self.vadd(b);}
    fn vassign_sub(&mut self, b:&Self){ *self=self.vsub(b);}
    fn vadd(&self, b: &Self) -> Self;
    fn vsub(&self, b: &Self) -> Self;
}

pub trait VecCmpOps :Sized+HasElem {
	type CmpOutput;
    fn vassign_min(&mut self,b:&Self){*self=self.vmin(b);}
    fn vassign_max(&mut self,b:&Self){*self=self.vmax(b);}
	fn vmin(&self,b:&Self)->Self;
	fn vmax(&self,b:&Self)->Self;
	fn gt(&self,b:&Self)->Self::CmpOutput;
	fn lt(&self,b:&Self)->Self::CmpOutput;
    fn vclamp(&self,a:&Self,b:&Self)->Self{
        self.vmin(b).vmax(a)
    }
    //fn vclamp_s(&self,a:&Self)->Self{
    //    self.vclamp(a.vneg(),a)
   // }
//    fn vclamp_scalar(&self, a:Self::Elem)->Self{
    //       self.vclamp(Self::splat(-a),Self::splat(a))
//    }
    fn vclamp_scalar_range(&self, smin:Self::Elem,smax:Self::Elem)->Self{unimplemented!()}
    fn vclamp_scalar(&self, s:Self::Elem)->Self{unimplemented!()}
}



// select masks.
trait Select<V>{
	fn select(&self,iftrue:V,iffalse:V)->V;
}
impl<T:VElem> Select<Vec3<T>> for Vec3<bool> where bool:Select<T>{
	fn select(&self,a:Vec3<T>,b:Vec3<T>)->Vec3<T> {
		vec3( self.x .select(a.x,b.x), self.y .select(a.y,b.y), self.z .select(a.z,b.z) )
	}
}

impl<T:VElem,BOOL:VElem> Select<Vec4<T>> for Vec4<BOOL> where BOOL:Select<T>{
	fn select(&self,a:Vec4<T>,b:Vec4<T>)->Vec4<T> {
		vec4( self.x .select(a.x,b.x), self.y .select(a.y,b.y), self.z .select(a.z,b.z) , self.w .select(a.w,b.w) )
	}
}

impl<T:VElem,BOOL:VElem> Select<Vec8<T>> for Vec8<BOOL> where BOOL:Select<T>{
	fn select(&self,a:Vec8<T>,b:Vec8<T>)->Vec8<T> {
		Vec8( self.0 .select(a.0,b.0), self.1 .select(a.1,b.1), self.2 .select(a.2,b.2) , self.3 .select(a.3,b.3) ,
		self.4 .select(a.4,b.4), self.5 .select(a.5,b.5), self.6 .select(a.6,b.6) , self.7 .select(a.7,b.7) 
		)
	}
}
impl<T,BOOL> Select<Vec16<T>> for Vec16<BOOL> where BOOL:Select<T>{
	fn select(&self,a:Vec16<T>,b:Vec16<T>)->Vec16<T> {
		Vec16( self.0 .select(a.0,b.0), self.1 .select(a.1,b.1), self.2 .select(a.2,b.2) , self.3 .select(a.3,b.3) ,
		self.4 .select(a.4,b.4), self.5 .select(a.5,b.5), self.6 .select(a.6,b.6) , self.7 .select(a.7,b.7),
		self.8 .select(a.8,b.8), self.9 .select(a.9,b.9), self.10 .select(a.10,b.10) , self.11 .select(a.11,b.11) ,
		self.12 .select(a.12,b.12), self.13 .select(a.13,b.13), self.14 .select(a.14,b.14) , self.15 .select(a.15,b.15),
		)
	}
}

impl Select<f32> for bool {
	fn select(&self,a:f32,b:f32)->f32{ if *self{a}else{b}}	
}
impl Select<f64> for bool {
	fn select(&self,a:f64,b:f64)->f64{ if *self{a}else{b}}	
}

/// helper trait to extract inner type, if the vector was referenced generically without the component available. e.g. for Vec4<f32> 'Elem=f32'
/// allows bounding to assert that it's dealing with vectors.
/// Only valid for vector types with the same 'T' for each component.
/// TODO clean up confusion - HasElem vs HasElemFloat vs Permute vs VecOps
/// helper trait to say this has elements, only guarantees raw data (eg a number bits, doesn't say what operations it has..).
///
/// Only valid for types with the same 'T' per component.
/// TODO clean up confusion - HasElem vs HasElemFloat vs Permute vs VecOps
pub trait HasElem{
	type Elem : Clone;
	fn vget(&self,i:i32)->Self::Elem{
		unimplemented!()
	}
}

/// horizontal add = summing the elements, e.g. dot product can be = multiply elements and horizontal-add.
pub trait HorizAdd {
	type Output;
	fn horiz_add(&self)->Self::Output;
}
pub trait HorizMul {
	type Output;
	fn horiz_mul(&self)->Self::Output;
}
pub trait HorizOr {
	type Output;
	fn horiz_or(&self)->Self::Output;
}
pub trait HorizAnd {
	type Output;
	fn horiz_and(&self)->Self::Output;
}


macro_rules! impl_componentwise_reduction_vec_functions{
	($FnTrait:ident::$fnname:ident using $TraitOp:ident::$op:ident)=> 
	{
		impl<T:VElem> $FnTrait for Vec2<T>  where for<'a,'b> &'a T:$TraitOp<&'b T,Output=T> {
			type Output=T;
			fn $fnname(&self)->T{self.x.$op(&self.y)}
		}
		impl<T:VElem> $FnTrait for Vec3<T> where for<'a,'b> &'a T:$TraitOp<&'b T,Output=T> {
			type Output=T;
			fn $fnname(&self)->T{(&self.x).$op(&self.y).$op(&self.z)}
		}
		impl<T:VElem> $FnTrait for Vec4<T>  where for<'a,'b> &'a T:$TraitOp<&'b T,Output=T> {
			type Output=T;
			fn $fnname(&self)->T{(self.x.$op(&self.y)).$op(&self.z.$op(&self.w))}
		}
	}
}

impl_componentwise_reduction_vec_functions!(HorizAdd::horiz_add using Add::add);
impl_componentwise_reduction_vec_functions!(HorizMul::horiz_mul using Mul::mul);
impl_componentwise_reduction_vec_functions!(HorizOr::horiz_or using BitOr::bitor);
impl_componentwise_reduction_vec_functions!(HorizAnd::horiz_and using BitAnd::bitand);


/// wraps any vector type to imply that it is Normalized.
/// exposes only versions of operations that make sense for normalized vectors.
/// e.g. dont need '.normalize()' because it could only have been generated by that.
/// TODO: properly imply the dimensionless-ness of the inner float value
pub struct Normal<V=Vec3<f32>>(pub V);

impl<T:Float,V:VecOps<Elem=T>> Normal<V> {
	pub fn as_vec(&self)->V{self.0.clone()}
	pub fn vlerp_norm(&self,b:&Self,f:V::Elem)->Self { Normal(self.0.vlerp(&b.0, f).vnormalize()) }
	pub fn vmadd_norm(&self,b:&Self,f:V::Elem )->Self { Normal(self.0.vmadd(&b.0,f).vnormalize()) }
	pub fn vmadd2_norm(&self,b:&Self,fb:V::Elem,c:&Self,fc:V::Elem )->Self { Normal(self.0.vmadd(&b.0,fb).vmadd(&c.0,fc).vnormalize()) }
	pub fn vcross_norm(&self,b:&Self)->Self{ Normal(self.0 .vcross(&b.0).vnormalize() )}
	pub fn vscale(&self,f:V::Elem)->V{self.0.vscale(f)}
	pub fn vsub(&self,b:&Self)->V{self.0.vsub(&b.0)}
	pub fn vdot_with_normal(&self,b:&Self)->V::Elem { self.0 .vdot(&b.0)}
	pub fn vdot_with_vec(&self,b:&V)->V::Elem { self.0 .vdot(&b)}
}

/// Wraps any vector type to imply that it is a Point
/// hides methods that are inapplicable to Points, etc.
pub struct Point<V=Vec3<f32>>(pub V);
impl<T:Float,V:VMath<Elem=T>> Point<V> {
	pub fn vsub(&self,b:&Self)->Vector<V>{ Vector(self.0 .vsub(&b.0)) }
	pub fn vadd(&self,b:&Vector<V>)->Point<V>{ Point(self.0 .vadd(&b.0)) }
	pub fn vsub_norm(&self,b:&Self)->Normal<V>{ Normal(self.0 .vsub_norm(&b.0)) }
	pub fn vmadd(&self,b:&Vector<V>, f:V::Elem)->Point<V>{ Point(self.0 .vmadd(&b.0, f)) }
	pub fn vlerp(&self,b:&Point<V>, f:V::Elem)->Point<V>{ Point(self.0 .vlerp(&b.0, f)) }
	pub fn vtriangle_norm(&self,b:&Self,c:&Self)->Normal<V> { Normal(self.0 .vsub(&b.0) .vcross_norm(&self.0 .vsub(&c.0)))}
	pub fn vdist(&self,b:&Self)->V::Elem{ (self.0.vsub(&b.0)).vlength() }
	pub fn vmax(&self,b:&Self)->Point<V>{ Point(self.0.vmax(&b.0)) }
	pub fn vmin(&self,b:&Self)->Point<V>{ Point(self.0.vmin(&b.0)) }
}


/// Wraps any vector type to imply that it is a Vector
/// hides methods that are specific to points, normals, ..
pub struct Vector<V>(pub V);
impl<T:Float,V:VMath<Elem=T>> Vector<V> {
	pub fn vdot(&self,b:&Self)->V::Elem { self.0 .vdot(&b.0) }
	pub fn vdot_normal(&self,b:&Normal<V>)->V::Elem { self.0 .vdot(&b.0) }
	pub fn vsub(&self,b:&Self)->V { self.0 .vsub(&b.0) }
	pub fn vsub_norm(&self,b:&Self)->Normal<V>{ Normal(self.0 .vsub_norm(&b.0)) }
	pub fn vcross(&self,b:&Self)->Vector<V>{ Vector(self.0.vcross(&b.0))}
	pub fn vnormalize(&self)->Normal<V>{ Normal(self.0.vnormalize())}
	pub fn vcross_norm(&self,b:&Self)->Normal<V>{ Normal(self.0.vcross(&b.0).vnormalize())}
	pub fn vmadd(&self,b:&V, f:V::Elem)->Point<V>{ Point(self.0 .vmadd(b, f)) }
	pub fn vlerp(&self,b:&Point<V>, f:V::Elem)->Point<V>{ Point(self.0 .vlerp(&b.0, f)) }
	pub fn vlength(&self)->V::Elem { self.0 .vlength() }
	pub fn vdist(&self,b:&Self)->V::Elem{ (self.0.vsub(&b.0)).vlength() }
	pub fn vmax(&self,b:&Self)->Vector<V>{ Vector(self.0.vmax(&b.0)) }
	pub fn vmin(&self,b:&Self)->Vector<V>{ Vector(self.0.vmin(&b.0)) }
}
//use super::vec_xyzw::conversions::IsNot;
macro_rules! impl_conversion_vecn{
	($Wrapper:ident<V>,$VecN:ident<$T:ident>)=>{
		impl<V:IsNot<$VecN<$T>>> From<$Wrapper<$VecN<$T>>> for $Wrapper<V> where $VecN<$T>:Into<V>{
			fn from(src:$Wrapper<$VecN<$T>>)->$Wrapper<V>{
				$Wrapper(src.0.into())
			}
		}
/*
		impl<V:IsNot<$VecN<$T>>> From<$Wrapper<V>> for $Wrapper<$VecN<$T>> where V:Into<$VecN<$T>>{
			fn from(src:$Wrapper<V>)->$Wrapper<$VecN<$T>>{
				$Wrapper(src.0.into())
			}
		}
*/
	} 
}
/// Implment conversions from wrappers of Vec2<f32>,Vec3<f32>,Vec4<f32> to/from <V>
///- workaround for 'core reflexivity impl clash'
macro_rules! impl_conversion{
	($Wrapper:ident<V>)=>{ 
		impl_conversion_vecn!($Wrapper<V>,Vec2<f32>);
		impl_conversion_vecn!($Wrapper<V>,Vec3<f32>);
		impl_conversion_vecn!($Wrapper<V>,Vec4<f32>);
		impl_conversion_vecn!($Wrapper<V>,Vec2<f64>);
		impl_conversion_vecn!($Wrapper<V>,Vec3<f64>);
		impl_conversion_vecn!($Wrapper<V>,Vec4<f64>);
	}
}
/*
doesn't seem to work :(
impl<T:IsNot<f32>> IsNot< Vec2<f32> >  for Vec2<T>{}
impl<T:IsNot<f64>> IsNot< Vec2<f64> >  for Vec2<T>{}
impl<T:IsNot<f32>> IsNot< Vec3<f32> >  for Vec3<T>{}
impl<T:IsNot<f64>> IsNot< Vec3<f64> >  for Vec3<T>{}
impl<T:IsNot<f32>> IsNot< Vec4<f32> >  for Vec4<T>{}
impl<T:IsNot<f64>> IsNot< Vec4<f64> >  for Vec4<T>{}
*/

//impl_conversion!(Vector<V>);
//impl_conversion!(Normal<V>);
//impl_conversion!(Point<V>);


/// vector type guaranteed to hold bool values, e.g. select-mask
pub struct VBool<V=Vec3<bool>>(V);

/// logic ops on vector bool type..
impl<V:VecBitOps> VBool<V> {
	fn vand(&self,other:&VBool<V>)-> Self{VBool(self.0 .vand(&other.0)) }
	fn vor(&self,other:&VBool<V>)-> Self{VBool(self.0 .vor(&other.0)) }
	fn vxor(&self,other:&VBool<V>)-> Self{VBool(self.0 .vxor(&other.0)) }
	fn vnot(&self)-> Self{VBool(self.0 .vnot()) }

	
	fn vselect<C,B>(&self, a:&C, b:&C)->C
		where
			C:IsWrappedV<B>, V:VSelect<B>
	{
		C::wrap( &self.0 .vselect(&a.as_vec(),&b.as_vec()) )
	}

//	fn vselect<V:IsWrapped<V>>()
}


/// Helper trait that says there's an inner vector, semantically wrapped.
/// todo .. do we want to return a ref? implement for copy types first.
pub trait IsWrappedV<V> {
	fn as_vec(&self)->V;
	fn wrap(x:&V)->Self;
}
impl<V:VElem> IsWrappedV<V> for Point<V>{
	fn as_vec(&self)->V { self.0 }
	fn wrap(s:&V)->Self { Point(*s)  }
}

impl<V:VElem> IsWrappedV<V> for Quaternion<V>{
	fn as_vec(&self)->V { self.0 }
	fn wrap(s:&V)->Self { Quaternion(*s)  }
}
impl<V:VElem> IsWrappedV<V> for VScalar<V>{
	fn as_vec(&self)->V { self.0 }
	fn wrap(s:&V)->Self { VScalar(*s)  }
}
impl<V:VElem> IsWrappedV<V> for Vector<V>{
	fn as_vec(&self)->V { self.0 }
	fn wrap(s:&V)->Self { Vector(*s) }
}
impl<V:VElem> IsWrappedV<V> for Normal<V>{
	fn as_vec(&self)->V { self.0 }
	fn wrap(s:&V)->Self { Normal(*s)  }
}
impl<V:VElem> IsWrappedV<V> for VBool<V>{
	fn as_vec(&self)->V { self.0 }
	fn wrap(s:&V)->Self { VBool(*s)  }
}

/// wrapped vector representing a [Quaternion](https://en.wikipedia.org/wiki/Quaternion).
/// need operations on float4, and conversions to and from packed forms
#[derive(Clone,Debug)]
pub struct Quaternion<V=Vec4<f32>>(V);
impl<T:Float,V:VecOps<Elem=T>> Quaternion<V> {
}

/// vector representing an ARGB color value
#[derive(Clone,Debug)]
pub struct RGBA<V=Vec4<f32>>(V);
impl<V:VElem> IsWrappedV<V> for RGBA<V>{
	fn as_vec(&self)->V { self.0 }
	fn wrap(s:&V)->Self { RGBA(*s)  }
}

///
/// vector type containin a single splatted vector value
/// depending on implementation it may be plain scalar, or a vector register with guaranteed x=y=z=w
pub struct VScalar<V>(V);

// aliases for the most common types.
pub type Point2<T=f32>=Point<Vec2<T>>;
pub type Point3<T=f32>=Point<Vec3<T>>;
pub type Point4<T=f32>=Point<Vec4<T>>;
pub type Normal2<T=f32>=Normal<Vec2<T>>;
pub type Normal3<T=f32>=Normal<Vec3<T>>;
pub type Normal4<T=f32>=Normal<Vec4<T>>;

fn Point2<T:VElem>(x:T,y:T)->Point<Vec2<T>>{ Point(vec2(x,y)) }
fn Point3<T:VElem>(x:T,y:T,z:T)->Point<Vec3<T>>{ Point(vec3(x,y,z)) }
fn Point4<T:VElem>(x:T,y:T,z:T,w:T)->Point<Vec4<T>>{ Point(vec4(x,y,z,w)) }

/// Trait for component multiply access, some platforms dont like moving from vec to scalar pipes..
pub trait VecBroadcastOps:Sized {

	fn vmul_x(&self,b:&Self)->Self;
	fn vmul_y(&self,b:&Self)->Self;
	fn vmul_z(&self,b:&Self)->Self;
	fn vmul_w(&self,b:&Self)->Self;

	fn vmadd_x(&self,a:&Self,b:&Self)->Self;
	fn vmadd_y(&self,a:&Self,b:&Self)->Self;
	fn vmadd_z(&self,a:&Self,b:&Self)->Self;
	fn vmadd_w(&self,a:&Self,b:&Self)->Self;

	/// self * b.x self*b.y, self*b.z, self*b.w
	fn vmul_xyzw(&self,b:&Self)->(Self,Self,Self,Self){(self.vmul_x(b),self.vmul_y(b),self.vmul_z(b),self.vmul_w(b))}
}


/// Types with vector operations; no semantic meaning. abstraction of raw vector register which may hold any data (rgba,normals,points etc). These are wrapped in Point<V>, Normal<V> etc to provide semantic meaning. 
/// TODO: complete hiding of componentwise access. In some processors with SIMD, the raw components may not necaserily play well with the scalar float types of the CPU

// matrix * vector = (matrix.0 * vector).sum_elems    \   these are the same thing?
// vector dot product = vector * vector . sum_elems   /    'componentwise product, and sum'
//
// matrix * matrix = 'matrix* vector' per matrix component..



/// implement vector operations for tuples, interface with plain tuple-based vecmaths

impl<T:One+Zero+Clone+PartialOrd+Num,V:HasXYZ<Elem=T>> VecConsts for V{
	fn vaxis(i:int)->Self{
		match i{
			0=> V::from_xyz(one::<T>(),zero::<T>(),zero::<T>()),
			1=> V::from_xyz(zero::<T>(),one::<T>(),zero::<T>()),
			_=> V::from_xyz(zero::<T>(),zero::<T>(),one::<T>())

		}
	}
	fn origin()->Self{ V::from_xyz(zero::<T>(),zero::<T>(),zero::<T>()) }
	fn one()->Self{ V::from_xyz(one::<T>(),one::<T>(),one::<T>()) }
}
impl<F:Float> VecCmpOps for (F,F,F){
	type CmpOutput=(bool,bool,bool);
	fn vmin(&self,b:&Self)->Self	{
		(
			min_ref(&self.0,&b.0),
			min_ref(&self.1,&b.1),
			min_ref(&self.2,&b.2))}
	fn vmax(&self,b:&Self)->Self	{
		(
			max_ref(&self.0,&b.0),
			max_ref(&self.1,&b.1),
			max_ref(&self.2,&b.2))}

	fn gt(&self,b:&Self)->(bool,bool,bool){
		(self.0>b.0,self.1>b.1,self.2>b.2)
	}
	fn lt(&self,b:&Self)->(bool,bool,bool){
		(self.0<b.0,self.1<b.1,self.2<b.2)
	}
}

impl<F,V> Zero for V where V:HasXYZ<Elem=F>,F:Zero+Num{
	fn zero()->Self{ V::from_xyz(zero::<F>(),zero::<F>(),zero::<F>())}
	fn is_zero(self)->bool {self.x() .is_zero() && self.y() .is_zero() && self.z() .is_zero()}
}


//todo - how?
//impl<F:Float,V:HasXY<F>> VecOps for V{

impl<F:VElem+Zero> VecAccessors for (F,F,F){
	fn vx(&self)->F{self.0.clone()}
	fn vy(&self)->F{self.1.clone()}
	fn vz(&self)->F{self.2.clone()}
	fn vw(&self)->F{zero::<F>()}
	fn splat_ref(f:&F)->Self {(f.clone(), f.clone(), f.clone())}
}

impl<T:Clone> HasElem for (T,){
	type Elem=T;
}

impl<T:Clone> HasElem for (T,T){
	type Elem=T;
}
impl<T:Clone> HasElem for (T,T,T){
	type Elem=T;
}
impl<T:Clone> HasElem for (T,T,T,T){
	type Elem=T;
}

impl<T,V> VecNumOps for V where V:HasXYZ<Elem=T>,T:Num{
    fn vadd(&self, b: &Self) -> Self { V::from_xyz(self.x() + b.x(), self.y() + b.y(), self.z() + b.z()) }
    fn vsub(&self, b: &Self) -> Self { V::from_xyz(self.x() - b.x(), self.y() - b.y(), self.z() - b.z()) }
}



// the arithmetic ops that dont need floating point, e.g. useable for int screen coords
pub trait HasFloatElem :HasElem{}
impl<T:Float,V:HasElem<Elem=T>> HasFloatElem for V{}


pub trait VecOps: Clone+
		Zero+
		VecNumOps+
		//VecCmpOps+
		HasElem+			// todo this dependance is awkward; we want 'vecops' with no scalar
		where Self::Elem : One+Float
{
	// TODO Scalar Type, default = Self::ElemF, but could be VScalar<Self>

	fn wrap_as<V:IsWrappedV<Self>>(&self)->V { IsWrappedV::wrap(self) }
	fn vscale(&self,f: Self::Elem)->Self	{unimplemented!()}
	fn vassign_scale(&mut self,f: Self::Elem)->&mut Self	{*self=self.vscale(f);self}
	fn vmul(&self,b:&Self)->Self			{unimplemented!()}
	fn vsum_elems(&self)-> Self::Elem		{unimplemented!()}
	fn vmul_elems(&self)-> Self::Elem		{unimplemented!()}
	fn vcross(&self,b:&Self)->Self			{unimplemented!()}
	fn vdot(&self,b:&Self)-> Self::Elem	{self.vmul(b).vsum_elems()}
    // 'multiply-add' a+b*c operation. receiver, the asymetrical one, is the add operand .. scalar operand
	// todo .. the naming pattern here is a little unusual,
	// unfortunately the default is .. what?
	fn vmadd(&self,b:&Self,f: Self::Elem)->Self	{self.vadd(&b.vscale(f))}
	fn vmadd7v(&self,b:&Self,c:&Self)->Self	{self.vadd(&b.vmul(&c))}
    fn vmsub(&self,b:&Self,f: Self::Elem)->Self	{self.vsub(&b.vscale(f))}
    // 'multiply-accumulate' - mutates in place, the receiver is an accumulator
    fn vmacc(&mut self, src1:&Self, src2:&Self)->&mut Self {self.vassign_add(&src1.vmul(src2)); self}
    fn vassign_mul_sub(&mut self, src1:&Self, src2:&Self)->&mut Self {self.vassign_sub(&src1.vmul(src2)); self}
    fn vassign_mul_add(&mut self, src1:&Self, src2:&Self)->&mut Self {self.vassign_add(&src1.vmul(src2)); self}

	fn vpara(&self,vaxis:&Self)->Self	{  	let dotp=self.vdot(vaxis); vaxis.vscale(dotp) }
	fn vneg(&self)->Self				;//{self.vscale(-one::<Self::ElemF> ())}
	fn vavr(&self,b:&Self)->Self		{self.vadd(b).vscale(one::<Self::Elem>()/(one::<Self::Elem>()+one::<Self::Elem>()))}
	fn vlerp(&self,b:&Self,f: Self::Elem)->Self	{self.vmadd(&b.vsub(self),f)}

	fn vsqr(&self)->Self::Elem			{ self.vdot(self)}
	fn vlength(&self)->Self::Elem		{ self.vsqr().sqrt()} //vlength!=vec.len ..
    fn vmagnitude(&self)->Self::Elem   {self.vlength()}    //synonym
	fn vreciprocal_magnitude(&self)->Self::Elem	{ one::<Self::Elem>()/(self.vsqr().sqrt())}
	fn vto_length(&self,length:Self::Elem)->Self { self.vscale(length/(self.vsqr().sqrt())) }
	fn vassign_to_length(&mut self,new_length:Self::Elem)->&Self { let nv={self.vscale(new_length/(self.vsqr().sqrt()))};*self=nv;self }
	fn vnormalize(&self)->Self		{ self.vscale(one::<Self::Elem>()/(self.vsqr().sqrt())) }
	fn vperp(&self,axis:&Self)->Self	{ let vpara =self.vpara(axis); self.vsub(&vpara)}
	fn vcross_norm(&self, b:&Self)->Self { self.vcross(b).vnormalize() }
	fn vsub_norm(&self,b:&Self)->Self	{ self.vsub(b).vnormalize() }
	fn vtriangle_norm(&self,b:&Self,c:&Self)->Self{
		b.vsub(self).vcross_norm(&c.vsub(self))		
	}
	// same function, different names.
	fn vassign_norm(&mut self)->&mut Self{ *self=self.vnormalize();self}
	fn vassign_normalize(&mut self)->&mut Self{self.vassign_norm()}

	fn vpara_perp(&self,vaxis:&Self)->(Self,Self) {
		let vpara=self.vpara(vaxis);
		(vpara.clone(),self.vsub(&vpara))
	}
	fn vcross_to_vec3(&self,b:&Self)->Vec3<Self::Elem> {unimplemented!()}
	fn vfrom_xyz_f(x:Self::Elem,y:Self::Elem,z:Self::Elem)->Self  {unimplemented!()}
}

// free function interface to vec maths


macro_rules! impl_vec_method_sametypes{
	($OpTrait:ident,$Bound:ident , $op:ident)=>{

		impl<T:$OpTrait+$Bound> $OpTrait for Vec2<T> {
			type Output=Vec2<T>;
			fn $op(self,rhs:Vec2<T>)->Vec2<T> { 
				vec2(self.x.$op(rhs.x)   , self.y.$op(rhs.y))
			}
		}

		impl<T:$OpTrait $OpTrait for Vec3<T> {
			type Output=Vec3<T>;
			fn $op(self,rhs:Vec3<T>)->Vec3<T> { 
				Vec3(self.x.$op(rhs.x)   , self.y.$op(rhs.y), self.z.$op(rhs.z))
			}
		}

		impl<T:$OpTrait $OpTrait for Vec4<T> {
			type Output=Vec4<T>;
			fn $op(self,rhs:Vec4<T>)->Vec4<T> { 
				vec4(self.x.$op(rhs.x)   , self.y.$op(rhs.y), self.z.$op(rhs.z), self.w.$op(rhs.w))
			}
		}

	}
}

macro_rules! impl_vec_operator{
	($OpTrait:ident :: $op:ident)=>{
		// Implement operators for values
/* got it from vec_xyzw
		impl<A:$OpTrait<B,Output=C>,B,C> $OpTrait<Vec2<B>> for Vec2<A> {
			type Output=Vec2<C>;
			fn $op(self,rhs:Vec2<B>)->Vec2<C> { 
				vec2(self.x.$op(rhs.x)   , self.y.$op(rhs.y))
			}
		}

		impl<A,B,C> $OpTrait<Vec3<B>> for Vec3<A> where A:$OpTrait<B,Output=C>{
			type Output=Vec3<C>;
			fn $op(self,rhs:Vec3<B>)->Vec3<C> {
				Vec3(self.x.$op(rhs.x)   , self.y.$op(rhs.y), self.z.$op(rhs.z))
			}
		}

		impl<A:$OpTrait<B,Output=C>,B,C> $OpTrait<Vec4<B>> for Vec4<A> {
			type Output=Vec4<C>;
			fn $op(self,rhs:Vec4<B>)->Vec4<C> { 
				Vec4::ew(self.x.$op(rhs.x)   , self.y.$op(rhs.y), self.z.$op(rhs.z), self.w.$op(rhs.w))
			}
		}
*/
		impl<A,B,C> $OpTrait<Vec8<B>> for Vec8<A> where 
			A:$OpTrait<B, Output= C>
		{
			type Output=Vec8<C>;
			fn $op(self,rhs:Vec8<B>)->Vec8<C> { 
				Vec8(
					self.0 .$op(rhs.0)   , self.1 .$op(rhs.1), self.2 .$op(rhs.2), self.3 .$op(rhs.3),
					self.4 .$op(rhs.4)   , self.5 .$op(rhs.5), self.6 .$op(rhs.6), self.7 .$op(rhs.7)
				)
			}
		}


		impl<A,B,C> $OpTrait<Vec16<B>> for Vec16<A> where 
			A:$OpTrait<B, Output= C>
		{
			type Output=Vec16<C>;
			fn $op(self,rhs:Vec16<B>)->Vec16<C> { 
				Vec16(
					self.0 .$op(rhs.0)   , self.1 .$op(rhs.1), self.2 .$op(rhs.2), self.3 .$op(rhs.3),
					self.4 .$op(rhs.4)   , self.5 .$op(rhs.5), self.6 .$op(rhs.6), self.7 .$op(rhs.7),
					self.8 .$op(rhs.8)   , self.9 .$op(rhs.9), self.10 .$op(rhs.10), self.11 .$op(rhs.11),
					self.12 .$op(rhs.12)   , self.13 .$op(rhs.13), self.14 .$op(rhs.14), self .15 .$op(rhs.15)
				)
			}
		}
/*
		// Implement operators for references
		impl<'a, A,B,C> $OpTrait<&'a Vec2<B>> for &'a Vec2<A> where
			&'a A:$OpTrait<&'a B, Output= C>
		{
			type Output=Vec2<C>;
			fn $op(self,rhs:&'a Vec2<B>)->Vec2<C> { 
				vec2(self.x.$op(&rhs.x)   , self.y.$op(&rhs.y))
			}
		}

		impl<'a, A,B,C> $OpTrait<&'a Vec3<B>> for &'a Vec3<A> where  
			&'a A:$OpTrait<&'a B, Output= C>
		{
			type Output=Vec3<C>;
			fn $op(self,rhs:&'a Vec3<B>)->Vec3<C> { 
				Vec3(self.x.$op(&rhs.x)   , self.y.$op(&rhs.y), self.z.$op(&rhs.z))
			}
		}

		impl<'a, A,B,C> $OpTrait<&'a Vec4<B>> for &'a Vec4<A> where 
			&'a A:$OpTrait<&'a B, Output= C>
		{
			type Output=Vec4<C>;
			fn $op(self,rhs:&'a Vec4<B>)->Vec4<C> { 
				vec4(self.x.$op(&rhs.x)   , self.y.$op(&rhs.y), self.z.$op(&rhs.z), self.w.$op(&rhs.w))
			}
		}
*/
		impl<'a, A,B,C> $OpTrait<&'a Vec8<B>> for &'a Vec8<A> where 
			&'a A:$OpTrait<&'a B, Output= C>
		{
			type Output=Vec8<C>;
			fn $op(self,rhs:&'a Vec8<B>)->Vec8<C> { 
				Vec8(
					self.0 .$op(&rhs .0)   , self.1 .$op(&rhs.1), self.2 .$op(&rhs.2), self.3 .$op(&rhs.3),
					self.4 .$op(&rhs.4)   , self.5 .$op(&rhs.5), self.6 .$op(&rhs.6), self.7 .$op(&rhs.7)
				)
			}
		}

		impl<'a, A,B,C> $OpTrait<&'a Vec16<B>> for &'a Vec16<A> where 
			&'a A:$OpTrait<&'a B, Output= C>
		{
			type Output=Vec16<C>;
			fn $op(self,rhs:&'a Vec16<B>)->Vec16<C> { 
				Vec16(
					self.0 .$op(&rhs.0)   , self.1 .$op(&rhs.1), self.2 .$op(&rhs.2), self.3 .$op(&rhs.3),
					self.4 .$op(&rhs.4)   , self.5 .$op(&rhs.5), self.6 .$op(&rhs.6), self.7 .$op(&rhs.7),
					self.8 .$op(&rhs.8)   , self.9 .$op(&rhs.9), self.10 .$op(&rhs.10), self.11 .$op(&rhs.11),
					self.12 .$op(&rhs.12)   , self.13 .$op(&rhs.13), self.14 .$op(&rhs.14), self.15 .$op(&rhs.15)
				)
			}
		}
	}
}

impl_vec_operator!(Mul::mul);
impl_vec_operator!(Div::div);
impl_vec_operator!(Add::add);
impl_vec_operator!(Sub::sub);
/*
impl<V:VecNumOps> Sub<V> for V{
    type Output=V;
    fn sub(&self,rhs:&V)->V{self.vsub(rhs)}
}
*/




macro_rules! impl_vec_component_method{
	($Trait:ident::$method:ident where T:$Bound:ident,  for VecN<T>)=>{
		impl<T:$Trait+$Bound+VElem> $Trait for Vec2<T>{
			fn $method(self,rhs:Vec2<T>)->Vec2<T>{
				vec2(
					self.x.$method(rhs.x),
					self.y.$method(rhs.y)
				)
			}
		}

		impl<T:$Trait+$Bound+VElem> $Trait for Vec3<T>{
			fn $method(self,rhs:Vec3<T>)->Vec3<T>{
				vec3(
					self.x.$method(rhs.x),
					self.y.$method(rhs.y),
					self.z.$method(rhs.z),
				)
			}
		}
		impl<T:$Trait+$Bound+VElem> $Trait for Vec4<T>{
			fn $method(self,rhs:Vec4<T>)->Vec4<T>{
				Vec4::<T>{
					x: self.x.$method(rhs.x),
					y: self.y.$method(rhs.y),
					z: self.z.$method(rhs.z),
					w: self.w.$method(rhs.w),
				}
			}
		}
		impl<T:$Trait+$Bound> $Trait for Vec8<T>{
			fn $method(self,rhs:Vec8<T>)->Vec8<T>{
				Vec8::<T>(
					self.0 .$method(rhs.0),
					self.1 .$method(rhs.1),
					self.2 .$method(rhs.2),
					self.3 .$method(rhs.3),
					self.4 .$method(rhs.4),
					self.5 .$method(rhs.5),
					self.6 .$method(rhs.6),
					self.7 .$method(rhs.7),
				)
			}
		}
		impl<T:$Trait+$Bound> $Trait for Vec16<T>{
			fn $method(self,rhs:Vec16<T>)->Vec16<T>{
				Vec16::<T>(
					self.0 .$method(rhs.0),
					self.1 .$method(rhs.1),
					self.2 .$method(rhs.2),
					self.3 .$method(rhs.3),
					self.4 .$method(rhs.4),
					self.5 .$method(rhs.5),
					self.6 .$method(rhs.6),
					self.7 .$method(rhs.7),
					self.8 .$method(rhs.8),
					self.9 .$method(rhs.9),
					self.10 .$method(rhs.10),
					self.11 .$method(rhs.11),
					self.12 .$method(rhs.12),
					self.13 .$method(rhs.13),
					self.14 .$method(rhs.14),
					self.15 .$method(rhs.15),
				)
			}
		}
	}
}

impl_vec_component_method!(Min::min where T:PartialOrd, for VecN<T>);
impl_vec_component_method!(Max::max where T:PartialOrd, for VecN<T>);


//impl<T:Float> Permute for Vec2<T> {}

// todo-trait VecPrimOps
impl<T:VElem+Num> VecNumOps for Vec2<T> {
    fn vadd(&self, b: &Vec2<T>) -> Vec2<T> { Vec2{x:self.x + b.x, y:self.y + b.y} }
    fn vsub(&self, b: &Vec2<T>) -> Vec2<T> { Vec2{x:self.x - b.x, y:self.y - b.y} }
}

/// these are only really interesting where .vx,.vy,.vz are not available,
/// some platforms dont like moving between vector and scalar registers.
/// some past platforms have had direct instructions for these functions.
impl<T:Float,V:VecOps+VecAccessors+HasElem<Elem=T>> VecBroadcastOps for V where V:HasElem<Elem=T>{
	fn vmul_x(&self,b:&Self)->Self{ self.vscale(b.vx()) }
	fn vmul_y(&self,b:&Self)->Self{ self.vscale(b.vy()) }
	fn vmul_z(&self,b:&Self)->Self{ self.vscale(b.vz()) }
	fn vmul_w(&self,b:&Self)->Self{ self.vscale(b.vw()) }
	fn vmadd_x(&self,b:&Self,c:&Self)->Self{ self.vmadd(b,c.vx()) }
	fn vmadd_y(&self,b:&Self,c:&Self)->Self{ self.vmadd(b,c.vy()) }
	fn vmadd_z(&self,b:&Self,c:&Self)->Self{ self.vmadd(b,c.vz()) }
	fn vmadd_w(&self,b:&Self,c:&Self)->Self{ self.vmadd(b,c.vw()) }
}

/// master 'Vector' trait, handles the most common interfaces we expect to use.
pub trait VMath : VecOps + VecCmpOps+VecConsts+
    Permute + VecAccessors+
    Siblings
    where Self::Elem : Float
{}
//impl<T:Float+VElem, V:VecOps+VecCmpOps+VecConsts+Permute+VecAccessors+HasElem<Elem=T>> VMath for V{
//}

impl<T:Float+VElem> VMath for Vec4<T>{}
impl<T:Float+VElem> VMath for Vec3<T>{}
impl<T:Float+VElem> VMath for Vec2<T>{}
//impl<T:Float+VElem> VMath for Vec1<T>{}


impl<T:VElem+Float> VecOps for Vec2<T> {
	fn vscale(&self,f:T)->Vec2<T>		{vec2(self.x*f,self.y*f)}
	fn vmul(&self,b:&Vec2<T>)->Vec2<T>	{vec2(self.x*b.x,self.y*b.y)}
	fn vsum_elems(&self)->T	{self.x+self.y}
    fn vneg(&self)->Self{vec2(-self.x,-self.y)}
	// todo .. not entirely happy with this interface.
	// cross product for a vector type returning its own type seems awkward
	// perhaps 'crossToSelf and crossToVec3' .. and 'cross' for vec3 only?
	fn vcross(&self,_:&Vec2<T>)->Vec2<T>{unimplemented!()}
	fn vcross_to_vec3(&self,b:&Vec2<T>)->Vec3<T>	{vec3(Zero::zero(),Zero::zero(),self.vcross_z(b))}
//	pub fn axisScale(i:int,f:VScalar)->Vec2 { vecAxisScale(i,f) } 
	fn vfrom_xyz_f(x:Self::Elem,y:Self::Elem,z:Self::Elem)->Self {vec2(x,y)}
}


impl<T:Num+VElem+PartialEq> HasXY for Vec2<T>{
    fn x(&self)->T{self.x.clone()}
    fn y(&self)->T{self.y.clone()}
    fn from_xy(x:T,y:T)->Self{ Vec2{x:x.clone(),y:y.clone()}}
}

impl<T:Num+VElem+PartialEq> HasXYZ for Vec3<T>{
	type Appended=Vec4<T>;
    fn x(&self)->T{self.x.clone()}
    fn y(&self)->T{self.y.clone()}
    fn z(&self)->T{self.z.clone()}
    fn from_xyz(x:T,y:T,z:T)->Self{ Vec3{x:x.clone(),y:y.clone(),z:z.clone()}}
}
impl<T:Num+VElem+PartialEq> HasXYZW for Vec4<T>{
    fn x(&self)->T{self.x.clone()}
    fn y(&self)->T{self.y.clone()}
    fn z(&self)->T{self.z.clone()}
    fn w(&self)->T{self.w.clone()}
    fn from_xyzw(x:T,y:T,z:T,w:T)->Self{ Vec4{x:x.clone(),y:y.clone(),z:z.clone(),w:w.clone()}}
}
/*
yikes, this seems to need exclusivity. can't have VecConsts for HasXY, VecConsts for HasXYZ because they *might* overlap?!
impl<T,V> VecConsts for V where V:HasXY<Elem=T>,T:Clone+Zero+One {
	fn one()->V	{V::from_xy(one::<T>(),one::<T>())}
	fn origin()->V	{V::from_xy(zero::<T>(),zero::<T>())}
	fn vaxis(i:int)->V {
		match i{
            0=>V::from_xy(one::<T>(),zero::<T>()),
            1=>V::from_xy(zero::<T>(),one::<T>()),
            _=>V::from_xy(zero::<T>(),zero::<T>())
        }
	}
}
*/

macro_rules! impl_vecn_fmap{
	// works for named elements
	($vectype:ident{$($elem:ident),*})=>{
		impl<A> VecFMap $vectype<A>{
			// dont know which to pick :( lack of C++ references.
			pub fn fmap<R,F:Fn(&A)->R>(&self,fnc:F)->$vectype<R>{
				$vectype{ $( $elem:fnc(&self.$elem) ),* }
			}
			pub fn zip_with<B,R,F:Fn(&A,&B)->R>(&self,b:&$vectype<B>,fnc:F)->$vectype<R>{
				$vectype( $( fnc(&self.$elem, &b.$elem) ),*)
			}
			pub fn foldr<B:Clone,F:Fn(&A,&B)->B>(&self, fnc:F, input:&B)->B{
				let mut acc=input.clone();
				$(acc=fnc(&self.$elem, &acc); )*;
				acc
			}
		}
	}
}
//impl_vecn_fmap!{Vec2{x,y}}
//impl_vecn_fmap!{Vec3{x,y,z}}
//impl_vecn_fmap!{Vec4{x,y,z,w}}

macro_rules! impl_vec_cmp_ops{
	($VecN:ident{$($elem:ident),*})=>{
		impl<T:VElem+PartialOrd+Neg<Output=T>> VecCmpOps for $VecN<T> {
			type CmpOutput=$VecN<bool>;	// todo: figure out nesting, e.g. impl this for scalars to terminate
			fn vmin(&self, b:&Self)->Self	{
				$VecN{$( $elem : min_ref(&self.$elem, &b.$elem) ),*}
			}
			fn vmax(&self, b:&Self)->Self	{
				$VecN{$( $elem : max_ref(&self.$elem, &b.$elem) ),*}
			}
			fn lt(&self, b:&Self)->$VecN<bool>{
				$VecN{$( $elem : self.$elem < b.$elem ),*}
			}
			fn gt(&self, b:&Self)->$VecN<bool>{
				$VecN{$( $elem : self.$elem > b.$elem ),*}
			}
			fn vclamp_scalar_range(&self,a:T,b:T)->Self{
			    $VecN{$( $elem : self.$elem.clamp(a,b) ),*}
			}
			fn vclamp_scalar(&self,a:T)->Self{
			    $VecN{$( $elem : self.$elem.clamp(-a,a) ),*}
			}
   		}
	}
}
impl_vec_cmp_ops!(Vec2{x,y});
impl_vec_cmp_ops!(Vec3{x,y,z});
impl_vec_cmp_ops!(Vec4{x,y,z,w});

impl<T:Float+VElem,V> VecOps for V where V:HasXYZ<Elem=T> {
    // todo-trait VecPrimOps
    fn vscale(&self,f:T)->V		{V::from_xyz(self.x()*f, self.y()*f, self.z()*f)}
    fn vmul(&self,b:&V)->V	{V::from_xyz(self.x()*b.x(), self.y()*b.y(), self.z()*b.z())}
    fn vsum_elems(&self)->T	{self.x()+self.y()+self.z()}
    fn vmul_elems(&self)->T	{self.x()*self.y()*self.z()}
    fn vcross(&self,b:&V)->V	{V::from_xyz(self.y()*b.z()-self.z()*b.y(), self.z()*b.x()-self.x()*b.z(), self.x()*b.y()-self.y()*b.x())}

    fn vcross_to_vec3(&self,b:&V)->Vec3<T>	{let v=self.vcross(b); vec3(v.x(),v.y(),v.z())}
    fn vfrom_xyz_f(x:T,y:T,z:T)->V {V::from_xyz(x,y,z)}
    fn vneg(&self)->Self{V::from_xyz(-self.x(),-self.y(),-self.z())}
}

/// the vector has siblings with 1,2,3,4 elements.
pub trait Siblings {
	type V1:Clone;
	type V2:Clone;
	type V3:Clone;
	type V4:Clone;
}
impl<T:VElem> Siblings for Vec1<T>{
	type V1=Vec1<T>;
	type V2=Vec2<T>;
	type V3=Vec3<T>;
	type V4=Vec4<T>;
}
impl<T:VElem> Siblings for Vec2<T>{
	type V1=Vec1<T>;
	type V2=Vec2<T>;
	type V3=Vec3<T>;
	type V4=Vec4<T>;
}
impl<T:VElem> Siblings for Vec3<T>{
	type V1=Vec1<T>;
	type V2=Vec2<T>;
	type V3=Vec3<T>;
	type V4=Vec4<T>;
}
impl<T:VElem> Siblings for Vec4<T>{
	type V1=Vec1<T>;
	type V2=Vec2<T>;
	type V3=Vec3<T>;
	type V4=Vec4<T>;
}

impl<T:Clone> Siblings for (T,){
	type V1=(T,);
	type V2=(T,T);
	type V3=(T,T,T);
	type V4=(T,T,T,T);
}
impl<T:Clone> Siblings for (T,T){
	type V1=(T,);
	type V2=(T,T);
	type V3=(T,T,T);
	type V4=(T,T,T,T);
}
impl<T:Clone> Siblings for (T,T,T){
	type V1=(T,);
	type V2=(T,T);
	type V3=(T,T,T);
	type V4=(T,T,T,T);
}
impl<T:Clone> Siblings for (T,T,T,T){
	type V1=(T,);
	type V2=(T,T);
	type V3=(T,T,T);
	type V4=(T,T,T,T);
}



/// accessors for a contained type guaranteed to support floatmath
/// we just simplified this to padding out.
pub trait VecAccessors : HasElem+Sized {
	fn vx(&self)->Self::Elem;
	fn vy(&self)->Self::Elem;
	fn vz(&self)->Self::Elem;
	fn vw(&self)->Self::Elem;
	fn splat(s:Self::Elem)->Self{Self::splat_ref(&s)}
    fn splat_ref(&Self::Elem)->Self;
	fn splat_x(&self)->Self{ Self::splat(self.vx())}	// can't say which type, so no default.
	fn splat_y(&self)->Self{ Self::splat(self.vy())}
	fn splat_z(&self)->Self{ Self::splat(self.vz())}
	fn splat_w(&self)->Self{ Self::splat(self.vw())}
}
pub trait SplatToVec :VElem{
    fn splat2(&self)->Vec2<Self>;
    fn splat3(&self)->Vec3<Self>;
    fn splat4(&self)->Vec4<Self>;
}
impl<T:VElem> SplatToVec for T{
    fn splat2(&self)->Vec2<T>{vec2(self.clone(),self.clone())}
    fn splat3(&self)->Vec3<T>{vec3(self.clone(),self.clone(),self.clone())}
    fn splat4(&self)->Vec4<T>{vec4(self.clone(),self.clone(),self.clone(),self.clone())}
}
pub trait Splat<V:HasElem<Elem=Self>> :VElem {
    fn splat_to_vec(&self) -> V;
}

macro_rules! impl_splat{
    ($vec:ident{$($elem:ident),*})=>
    {   impl<T: VElem> Splat<$vec<T>> for T {
            fn splat_to_vec(&self) -> $vec<T> {
                $vec{$($elem:self.clone()),*}
            }
        }
    }
}
impl_splat!{Vec2{x,y}}
impl_splat!{Vec3{x,y,z}}
impl_splat!{Vec4{x,y,z,w}}


pub fn normal(x:f32,y:f32,z:f32)->Vec3<f32>{vec3(x,y,z).vnormalize()}

impl<T:Zero+VElem> VecAccessors for Vec2<T>
{
	fn vx(&self)->T	{ self.x.clone()}
	fn vy(&self)->T	{ self.y.clone()}
	fn vz(&self)->T	{ zero::<T>()}
	fn vw(&self)->T	{ zero::<T>()}
	fn splat(f:T)->Self{ vec2(f.clone(),f.clone())}
	fn splat_ref(f:&T)->Self{ vec2(f.clone(),f.clone())}
}

impl<T:Zero+VElem> VecAccessors for Vec3<T> {
	fn vx(&self)->T	{ self.x.clone()}
	fn vy(&self)->T	{ self.y.clone()}
	fn vz(&self)->T	{ self.z.clone()}
	fn vw(&self)->T	{ zero::<T>()}
	fn splat(f:T)->Self{ vec3(f.clone(),f.clone(),f.clone())}
    fn splat_ref(f:&T)->Self{ vec3(f.clone(),f.clone(),f.clone())}
}

impl<T:Zero+VElem> VecAccessors for Vec4<T>
{
	fn vx(&self)->T	{ self.x.clone()}
	fn vy(&self)->T	{ self.y.clone()}
	fn vz(&self)->T	{ self.z.clone()}
	fn vw(&self)->T	{ self.w.clone()}
	fn splat(f:T)->Self{ vec4(f.clone(),f.clone(),f.clone(),f.clone())}
    fn splat_ref(f:&T)->Self{ vec4(f.clone(),f.clone(),f.clone(),f.clone())}
}

impl<T:Zero+VElem> Zero for Vec4<T> {
	fn zero()->Vec4<T>{vec4(zero::<T>(),zero::<T>(),zero::<T>(),zero::<T>())}
	fn is_zero(self)->bool  {self.x.is_zero() && self.y.is_zero() && self.z.is_zero() && self.w.is_zero()}
}
impl<T:Zero+VElem> Zero for Vec2<T> {
    fn zero()->Vec2<T>{vec2(zero::<T>(),zero::<T>())}
    fn is_zero(self)->bool  {self.x.is_zero() && self.y.is_zero()}
}

pub trait Project<T:VElem=f32>{
    fn project_to_vec3(&self)->Vec3<T>;
    fn project(&self)->Self;
}

impl<T:Float+VElem> Project<T> for Vec4<T>{
    fn project_to_vec3(&self)->Vec3<T>{
        let inv=self.w.recip();
        vec3(self.x*inv,self.y*inv,self.z*inv)
    }
    fn project(&self)->Vec4<T>{
        let inv=self.w.recip();
        vec4(self.x*inv,self.y*inv,self.z*inv,one())
    }
}


impl<T:Zero+One+VElem> VecConsts for Vec2<T> {

    fn one()->Vec2<T>	{vec2(one::<T>(),one::<T>())}
    fn origin()->Vec2<T>	{vec2(zero::<T>(),zero::<T>())}
    fn vaxis(i:int)->Vec2<T>{
        match i{
            0=>vec2(one::<T>(),zero::<T>()),
            _=>vec2(zero::<T>(),one::<T>()),
        }
    }
}

pub trait ToVec2<T:VElem> {
    fn to_vec2(&self)->Vec2<T>;
}

pub trait ToVec3<T:VElem> {
    fn to_vec3(&self)->Vec3<T>;
}
impl<T:Float+VElem,V:HasXYZ<Elem=T>> ToVec3<T> for V{
	fn to_vec3(&self)->Vec3<T>{vec3(self.x(),self.y(),self.z())}
}
impl<T:Float+Clone> ToVec2<T> for (T,T){
    fn to_vec2(&self)->Vec2<T>{vec2(self.0.clone(),self.1.clone())}
}

//impl<T:Float> ToVec3<T> for (T,T,T){
//    fn to_vec3(&self)->Vec3<T>{Vec3(self.0,self.1,self.2)}
//}
impl<T:Float> ToVec4<T> for (T,T,T,T){
    fn to_vec4(&self)->Vec4<T>{vec4(self.0,self.1,self.2,self.3)}
}
//impl<T,V:HasXYZ<Elem=T>> ToVec3<T> for V{
//	fn to_vec3(&self)->Vec3<T>{vec3(self.x(),self.y(),self.z())}
//}
pub trait ToTuple<T>{
    type Output;
    fn to_tuple2(&self)->(T,T);
    fn to_tuple3(&self)->(T,T,T);
    fn to_tuple4(&self)->(T,T,T,T);
    fn to_tuple(&self)->Self::Output;
}
impl<T:Float> ToTuple<T> for Vec3<T>{
    type Output=(T,T,T);
    fn to_tuple(&self)->Self::Output{(self.x,self.y,self.z)}
    fn to_tuple2(&self)->(T,T){ (self.x,self.y)}
    fn to_tuple3(&self)->(T,T,T){ (self.x,self.y,self.z)}
    fn to_tuple4(&self)->(T,T,T,T){ (self.x,self.y,self.z,zero())}
}
impl<T:Float> ToTuple<T> for Vec4<T>{
    type Output=(T,T,T,T);
    fn to_tuple(&self)->Self::Output{(self.x,self.y,self.z,self.w)}
    fn to_tuple2(&self)->(T,T){ (self.x,self.y)}
    fn to_tuple3(&self)->(T,T,T){ (self.x,self.y,self.z)}
    fn to_tuple4(&self)->(T,T,T,T){ (self.x,self.y,self.z,self.w)}
}
//impl<T:Float> ToVec3<T> for Vec4<T>{
//    fn to_vec3(&self)->Vec3<T>{ Vec3(self.x,self.y,self.z)}
//}
impl<T:Float> ToVec2<T> for Vec2<T>{
    fn to_vec2(&self)->Vec2<T>{ self.clone()}
}

//impl<T:Float> ToVec3<T> for Vec3<T>{
//    fn to_vec3(&self)->Vec3<T>{ self.clone()}
//}
impl<T:Float> ToVec4<T> for Vec4<T>{
    fn to_vec4(&self)->Vec4<T>{ self.clone()}
}

impl<'a,T:VElem> From<&'a [T;3]> for Vec3<T> {
    fn from(src:&'a [T;3])->Vec3<T>{ vec3(src[0].clone(),src[1].clone(),src[2].clone()) }
}
impl<'a,T:VElem> From<&'a [T;4]> for Vec4<T> {
    fn from(src:&'a [T;4])->Vec4<T>{ vec4(src[0].clone(),src[1].clone(),src[2].clone(),src[3].clone()) }
}



/*
fn unpack_sub<F:Float,V:HasXYZW<Elem=F>>(src:u32, centre:u32, scale:F )->V{
    V::from_xyzw(
        ((src&255)-centre) as F*scale,
        (((src>>8)&255)-centre) as F*scale,
        (((src>>16)&255)-centre) as F*scale,
        (((src>>24)&255)-centre) as F*scale);

}
fn pack_sub<F:Float,V:HasXYZW<Elem=F>>(src:&V, centre:F, scale:F )->u32{
    clamp((src.x()+centre)*scale as u32,0,255)|
        (clamp( (src.y()+centre)*scale as u32,0,255)<<8)|
        (clamp( (src.z()+centre)*scale as u32,0,255)<<16)|
              (clamp( (src.w()+centre)*scale as u32,0,255)<<24)
}
impl<F:Float,V:HasXYZW<Elem=F>> From<PackedARGB> for V{
    fn from(src:PackedARGB)->Self{
        unpack_sub(src.0, 0,1.0/255.0)
    }
}
impl<F:Float,V:HasXYZW<Elem=F>> From<V> for PackedARGB {
    fn from(src:&V)->Self{
        PackedARGB(pack_sub(src,0.0, 255.0))
    }
}

impl<F:Float,V:HasXYZW<Elem=F>> From<PackedS8x4> for V{
    fn from(src:PackedARGB)->Self{
        unpack_sub(src.0, 128,1.0/128.0)
    }
}
impl<F:Float,V:HasXYZW<Elem=F>> From<V> for PackedS8x4 {
    fn from(src:&V)->Self{
        PackedS8x4(pack_sub(src,0.5, 127.0))
    }
}
*/


impl<T:Zero+One+VElem> VecConsts for Vec4<T> {

	fn one()->Vec4<T>	{vec4(one::<T>(),one::<T>(),one::<T>(),one::<T>())}
	fn origin()->Vec4<T>	{vec4(zero::<T>(),zero::<T>(),zero::<T>(),one::<T>())}
	fn vaxis(i:int)->Vec4<T>{
		match i{
			0=>vec4(one::<T>(),zero::<T>(),zero::<T>(),zero::<T>()),
			1=>vec4(zero::<T>(),one::<T>(),zero::<T>(),zero::<T>()),
			2=>vec4(zero::<T>(),zero::<T>(),one::<T>(),zero::<T>()),
			3=>vec4(zero::<T>(),zero::<T>(),zero::<T>(),one::<T>()),
			_=>vec4(zero::<T>(),zero::<T>(),zero::<T>(),zero::<T>())
		}
	}
}
/*
impl<T:Float> VecPermute for Vec4<T> {
	// nothing to pad when it's Vec4->Vec4
	fn to_vec4_pad0000(&self)->Self{self.clone()}
	fn to_vec4_pad0001(&self)->Self{self.clone()}
}
*/

impl<T:Num+VElem> VecNumOps for Vec4<T> {
    fn vadd(&self, b: &Vec4<T>) -> Vec4<T> { vec4(self.x + b.x, self.y + b.y, self.z + b.z, self.w + b.w) }
    fn vsub(&self, b: &Vec4<T>) -> Vec4<T> { vec4(self.x - b.x, self.y - b.y, self.z - b.z, self.w - b.w) }

}

/*
needs mutually-exclusive-traits
impl<T:Num+Default,V:HasXYZW<Elem=T>> VecNumOps for V {
    fn vadd(&self, b: &V) -> V { V::from_xyzw(self.x() + b.x(), self.y() + b.y(), self.z() + b.z(), self.w() + b.w()) }
    fn vsub(&self, b: &V) -> V { V::from_xyzw(self.x() - b.x(), self.y() - b.y(), self.z() - b.z(), self.w() - b.w()) }
}
*/


impl<T:Float> VecOps for Vec4<T> {
	// todo-trait VecPrimOps
	fn vscale(&self,f:T)->Vec4<T>		{vec4(self.x*f,self.y*f,self.z*f,self.w*f)}
	fn vmul(&self,b:&Vec4<T>)->Vec4<T>	{vec4(self.x*b.x,self.y*b.y,self.z*b.z,self.w*b.w)}
	fn vsum_elems(&self)->T	{self.x+self.y+self.z+self.w}
	fn vmul_elems(&self)->T	{self.x*self.y*self.z*self.w}

	fn vcross(&self,b:&Vec4<T>)->Vec4<T>	{vec4(self.y*b.z-self.z*b.y,self.z*b.x-self.x*b.z,self.x*b.y-self.y*b.x,zero::<T>())}

	fn vcross_to_vec3(&self,b:&Vec4<T>)->Vec3<T>	{self.vcross(b).permute_xyz()}
	fn vfrom_xyz_f(x:T,y:T,z:T)->Vec4<T>{vec4(x,y,z,zero::<T>())}
    fn vneg(&self)->Self{vec4(-self.x(),-self.y(),-self.z(),-self.w())}
}
/*
#[derive(Clone,Debug)]
pub struct MinMax<A> {  
	pub min:A,pub max:A
}
impl MinMax<Vec3>
	where 
//		T:PartialOrd+Num+Clone,
//		V:VecOps<T>
{ 
	fn size(&self)->Vec3 { self.max.vsub(&self.min) }
	fn centre(&self)->Vec3 { self.min.vadd(&self.max).vscale(0.5f32) }
}
*/
//impl<T:PartialOrd+Num+Float,V:VecOps<T>> MinMax<V>
//impl MinMax<Vec3>
//{ 
//	fn centre(&self)->V { self.min.vadd(&self.max).vscale(One::one::<T>()/(One::one::<T>()+One::one::<T>())) }
//}

pub trait UpdateExtents{
	fn update_extents(&self, lo:&mut Self, hi:&mut Self);
}

impl<T:PartialOrd+VElem+Num> UpdateExtents for  Vec3<T>{
	fn update_extents(&self, lo:&mut Self, hi:&mut Self){
		*lo=lo.min(*self);
		*hi=hi.max(*self);
	}
}
/// return a clone of the minimum of 2 references
fn min_ref<'l,T>(a:&'l T,b:&'l T)->T
	where for<'a> &'a T:PartialOrd,T:Clone
{
	if a<b{a.clone()}else{b.clone()}
}

/// return a clone of the maximum of 2 references
fn max_ref<'l,T>(a:&'l T,b:&'l T)->T
	where for<'a> &'a T:PartialOrd,T:Clone
{
	if a>b{a.clone()}else{b.clone()}
}
