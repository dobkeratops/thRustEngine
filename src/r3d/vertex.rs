use super::*;

// N normal
// C packed-color
//  FC=Float Color.
// T texcoord
// P position
// D DiffuseColor
// W weights
// B bones
// S specularColor
// FC=FloatColor, the unusual attribute

// packed color vs float color
// blend weights

// Todo .. 
// would we make Vertex<POS,Color=Vec4f, Tex=Vec2f, ()> 
// .. and typedef it to each?

#[derive(Clone,Debug)]
pub struct	VertexNFCT
{
	pub pos:Vec3<f32>,
	pub norm:Vec3<f32>,
	pub color:Vec4<f32>,	// TODO: should be packed format!!!
	pub tex0:Vec2<f32>
}

/// simple vertex, position without attributes,but semantically distinct from 'vector'
#[derive(Clone,Debug)]
pub struct	VertexP
{
	pub pos:Vec3<f32>,
}

impl Pos<Vec3> for VertexP {
	type Output=Vec3;
	fn pos(&self)->Vec3{self.pos}
}

impl Pos<Vec3> for VertexNFCT{	
	type Output=Vec3;
	fn pos(&self)->Vec3 {Vec3(self.pos.x,self.pos.y,self.pos.z)}
	fn set_pos(&mut self,v:&Vec3) {self.pos=v.clone();}
}

impl Norm<Vec3> for VertexNFCT{	
	type Output=Vec3;
	fn norm(&self)->Vec3 {Vec3(self.pos.x,self.pos.y,self.pos.z)}
	fn set_norm(&mut self,v:&Vec3) {self.pos=v.clone();}
}

/// trait for objects with vertex-arrays
/// TODO - should it just return a slice? 
/// ...you dont want indexing interface unless it's efficient for indexing..
/// todo: return enumerated vertex iterator aswell, whats the protocol?

/// TODO - how to make vertex index generic? gen 'to_usize'/'from_usize'
/// for the moment,just throw hardcoded i32 around as a universal vertex index.

type VTI=i32;
pub trait HasVertices<V:Pos>{
	fn num_vertices(&self )->VTI;
	fn vertex<'a>(&'a self,i:VTI)->&'a V;

	/// compute object's local extents of position vectors
	fn bounding_box(&self)->Extents<Vec3f>{
		let mut e=Extents::new();	
		for i in 0..self.num_vertices() as i32{
			e.include(&self.vertex(i).pos());
		}
		e
	}

	///Compute the AABB of all vertex attributes
	/// TODO - not so sure about this.
	/// not all vertices will have linearly combinable attributes.
	/// we might need 2 types of vertex.
//	fn vertex_extents()->Extents<V>{//of all vertex params
//	}

	fn map_vertices<R,F:Fn(VTI,&V)->R>(&self,f:F)->Array<R>{
		let mut result=Array::new();
		for i in 0..self.num_vertices(){ result.push( f(i,self.vertex(i))) }
		result
	}
	fn fold_vertices<A,R,F:Fn(VTI,A,& V)->A>(&self,input:A,f:F)->A{
		let mut acc=input;
		for i in 0..self.num_vertices(){ acc=f(i,acc,self.vertex(i))}
		acc
	}
}


/*
struct PT(Vec3,Vec2);
struct PC(Vec3,PackedARGB);
struct PNCT(Vec3,Vec3,PackedARGB,Vec2);

impl Pos<Vec3> for VertexPNCT {
	type Output=Vec3;
	fn pos(&self)->Vec3{self.pos}
	fn set_pos(&self,v:&Vec3)->Vec3{self.pos=v}
}
impl Tex0<Vec2> for VertexPNCT {
	type Output=Vec2;
	fn tex0(&self)->Vec2{self.tex0.clone()}
	fn set_tex0(&self,v:&Vec2)->Vec2{self.tex0=v.clone();}
}
impl Norm<Vec3> for VertexPNCT {
	type Output=Vec3;
	fn norm(&self)->Vec3{self.norm.clone()}
	fn set_norm(&self,v:&Vec3)->Vec3{self.norm=v.clone();}
}
impl Color<Vec4> for VertexPNCT {
	type Output=Vec4;
	fn color(&self)->Vec4{self.norm.clone()}
	fn set_color(&self,v:&Vec4)->Vec4{self.norm=v.clone();}
}
impl VertexNCT {
	fn bind_attribs(){
	}
}

// we dont need a vertex structure for this
// we can just allocate.
enum VertexType{
	VertexPNCT,		// main mode
	VertexPC,		// debug, color only.
	VertexPT,		// positional/texture
	VertexC,		// color channel only.
}
enum ElemType{
	Vec2h,
	Vec2f,
	Vec3f,
	Vec3h,
	Vec4f,
	Packed8888,
	PackedNormal,
}

struct VertexElem{
	attr:VertexAttrIndex,
	elemtype:ElemType,
	offset:usize,
};
struct VertexDesc{
	size:usize,
	elems:Vec<VertexElem>
};
static mut g_vertex_formats:Option<Vec<VertexDesc>>=None;
*/

/*
can't work without concat_idents!
macro_rules! define_vertex{
	{struct $vertex_name:ident{$($elem_name:ident:$elem_type:ty),*}}=>{
		#[derive(Clone,Debug)]
		struct $vertex_name {
			$(pub $elem_name:$elem_type,)*
		}
		$(
			impl $elem_name for $vertex_name {
				type Output=$elem_type;
				fn $elem_name(&self)->$elem_type {self.$elem_name.clone()}
				fn concat_idents!(set_,$elem_name)(&mut self,v:&$elem_type) {self.$elem_name=v.clone();}
			}
		)*
	}
}

macro_rules! define_elem_trait{
	($elem_name:ident,$setter:ident)=>{
		pub trait $elem_name<V> {
			type Output;
			fn $elem_name(&self)->V;
			fn $setter(&self)->V;
		}
	}
}

define_elem_trait!(pos,set_pos);
define_elem_trait!(norm,set_norm);
define_elem_trait!(tex0,set_tex0);
define_elem_trait!(tex1,set_tex1);
define_elem_trait!(color,set_color);
define_elem_trait!(binormal,set_binormal);
define_elem_trait!(tangent,set_tangent);
define_elem_trait!(blend_weights,set_blend_weights);
define_elem_trait!(bones,set_bones);

define_vertex! {struct VertexPNCT{ pos:Vec3<f32>,norm:Vec3<f32>,color:Vec4<f32>,tex0:Vec2<f32> }}
define_vertex! {struct VertexPC{ pos:Vec3<f32>,color:Vec4<f32>, }}
*/
// todo - map vertex attrib
