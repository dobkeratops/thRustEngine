use super::*;

#[derive(Clone,Debug)]
pub struct	VertexNCT
{
	pub pos:Vec3<f32>,
	pub norm:Vec3<f32>,
	pub color:Vec4<f32>,	// TODO: should be packed format!!!
	pub tex0:Vec2<f32>
}

impl Pos<Vec3> for VertexNCT{	
	type Output=Vec3;
	fn pos(&self)->Vec3 {Vec3(self.pos.x,self.pos.y,self.pos.z)}
	fn set_pos(&mut self,v:&Vec3) {self.pos=v.clone();}
}

impl Norm<Vec3> for VertexNCT{	
	type Output=Vec3;
	fn norm(&self)->Vec3 {Vec3(self.pos.x,self.pos.y,self.pos.z)}
	fn set_norm(&mut self,v:&Vec3) {self.pos=v.clone();}
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
