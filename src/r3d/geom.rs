use super::*;

/// newtype for an 'offset vector' - difference of points, could be 2d,3d,4d
/// Ofs(Vec3) should coerce to 'Vec4' with W=0'
/// another way to implement these may be a version of Vec2/3/4 which take a different type
/// for the last value, allowing plugging in a ZeroVal or OneVal type.

/// newtype for a 'normal vector' - guaranteed magnitude =1, dimensionless. could be 2d,3d,4d
/// 3d would coerce to 4d with W=0
//pub struct Normal<V>(pub V);

/*
macro_rules! impl_operator{
	//binary operator
	( $T:ident $op:ident($a:ident,$b:ident)->$c:ident)=>{
		
	}
	//unary operator
	( $T:ident::$op:ident($a:type)->$c:type)=>{
		
	}
};
impl_operator!{Sub::sub(Point<V>,Point<V>)->Ofs<V>};
impl_operator!{Add::add(Point<V>,Ofs<V>)->Point<V>};
impl_operator!{Add::add(Ofs<V>,Point<V>)->Point<V>};
impl_operator!{Add::add(Ofs<V>,Ofs<V>)->Ofs<V>};
impl_operator!{Normalize::normalize(Ofs<V>)->Normal<V>};
impl_operator!{Add::add(Normal<V>,Normal<V>)->Ofs<V>};
impl_operator!{Sub::sub(Normal<V>,Normal<V>)->Ofs<V>};
impl_operator!{Mul::mul(Normal<V>,T)->Ofs<V>};
impl_operator!{Mul::mul(Ofs<V>,T)->Ofs<V>};//todo.. all the options.. accel, yada..
*/

pub type Idx=uint;
pub trait	VertexArray<V:VecCmpOps> {
	fn num_vertices(&self)->Idx;
	fn vertex(&self,i:Idx)->V;
	fn aabb(&self)->Extents<V> {
		let mut vmin=self.vertex(0); let mut vmax=self.vertex(0);
		let mut i=self.num_vertices()-1;
		while i>0 { i-=1; vmin=vmin.vmin(&self.vertex(i)); vmax=vmax.vmax(&self.vertex(i)); }
		Extents{min:vmin,max:vmax}
	}
}

#[derive(Clone,Copy,Debug)]
pub struct Line<V:Copy> {
	pub vertex:[V;2]
}
pub fn Line<V:Copy>(a:V,b:V)->Line<V>{Line{vertex:[a,b]}}

#[derive(Clone,Copy,Debug)]
pub struct Triangle<V:Copy> {
	pub vertex:[V;3]
}
pub fn Triangle<V:Copy>(a:V,b:V,c:V)->Triangle<V>{Triangle{vertex:[a,b,c]}}

#[derive(Clone,Copy,Debug)]
pub struct Quad<V:Copy> {
	pub vertex:[V;4]
}

#[derive(Clone,Copy,Debug)]
pub struct Tetrahedron<V:Copy>{
	pub vertex:[V;4]
}

type AABB<T> =Extents<Vec3<T>>;
//type Rect<T> =MinMax<Vec2<T>>;

#[derive(Clone,Debug)]
pub struct Sphere<T:VElem> {
	pub pos:Vec3<T>, pub radius:T
}
#[derive(Clone,Debug)]
pub struct NormalCone<T:VElem+Float>(Vec3<T>,T);

#[derive(Clone,Debug)]
pub struct OOBB<T:VElem> {
	pub matrix:Matrix4<Vec3<T>>,
	pub size:Vec3<T>
}
#[derive(Clone,Debug)]
pub struct Extents<V:Sized=Vec3<f32>> {  
	pub min:V,pub max:V
}

// 'Position' trait for anything with a spatial centre/position.
// position should be an x,y,z or x,y,z,1
pub trait Pos<V=Vec3<f32>> {
	type Output;
	fn pos(&self)->V;
	fn set_pos(&mut self,v:&V) {unimplemented!()}
}

pub trait Norm<V=Vec3<f32>> {
	type Output;
	fn norm(&self)->V;
	fn set_norm(&mut self,v:&V) {unimplemented!()}
}
pub trait Color<V=Vec4<f32>> {
	type Output;
	fn color(&self)->V;
	fn set_color(&mut self,v:&V) {unimplemented!()}
}
pub trait Tex0<V=Vec2<f32>> {
	type Output;
	fn tex0(&self)->V;
	fn set_tex0(&mut self,v:&V) {unimplemented!()}
}
// todo - f16 type for sanity.


impl Extents<Vec3<f32>>{
	pub fn new()->Extents<Vec3<f32>> {
        // negative extents - empty
		let f=1000000.0f32;//todo: FLT_MAX
		Extents{min:Vec3::splat(f),max:Vec3::splat(-f)}
	}
	pub fn from_vertices<V:Pos>(vertices:&[V])->Extents {
		let mut m=Extents::new();
		for v in vertices.iter() {
			m.include(&v.pos());
		}
		m
	}
}

pub fn Extents<V:VecCmpOps+Clone>(a:&V,b:&V)->Extents<V>{ Extents{min:a.vmin(b),max:a.vmax(b)}}
pub type Rect=Extents<Vec2f>;
pub type Cuboid=Extents<Vec3f>;

impl<V:VecCmpOps> Extents<V> { 
	pub fn include(&mut self, v:&V) {
		self.min=self.min.vmin(v);
		self.max=self.max.vmax(v);
	}
}
impl<T:Float,V:VMath<Elem=T>> Extents<V> {
    pub fn size(&self)->V {
        self.max.vsub(&self.min)
    }
}

pub trait Centre<V:Clone> {
	fn centre(&self)->V;
}
impl<V:VecOps> Centre<V> for Extents<V> 
	where <V as HasElem>::Elem:Float
{
    fn centre(&self)->V { self.min.vavr(&self.max)}
}


//pub type AABB=Extents<Vec3<f32>>;

pub trait ExtentsTrait<T:Sized+Copy> {
	fn min(&self)->T;
	fn max(&self)->T;
	fn overlap(&self,other:&Extents<T>)->bool;
	fn contains(&self,other:&Extents<T>)->bool;
}

pub trait Polygon<V:Copy> 
{
	fn num_vertices()->usize;
	fn normal()->V;
	fn edge(i:Idx)->Line<V>;
	fn aabb(&self)->Extents<V>;
}

/// triangle normal trait. gives method for calculating and bound for normal from a point
/// e.g. TriangleNormal<Point>::Output
pub trait TriangleNormal :Copy{
	type Output;
	type Edge:Cross<Self::Edge>;
	type EdgeCrossEdge;
	fn triangle_normal(self,b:Self,c:Self)->Self::Output;
}
impl<Point,Normal,E,C> TriangleNormal for Point where
	Point:Sub<Point,Output=E>+Copy,
	E:Cross<E,Output=C>+Copy,
	C:Normalize<Output=Normal>+Copy,
{
	type Output=Normal;
	type Edge=E;
	type EdgeCrossEdge=C;
	/// (self,b,c) form a triangle. calcualte the normal from the cross of both edge vectors eminating from 'self'
	fn triangle_normal(self,b:Point,c:Point)->Normal {
		(b-self).cross(c-self).normalize()
	}
}

/// free-function wrapper for triangle normal
fn triangle_normal<P:TriangleNormal<Output=N>,N>(a:P,b:P,c:P)->N{
	a.triangle_normal(b,c)
}

//fn Triangle<V:Copy>(a:V,b:V,c:V)->Triangle<V>{ Tri{pos:[a,b,c]} }
pub trait Normal {
	type Output;
	fn normal(&self)->Self::Output;
}

impl<V:TriangleNormal<Output=N>,N> Normal for Triangle<V> {
	type Output=N;
	fn normal(&self)->N { self.vertex[0].triangle_normal(self.vertex[1],self.vertex[2]) }
}

/// plane at a vertex, for precision
pub struct PlaneThruPoint<P,N>(P,N);
impl<P,N:Copy> Normal for PlaneThruPoint<P,N> { type Output=N; fn normal(&self)->N{ self.1 }}

struct NormalHit{
}

type Radius=f32;
pub struct Capsule<V:Copy>(pub Line<V>,pub Radius);

pub trait Intersection<B> {
	type Output;
	fn intersect(&self, other:&B)->Self::Output;
}

pub trait GetClosestFeatures<B> {
	type Output;
	fn get_closest_features(&self,other:&B)->Self::Output;
}

/// result of get_closest_features; contains the two points on the objects having been tested, their seperation and the normal
pub struct ClosestFeatures<S=f32,V=Vec3<S>>{
	pub points:[V;2],	//
	pub normal:V,
	pub seperation:S			// >0 means no collision, <0 means the objects intersect
}
/*
// point-vs-point 'collision test' = none.
impl<T:MyFloat+Copy+Sized> GetClosestFeatures<Self> for Vec3<T> where
{
	type Output=ClosestFeatures<Vec3<T>,T>;
	fn get_closest_features(&self,other:&Vec3<T>)->Self::Output {
		ClosestFeatures{
			points:[*self,*other],
			normal:other.sub_norm(*self),
			seperation:other.distance(self)
		}
	}
}
*/


