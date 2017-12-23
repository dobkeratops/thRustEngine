//use super::*;
pub type Idx=i32;
use std::ops::Range;
use std::ops::{Add,Sub,Mul,Div,Rem,BitOr,BitAnd,BitXor,Index,IndexMut};

// local mini maths decouples
#[derive(Copy,Debug,Clone)]
pub struct Vec2<T>{pub x:T,pub y:T} // array3d::Vec3 should 'into()' into vector::Vec3 , etc.
#[derive(Copy,Debug,Clone)]
pub struct Vec3<T>{pub x:T,pub y:T,pub z:T} // array3d::Vec3 should 'into()' into vector::Vec3 , etc.
#[derive(Copy,Debug,Clone)]
pub struct Vec4<T>{pub x:T,pub y:T,pub z:T,pub w:T} // array3d::Vec3 should 'into()' into vector::Vec3 , etc.
pub type V2i=Vec2<Idx>;
pub type V3i=Vec3<Idx>;
pub type V4i=Vec4<Idx>;//TODO
type Axis_t=i32;
const XAxis:Axis_t=0; const YAxis:Axis_t=1; const ZAxis:Axis_t=2;

impl<T> Index<i32> for Vec3<T>{
	type Output=T;
	fn index(&self,i:Axis_t)->&T{match i{ XAxis=>&self.x,YAxis=>&self.y,ZAxis=>&self.z,_=>panic!("Vec3 index out of range")}}
}
impl<T> IndexMut<i32> for Vec3<T>{
	fn index_mut(&mut self,i:Axis_t)->&mut T{match i{ XAxis=>&mut self.x,YAxis=>&mut self.y,ZAxis=>&mut self.z,_=>panic!("Vec3 index out of range")}}
}

pub fn v2i(x:i32,y:i32)->V2i{Vec2{x:x,y:y}}
pub fn v3i(x:i32,y:i32,z:i32)->V3i{Vec3{x:x,y:y,z:z}}
pub fn v3izero()->V3i{v3i(0,0,0)}
pub fn v3ione()->V3i{v3i(1,1,1)}
pub fn v3iadd_axis(a:V3i,axis:i32, value:i32)->V3i{let mut r=a; r[axis]+=value; r}
pub struct Neighbours<T>{pub prev:T,pub next:T}
type Neighbours2d<T>=Vec2<Neighbours<T>>;
type Neighbours3d<T>=Vec3<Neighbours<T>>;
type Neighbours4d<T>=Vec4<Neighbours<T>>;

macro_rules! v3i_operators{[$(($fname:ident=>$op:ident)),*]=>{
	$(pub fn $fname(a:V3i,b:V3i)->V3i{v3i(a.x.$op(b.x),a.y.$op(b.y),a.z.$op(b.z))})*
}}
macro_rules! v3i_permute_v2i{[$($pname:ident($u:ident,$v:ident)),*]=>{
	$(pub fn $pname(a:V3i)->V2i{v2i(a.$u,a.$v)})*
}}

pub trait MyMod :Add+Sub+Div+Mul+Sized{
	fn mymod(&self,b:Self)->Self;
}
impl MyMod for i32{
	fn mymod(&self,b:Self)->Self{ if *self>=0{*self%b}else{ b-((-*self) %b)} }
}
v3i_operators![(v3iadd=>add),(v3isub=>sub),(v3imul=>mul),(v3idiv=>div),(v3irem=>rem),(v3imin=>min),(v3imax=>max),(v3imymod=>mymod)];
v3i_permute_v2i![v3i_xy(x,y), v3i_yz(y,z), v3i_xz(x,z)];

pub fn v3i_hmul_usize(a:V3i)->usize{ a.x as usize*a.y as usize *a.z as usize}
pub fn v2i_hmul_usize(a:V2i)->usize{ a.x as usize*a.y as usize}

pub struct Array2d<T>{pub shape:V2i,pub data:Vec<T>}
pub struct Array3d<T>{pub shape:V3i,pub data:Vec<T>}
pub struct Array4d<T>{pub shape:V4i,pub data:Vec<T>}

impl<T:Clone> Array2d<T>{
	pub fn new()->Self{Array2d{shape:v2i(0,0),data:Vec::new()}}
	pub fn len(&self)->usize{ v2i_hmul_usize(self.shape) }
	pub fn linear_index(&self, pos:V2i)->usize{
		// now this *could* exceed 2gb.
		(pos.x as usize)+
		(self.shape.x as usize) *(pos.y as usize)
	}
	pub fn map_pos<B:Clone,F:Fn(V2i,&T)->B> (&self,f:F) -> Array2d<B>{
		// todo xyz iterator
		let mut r=Array2d::new();
		r.data.reserve(self.data.len());
		
		for y in 0..self.shape.y { for x in 0.. self.shape.x{
			let pos=v2i(x,y);
			r.data.push( f(pos,self.index(pos)) );
		}}
		r
	}
	pub fn from_fn<F:Fn(V2i)->T> (s:V2i, f:F)->Array2d<T>{
		let mut d=Array2d{shape:s,data:Vec::new()};
		d.data.reserve(s.x as usize * s.y as usize);
		for y in 0..s.y{ for x in 0..s.x{
				d.data.push(f(v2i(x,y)))
			}
		}
		d
	}
}

impl<T:Clone> Index<V2i> for Array2d<T>{
	type Output=T;
	fn index(&self, pos:V2i)->&T{
		let i=self.linear_index(pos);
		&self.data[i]
		
	}
}

impl<T:Clone> IndexMut<V2i> for Array2d<T>{
	fn index_mut(&mut self, pos:V2i)->&mut T{
		let i=self.linear_index(pos);
		&mut self.data[i]
	}
}

impl<T:Clone> Array3d<T>{	
	pub fn from_fn<F:Fn(V3i)->T> (s:V3i,f:F) -> Array3d<T> {
		let mut a=Array3d{shape:s, data:Vec::new()};
		a.data.reserve(v3i_hmul_usize(s));
		for z in 0..a.shape.z{ for y in 0..a.shape.y { for x in 0.. a.shape.x{
			a.data.push( f(v3i(x,y,z)) )
		}}}		
		a
	}

	/// production from a function with expansion,
	/// todo - make this more general e.g.'production in blocks'
	/// the motivation is to share state across the production of
	///  a block of adjacent cells
	pub fn from_fn_doubled<F:Fn(V3i)->(T,T)> (sz:V3i,axis:i32, f:F)->Array3d<T>{
		let mut scale=v3i(1,1,1); scale[axis]*=2;
		let mut d=Array3d{shape:v3imul(sz,scale),data:Vec::new()};
		d.data.reserve(sz.x as usize * sz.y as usize * sz.z as usize);
		for z in 0..sz.z {for y in 0..sz.y{ for x in 0..sz.x{
					let (v0,v1)=f(v3i(x,y,z));
					d.data.push(v0);
					d.data.push(v1);
				}
			}
		}
		d
	}

	pub fn fill_val(size:V3i,val:T)->Array3d<T>{let pval=&val;Self::from_fn(size,|_|pval.clone())}



	pub fn len(&self)->usize{ v3i_hmul_usize(self.shape) }
	pub fn linear_index(&self, pos:V3i)->usize{
		// now this *could* exceed 2gb.
		(pos.x as usize)+
		(self.shape.x as usize)*( 
			(pos.y as usize)+
			(pos.z as usize)*(self.shape.y as usize)
		)
	}
	pub fn size(&self)->V3i{self.shape}
	/// produce a new array3d by applying a function to every element
	pub fn map_xyz<B:Clone,F:Fn(V3i,&T)->B> (&self,f:F) -> Array3d<B>{
		Array3d::from_fn(self.shape,
			|pos:V3i|f(pos,self.index(pos))
		)
	}
	pub fn map_strided_region<B:Clone,F:Fn(V3i,&T)->B> 
		(&self,range:Range<V3i>,stride:V3i, f:F) -> Array3d<B>
	{
		Array3d::from_fn(v3idiv(v3isub(range.end,range.start),stride),
			|outpos:V3i|{
				let inpos=v3iadd(v3imul(outpos,stride),range.start);
				f(inpos,self.index(inpos))
			}
		)
	}

	/// internal iteration with inplace mutation
	pub fn for_each<F:Fn(V3i,&mut T)> (&mut self,f:F){
		for z in 0..self.shape.z{ for y in 0..self.shape.y { for x in 0.. self.shape.x{
			let pos=v3i(x,y,z);
			f(pos,self.index_mut(pos))
		}}}
	}

	// mappers along each pair of axes,
	// form primitive for reducers along the axes
	// or slice extraction
	pub fn map_xy<F,B>(&self, a_func:F)->Array2d<B>
		where F:Fn(&Self,i32,i32)->B, B:Clone
	{
		Array2d::from_fn(v3i_xy(self.shape),
			|pos:V2i|{a_func(self,pos.x,pos.y)}
		)		
	}
	pub fn map_xz<F,B>(&self, a_func:F)->Array2d<B>
		where F:Fn(&Self,i32,i32)->B, B:Clone
	{
		Array2d::from_fn(v3i_xz(self.shape),
			|pos:V2i|{a_func(self,pos.x,pos.y)}
		)		
	}
	pub fn map_yz<F,B>(&self, a_func:F)->Array2d<B>
		where F:Fn(&Self,i32,i32)->B, B:Clone
	{
		Array2d::from_fn(v3i_yz(self.shape),
			|pos:V2i|{a_func(self,pos.x,pos.y)}
		)		
	}

	pub fn reshape(&mut self, s:V3i){
		// todo: do allow truncations etc
		// todo: reshape to 2d, 3d
		assert!(v3i_hmul_usize(self.shape)==v3i_hmul_usize(s));
		self.shape=s;
	}

	// TODO ability to do the opposite e.g. map to vec's which become extra dim.
	// fold along axes collapse the array ?
	// e.g. run length encoding,packing, whatever.
	pub fn fold_z<B,F> (&self,init_val:B, f:F) -> Array2d<B>
		where B:Clone,F:Fn(V3i,B,&T)->B
	{
		self.map_xy(|s:&Self,x:i32,y:i32|{
			let mut acc=init_val.clone();
			for z in 0..s.shape.z{
				let pos=v3i(x,y,z);
				acc=f(pos,acc,self.index(pos))
			}
			acc
		})
	}
/*
		let mut out=Array2d::new();
		out.data.reserve(self.shape.y as usize *self.shape.z as usize);
		for y in 0..self.shape.y { for x in 0.. self.shape.x{
			let mut acc=input.clone();
			for z in 0..self.shape.z{
				let pos=Vec3(x,y,z);
				acc=f(pos,acc,self.index(pos));
			}
			out.data.push(acc);
		}}		
		out.shape=Vec2(self.shape.x,self.shape.y);
		out
	}
*/
	/// produce a 2d array by folding along the X axis
	pub fn fold_x<B:Clone,F:Fn(V3i,B,&T)->B> (&self,input:B, f:F) -> Array2d<B>{
		let mut out=Array2d::new();
		out.data.reserve(self.shape.y as usize *self.shape.z as usize);
		for z in 0..self.shape.z {
			for y in 0.. self.shape.y{
				let mut acc=input.clone();
				for x in 0..self.shape.x{
					let pos=v3i(x,y,z);
					acc=f(pos,acc, self.index(pos));
				}
				out.data.push(acc);
			}
		}		
		out.shape=v2i(self.shape.y,self.shape.z);
		out		
	}
	/// fold values along x,y,z in turn without intermediate storage
	pub fn fold_xyz<A,B,C,FOLDX,FOLDY,FOLDZ>(
		&self,
		(input_x,fx):(A,FOLDX), (input_y,fy):(B,FOLDY), (input_z,fz):(C,FOLDZ)
	)->A
	where
				A:Clone,B:Clone,C:Clone,
				FOLDZ:Fn(V3i,C,&T)->C,
				FOLDY:Fn(i32,i32,B,&C)->B,
				FOLDX:Fn(i32,A,&B)->A,
	{
		let mut ax=input_x.clone();
		for x in 0..self.shape.x{
			let mut ay=input_y.clone();
			for y in 0..self.shape.y{
				let mut az=input_z.clone();//x accumulator
				for z in 0..self.shape.z{
					let pos=v3i(x,y,z);
					az=fz(pos,az,self.index(pos));
				}
				ay=fy(x,y,ay,&az);
			}
			ax= fx(x, ax,&ay);
		}
		ax
	}
	/// fold values along z,y,x in turn without intermediate storage
	pub fn fold_zyx<A,B,C,FOLDX,FOLDY,FOLDZ>(
		&self,
		(input_x,fx):(A,FOLDX), (input_y,fy):(B,FOLDY), (input_z,fz):(C,FOLDZ)
	)->C
	where
				A:Clone,B:Clone,C:Clone,
				FOLDZ:Fn(i32,C,&B)->C,
				FOLDY:Fn(i32,i32,B,&A)->B,
				FOLDX:Fn(V3i,A,&T)->A,
	{
		let mut az=input_z.clone();
		for z in 0..self.shape.z{
			let mut ay=input_y.clone();
			for y in 0..self.shape.y{
				let mut ax=input_x.clone();//x accumulator
				for x in 0..self.shape.x{
					let pos=v3i(x,y,z);
					ax=fx(pos,ax,self.index(pos));
				}
				ay=fy(y,z,ay,&ax);
			}
			az= fz(z, az,&ay);
		}
		az
	}

	/// fold the whole array to produce a single value
	pub fn fold<B,F> (&self,input:B, f:F) -> B
	where F:Fn(V3i,B,&T)->B,B:Clone
	{
		let mut acc=input;
		for z in 0..self.shape.z { for y in 0.. self.shape.y{ for x in 0..self.shape.x{
			let pos=v3i(x,y,z);
			acc=f(pos, acc,self.index(pos));
		}}}
		acc
	}

	/// produce tiles by applying a function to every subtile
	/// output size is divided by tilesize
	/// must be exact multiple.
	pub fn fold_tiles<B,F>(&self,tilesize:V3i, input:B,f:&F)->Array3d<B>
		where F:Fn(V3i,B,&T)->B,B:Clone
	{
		self.map_strided(tilesize,
			|pos,_:&T|{self.fold_region(pos..v3iadd(pos,tilesize),input.clone(),f)})
	}

	/// subroutine for 'fold tiles', see context
	/// closure is borrowed for multiple invocation by caller
	pub fn fold_region<B,F>(&self,r:Range<V3i>, input:B,f:&F)->B
		where F:Fn(V3i,B,&T)->B, B:Clone
	{
		let mut acc=input.clone();
		for z in r.start.z..r.end.z{
			for y in r.start.y..r.end.y{
				for x in r.start.x..r.end.x{
					let pos=v3i(x,y,z);
					acc=f(pos,acc,self.index(pos));
				}
			}
		}
		acc
	}
	pub fn get_indexed(&self,pos:V3i)->(V3i,&T){(pos,self.index(pos))}
	pub fn region_all(&self)->Range<V3i>{v3i(0,0,0)..self.shape}
	pub fn map_region_strided<F,B>(&self,region:Range<V3i>,stride:V3i, f:F)->Array3d<B>
		where F:Fn(V3i,&T)->B, B:Clone{
		Array3d::from_fn(v3idiv(v3isub(region.end,region.start),stride),
			|outpos:V3i|{
				let inpos=v3iadd(region.start,v3imul(outpos,stride));
				f(inpos,self.index(inpos))  
			}
		)
	}
	pub fn map_strided<F,B>(&self,stride:V3i,f:F)->Array3d<B>
		where F:Fn(V3i,&T)->B, B:Clone{
		self.map_region_strided(self.region_all(),stride,f)
	}
	pub fn map_region<F,B>(&self,region:Range<V3i>,f:F)->Array3d<B>
		where F:Fn(V3i,&T)->B, B:Clone{
		self.map_region_strided(region, v3ione(), f)
	}
	/// _X_     form of convolution  
	/// XOX		passing each cell and it's
	/// _X_		immiediate neighbours on each axis
	pub fn convolute_neighbours<F,B>(&self,f:F)->Array3d<B>
		where F:Fn(&T,Vec3<Neighbours<&T>>)->B ,B:Clone
	{
		self.map_region(v3ione()..v3isub(self.shape,v3ione()),
			|pos:V3i,current_cell:&T|{
				f(	current_cell,
					self::Vec3{
						x:Neighbours{
							prev:self.index(v3i(pos.x-1,pos.y,pos.z)),
							next:self.index(v3i(pos.x+1,pos.y,pos.z))},
						y:Neighbours{
							prev:self.index(v3i(pos.x,pos.y-1,pos.z)),
							next:self.index(v3i(pos.x,pos.y+1,pos.z))},
						z:Neighbours{
							prev:self.index(v3i(pos.x,pos.y,pos.z-1)),
							next:self.index(v3i(pos.x,pos.y,pos.z+1))}})
		})
	}
	pub fn index_wrap(&self,pos:V3i)->&T{self.get_wrap(pos)}
	pub fn get_wrap(&self,pos:V3i)->&T{
		self.index( v3imymod(pos, self.shape) )
	}
	pub fn get_ofs_wrap(&self,pos:V3i,dx:i32,dy:i32,dz:i32)->&T{
		self.get_wrap(v3iadd(pos, v3i(dx,dy,dz)))
	}
	pub fn convolute_neighbours_wrap<F,B>(&self,f:F)->Array3d<B>
		where F:Fn(&T,Vec3<Neighbours<&T>>)->B,B:Clone 
	{
		// TODO - efficiently, i.e. share the offset addresses internally
		// and compute the edges explicitely
		// niave implementation calls mod for x/y/z individually and forms address
		self.map_region(v3izero()..self.shape,
			|pos:V3i,current_cell:&T|{
				f(	current_cell,
					Vec3{
						x:Neighbours{
							prev:self.get_wrap(v3i(pos.x-1,pos.y,pos.z)),
							next:self.get_wrap(v3i(pos.x+1,pos.y,pos.z))},
						y:Neighbours{
							prev:self.get_wrap(v3i(pos.x,pos.y-1,pos.z)),
							next:self.get_wrap(v3i(pos.x,pos.y+1,pos.z))},
						z:Neighbours{
							prev:self.get_wrap(v3i(pos.x,pos.y,pos.z-1)),
							next:self.get_wrap(v3i(pos.x,pos.y,pos.z+1))}})
		})
	}
	/// special case of convolution for 2x2 cells, e.g. for marching cubes
	pub fn convolute_2x2x2_wrap<F,B>(&self,f:F)->Array3d<B>
		where F:Fn(V3i,[[[&T;2];2];2])->B,B:Clone
	{
		self.map_region(v3izero()..self.shape,|pos,_|{
			f(pos,[
				[	[self.get_ofs_wrap(pos,0, 0, 0),self.get_ofs_wrap(pos, 1, 0, 0)],
					[self.get_ofs_wrap(pos,0, 1, 0),self.get_ofs_wrap(pos, 1, 1, 0)]
				],
				[	[self.get_ofs_wrap(pos,0, 0, 1),self.get_ofs_wrap(pos, 1, 0, 1)],
					[self.get_ofs_wrap(pos,0, 1, 1),self.get_ofs_wrap(pos, 1, 1, 1)]
				]
			])
		})
	}

	/// take 2x2x2 blocks, fold to produce new values
	pub fn fold_half<F,B>(&self,fold_fn:F)->Array3d<B>
		where F:Fn(V3i,[[[&T;2];2];2])->B,B:Clone
	{
		Array3d::from_fn( v3idiv(self.shape,v3i(2,2,2)), |dpos:V3i|{
			let spos=v3imul(dpos,v3i(2,2,2));
			fold_fn(dpos,
					[	[	[self.index(v3iadd(spos,v3i(0,0,0))),self.index(v3iadd(spos,v3i(1,0,0)))],
							[self.index(v3iadd(spos,v3i(0,1,0))),self.index(v3iadd(spos,v3i(1,1,0)))]
						],
						[	[self.index(v3iadd(spos,v3i(0,0,0))),self.index(v3iadd(spos,v3i(1,0,0)))],
							[self.index(v3iadd(spos,v3i(0,0,0))),self.index(v3iadd(spos,v3i(1,0,0)))]
						]
					]
			)
		})
	}
}


/*
fn avr<Diff,T>(a:&T,b:&T)->T where
	for<'u> Diff:Mul<f32,Output=Diff>+'u,
	for<'u,'v>&'u T:Sub<&'v T,Output=Diff>,
	for<'u,'v>&'u T:Add<&'v Diff,Output=T>
{
	a.add(&a.sub(b).mul(0.5f32))
}
*/
pub trait Lerp<F=f32> {
	fn lerp(&self,b:&Self,factor:&F)->Self;
}
// generic implementation which should work for propogation of
// dimensional intermediate types, fraction/fixed point types, etc

impl<T,Diff,Scaled,Factor> Lerp<Factor> for T where
	for<'u,'v> &'u T:Sub<&'v T,Output=Diff>,
	for<'u,'v> &'u Diff:Mul<&'v Factor,Output=Scaled>,
//	for<'u,'v> &'u DiffScaled:Add<&'v T,Output=T>,
	for<'u,'v,'w> &'u T:Add<&'v Scaled,Output=T>,   
	//<Diff as Mul<&'w Factor>>::Output
{
	fn lerp(&self,b:&Self,factor:&Factor)->Self{
		let diff=self.sub(b);
		let scaled=diff.mul(factor);
		self.add(&scaled)
	}
}	
pub fn avr<T:Lerp>(a:&T,b:&T)->T{ a.lerp(b,&0.5f32) }
// for types T with arithmetic,
impl<T:Clone+Lerp> Array3d<T>
{

	/// downsample, TODO downsample centred on alternate cells
	fn downsample_half(&self)->Array3d<T>{
		self.fold_half(|pos:V3i,cell:[[[&T;2];2];2]|->T{
			
			avr(
				&avr(
					&avr(&cell[0][0][0],&cell[1][0][0]),
					&avr(&cell[0][1][0],&cell[1][1][0])),
				&avr(
					&avr(&cell[0][0][1],&cell[1][0][1]),
					&avr(&cell[0][1][1],&cell[1][1][1])
				)
			)

		})
	}

	/// expand 2x with simple interpolation (TODO , decent filters)
	fn upsample_double_axis(&self,axis:i32)->Array3d<T>{
		Array3d::from_fn_doubled(self.shape,axis,|pos:V3i|->(T,T){
			let v0=self.index(pos); let v1=self.get_wrap(v3iadd_axis(pos,axis,1));
			let vm=avr(v0,v1);
			(v0.clone(),vm)
		})
	}
	/// expand 2x in all axes (TODO, in one step instead of x,y,z composed)
	fn upsample_double_xyz(&self)->Array3d<T>{
		// todo - should be possible in one step, without 3 seperate buffer traversals!
		self.upsample_double_axis(XAxis).upsample_double_axis(YAxis).upsample_double_axis(ZAxis)
	}

}

impl<T:Clone> Index<V3i> for Array3d<T>{
	type Output=T;
	fn index(&self, pos:V3i)->&T{
		let i=self.linear_index(pos);
		&self.data[i]
		
	}
}

impl<T:Clone> IndexMut<V3i> for Array3d<T>{
	fn index_mut(&mut self, pos:V3i)->&mut T{
		let i=self.linear_index(pos);
		&mut self.data[i]
 	}
}





