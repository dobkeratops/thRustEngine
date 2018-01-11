use super::*;

/// 4x4x4 cell Tile for simple compression:
/// Has option of a single repeated value, or a real value;
///
/// TODO - sparse content option
#[derive(Clone,Debug)]
pub enum Tile4<T> {
	Fill(T),Detail(Box<[[[T;4];4];4]>)
}

impl<T:Clone+PartialEq> Array3d<T>{
	fn iter_from_to(&self,start:V3i, end:V3i)->IterXYZ{
		IterXYZ::new(start,end, self.linear_index(start),self.linear_stride())
	}
	fn iter_at_sized(&self,start:V3i, size:V3i)->IterXYZ{
		IterXYZ::new(start,v3iadd(start,size), self.linear_index(start),self.linear_stride())
	}
	/// check if all values in a region are equal to the given value
	pub fn is_region_all_eq(&self,val:&T,start:V3i,end:V3i)->bool{
		for (pos,i) in self.iter_from_to(start,end){
			if self.at_linear_index(i)!=val{ return false}
		}
		return true;
	}
	/// check if all the values in a region are the same. if so return the common value; else return None.
	pub fn get_homogeneous_value(&self,start:V3i,end:V3i)->Option<T>{
// grab 4 consecutive elements in the x axis
		let ref_val=self.index(start);
		if self.is_region_all_eq(ref_val,start,end){ Some(ref_val.clone())}
		else{None}
	}

	/// grab 4 consecutive elements in the x axis
	fn copy4_x(&self,pos:V3i)->[T;4]{
		assert!(pos.x+4<=self.shape.x);
		let i=self.linear_index(pos);
		// brute-force -construct the array immiediately-
		//avoids overwriting temporary
		[self.at_linear_index(i).clone(),self.at_linear_index(i+1).clone(),
		self.at_linear_index(i+2).clone(),self.at_linear_index(i+3).clone()]
	}
	/// assemble 4 X's into 4x4
	fn copy4x4_xy(&self,pos:V3i)->[[T;4];4]{
		assert!(pos.y+4<=self.shape.y);
		[self.copy4_x(pos),self.copy4_x(v3iadd(pos,v3i(0,1,0))),
		self.copy4_x(v3iadd(pos,v3i(0,2,0))),self.copy4_x(v3iadd(pos,v3i(0,3,0)))]
	}

	/// assemble 4 XYs into 4x4x4
	fn copy4x4x4(&self,pos:V3i)->[[[T;4];4];4]{
		//setup and overwrite :(
		// to do cleanly+efficient.. need array4 constructor and repeated calls.
		assert!(pos.z+4<=self.shape.z);
		[self.copy4x4_xy(pos),self.copy4x4_xy(v3iadd(pos,v3i(0,0,1))),
		self.copy4x4_xy(v3iadd(pos,v3i(0,0,2))),self.copy4x4_xy(v3iadd(pos,v3i(0,0,3)))]
	}

	/// compressed tiling
	/// macro cells include either single value or detail
	/// 2x2x2 and 4x4x4 manually written.. awaiting <T,N>
	fn tile4(&self)->Array3d<Tile4<T>>{
		Array3d::from_fn(
			v3idiv_s(self.index_size(),4),
			|pos|{
				let srcpos=v3imul_s(pos,4);
				match self.get_homogeneous_value(srcpos,v3iadd_s(srcpos,4)){
					Some(cell)=>Tile4::Fill(cell),
					None=>Tile4::Detail(Box::new(self.copy4x4x4(srcpos)))
				}
			}
		)
	}
}
/// 3d array composed of 4x4x4 tiles, with simple compression;
///
/// tiles contain either a single fill value or 4x4x4 individual values
///
/// rationale:
/// 4x4x4 is a sweetspot; between pointer overhead and precision. 4x4x4 has 64cells, similar to the common 8x8 tile size in 2d.

pub struct Array3dTiled4<T>(pub Array3d<Tile4<T>>);
impl<'a,T:Clone+PartialEq+Default> From<&'a Array3d<T>> for Array3dTiled4<T>{
	fn from(s:&Array3d<T>)->Self{ Array3dTiled4(s.tile4()) }
}
/// expand out raw array from, TODO - efficiently,
/// there should be a tiled constructor for Array3d<T> that works inplace
impl<'a,T:Clone> From<&'a Array3dTiled4<T>> for Array3d<T> {
	fn from(src:&Array3dTiled4<T>)->Array3d<T>{
		Array3d::from_fn(v3imul_s(src.0.index_size(),4),|pos|{src[pos].clone()})
	}
}
/// read access to tile4 array
impl<T> Index<V3i> for Array3dTiled4<T>{	
	type Output=T;
	fn index(&self,pos:V3i)->&T{
		let (tpos,sub)=v3itilepos(pos,2);
		match self.0.index(tpos){
			&Tile4::Fill(ref x)=>&x,
			&Tile4::Detail(ref tiledata)=>{
				&tiledata[sub.z as usize][sub.y as usize][sub.x as usize]
			}
		}
	}
}

/// helper function - clone a value 4 times to produce a rust array; Saves restricting Tile helper code to Copy types
pub fn clone4<T:Clone>(t:T)->[T;4]{
	[t.clone(),t.clone(),t.clone(),t]
}
pub fn clone2<T:Clone>(t:T)->[T;2]{
	[t.clone(),t]
}
impl<T:Clone> Array3dTiled4<T>{
	fn from_val(size:V3i,val:&T)->Self{
		Array3dTiled4(Array3d::from_val(v3idiv_s(size,4),&Tile4::Fill(val.clone())))
	}
}
/// Write access to tile4 array,converting any written 'compressed tiles' to uncompressed; TODO detect for writes that leave clear for compression.
impl<T:Clone> IndexMut<V3i> for Array3dTiled4<T>{	
	fn index_mut(&mut self,pos:V3i)->&mut T{
		let (tpos,sub)=v3itilepos(pos,2);
		let a=self.0.index_mut(tpos);
		// convert the tile to mutable contents
		// unfortunately we can't erase yet
		// 2 steps to appease borrowck
		let newval=if let &mut Tile4::Fill(ref val)=a{
			Some(val.clone())
		} else {None};
		if let Some(v)=newval{
			*a=Tile4::Detail(Box::new(clone4(clone4(clone4(v)))));
		};
		// by now 'a' must be 'Right',i.e. a defined tile
		match *a{
			Tile4::Fill(_)=>panic!("tile should be defined now"),
			Tile4::Detail(ref mut p)=>&mut (*p)[sub.z as usize][sub.y as usize][sub.x as usize]
		}
		// after writes you must cleanup	
	}
}

#[test]
fn try_tiles(){
	let mut tiles=Array3dTiled4::<i32>::from_val(v3i(8,8,8),&0);
	// start with empty tiles, write some values, see what we have..
	let pos1=v3i(1,3,1);
	let pos2=v3i(5,3,7);
	let pos3=v3i(2,3,3);
	tiles[pos1]=1;
	tiles[pos2]=2;
	assert_eq!(tiles[pos1],1);
	assert_eq!(tiles[pos2],2);
	assert_eq!(tiles[pos3],0);
}


