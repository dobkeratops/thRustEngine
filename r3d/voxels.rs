use super::*;
use std::ops::Range;
type Index3d=[i32;3];

/// 3d array interface
/// todo: wrapround/non wrapround versions
pub trait Array3d<T>{
	fn index_range(&self)->Range<Index3d>;
	fn fill_with(&mut self, index:Index3d, fill_with:&Fn(Index3d)->T);
	fn get(&self, index:Index3d)->&T;
	fn get_wrap(&self,index:Index3d)->&T{
		let ir=self.index_range(); let isz=[ ir.end[0]-ir.start[0],ir.end[1]-ir.start[1],ir.end[2]-ir.start[2]];
		let ijk=[mymod((index[0]-ir.start[0]),isz[0])+ir.start[0],
		mymod((index[1]-ir.start[1]),isz[1])+ir.start[1],
		mymod((index[2]-ir.start[2]),isz[2])+ir.start[2]];
		self.get(ijk)
	}
	fn set(&mut self,index:Index3d,newval:T){*self.get_mut(index)=newval;}
	fn get_mut(&mut self, index:Index3d)->&mut T;
	fn replace(&mut self, index:Index3d, newval:T)->T{
		mem::replace( self.get_mut(index), newval)
	}
	fn foreach<F:FnMut(Index3d,&mut T)>(&mut self,mut f:F) {			// 
		let r=self.index_range();
		// todo iterator yielding (x,y,z)
		for z in r.start[2]..r.end[2]{
			for y in r.start[1]..r.end[1]{
				for x in r.start[0]..r.end[0]{
					let xyz=[x,y,z];
					f(xyz,self.get_mut(xyz))
				}
			}
		}		
	}
	fn visit_mut(&mut self, f:&mut FnMut(Index3d,&mut T));	// 'visit'.. takes a function object.
	fn visit(&self, f:&mut FnMut(Index3d,&T));
	//	   visit comparing an item with neighbours [[left,right],[fwd,back],[down,up]]
	// special case of minimal convolution.
	// todo - general convolution
	// todo parallel
	fn get_neighbours_wrap(&self,index:Index3d)->(&T,[[&T;2];3]){
		let x=index[0];let y=index[1]; let z=index[2];
		(self.get_wrap([x,y,z]),
			[	[self.get_wrap([x-1,y,z]),self.get_wrap([x+1,y,z])],
				[self.get_wrap([x,y-1,z]),self.get_wrap([x,y+1,z])],
				[self.get_wrap([x,y,z-1]),self.get_wrap([x,y,z+1])]])
	}
	fn visit_with_neighbours_wrap(&self, f:&mut FnMut(Index3d,&T,[[&T;2];3])){ 
		// todo - can the be written in terms of the lambda above, 
		//or will the borrow-checker freak out
		let r=self.index_range();
		for z in r.start[2]..r.end[2]{
			for y in r.start[1]..r.end[1]{
				for x in r.start[0]..r.end[0]{
					// todo: efficient generation of the wrap-round and neighbour addresses.
					let (c,n)=self.get_neighbours_wrap([x,y,z]);
					f([x,y,z], c, n);					
				}
			}
		}		
	}

	// TODO - array -> array map... does it need a concrete output type.
	// do it for just 'FlatArray3d' until we have impl-trait?

}

/// simple linear 3d array; 
///TODO - tiled ,etc; Access as a crop in another array, with min index
#[derive(Clone,Debug)]
pub struct FlatArray3d<T>{
	pub size:Index3d,	// we will not have >4billion^3 even on 64bit.
	pub data:Vec<T>,
}

impl<T:Debug+Clone+Sized> FlatArray3d<T>{
	pub fn new()->Self{
		FlatArray3d::<T>{size:[0,0,0],data:Vec::new()}
	}
	pub fn from_fn(size:Index3d, f:&Fn(Index3d)->T)->Self{
		let mut ret=FlatArray3d::new();
		ret.fill_with(size,f);
		ret
	}
	pub fn fill_value(size:Index3d, val:T)->Self{
		FlatArray3d{size:size,data:vec![val;(size[0]*size[1]*size[2])as usize]}		
	}
	pub fn linear_size(size:Index3d)->usize{ size[0]as usize*size[1] as usize* size[2] as usize }
	pub fn linear_index(&self,index:Index3d)->usize{
		assert!(index[0]>=0 && index[1]>=0 && index[2]>=0);
		(index[0] as usize +self.size[0] as usize*(index[1] as usize+self.size[1] as usize*(index[2] as usize))) as usize
	}
	pub fn map<Y>(&self, f:&Fn(Index3d,&T)->Y)-> FlatArray3d<Y>{
		let mut ret= FlatArray3d::<Y>{size:self.size, data:Vec::new()};
		ret.data.reserve(self.data.len());
		let mut n=0;
		for k in 0..self.size[2]{
			for j in 0..self.size[1]{
				for i in 0..self.size[0]{
					// apply the mapper function and write result in new array
					ret.data.push( f([i,j,k],&self.data[n]) );
					n+=1;					
				}
			}
		}
		ret
	}
	// TODO - figure out how to customize 'Debug'
	pub fn dump(&self){
		println!("[");
		for k in 0..self.size[2]{	
			println!("\t[");
			for j in 0..self.size[1]{
				print!("\t\t[\t");
				for i in 0..self.size[0]{
					print!("{:?},",self.get([i,j,k]));
				}
				print!("\n");
				println!("\t\t],");
			}
			println!("\t],");
		}
		println!("],");
							
	}
}

impl<T:Debug+Clone+Sized> Array3d<T>for FlatArray3d<T>{
	// TODO array iterator.
	// resize not supported yet!
	fn index_range(&self)->Range<Index3d>{Range{start:[0,0,0],end:self.size}}
	fn fill_with(&mut self, size:Index3d, fill_with:&Fn(Index3d)->T){
		self.size=size;
		let lnsz=Self::linear_size(size);
		self.data.truncate(0);
		self.data.reserve(lnsz);
		for k in 0..size[2]{
			for j in 0..size[1]{
				for i in 0..size[0]{
					self.data.push( fill_with([i,j,k]) );
				}
			}
		}
	}
	
	fn get(&self, index:Index3d)->&T {
		&self.data[self.linear_index(index)]
	}
	fn get_mut(&mut self,index:Index3d)->&mut T{let li={self.linear_index(index)}; &mut self.data[li] }
	fn visit_mut(&mut self, f:&mut FnMut(Index3d,&mut T)){
		let mut n=0;
		for k in 0..self.size[2]{
			for j in 0..self.size[1]{
				for i in 0..self.size[0]{
					f([i,j,k],&mut self.data[n]);
					n+=1;
				}
			}
		}
	}
	fn visit(&self, f:&mut FnMut(Index3d,&T)){
		let mut n=0;
		for k in 0..self.size[2]{
			for j in 0..self.size[1]{
				for i in 0..self.size[0]{
					f([i,j,k],&self.data[n]);
					n+=1;
				}
			}
		}
	}
}
// -3  -2  -1  0  1  2
//  0   1   2  0  1  2
pub fn test_array3d(){
	println!("{} {} {} {}",mymod(-3,3), mymod(-2,3),mymod(-1,3),mymod(0,3));
	let aa=FlatArray3d::<u8>::new();
	aa.dump();
	let bb=FlatArray3d::<u8>::from_fn([2,2,2],&|pos|((pos[0]+pos[1]*10+pos[2]*20)as _));
	bb.dump();
	for x in -1..3{for y in -1..3 {for z in -1..3{ print!("{:}\t",bb.get_wrap([x,y,z]));}print!("\n")}}
}

