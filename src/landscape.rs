use super::*;

pub struct LandscapeDef {
	dimension:f32,
	power:f32,
}

pub struct Landscape {
	pts:Option<[Vec3;4]>	// landscape is a tree of quads.
}

// TODO: as a primitive in the Editor

pub trait IndexWrap{
	type Output : Sized;
	fn index_wrap(&self, i:i32)->&Self::Output;
	fn index_wrap_mut(&mut self, i:i32)->&mut Self::Output;
}
use std::ops::{Index,IndexMut};
/*
TODO - needs generic length?
impl<V:IndexMut<usize>> IndexWrap for V where <V as Index<usize>>::Output : Sized
{
	type Output=<V as Index<usize>>::Output;
	fn index_wrap(&self, i:i32)->&Self::Output{
		return self.index((i % self.len()) as usize);
	}
	fn index_wrap_mut(&mut self, i:i32)->&mut Self::Output{
		return self.index_mut((i % self.len()) as usize);
	}	
}
*/
impl<T:Sized> IndexWrap for Vec<T>{
	type Output=T;
	fn index_wrap(&self, i:i32)->&Self::Output{
		return self.index(((i as usize)% self.len()) as usize);
	}
	fn index_wrap_mut(&mut self, i:i32)->&mut Self::Output{
		let l=self.len();
		return self.index_mut(((i as usize) % l) as usize);
	}	
}
trait WrappedArray2d<T,I>{
	fn get(&self,x:I,y:I)->&T;
	fn set(&mut self,x:I,y:I,v:T);
}
impl<T> WrappedArray2d<T,i32> for Vec<Vec<T>>{
	fn get(&self,i:i32,j:i32)->&T{
		self.index_wrap(j).index_wrap(i)
	}
	fn set(&mut self,i:i32,j:i32,v:T){
		*(self.index_wrap_mut(j).index_wrap_mut(i))=v;
	}
}


/// generate by filling array inplace. repeating on square.
fn generate(log_2_size:i32,init_amp:f32,dimension:f32,power:f32,iseed:i32)->Vec<Vec<f32>>{
	type T=f32;
	let mut seed=iseed;
	// todo - version using expansion/smoothing/noise. 
	// that would be more cachefriendly, and yields other useful parts.
	let size=(1<<log_2_size) as i32;
	let mut ht:Vec<Vec<f32>> = Vec::new();


	for j in 0..size{
		let mut x:Vec<f32>= vec![zero();size as usize];
		let row:&mut Vec<f32>= &mut ht[j as usize];
		*row=x;
	}
	let mut amp=init_amp;
	// classic diamond-square algorithm
	let mut step=size as i32;
	let inv4=T::half()*T::half();
	let sqrt2= (one::<T>()+one::<T>()).sqrt();
	while step >2 {
		let hstep=step/2;
		for x in (0..size).step(step as usize){
			for y in (0..size).step(step as usize){
				let x1=x+step; let y1=y+step;
				let mp=(*ht.get(x,y)+*ht.get(x1,y)+*ht.get(x,y1)+*ht.get(x1,y1))*inv4;
				let mut disp:f32;
				let sd=frands(seed);
				disp=sd.1; seed=sd.0;
				ht.set(x+hstep,y+hstep,mp+amp*(disp as f32));
			}
		}
		//  x   +   x
		//     
		//  +   x   .
		//  
		//  x   .   x
		amp*=sqrt2;
		for x in (0..size+1).step(step as usize){
			for y in (0..size+1).step(step as usize){
				let mut fill_pt=|x1,y1|{
					let mp=(*ht.get(x+hstep,y)+*ht.get(x-hstep,y)+*ht.get(x,y+hstep)+*ht.get(x,y-hstep))*inv4;
					let sd=frands(seed);
					ht.set(x,y,mp+amp*sd.1);
					seed=sd.0;
				};
				fill_pt(x+hstep,y);
				fill_pt(x,y+hstep);
			}
		}
		amp*=sqrt2;
		step=hstep;
	}
	ht
}

