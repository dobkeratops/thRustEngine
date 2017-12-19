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
	fn get_wrap(&self,x:I,y:I)->&T;
	fn set_wrap(&mut self,x:I,y:I,v:T);
}
impl<T> WrappedArray2d<T,i32> for Vec<Vec<T>>{
	fn get_wrap(&self,i:i32,j:i32)->&T{
		self.index_wrap(j).index_wrap(i)
	}
	fn set_wrap(&mut self,i:i32,j:i32,v:T){
		*(self.index_wrap_mut(j).index_wrap_mut(i))=v;
	}
}


/// generate by filling array inplace. repeating on square.
pub fn generate(log_2_size:i32,init_amp:f32,dimension:f32,power:f32,iseed:i32)->Vec<Vec<f32>>{
	type T=f32;
	let mut seed=iseed;
	// todo - version using expansion/smoothing/noise. 
	// that would be more cachefriendly, and yields other useful parts.
	let size=(1<<log_2_size) as i32;
	let mut ht:Vec<Vec<f32>> = Vec::new();
	ht.reserve(size as usize);


	for j in 0..size{
		let mut row:Vec<f32>= vec![0.0;size as usize];
		ht.push(row);
	}
	ht[0][0]=0.0f32;
	let mut amp=init_amp;
	// classic diamond-square algorithm
	let mut step=size as i32;
	let inv4=T::half()*T::half();
	let ampscale= (dimension/(one::<T>()+one::<T>())).sqrt();
	let mut visited:Vec<isize> =vec![0;(size*size) as usize];
	let mut pass=1;
	while step >=2 {
		let hstep=step/2;
		for x in (0..size).step(step as usize){
			for y in (0..size).step(step as usize){
				let x1=x+step; let y1=y+step;
				let mp=(*ht.get_wrap(x,y)+*ht.get_wrap(x1,y)+*ht.get_wrap(x,y1)+*ht.get_wrap(x1,y1))*inv4;
				let mut disp:f32;
				let sd=frands(seed);
				seed=sd.0;
				ht.set_wrap(x+hstep,y+hstep,mp+amp*sd.1);
				visited[(x+hstep+(y+hstep)*size) as usize]=pass;
			}
		}
		//  x   +   x
		//     
		//  +   x   .
		//  
		//  x   .   x
		amp*=ampscale;
		for xx in (0..size).step(step as usize){
			for yy in (0..size).step(step as usize){
				let mut fill_pt=|x,y|{
					let mp=(*ht.get_wrap(x+hstep,y)+*ht.get_wrap(x-hstep,y)+*ht.get_wrap(x,y+hstep)+*ht.get_wrap(x,y-hstep))*inv4;
					let sd=frands(seed);
					ht.set_wrap(x,y,mp+amp*sd.1);
					seed=sd.0;
					visited[(x+y*size) as usize]=pass+1;
				};
				fill_pt(xx+hstep,yy);
				fill_pt(xx,yy+hstep);
			}
		}
		//break;
		pass+=2;
		amp*=ampscale;
		step=hstep;
	}
	let ax=6.248f32/(size as f32);
	let ay=6.248f32/(size as f32);
	//for y in 0..size{for x in 0..size{ print!("{}",visited[(x+y*size) as usize]);};print!("\n");}
	//panic!();
	// fill with sin x*sin y for test poly gen
	//for y in (0..size){for x in 0..size{ ht[y as usize][x as usize]=sin(x as f32*ax)*sin(y as f32*ay)*init_amp}}
	ht
}

