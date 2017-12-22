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
impl<T:Sized> IndexWrap for Array<T>{
	type Output=T;
	fn index_wrap(&self, i:i32)->&Self::Output{
		return self.index(mymod(i,self.num_elems()));
	}
	fn index_wrap_mut(&mut self, i:i32)->&mut Self::Output{
		let l=self.num_elems();
		return self.index_mut(mymod(i,l));
	}	
}

trait WrappedArray2d<T,I>{
	fn get_wrap(&self,x:I,y:I)->&T;
	fn set_wrap(&mut self,x:I,y:I,v:T);
}
impl<T> WrappedArray2d<T,i32> for Array<Array<T>>{
	fn get_wrap(&self,i:i32,j:i32)->&T{
		self.index_wrap(j).index_wrap(i)
	}
	fn set_wrap(&mut self,i:i32,j:i32,v:T){
		*(self.index_wrap_mut(j).index_wrap_mut(i))=v;
	}
}
trait WrappedArray3d<T,I>{
	fn get_wrap3(&self,x:I,y:I,z:I)->&T;
	fn set_wrap3(&mut self,x:I,y:I,z:I,v:T);
}
impl<T> WrappedArray3d<T,i32> for Array<Array<Array<T>>>{
	fn get_wrap3(&self,i:i32,j:i32,k:i32)->&T{
		self.index_wrap(k).index_wrap(j).index_wrap(i)
	}
	fn set_wrap3(&mut self,i:i32,j:i32,k:i32,v:T){
		*(self.index_wrap_mut(k).index_wrap_mut(j).index_wrap_mut(i))=v;
	}
}
fn dump_row_sizes(ht:&Array<Array<f32>>){
	for i in 0..ht[0i32].num_elems(){
		println!("size[row {:?}]={:?}",i,ht[i]);
	}
}
/// generate by filling array inplace. repeating on square.
pub fn generate2d(log_2_size:i32,init_amp:f32,dimension:f32,power:f32,iseed:i32)->Array<Array<f32>>{
	type T=f32;
	let mut seed=iseed;
	// todo - version using expansion/smoothing/noise. 
	// that would be more cachefriendly, and yields other useful parts.
	let size=(1<<log_2_size) as i32;
	let mut ht:Array<Array<f32>> = Array::new();
	ht.reserve(size);


	for j in 0..size{
		let mut row=Array::from_val_n(0.0,size);
		ht.push(row);
	}
		
	ht[0i32][0i32]=0.0f32;
	//let mut row:&Array<f32>=&mut ht[0];
	//row[0]=0.0f32;
	let mut amp=init_amp;
	// classic diamond-square algorithm
	let mut step=size as i32;
	let inv4=T::half()*T::half();
	let ampscale= (dimension/(one::<T>()+one::<T>())).sqrt();
	let mut pass=1;
	while step >=2 {
		trace!();
		let hstep=step/2;
		for x in (0..size).step(step as usize){
			for y in (0..size).step(step as usize){
				let x1=x+step; let y1=y+step;
				let mp=(*ht.get_wrap(x,y)+*ht.get_wrap(x1,y)+*ht.get_wrap(x,y1)+*ht.get_wrap(x1,y1))*inv4;
				ht.set_wrap(x+hstep,y+hstep,mp+amp*frandsm(&mut seed));
			}
		}
		//  x   +   x
		//     
		//  +   x   .
		//  
		//  x   .   x
		amp*=ampscale;
		trace!();
		for xx in (0..size).step(step as usize){
			for yy in (0..size).step(step as usize){
				let mut fill_pt=|x,y|{
					let mp=(*ht.get_wrap(x+hstep,y)+*ht.get_wrap(x-hstep,y)+*ht.get_wrap(x,y+hstep)+*ht.get_wrap(x,y-hstep))*inv4;
					ht.set_wrap(x,y,mp+amp*frandsm(&mut seed));
				};
				fill_pt(xx+hstep,yy);
				fill_pt(xx,yy+hstep);
			}
		}
		trace!();
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

// todo - a single allocation eg ([usize;3],Vec<T>)
pub type Array3d<T>=Array<Array<Array<T>>>;

pub fn array3d_foreach<T>( a:&mut Array3d<T>, f:&Fn([i32;3], &mut T)){
	let nz:i32=a.num_elems();
	let ny:i32=a[0i32].num_elems();
	let nx:i32=a[0i32][0i32].num_elems();
	for z in 0..nz{ for y in 0..ny{for x in 0..nz{
		f([x,y,z],&mut a[z][y][x])
	}}}
}

/// 3d volume noise texture.
pub fn generate3d(log_2_size:i32,first_disp_scale:i32,init_amp:f32,dimension:f32,power:f32,iseed:i32)->Array<Array<Array<f32>>>{
	type T=f32;
	let mut seed=iseed;
	// todo - version using expansion/smoothing/noise. 
	// that would be more cachefriendly, and yields other useful parts.
	let size=(1<<log_2_size) as i32;
	let mut ht:Array<Array<Array<f32>>> = Array::new();
	ht.reserve(size);

	for k in 0..size{
		let mut jslice:Array<Array<f32>>= Array::new();
		for j in 0..size{
			let mut islice= Array::<f32>::from_val_n(0.0,size);
			jslice.push(islice);
		}
		ht.push(jslice);
	}
	ht[0i32][0i32][0i32]=0.0f32;

	let mut amp=init_amp;
	// classic diamond-square algorithm ext to 3d
	let mut step=size as i32;
	let inv4=1.0/4.0;//T::half()*T::half();
	let inv8=1.0/8.0; //T::half()*T::half()*T::half();
	let inv6=1.0/6.0;//T::half()/(T::one()+T::one()+T::one()); //1/6 = 0.5/3
	let ampscale= (dimension/(one::<T>()+one::<T>())).powf(0.333f32);
	let mut pass=1;
	let mut num_writes=1;
	let mut total_disp=0.0f32;
	while step >=2 {
		let hstep=step/2;
		if step<=(size/2){
		// to subdivide 2x2x2 - must write 7 values?
		for z in (0..size).step(step as usize){
			for y in (0..size).step(step as usize){
				for x in (0..size).step(step as usize){
					let x1=x+step; let y1=y+step; let z1=z+step;
					let mp=(*ht.get_wrap3(x,y,z)+*ht.get_wrap3(x1,y,z)
							+*ht.get_wrap3(x,y1,z)+*ht.get_wrap3(x1,y1,z)
							+*ht.get_wrap3(x,y,z1)+*ht.get_wrap3(x1,y,z1)
							+*ht.get_wrap3(x,y1,z1)+*ht.get_wrap3(x1,y1,z1)
							)*inv8;
					let disp=frandsm(&mut seed); total_disp+=disp;
					ht.set_wrap3(x+hstep,y+hstep,z+hstep,mp+amp*disp);
					num_writes+=1;
				}
			}
		}
		amp*=ampscale;
		for z in (0..size).step(step as usize){
			for y in (0..size).step(step as usize){
				for x in (0..size).step(step as usize){
					let x1=x+step; let y1=y+step; let z1=z+step;
					//xysquare/updown
					let mp=(*ht.get_wrap3(x,y,z)+*ht.get_wrap3(x1,y,z)
							+*ht.get_wrap3(x,y1,z)+*ht.get_wrap3(x1,y1,z)
							+*ht.get_wrap3(x+hstep,y+hstep,z+hstep)+*ht.get_wrap3(x+hstep,y+hstep,z-hstep)
							
							)*inv6;
					let disp=frandsm(&mut seed); total_disp+=disp;
					ht.set_wrap3(x+hstep,y+hstep,z,mp+amp*disp);
					num_writes+=1;

					//xzsquare / +/-y
					let mp=(*ht.get_wrap3(x,y,z)+*ht.get_wrap3(x1,y,z)
							+*ht.get_wrap3(x,y,z1)+*ht.get_wrap3(x1,y,z1)
							+*ht.get_wrap3(x+hstep,y+hstep,z+hstep)+*ht.get_wrap3(x+hstep,y-hstep,z+hstep)
							
							)*inv6;
					frandsm(&mut seed); total_disp+=disp;
					ht.set_wrap3(x+hstep,y,z+hstep,mp+amp*disp);
					num_writes+=1;

					//yzsquare / +/- x
					let mp=(*ht.get_wrap3(x,y,z)+*ht.get_wrap3(x,y1,z)
							+*ht.get_wrap3(x,y,z1)+*ht.get_wrap3(x,y1,z1)
							+*ht.get_wrap3(x-hstep,y+hstep,z+hstep)+*ht.get_wrap3(x+hstep,y+hstep,z+hstep)
							
							)*inv6;
					let disp=frandsm(&mut seed); total_disp+=disp;
					ht.set_wrap3(x,y+hstep,z+hstep,mp+amp*disp);
					num_writes+=1;
				}
			}
		}

		//  x   +   x
		//     
		//  +   x   .
		//  
		//  x   .   x
		amp*=ampscale;
		for zz in (0..size).step(step as usize){
			for yy in (0..size).step(step as usize){
				for xx in (0..size).step(step as usize){

					let mut fill_pt=|x,y,z|{
						let mp=(*ht.get_wrap3(x+hstep,y,z)+
								*ht.get_wrap3(x-hstep,y,z)+
								*ht.get_wrap3(x,y+hstep,z)+
								*ht.get_wrap3(x,y-hstep,z)+
								*ht.get_wrap3(x,y,z-hstep)+
								*ht.get_wrap3(x,y,z+hstep))*inv6;
						let disp=frandsm(&mut seed); total_disp+=disp;
						ht.set_wrap3(x,y,z,mp+amp*disp);
					};
					fill_pt(xx+hstep,yy,zz);
					fill_pt(xx,yy+hstep,zz);
					fill_pt(xx,yy,zz+hstep);
					num_writes+=3;
				}
			}
		}
		//break;
		pass+=2;
		amp*=ampscale;
		}else{amp*=0.5f32};
		step=hstep;
	}
	let ax=6.248f32/(size as f32);
	let ay=6.248f32/(size as f32);
	// dump slices
	#[cfg(debug_voxels)]
	{
		for z in (0..size).step(4){
			for x in 0..size{
				for y in 0..size {
					print!("{}",if *ht.get_wrap3(x,y,size/2)>0.0{"O"}else{"."});
				}
				print!("\n\n");
			}
		}
		dump!(ampscale,ampscale*ampscale*ampscale,total_disp/((num_writes-1) as f32));
		let mut acc=0.0f32;
		for x in 0..40{
			let rnd=frandsm(&mut seed);acc+=rnd;print!("{} ",rnd);
		}
		println!("ok avrrnd={}",acc*(1.0/40.0));
	}
	//assert!(num_writes==size*size*size,"num writes {} size^3={} size={}",num_writes,size*size*size,size);
	ht
}

