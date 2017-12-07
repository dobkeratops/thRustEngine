pub use self::mesh::*;
mod mesh;

pub type Height=f32;

pub struct Landscape<Cell> {
	cellSize:VScalar,
	size:(int,int),
	tile:~[Cell],
	height:~[Height]
}

trait Grid<V>{
	pub fn numUV(&self)->(int,int);
	pub fn vertex(&self,ij:(int,int))->V;
}

impl<V:VecOps,Cell> Grid<V> for Landscape<Cell> {
	pub fn numUV(&self)->(int,int) {(self.numU,self.numV)}
	pub fn vertex(&self,(i,j):(int,int))->V{
		let x=self.cellSize.to_f32()*i.to_f32();
		let y=self.cellSize.to_f32()*j.to_f32();
		let z=self.height[i+j*self.numU];
		VecOps::fromXYZ(x,y,z)
	}
}






