//pub use r3d::mesh::*;
//use std::vec::*;
/*
pub struct HeightField<T,Cell> {
	cellSize:T,
	numuv:(Index,Index),
	tile:~[Cell],
	height:~[T]
}

type FuncGenHeight<'self,T> =&'self fn(ij:(Index,Index))->T;
fn from_fn_2d<T>((numi,numj):(Index,Index), f:FuncGenHeight<T>)->~[T]
{
	from_fn(numi*numj, |i|{f(i.div_mod_floor(&numi))})
}

impl HeightField<int> {
	fn from_fn(s:(Index,Index), cellSize:VScalar,f:FuncGenHeight)->HeightField<int>
	{
		let v:HeightField<int> =
		HeightField{
			cellSize:cellSize,
			numuv:s,
			tile:~[],
			height:from_fn_2d(s,f)
		};
		v
	}
}


//interface for a mesh with grid topology, f(u,v)->(x,y,z)

pub trait IVertexGrid<V>{
	fn size(&self)->(Index,Index);
	fn vertex(&self,ij:(Index,Index))->V;
}

impl<V:LandscapeVertex,Cell> IVertexGrid<V> for HeightField<Cell> {
	fn size(&self)->(Index,Index) {self.numuv}
	fn vertex(&self,(i,j):(Index,Index))->V{
		let (sizei,_)=self.numuv;
		let x=self.cellSize.to_f32()*i.to_f32();
		let y=self.cellSize.to_f32()*j.to_f32();
		let z=self.height[i+j*sizei];
		LandscapeVertex::fromXYZAttr(x,y,z,0)
	}
}

pub trait LandscapeVertex {
	fn fromXYZAttr<A>(x:VScalar,y:VScalar,z:VScalar,attr:A)->Self;
}

// TODO - we wanted to implemnt TriangleMesh for VertexGrid ..
// but this generated conflicing implementation. with ITriangleMesh for TriangleMesh... we dont know why; this would have been handled with 
// selection of specific cases in c++


fn numTrianglesOfGrid<V,T:IVertexGrid<V>>(a:&T)->Index {
	let (numi,numj) = a.size();
	(numi-1)*(numj-1)*2
}


fn triVertexOfGrid<V, T:IVertexGrid<V>>(a:&T,ti:Index,tvi:Index)->V {
	let (numi,_)=a.size();
	let (i,j)=(ti>>1).div_mod_floor(&(numi-1));
	// odd, even triangles in this quad...
	let quadVertexIndices=[(i,j),(i+1,j),(i,j+1),(i+1,j+1)];
	let subTriVertexIndices2=[[0,1,2],[1,3,2]];
	let subTriVertexIndices=subTriVertexIndices2[ti&1];

	a.vertex( quadVertexIndices[subTriVertexIndices[tvi]] )
}


impl<V:LandscapeVertex,Cell> ITriangleMesh<V> for HeightField<Cell>
{
	fn numTriangles(&self)->Index {
		let (numi,numj)=self.numuv;	
		((numi-1)*(numj-1))*2
	}
	fn triVertex(&self,ti:Index, tvi:Index)->V {
		let (numi,_)=self.numuv;
		let (i,j)=(ti>>1).div_mod_floor(&(numi-1));
		// odd, even triangles in this quad...
		let quadVertexIndices=[(i,j),(i+1,j),(i,j+1),(i+1,j+1)];
		let subTriVertexIndices2=[[0,1,2],[1,3,2]];
		let subTriVertexIndices=subTriVertexIndices2[ti&1];

		self.vertex( quadVertexIndices[subTriVertexIndices[tvi]] )
	}	
}
*/




/*
impl<V:LandscapeVertex,T:IVertexGrid<V>> ITriangleMesh<V> for T
{
	pub fn numTriangles(&self)->Index {
		let (numi,numj)=self.size();	
		((numi-1)*(numj-1))*2
	}
	pub fn triVertex(&self,ti:Index, tvi:Index)->V {
		let (numi,_)=self.size();
		let (i,j)=(ti>>1).div_mod_floor(&(numi-1));
		// odd, even triangles in this quad...
		let quadVertexIndices=[(i,j),(i+1,j),(i,j+1),(i+1,j+1)];
		let subTriVertexIndices2=[[0,1,2],[1,3,2]];
		let subTriVertexIndices=subTriVertexIndices2[ti&1];

		self.vertex( quadVertexIndices[subTriVertexIndices[tvi]] )
	}
}
*/
