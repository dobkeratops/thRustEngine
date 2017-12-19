use super::*;

// todo: generalize primitive type
// e.g. edgemesh, trimesh, quadmesh, polymesh.
// 

type VtIdx=i32;
/// triangle mesh; 'vertex' doesn't need an attr, you'd just extend 'V' itself.
#[derive(Clone,Debug)]
pub struct TriMesh<V,ATTR=()> {
	pub vertices:Vec<V>,
	pub indices:Vec<[VtIdx;3]>,
	pub attr:Vec<ATTR>,
}
// TODO: 'VertexAttrMesh'
// -shared positions
// surface vertices indexed by the triangles
// triangles connecting the attrvertices

// generation of common mesh primitives
pub fn add_quad(tris:&mut Vec<[VtIdx;3]>, quad:[VtIdx;4]){
	tris.push([quad[0],quad[1],quad[3]]);
	tris.push([quad[1],quad[2],quad[3]]);
}
impl<V> TriMesh<V,()>{
	pub fn grid(is:VtIdx,js:VtIdx,f:&Fn(VtIdx,VtIdx)->V)->Self{
		Self::grid_wrapped(is,js,false,false,f)
	}
	pub fn grid_wrapped(is:VtIdx,js:VtIdx,wrapi:bool,wrapj:bool,f:&Fn(VtIdx,VtIdx)->V)->Self{
		let imax=if wrapi{is}else{is-1};
		let jmax=if wrapj{js}else{js-1};
		let mut vts=Vec::new(); vts.reserve((is*js) as usize);
		let mut tris=Vec::new(); tris.reserve((imax*jmax*2)as usize);
		for j in 0..js{
			for i in 0..is{
				vts.push(f(i,j));
			}
		}
		for j in 0..(js-1){
			for i in 0..(is-1){
				let base0=j*is + i;
				let base1=(j+1)*is + i;
				add_quad(&mut tris,[base0,base0+1,base1+1,base1]);
			}
			if wrapi {
				let base=j*is + (is-1);
				add_quad(&mut tris,[base+0+0,base+(1-is)+0,base+(1-is)+is,base+0+is]);
			}
		}
		if wrapj{
			for i in 0..(is-1){
				let base0=(js-1)*is + i;
				let base1=0 + i;
				add_quad(&mut tris,[base0,base0+1,base1+1,base1]);
			}
			if wrapi {
				let base0=(js-1)*is + (is-1);
				let base1=(is-1);
				add_quad(&mut tris,[base0,base0+(1-is),base1+(1-is),base1]);
			}
		}
		//println!("w={} h={} prod1 {} prod2 {} made: vtc={} tris={}",is,js, is*js, (is-1)*(js-1),vts.len(),tris.len()); panic!();
		// rely on call to 'map attributes' to compute stuff using pos
		// todo 'with_attr' version
		let len=tris.len();
		TriMesh{vertices:vts, indices:tris, attr:vec![();len]}
	}
}

impl<V:Sized,ATTR> TriMesh<V,ATTR>{
	pub fn foreach_triangle(&self, f:&mut FnMut([&V;3],&ATTR)->() ){
		for (i,t) in self.indices.iter().enumerate(){
			f([&self.vertices[t[0] as usize],
				&self.vertices[t[1] as usize],
				&self.vertices[t[2] as usize]],
			&self.attr[i]);
		}
	}

	pub fn map_triangles<B>(&self, f:&Fn([&V;3],&ATTR)->B )->Vec<B>{
		let mut out=Vec::new(); out.reserve(self.indices.len());
		for (i,t) in self.indices.iter().enumerate(){
			out.push(
				f([&self.vertices[t[0] as usize],
					&self.vertices[t[1] as usize],
					&self.vertices[t[2] as usize]],
				&self.attr[i]));
		}
		out
	}

	//  consume recomputing triangle attributes, make a new mesh.
	pub fn map_attr<NEW_ATTR>(self,mapper:&Fn([&V;3],&ATTR)->NEW_ATTR)->TriMesh<V,NEW_ATTR>{
		let new_attr=self.map_triangles(mapper);
		TriMesh{vertices:self.vertices, indices:self.indices, attr:new_attr}
	}

	// consume, filtering the triangle list.
	// ditch any unused vertices
	pub fn filter_triangles(self, predicate:&Fn([&V;3],&ATTR)->bool)->TriMesh<V,ATTR>{
		unimplemented!()
	}

//	fn foreach_vertex_of_triangle(&self, f:FnMut(
}



impl TriMesh<Vec3,()> {
	// generates xy-plane heightfield z is up. 
	// must rotate it after if you want something else
	pub fn from_heightfield(ht:&Vec<Vec<f32>>,size:f32)->Self{
		let width=ht[0].len();
		let height=ht.len();
		for col in ht.iter(){
			assert!(col.len()==width);
		}
		let mut vertices:Vec<Vec3> = Vec::new();
		let cellxsize=size/(width as f32);
		let cellysize=size/(height as f32);
		let mut fy=-0.5f32*cellysize;
		for y in 0..height{
			let mut fx=-0.5f32*cellxsize;
			for x in 0..width{
				vertices.push(Vec3(fx,fy,ht[y][x]));
				fx+=cellxsize;
			}
			fy+=cellysize;
		}
		TriMesh::grid(width as VtIdx,height as VtIdx,&|i,j|vertices[(j as usize)*width+(i as usize)].clone())
	}
	pub fn triangle_normals(&self)->Vec<Vec3>{
		self.map_triangles(&|tri,_|tri[0].vtriangle_norm(tri[1],tri[2]))
	}
	pub fn vertex_normals(&self)->Vec<Vec3>{
		let mut ret=vec![Vec3::zero(); self.vertices.len()];
		let tnorm=self.triangle_normals();
		for (i,t) in self.indices.iter().enumerate(){
			ret[t[0] as usize].vassign_add(&tnorm[i]);
			ret[t[1] as usize].vassign_add(&tnorm[i]);
			ret[t[2] as usize].vassign_add(&tnorm[i]);
		}
		for x in ret.iter_mut(){x.vassign_norm();}
		ret
	}


}
