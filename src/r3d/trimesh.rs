use super::*;
use std::collections::HashMap;

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
	pub fn new()->Self{
		TriMesh{
			vertices:Vec::new(),
			indices:Vec::new(),
			attr:Vec::new(),
		}
	}
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

/// push an unshared cyclic orderquad into a mesh. 
impl<V,ATTR:Clone> TriMesh<V,ATTR>{
	pub fn push_quad(&mut self,vs:(V,V,V,V),a:ATTR){ 
		// todo: parallel arrays?
		let index=self.vertices.len() as i32;
		self.vertices.push(vs.0);
		self.vertices.push(vs.1);
		self.vertices.push(vs.2);
		self.vertices.push(vs.3);
		self.indices.push([index+0,index+1,index+3]);
		self.indices.push([index+1,index+2,index+3]);
		self.attr.push(a.clone());
		self.attr.push(a);
		assert!(self.indices.len()==self.attr.len(),"primitive and attr arrays must be same length")
	}
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
		// todo area weighting, smoothing..
		for (i,t) in self.indices.iter().enumerate(){
			ret[t[0] as usize].vassign_add(&tnorm[i]);
			ret[t[1] as usize].vassign_add(&tnorm[i]);
			ret[t[2] as usize].vassign_add(&tnorm[i]);
		}
		for x in ret.iter_mut(){x.vassign_norm();}
		ret
	}

	fn edges(&self)->Vec<[VtIdx;2]> {
		// brute force.. hashmap *per vertex*, vertex->vertex connect?
		// or do it the C way..
		// TODO - we need to discover the best way using hashmaps etc
		unsafe {unsafe_edgebuilder(self.vertices.len() as i32,&self.indices)}
	}
}

fn add_axis(pos:[usize;3],axis:usize,disp:isize)->[usize;3]{
	let mut ret=pos; ret[axis]=ret[axis].wrapping_add(disp as usize);
	ret
}
fn geti3<T:Copy>(s:&Vec<Vec<Vec<T>>>,xyz:[usize;3])->T{
	s[xyz[2]][xyz[1]][xyz[0]]
}
impl<V:Debug+Pos> TriMesh<V,()>{
	pub fn dump_info(&self){
		println!("mesh: vertices{} triangles{} extents {:?}",self.vertices.len(),self.indices.len(),self.extents());
		
	}
	pub fn extents(&self)->Extents<Vec3>{
		let mut ex=Extents::new();
		for v in self.vertices.iter(){
			ex.include(&v.pos())
		}
		ex
	}
}
impl TriMesh<VertexNCT,()>{

	//todo - customize with lambdas
	pub fn from_voxels(voltex:&Vec<Vec<Vec<f32>>>, size:f32)->Self{
		let mut mesh = TriMesh::new();
		// needed to constrain lifetimes
		{
			let cell_size=size/(voltex.len() as f32);
			let make_vertex=|ipos:[usize;3],normal_axis_index,(u,v)|{
				let mut norm=Vec3::zero(); norm[normal_axis_index]=1.0;
				let s=voltex[ipos[2]][ipos[1]][ipos[0]];// pick shade from tex
				VertexNCT{
					pos:Vec3(ipos[0].fmul(cell_size),ipos[1].fmul(cell_size),ipos[2].fmul(cell_size)),
					color:Vec4(s,s,s,1.0),
					norm:norm,
					tex0:Vec2(u,v)
				}
			};

			let mut seed=0x92824;
			let mut fn_cmpcell=|ipos:[_;3],axis,uaxis,vaxis|{
				let mut cmppos=add_axis(ipos,axis,1);
				// place polys on transition from -ve to posative.
				let a=geti3(voltex,ipos);let b=geti3(voltex,cmppos);

				// render all transition planes.
				if (a>0.0 && b<0.0) || (a<0.0 && b>0.0){
					// quad vertex index positions
					let qipos00=cmppos;
					let qipos01=add_axis(qipos00, uaxis,1);

					let qipos10=add_axis(qipos00, vaxis,1);
					let qipos11=add_axis(qipos01, vaxis,1);

					// todo less cut-pasty.. 'map qpos make_vpos'
					// todo - consider cell for texture info
					let mut v00=make_vertex(qipos00,axis,(0.0,0.0));
					let mut v01=make_vertex(qipos01,axis,(1.0,0.0));
					let mut v10=make_vertex(qipos10,axis,(0.0,1.0));
					let mut v11=make_vertex(qipos11,axis,(1.0,1.0));
					mesh.push_quad((v00,v01,v11,v10),());
				}
			};

			for z in 0..voltex.len()-1{
				for y in 0..voltex[0].len()-1{
					for x in 0..voltex[0][0].len()-1{
						let pos=[x,y,z];
						fn_cmpcell(pos, 0, 1,2);		
						fn_cmpcell(pos, 1, 0,2);		
						fn_cmpcell(pos, 2, 0,1);		
					}
				}
			}
			// runthe fill algorithm in the console.
		}
		mesh.dump_info();
		return mesh;
	}
}

pub type TriInd=VtIdx;
pub struct EdgeLink{
	pub vertex:[VtIdx;2],
	pub next:[*mut EdgeLink;2],
	pub edge_tris:Vec<TriInd>,
}
impl EdgeLink{
	pub fn is_edge(&self,sv:VtIdx,ev:VtIdx)->bool{
		(self.vertex[0]==sv && self.vertex[1]==ev)||
		(self.vertex[1]==sv && self.vertex[0]==ev)
	}
	pub unsafe fn next_of(&mut self, v:VtIdx)->*mut Self{
		if v==self.vertex[0] {self.next[0]} else if v==self.vertex[1]{self.next[1]} else {panic!("edge doesnt have vertex")}
	}
}

pub fn edgebuilder(num_vertices:VtIdx,tris:&[[VtIdx;3]])->Vec<[VtIdx;2]>{
	let mut vertex_num_edges=vec![0 as VtIdx;num_vertices as usize];
	for tri in tris.iter(){
		for &vi in tri.iter(){ vertex_num_edges[vi as usize]+=2;}// each tri contributes 2 edges to a vertex
	}
	//allocate array space by count, fill etc
	unimplemented!()
}

unsafe fn unsafe_edgebuilder(num_vertices:VtIdx, tris:&[[VtIdx;3]])->Vec<[VtIdx;2]>{
	let mut vertex_edges=vec![0 as *mut EdgeLink;num_vertices as usize];
	let mut edges:vecbox<EdgeLink>=Vec::new();
	edges.reserve((num_vertices*3) as usize);
	let mut final_edges:Vec<[VtIdx;2]>=Vec::new();

	// todo - generalize this pattern to n-prims			
	for (tri_index,tri) in tris.iter().enumerate() {
		for i in 0..2{
			let ii=(i+1)%3;
			// todo - could linklists be done with Option<&'>?
			// search ..
			let edge_start_vt=tri[i];
			let  edge_end_vt=tri[ii];
			let mut e=vertex_edges[edge_start_vt as usize];
			let mut found_edge=None;
			while e!=0 as *mut EdgeLink{
				if (&*e).is_edge(edge_start_vt,edge_end_vt){
					found_edge=Some(e);
					break;
				}
				e=(&mut *e).next_of(edge_start_vt);
			}
			if let Some(ef)=found_edge{
				(&mut *ef).edge_tris.push(tri_index as TriInd);
			}else{
				// create the edge.
				let mut edge=Box::new(EdgeLink{
					vertex:[edge_start_vt,edge_end_vt],
					next:[0 as *mut EdgeLink;2],
					edge_tris:vec![tri_index as TriInd],
				});
				{
					let mut e=&mut *edge as *mut EdgeLink;
					//link list push for start vertex
					//link thru the edge start
					(&mut *e).next[0]=vertex_edges[edge_start_vt as usize];
					vertex_edges[edge_start_vt as usize]=e;

					//link list push for end vertex
					//link thru the edge end
					(&mut *e).next[1]=vertex_edges[edge_end_vt as usize];
					vertex_edges[edge_end_vt as usize]=e;
				}

				edges.push(edge);
				// link it..
			}		
		}
	}
	// collect the edges
	for edge in edges {
		final_edges.push([edge.vertex[0],edge.vertex[1]]);
	}
	final_edges
}



