use super::*;
use std::collections::HashMap;

// todo: generalize primitive type
// e.g. edgemesh, trimesh, quadmesh, polymesh.
// 

type VtIdx=i32;
/// all attributes on the vertex.
/// triangle mesh; 'vertex' doesn't need an attr, you'd just extend 'V' itself.
#[derive(Clone,Debug)]
pub struct TriMesh<V:Pos,ATTR=()> {
	pub vertices:Array<V>,
	pub indices:Array<Array3<VtIdx>>,
	pub attr:Array<ATTR>,
}

impl<V:Pos,ATTR> HasVertices<V> for TriMesh<V,ATTR>{
	fn num_vertices(&self)->VtIdx{self.vertices.len()}
	fn vertex(&self,i:i32)->&V{&self.vertices[i]}
	
}


// TODO: 'VertexAttrMesh'
// -shared positions
// surface vertices indexed by the triangles
// triangles connecting the attrvertices

// generation of common mesh primitives
pub fn add_quad(tris:&mut Array<Array3<VtIdx>>, quad:Array4<VtIdx>){
	tris.push(Array3(quad[0],quad[1],quad[3]));
	tris.push(Array3(quad[1],quad[2],quad[3]));
}
impl<V:Pos> TriMesh<V,()>{
	pub fn new()->Self{
		TriMesh{
			vertices:Array::new(),
			indices:Array::new(),
			attr:Array::new(),
		}
	}
	pub fn grid(is:VtIdx,js:VtIdx,f:&Fn(VtIdx,VtIdx)->V)->Self{
		Self::grid_wrapped(is,js,false,false,f)
	}
	pub fn grid_wrapped(is:VtIdx,js:VtIdx,wrapi:bool,wrapj:bool,f:&Fn(VtIdx,VtIdx)->V)->Self{
		let imax=if wrapi{is}else{is-1};
		let jmax=if wrapj{js}else{js-1};
		let mut vts=Array::new(); vts.reserve(is*js);
		let mut tris=Array::new(); tris.reserve(imax*jmax*2);
		for j in 0..js{
			for i in 0..is{
				vts.push(f(i,j));
			}
		}
		for j in 0..(js-1){
			for i in 0..(is-1){
				let base0=j*is + i;
				let base1=(j+1)*is + i;
				add_quad(&mut tris,Array4(base0,base0+1,base1+1,base1));
			}
			if wrapi {
				let base=j*is + (is-1);
				add_quad(&mut tris,Array4(base+0+0,base+(1-is)+0,base+(1-is)+is,base+0+is));
			}
		}
		if wrapj{
			for i in 0..(is-1){
				let base0=(js-1)*is + i;
				let base1=0 + i;
				add_quad(&mut tris,Array4(base0,base0+1,base1+1,base1));
			}
			if wrapi {
				let base0=(js-1)*is + (is-1);
				let base1=(is-1);
				add_quad(&mut tris,Array4(base0,base0+(1-is),base1+(1-is),base1));
			}
		}
		//println!("w={} h={} prod1 {} prod2 {} made: vtc={} tris={}",is,js, is*js, (is-1)*(js-1),vts.len(),tris.len()); panic!();
		// rely on call to 'map attributes' to compute stuff using pos
		// todo 'with_attr' version
		let len=tris.len() as i32;
		TriMesh{vertices:vts, indices:tris, attr:Array::from_val_n((),len)}
	}
}

impl<V:Pos+Sized,ATTR> TriMesh<V,ATTR>{
	pub fn foreach_triangle(&self, f:&mut FnMut([&V;3],&ATTR)->() ){
		for (i,t) in self.indices.iter().enumerate(){
			f([&self.vertices[t[0]],
				&self.vertices[t[1]],
				&self.vertices[t[2]]],
			&self.attr[i as i32]);
		}
	}

	pub fn map_triangles<B>(&self, f:&Fn([&V;3],&ATTR)->B )->Array<B>{
		let mut out=Array::new(); out.reserve(self.indices.len() as i32);
		for (i,t) in self.indices.iter().enumerate(){
			out.push(
				f([&self.vertices[t[0]],
					&self.vertices[t[1]],
					&self.vertices[t[2]]],
				&self.attr[i as i32]));
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
impl<V:Pos,ATTR:Clone> TriMesh<V,ATTR>{
	pub fn push_quad(&mut self,vs:(V,V,V,V),a:ATTR){ 
		// todo: parallel arrays?
		let index=self.vertices.len() as i32;
		self.vertices.push(vs.0);
		self.vertices.push(vs.1);
		self.vertices.push(vs.2);
		self.vertices.push(vs.3);
		self.indices.push(Array3(index+0,index+1,index+3));
		self.indices.push(Array3(index+1,index+2,index+3));
		self.attr.push(a.clone());
		self.attr.push(a);
		assert!(self.indices.len()==self.attr.len(),"primitive and attr arrays must be same length")
	}
}

impl TriMesh<VertexP,()> {
	// generates xy-plane heightfield z is up. 
	// must rotate it after if you want something else
	pub fn from_heightfield(ht:&Array<Array<f32>>,size:f32)->Self{
		let row:&Array<f32>=&ht[0];
		let width=row.len();
		let height=ht.len();
		for col in ht.iter(){
			assert!(col.len()==width);
		}
		let mut vertices:Array<VertexP> = Array::new();
		let cellxsize=size/(width as f32);
		let cellysize=size/(height as f32);
		let mut fy=-0.5f32*cellysize;
		for y in 0..height{
			let mut fx=-0.5f32*cellxsize;
			for x in 0..width{
				vertices.push(VertexP{pos:Vec3::new(fx,fy,ht[y][x])});
				fx+=cellxsize;
			}
			fy+=cellysize;
		}
		TriMesh::grid(width as VtIdx,height as VtIdx,&|i,j|vertices[j*width+i].clone())
	}
	pub fn triangle_normals(&self)->Array<Vec3f>{
		self.map_triangles(&|tri,_|tri[0].pos.vtriangle_norm(&tri[1].pos(),&tri[2].pos()))
	}
	pub fn vertex_normals(&self)->Array<Vec3f>{
		let mut ret={
			let mut ret=Array::<Vec3<f32>>::from_val_n(Vec3::zero(), self.vertices.len() as i32);
			let tnorm=self.triangle_normals();
			// todo area weighting, smoothing..
			for (i,t) in self.indices.iter().enumerate(){
				let i0:VtIdx=t[0];
				let	i1:VtIdx=t[1];
				let i2:VtIdx=t[2];
				ret[i0].vassign_add(&tnorm[i]);
				ret[i1].vassign_add(&tnorm[i]);
				ret[i2].vassign_add(&tnorm[i]);
			}
			ret
		};	
//		for  x in (&mut ret).iter_mut(){x.vassign_norm();}
		for x in 0..ret.len(){
			ret[x].vassign_norm();
		}
		ret
	}

	fn edges(&self)->Array<Array2<i32>> {
		// brute force.. hashmap *per vertex*, vertex->vertex connect?
		// or do it the C way..
		// TODO - we need to discover the best way using hashmaps etc
		let r:Array<Array2<i32>> =unsafe {unsafe_edgebuilder(self.vertices.len(),self.indices.as_slice())};
		return r;
	}
}

fn add_axis(pos:Array3<i32>,axis:i32,disp:i32)->Array3<i32>{
	let mut ret=pos; ret[axis]=ret[axis].wrapping_add(disp);
	ret
}
fn geti3<T:Copy>(s:&Array<Array<Array<T>>>,xyz:Array3<i32>)->T{	
	let (xi,yi,zi):(i32,i32,i32) = (xyz[0],xyz[1],xyz[2]);
	s[zi][yi][xi]
}
impl<V:Debug+Pos> TriMesh<V,()>{
	pub fn dump_info(&self){
		println!("mesh: vertices{} triangles{} extents {:?}",self.vertices.len(),self.indices.len(),self.extents());
		
	}
	pub fn extents(&self)->Extents<Vec3f>{
		let mut ex=Extents::new();
		for v in self.vertices.iter(){
			ex.include(&v.pos())
		}
		ex
	}
}

impl TriMesh<VertexNFCT,()>{

	//todo - customize with lambdas
	pub fn from_voxels(voltex:&Array<Array<Array<f32>>>, size:f32)->Self{
		let mut mesh = TriMesh::new();
		// needed to constrain lifetimes
		{
			let cell_size=size/(voltex.len() as f32);
			let make_vertex=|ipos:Array3<i32>,clip:f32,normal_axis_index:i32,(u,v)|{
				let mut norm:Vec3f=match normal_axis_index{
					0=>Vec3::new(1.0,0.0,0.0),
					1=>Vec3::new(0.0,1.0,0.0),
					2=>Vec3::new(0.0,0.0,1.0),
					_=>panic!()
				} ;
				let (pix,piy,piz):(i32,i32,i32)=(ipos[0],ipos[1],ipos[2]);
				let s:f32=voltex[piz][piy][pix];// pick shade from tex
				let fx:f32=ipos[0] as f32 * cell_size;
				let fy:f32=ipos[1] as f32 *cell_size;
				let fz:f32=ipos[2] as f32 *cell_size;
				let pv:Vec3<f32>=Vec3::new(fx,fy,fz);
				// todo - confusion about centreing ,0.0 or 0.05?
				VertexNFCT{
					pos:pv.vmadd(&norm,clip*cell_size),
					color:Vec4::new(s,s,s,1.0),
					norm:norm,
					tex0:Vec2::new(u,v)
				}
			};

			let mut seed=0x92824;
			let mut fn_cmpcell=|ipos:Array3<_>,axis:i32,uaxis:i32,vaxis:i32|{
				let mut cmppos=add_axis(ipos,axis,1);
				// place polys on transition from -ve to posative.
				let a=geti3(voltex,ipos);let b=geti3(voltex,cmppos);

				// render all transition planes.

				
				if (a>0.0 && b<0.0) || (a<0.0 && b>0.0){
					// find the clip position
					let fclip = (0.0-a)/(b-a);
					// quad vertex index positions
					let qipos00=cmppos;
					let qipos01=add_axis(qipos00, uaxis,1);

					let qipos10=add_axis(qipos00, vaxis,1);
					let qipos11=add_axis(qipos01, vaxis,1);

					// todo less cut-pasty.. 'map qpos make_vpos'
					// todo - consider cell for texture info
					let mut v00=make_vertex(qipos00,0.0,axis,(0.0,0.0));
					let mut v01=make_vertex(qipos01,0.0,axis,(1.0,0.0));
					let mut v10=make_vertex(qipos10,0.0,axis,(0.0,1.0));
					let mut v11=make_vertex(qipos11,0.0,axis,(1.0,1.0));
					mesh.push_quad((v00,v01,v11,v10),());
				}
			};

			let e0:&Array<Array<Array<f32>>>=&voltex;
			let e1:&Array<Array<f32>>=&e0[0];
			let e2:&Array<f32>=&e1[0];

			for z in 0..e0.len()-1{
				for y in 0..e1.len()-1{
					for x in 0..e2.len()-1{
						let pos=Array3(x,y,z);
						fn_cmpcell(pos, 0i32, 1i32,2i32);		
						fn_cmpcell(pos, 1i32, 0i32,2i32);		
						fn_cmpcell(pos, 2i32, 0i32,1i32);		
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
	pub edge_tris:Array<TriInd>,
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

pub fn edgebuilder(num_vertices:VtIdx,tris:&[[VtIdx;3]])->Array<[VtIdx;2]>{
	let mut vertex_num_edges=vec![0 as VtIdx;num_vertices as usize];
	for tri in tris.iter(){
		for &vi in tri.iter(){ vertex_num_edges[vi as usize]+=2;}// each tri contributes 2 edges to a vertex
	}
	//allocate array space by count, fill etc
	unimplemented!()
}

unsafe fn unsafe_edgebuilder(num_vertices:VtIdx, tris:&[Array3<VtIdx>])->Array<Array2<VtIdx>>{
	let mut vertex_edges=vec![0 as *mut EdgeLink;num_vertices as usize];
	let mut edges:Array<Box<EdgeLink>>=Array::new();
	edges.reserve(num_vertices*3);
	let mut final_edges:Array<Array2<VtIdx>>=Array::new();

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
					edge_tris:Array::from_val_n(tri_index as TriInd,1),
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
	for edge in edges.iter() {
		final_edges.push(Array2(edge.vertex[0],edge.vertex[1]));
	}
	final_edges
}



