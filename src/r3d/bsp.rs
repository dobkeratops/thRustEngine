use super::*;

pub fn main()
{
	unsafe {
		draw::init();
		let mut palette=Vec::<u8>::new();
		let x=File::open("/Users/walter/data/palette.lmp").unwrap().read_to_end(&mut palette);
        match x{
            Err(x)=>{println!("error loading file");}
            _=>{}
        };
		for i in 0..256*3 {g_palette[i]=palette[i];}


		let mut bsp=Blob::<BspHeader>::read(&Path::new("/Users/walter/data/e1m1.bsp"));
		let mut a=0.0f32;
		let mut tex_array=Vec::<GLuint>::new();
		// Load textures to GL
//		glutPostRedisplay();
		bsp.visit_textures_mut( &mut |i,_|{
				let (tx,img)=bsp.get_texture_image(i); 
				let txsize=(tx.width as u32,tx.height as u32);
				draw::image(txsize,&img, (((i&7)as f32)*(1.0/4.0)-1.0, (((i>>3)&7)as f32)*(1.0/4.0)-1.0) );
				let txi=draw::create_texture((tx.width,tx.height), &img,8);
				tex_array.push(txi);
			}
		);
		// show the map, isometric.
		for i in 0..200{
		glClearColor((i as f32) * 0.1f32,0.5f32,0.0f32,0f32);
		glClear(GL_COLOR_BUFFER_BIT);

		bsp.visit_textured_triangles(
			&|vertices,txi|{
				draw::set_texture(0,tex_array[txi as usize]);
				draw::tri_iso_tex(vertices[0],vertices[1],vertices[2],0xffffff,1.0/2000.0)
			}
		);

		println!("foo");
			draw::show();
		glutSwapBuffers();
		glutCheckLoop();
		}
	//	glutDisplayFunc(render_and_swap as *const u8);
	//	glutIdleFunc(idle as *const u8);

		//draw_win_loop();
	}
}





#[repr(C)]
pub struct Blob<HEADER> {
	header:[HEADER;0],
	data:Vec<u8>,
//    phantom: std::marker::PhantomData<HEADER>
}

impl<T> Deref for Blob<T> {
	type Target=T;
	fn deref<'s>(&'s self)->&'s T {
		unsafe {	&*(&self.data[0]as*const _ as*const T)
		}
	}
}

impl<T> Blob<T> {
	pub fn num_bytes(&self) -> uint { self.data.len() as uint }
	
	pub fn read(path:&Path)->Blob<T> {
//		let data=
		let mut data=Vec::<u8>::new();
			match File::open(path).unwrap().read_to_end(&mut data) {
				Ok(x)=>{
					println!("read {:?} {:?} bytes", /*path_to_str*/path.to_str(), x);	
				},
				Err(e)=>{
					println!("failed to read {:?}", path); 
					//~[0,..intrinsics::size_of::<Header>()]		// still returns an empty object, hmmm.
					//vec::from_elem(0,intrinsics::size_of::<Header>())
				}
			};
		Blob::<T>  {data:data,header:[]}
	}
}

pub struct DEntry<Header,T> { 
	// we got a nasty bug passing wrong base ptr without header typeinfo here
	// &self instead of self..
	offset:u32, 
	size:u32,
    phantom: (marker::PhantomData<Header>,marker::PhantomData<T>),
}
//unsafe fn byte_ofs_ref<'a,X,Y=X,I:Int=int>(base:&'a X, ofs:I)->&'a Y {
//	&*( (base as *_ as *u8).offset( ofs.to_int().unwrap() ) as *Y)
//}
pub type BspDEntry<T> =DEntry<BspHeader,T>;

impl<Header,T> DEntry<Header,T> {
	fn len(&self)->uint { unsafe {(self.size as usize /  size_of::<T>()) as uint} }
	fn get<'a>(&'a self, owner:&'a Header,i:uint) -> &'a T{
		// TODO: to REALLY be safe, the sub-elements need to check safety from the blob 'owner'
		// unfortunately 'bspheader' doesn't seem to have that, although the last elements' ofs & size could be used
		// for an assert?
		unsafe {
			&*(((owner as *const Header as *const u8).offset(self.offset as isize) as *const T).offset(i as isize))
//			&*(byte_ofs_ptr(owner, self.offset).offset(i as int))
		}
	}
	fn get_mut<'a>(&'a self, owner:&'a  Header,i:uint) -> &'a mut T{
		unsafe {
			let p=(owner as *const Header as *mut Header as *mut u8).offset(self.offset as isize)  as *mut T; 
			let p2=p.offset(i as isize) as *mut T;
			&mut *p
		}
	}
}

#[repr(C)]
pub struct BspHeader {
	pub version:u32,
	pub entities:BspDEntry<Entity>,
	pub planes:BspDEntry<Plane>,

	pub miptex:BspDEntry<MipHeader>,
	pub vertices:BspDEntry<BspVec3>,

	pub visibility:BspDEntry<VisiList>,
	pub nodes:BspDEntry<BspNode>,

	pub texinfo:BspDEntry<TexInfo>,

	pub faces:BspDEntry<Face>,

	pub lightmaps:BspDEntry<LightMap>,
	pub clipnodes:BspDEntry<ClipNode>,

	pub leafs:BspDEntry<BspLeaf>,

	pub marksurfaces:BspDEntry<i16>, //? no
	pub edges:BspDEntry<Edge>,

	pub surfedges:BspDEntry<i32>, // ? no
	pub models:BspDEntry<Model>,
}
macro_rules! get {
	($obj:ident . $field:ident [ $id:expr ] )=>($obj . $field . get( $obj , $id as uint ))
}
impl BspHeader {
	pub fn dump_vertices(&self) {	
		println!("vertices:{}(",self.vertices.len());
		let mut i:uint=0;
		let vtlen=self.vertices.len();
		while i<vtlen { 
			let vtref= self.vertices.get(self,i);
			println!("vertex{}/{}:owner={:p} vertex= {:p} ,({},{},{})",
				i,vtlen, self, vtref,
				vtref.0,vtref.1,vtref.2);
			i+=1;
			let v=*vtref;
		}
		println!("vertices:)");
	
	}
	pub fn dump(&self) {
		println!("ptrs: {:p}\t{:p}\t{:p}",self, &self.entities, &self.planes);
		println!("id: {}", self.version);
		println!("entities: {}{}", self.entities.offset, self.entities.size);
		println!("BSP info:-(");
		println!("entities: {:?}", self.entities.len());
		println!("planes: {:?}", self.planes.len());
		println!("miptex: {:?}", self.miptex.len());
		println!("vertices: {:?}", self.vertices.len());
		println!("nodes: {:?}", self.nodes.len());
		println!("faces: {:?}", self.faces.len());
		println!("lightmaps: {:?}", self.lightmaps.len());
		println!("BSP info:-)");
		self.dump_vertices();
	}
	pub fn visit_vertices_mut<'a>(&'a self, fv:&mut FnMut(int,&mut (f32,f32,f32))){
		for i in 0..self.vertices.len() {
			let mut v=self.vertices.get_mut(self,i);
			fv(i as int,v);
		}
	}
	pub fn visit_vertices<'a>(&'a self, fv:&Fn(int,&Vec3<f32>)){
		for i in 0..self.vertices.len() {
			let v=self.vertices.get(self,i);
			let vt=Vec3(v.0,v.1,v.2);
			fv(i as int,&vt);
		}
	}
	// some convinient accessors. - TODO autogenerate from a macro
	pub fn visit_triangles<'a,'b>(
			&'a self,
			fn_apply_to_tri:
				&'b Fn(	(uint,uint,uint),
						(&'a BspVec3,&'a BspVec3,&'a BspVec3),
						(uint,&'a TexInfo),
						(uint,&'a Plane),
						(uint,&'a Face))
			)
	{
		//let mut return_val:Vec<R> =Vec::new();	// todo: reserve
		for face_id in 0..self.faces.len() {
			let face=self.faces.get(self, face_id);
			let eii = face.firstedge;
			let first_ei= *get!{self.surfedges[eii]};
			let first_edge= get!{self.edges[if first_ei>=0{first_ei}else{-first_ei}]};
			let iv0=(if first_ei>=0 {first_edge.vertex0}else{first_edge.vertex1})  as uint;
			let v0 = self.vertices.get(self, iv0 as uint) ;
			
			// todo: iterate as strips, not fans.
			for esubi in 0..face.num_edges {
				let ei = *get!{self.surfedges[eii+esubi as i32]};
				let edge=get!{self.edges[ei]};
				let edge=get!{self.edges[if ei>0{ei}else{-ei}]};
				let (iv1,iv2)=if ei>=0{ 
					(edge.vertex0 as uint,edge.vertex1 as uint)
				} else {
					(edge.vertex1 as uint,edge.vertex0 as uint)
				};
				let mut v1=self.vertices.get(self, iv1 as uint);
				let mut v2=self.vertices.get(self, iv2 as uint);

				let tri_result=
				(*fn_apply_to_tri) (
					(iv0,iv1,iv2),
					(v0,v1,v2),	
					(face.texinfo as uint,	get!{self.texinfo[face.texinfo]} ),
					(face.plane as uint,	get!{self.planes[face.plane]} ),
					(face_id, face)
				);
				//return_val.push(tri_result);
			}
		}
		//return_val
	}

	fn get_textured_point<'a>(&self,point:&'a BspVec3, tx:&'a TexInfo)->(&'a BspVec3,(f32,f32)){
		let s=1.0f32/256.0f32;
		let u=v3dot(&tx.axis_s,point)+tx.ofs_s;
		let v=v3dot(&tx.axis_t,point)+tx.ofs_t;
		(point, (s*u,s*v))
	}

	pub fn visit_textured_triangles<'a>(&'a self,user_f:&Fn([(&'a BspVec3,(f32,f32));3],i32)) {
		self.visit_triangles(
			&mut |_,(v0,v1,v2),(_,txinfo),(_,plane),(face_id,_)| {
				user_f(
					[	self.get_textured_point(v0,txinfo),
						self.get_textured_point(v1,txinfo),
						self.get_textured_point(v2,txinfo)],
					txinfo.miptex as i32);
			}
		);
	}

	pub fn visit_faces<'a>(&'a self, f:&'a Fn(uint, &Face )) {
		for i in 0..self.faces.len() {
			(*f)(i, get!{self.faces[i]} );
		}
	}
	pub fn visit_faces_mut<'a>(&'a self, f:&'a mut FnMut(uint, &Face )) {
		for i in 0..self.faces.len() {
			(*f)(i, get!{self.faces[i]} );
		}
	}

	pub fn extents(&self)->(Extents<Vec3<f32>>,Vec3<f32>,f32) {
		let mut ext=Extents::new();
		self.visit_vertices_mut(&mut |index,pos|{
			ext.include(&Vec3(pos.0,pos.1,pos.2));
		});
		let c=ext.centre();
		let mut maxr2=0.0f32;
		self.visit_vertices_mut(&mut |_,pos|{
			let ofs=Vec3(pos.0,pos.1,pos.2).vsub(&c);
			let d2=ofs.vsqr();
			maxr2=max(d2,maxr2);
		});
		(ext,c,maxr2.sqrt())
	}

	pub fn get_used_textures(&self)->HashSet<uint> {
		let mut used_tx= HashSet::<uint>::new();
		self.visit_faces_mut( &mut |i:uint,face:&Face|{used_tx.insert(face.texinfo as uint);});
		used_tx
	}

	pub fn get_texture<'a>(&'a self, i:uint)->&'a MipTex {
		let txh=self.miptex.get(self,0);
		let tx = unsafe {&*(
			(txh as *const _ as *const u8).offset(*txh.miptex_offset.get_unchecked(i as usize) as isize) as *const MipTex
		)};
		tx
	}
	pub fn get_texture_size(&self,i:uint)->(u32,u32) {
		let t=self.get_texture(i);
		(t.width as u32 ,t.height as u32)
	}
	pub fn swap_yz(&mut self) {
		self.visit_vertices_mut(&mut |i,v|{ let (x,y,z)=v.clone(); *v=(x,z,y);})
	}

	pub fn visit_textures<'a>(&'a self, tex_fn:&'a Fn(uint,&MipTex)) {
		println!("visit textures self={:p}",&self);
		let txh =self.miptex.get(self,0);
		println!("visit textures txh={:p}",txh);
		for i in 0..txh.numtex {
			println!("{}/{}",i,txh.numtex);
			let tx=self.get_texture(i as uint);
			unsafe {
				println!("tx: {:?} {:?} {:?}",
					i,
					//CString::new(&tx.name[0]), 
					tx.width, tx.height);
			}
			(*tex_fn)( i as uint, tx );
		}
	}

	pub fn visit_textures_mut<'a>(&'a self, tex_fn:&'a mut FnMut(uint,&MipTex)) {
		println!("visit textures self={:p}",&self);
		let txh =self.miptex.get(self,0);
		println!("visit textures txh={:p}",txh);
		for i in 0..txh.numtex {
			println!("{}/{}",i,txh.numtex);
			let tx=self.get_texture(i as uint);
			unsafe {
				println!("tx: {:?} {:?} {:?}",
					i,
					//CString::new(&tx.name[0]), 
					tx.width, tx.height);
			}
			(*tex_fn)( i as uint, tx );
		}
	}


}

impl BspHeader {
	fn draw_edges(&self) {
		let scale=1.0f32/3000.0f32;
		let mut i=0;
		while i < self.edges.len() {
			let e= get!{self.edges[i]};
			let v0 = get!{self.vertices[e.vertex0]};
			let v1 = get!{self.vertices[e.vertex1]};
			draw::line_iso(v0,v1,0xffffff, scale);
			i+=1;
		}
	}
	fn draw_faces(&mut self) {
		let scale=1.0f32/3000.0f32;
		self.visit_triangles(
			&mut |(i0,i1,i2),(v0,v1,v2),(_,txinfo),_,(face_id,_)| draw::tri_iso(v0,v1,v2, draw::random_color(face_id), scale)
		);
		
	}
	fn draw_all_surface_edges(&self)
	{
		for i in 0..self.surfedges.len() {
			self.draw_edge(*(self.surfedges.get(self, i))  as int);
		}
	}
	fn draw_edge(&self, mut ei:int) {
		let scale=1.0f32/3000.0f32;
		if ei<0 {ei=-ei}
		let edge=self.edges.get(self, ei as uint);
		let v0 = self.vertices.get(self, edge.vertex0 as uint);
		let v1 = self.vertices.get(self, edge.vertex1 as uint);
		draw::line_iso(v0,v1,0xffffff, scale);
	}
}


pub type Point3s=(i16,i16,i16);
pub type BBox=(Point3s,Point3s);
pub struct Plane {
	pub normal:BspVec3,
	pub dist:f32,
	pub plane_type:u32	// 0,1,2 = axial planes x,y,z; 3,4,5 = x,y,z predominant..
}
pub struct MipTex {
	pub name:[c_char;16],
	pub width:u32, pub height:u32, pub offset1:u32, pub offset2:u32, pub offset4:u32, pub offset8:u32
}
pub struct MipHeader {
	pub numtex:u32, 
	pub miptex_offset:[u32;0]	// actual size is..
}
impl MipHeader {
	pub unsafe fn tex_offsets(&self)->*const u32 {
		(self as *const _).offset(1) as *const u32
	}
	pub unsafe fn tex_offset(&self, i:i32)->int {
		let ofs=self.tex_offsets();
		*ofs.offset(i as isize) as int
	}
	pub fn get_texture(&self, i:int)->&MipTex {
		unsafe {
			&*((self as*const _ as*const u8).offset( self.tex_offset(i as  i32) as isize) as*const MipTex)
		}
	}
}

pub type BspVec3=(f32,f32,f32);
pub type BspVec2=(f32,f32);
pub struct VisiList;
pub struct BspNode {
	plane_id:u32,
	children:[i16;2],
	bbox:BBox,
	firstface:u16,
	numfaces:u16
}
pub enum BspNodeChild {
	ChildNode(i16),ChildLeaf(i16)
}
impl BspNode {
	pub fn child_node(&self, i:int)->BspNodeChild {
		match self.children[i as usize] {
			x if x>=0 => BspNodeChild::ChildNode(x),
			x  =>BspNodeChild::ChildLeaf(-(self.children[i as usize]+1))
		}
	}
}

pub struct TexInfo {
	pub axis_s:BspVec3, pub ofs_s:f32,
	pub axis_t:BspVec3, pub ofs_t:f32,
	pub miptex:i32,
	pub flags:i32
}
pub struct Faces(u8);
pub struct LightMap(u8); //{ 	texels:[u8]} ??
pub struct ClipNode {
	pub planenum:u32,
	pub front:u16, pub back:u16,
}
pub struct BspLeaf {
	pub contents:u32, 
	pub visofs:u32, 
	pub min:Point3s,
	pub max:Point3s,
	pub firstmarksurface:u16,
	pub nummarksurfaces:u16,
	pub ambient_level:[u8;AmbientNum as usize]
}

pub struct Edge {
	pub vertex0:u16,pub vertex1:u16
}

//enum Max {
//	MaxMapHulls=4
//}
const AmbientNum:uint = 4;
const MaxMapHulls:uint = 4;
enum Ambient {
	AmbientWater=0,AmbientSky,AmbientSlime,AmbientLava
}
#[derive(Debug)]
pub struct Entity(u8);
pub struct Model {
	pub bound:BBox,
	pub origin:BspVec3,
	pub headnode:[i32;MaxMapHulls as usize],
	pub visileafs:i32,
	pub firstface:i32,
	pub numfaces:i32
}
pub struct Face {
	pub plane:u16,
	pub side:u16,
	pub firstedge:i32,
	pub num_edges:u16,
	pub texinfo:u16,
//	typelight:u8,
//	baselight:u8,
	pub light:[u8;2],
	pub lightmap_ofs:i32, // [styles*sursize] samples..
}

/// return a reference to a different type at a byte offset from the given base object reference
unsafe fn byte_ofs_ref<'a,X,Y>(base:&'a X, ofs:i32)->&'a Y {
	&*byte_ofs_ptr(base,ofs)
}
/// return a raw ptr to a different type at a byte offset from the given base object reference
unsafe fn byte_ofs_ptr<'a,FROM,TO>(base:&'a FROM, ofs:i32)->*const TO {
	byte_ofs(base as *const _, ofs)
}
/// offsets a raw pointer by a byte amount, and changes type based on return value inference.
unsafe fn byte_ofs<'a,FROM,TO>(base:*const FROM, ofs:i32)->*const TO {
	(base as *const u8).offset( ofs as isize ) as *const TO
}

static mut g_palette:[u8;256*3]=[0;256*3];
//incbin!("palette.lmp");

impl BspHeader {
	pub fn get_texture_image<'a>(&'a self, i:uint)->(&'a MipTex, Vec<u32>) {
		unsafe {
			let tx=self.get_texture(i);
			let mip0:*const u8=byte_ofs_ptr(tx, tx.offset1 as i32);

			let image = ::vec_from_fn(
				(tx.width*tx.height) as usize, 
				&|i|{
					let color_index = *mip0.offset(i as isize) as uint;
					let rgb_index=(color_index*3) as usize;
					let r=g_palette[(rgb_index+0)] as u32;
					let g=g_palette[(rgb_index+1)] as u32;
					let b=g_palette[(rgb_index+2)] as u32;
					(r|(g<<8)|(b<<16)|(if color_index<255{0xff000000}else{0})) as u32
				}
			);
			(tx,image)
		}
	}
}


