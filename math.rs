use super::*;

/// Roll the helper functions for a vector type - apply per-element function to 2 vectors, 1vector and a scalar broadcast, or 1 vector and a scalar per elem
macro_rules! impl_vector_functions{[vector_type:$vtype:ty,element_type:$elemt:ty,vector_constructor:$vcons:ident=>$(($fname:ident,$fname_s:ident,$fname_xyz:ident=>$op:ident)),*]=>{
	$(pub fn $fname(src1:$vtype, src2:$vtype)->$vtype{$vcons( src1.x.$op(src2.x),src1.y.$op(src2.y),src1.z.$op(src2.z)  )})*
	$(pub fn $fname_s(src1:$vtype, s:$elemt)->$vtype{$vcons(src1.x.$op(s),src1.y.$op(s),src1.z.$op(s))})*
	$(pub fn $fname_xyz(src1:$vtype, x:$elemt, y:$elemt, z:$elemt)->$vtype{$vcons(src1.x.$op(x),src1.y.$op(y),src1.z.$op(z))})*

}}

macro_rules! v3i_permute_v2i{[$($pname:ident($u:ident,$v:ident)),*]=>{
	$(pub fn $pname(a:V3i)->V2i{v2i(a.$u,a.$v)})*
}}

pub trait MyMod :Add+Sub+Div+Mul+Sized{
	fn mymod(&self,b:Self)->Self;
}
impl MyMod for i32{
	fn mymod(&self,b:Self)->Self{ if *self>=0{*self%b}else{ b-((-*self) %b)} }
}

impl_vector_functions![vector_type:V3i,element_type:i32,vector_constructor:v3i=>(v3iadd,v3iadd_s,v3iadd_xyz=>add),(v3isub,v3isub_s,v3isub_xyz=>sub),(v3imul,v3imul_s,v3imul_xyz=>mul),(v3idiv,v3idiv_s,v3idiv_xyz=>div),(v3irem,v3irem_s,v3irem_xyz=>rem),(v3imin,v3imin_s,v3imin_xyz=>min),(v3imax,v3imax_s,v3imax_xyz=>max),(v3iand,v3iand_s,v3iand_xyz=>bitand),(v3ior,v3ior_s,v3ior_xyz=>bitor),(v3ixor,v3ixor_s,v3ixor_xyz=>bitxor),(v3imymod,v3imymod_s,v3imymod_xyz=>mymod),(v3ishl,v3ishl_s,v3ishl_xyz=>shl),(v3ishr,v3ishr_s,v3ishr_xyz=>shr)];

v3i_permute_v2i![v3i_xy(x,y), v3i_yz(y,z), v3i_xz(x,z)];

pub fn v3itilepos(a:V3i,tile_shift:i32)->(V3i,V3i){
	(v3ishr_s(a,tile_shift),v3iand_s(a,(1<<tile_shift)-1))
}

pub fn v3i_hmul_usize(a:V3i)->usize{ a.x as usize*a.y as usize *a.z as usize}
pub fn v2i_hmul_usize(a:V2i)->usize{ a.x as usize*a.y as usize}
pub fn v2i_hadd_usize(a:V2i)->usize{ a.x as usize+a.y as usize}
pub fn v3i_hadd_usize(a:V3i)->usize{ a.x as usize+a.y as usize +a.z as usize}

