pub use r3d::*;
extern "C" {
	pub fn glVertexVec3f(pos:Vec3f);
	pub fn glNormalVec3f(n:Vec3f);
	pub fn glColorVec4f(color:Vec4f);
	pub fn glTexcoordVec2f(uv:Vec2f);

	pub fn glVertexC(pos:Vec3f,color:Vec4f);
	pub fn glVertexNC(pos:Vec3f,norm:Vec3f,color:Vec4f);
	pub fn glVertexCT(pos:Vec3f,color:Vec4f,tex0:Vec2f);
	pub fn glVertexNCT(pos:Vec3f,norm:Vec3f,color:Vec4f,tex0:Vec2f);
	pub fn glVertexN(pos:Vec3f,norm:Vec3f);
	pub fn glVertexNCT2(pos:Vec3f,norm:Vec3f,color:Vec4f,tex0:Vec2f,tex1:Vec2f);
}
