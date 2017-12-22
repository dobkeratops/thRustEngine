use super::*;

type VertexId=Idx;
type AttrVertexId=Idx;
type MaterialId=Idx;
type TexelScale=f32;    // scaling units to UV coords
type ShaderId=usize;
type BoneId=usize;
type Power=f32;
type TextureId=usize;

#[derive(Clone,Debug)]
enum UVSource {
    UVLayer0,
    UVLayer1,
    ReflectionMap,
    Normal,
    CameraFacingNormal,
    TriplanarXYZ     // pick xy xz yz planes according to normal
}

#[derive(Clone,Debug)]
enum AlphaMode {

}

/// Description of texture/shading blending tree
/// todo - dynamically generate a shader from this...
#[derive(Clone,Debug)]
enum TexNode {
    Tex2d(TextureId, UVSource,TexelScale),
    Tex3d(TextureId, UVSource,TexelScale),  // volume texture?
    VertexColor,                // vertex color channel..
    SurfaceNormal,
    DiffuseLight,
    SpecularLight,
    Fresnel(RGBA,Power),
    Constant(RGBA),          // ARGB value
    Overlay(Box<TexNode>,Box<TexNode>),
    Offset(Box<TexNode>,Box<TexNode>), //add mode blending,midgrey
    Multiply(Box<TexNode>,Box<TexNode>),
    Multiply2x(Box<TexNode>,Box<TexNode>), // playstation style mul where mid-grey=1
    Screen(Box<TexNode>,Box<TexNode>), // photoshop, kinda like inv mul
    HardLight(Box<TexNode>,Box<TexNode>),
    Distort(Box<TexNode>,Box<TexNode>), //
    BlendByMask(Box<TexNode>,Box<TexNode>,Box<TexNode>),
    NormalMap(Box<TexNode>)
}

// example  Multiply2x(VertexColor,Box<Tex2d(

#[derive(Clone,Debug)]
struct Material {
    program:ShaderId,
    texture:TexNode,
    source:String,     // texnodes represented in string format.
    textures:Vec<TextureId> // collected list of textures
}

/// rendermesh, optimized vertices.
///
#[derive(Clone,Debug)]
struct RMesh<V> {
    vertices:Vec<V>,
    triangles:Vec<VertexId>,

}

/// raw unsorted mesh
/// a mesh with positions sharable between multiple render-vertices,
/// e.g. a textured cube would have 8 positions but 24 rendervertices

type Idx=usize;		// TODO how to use i32 index without fighting rust
type PositionId=Idx;	


type Position=Vec3;
type Normal=Vec3;
type Texcoord=Vec2;

// channels: tex0, color, ..?
// TODO - channelmesh.

#[derive(Clone,Debug)]
struct IndexedVertex{
	pos_index:Idx,
	normal:Idx,
	tex0:Idx,
	color:Idx,	
}
type VertexIndex=Idx;

#[derive(Clone,Debug)]
pub struct Mesh {
	// attribute arrays
    positions:Vec<Position>,
	normals:Vec<Normal>,
	texcoords:Vec<Texcoord>,
	colors:Vec<Color>,
    vertex:Vec<IndexedVertex>,
    polygons:Vec<(MaterialId,Vec<[VertexIndex;3]>)>,
    weightmaps:Vec<(BoneId,Vec<(PositionId,f32)>)>,    // skinning weights
    materials:Vec<Material>
}


/// TODO - steps of mesh optimization
/// weightmap sorting, tristrips, ...
/// simple tristrip generator, not optimized
fn gen_tristrips(src:Vec<[VertexId;3]>)->Vec<VertexId>{
    unimplemented!()
}

///
fn quadrify(src:Vec<[VertexId;3]>)->Vec<[VertexId;3]>{
    unimplemented!()

}




