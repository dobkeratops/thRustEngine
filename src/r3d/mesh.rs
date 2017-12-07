use super::*;

type VertexId=usize;
type AttrVertexId=usize;
type MaterialId=usize;
type vec<T> = Vec<T>;   // todo: want replaceable vertex index.
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

    XYZ     // pick xy xz yz planes according to normal
}

#[derive(Clone,Debug)]
enum AlphaMode {

}

/// Description of texture/shading blending tree
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
    textures:vec<TextureId> // collected list of textures
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
#[derive(Clone,Debug)]
struct UMesh<POINT,ATTR>{
    vertices:Vec<POINT>,                // spatial position
    attrvertex:Vec<(VertexId,ATTR)>,    // pos+normal,texture etc
    triangles:Vec<(MaterialId,[VertexId;3])>,
    weightmaps:Vec<(BoneId,Vec<(VertexId,f32)>)>,    // skinning weights
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




