use super::*;

type Damping=f32;
type SpringConst=f32;
type SpringLength=f32;
type SceneNodeId=usize;
type Angvel=f32;

struct SceneNode{
	name:String,
	matrix: matrix::Mat44f
}

enum ConstraintType {
	Distance(f32),
	LookAt(),
	Spring(SpringConst,Damping,SpringLength),
	OwnedBy(),
	Blend(f32)
}

enum DriverType {
	Rotate(Normal,Angvel)

}

struct Constraint{
	target:SceneNodeId,
	source1:SceneNodeId,
	source2:SceneNodeId,
	ctype:ConstraintType
}

pub struct Scene {
	nodes:Vec<SceneNode>,
	matrix:Vec<matrix::Mat44f>,
	constraints:Vec<Constraint>
}


