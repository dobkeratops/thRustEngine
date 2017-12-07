use super::*;

// quaternions stored as a vec4,
// but dont support vec4 ops
// multiplies etc do something else.

impl<T:Float> convert::From<Mat44<T>> for Quaternion<T> {
	fn from(x:matrix::Matrix4<Vec4<T>>)->Self{
		unimplemented!()
	}
}

impl<T:Float> convert::From<Quaternion<T>> for matrix::Matrix4<Vec4<T>> {
	fn from(x:Quaternion<T>)->Self{
		unimplemented!()
	}
}



