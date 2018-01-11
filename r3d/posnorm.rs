use super::*;

/// Combined position and normal
/// handy to throw around
pub struct PosNorm<T> {
	pub pos:Vec3<T>;
	pub norm:Vec3<T>;	// TODO very common to store normals in reduced precision
};

impl<T:Float> LerpBy<T> for (PosNorm<T>,PosNorm<T>) {
	type Output=PosNorm<T>;
	pub fn lerp_by(self,r:Ratio)->PosNorm {
		PosNorm{
			pos:self.0.vlerp(self.1,r),
			norm:self.0.vlerp(self.1,r),
		}
	}
}

impl<T:Float> PosNorm<T> {
	pub fn blend_by(&self, b:&Self,t:T)->Self{
		PosNorm{
			pos: self.pos.vlerp(&b.pos,t),
			norm: self.norm.vlerp(&b.pos,t).normalize,
		}
	}
	fn midpoint(&self,b:&Self,t:T)->Self{
		self.blend_by(b, T::half())
	}
}
