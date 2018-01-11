pub use ::r3d::*;
//use super::{Matrix4,Matrix3,Matrix2,Matrix1,HasElem};

// todo: simplify, or verify our arithmetic will work with different types
// this needs hkt really?; Matrix4<V,AXISVAL,POSVAL=AXISVAL>  { ax:VEC<AXISVAL>,.. pos:VEC<POSVAL> }s
// last element can be another type.
// e.g. ax,ay,az dimensionless,  pos = position dimension
// todo - i think when it comes to inverting such a thing more thought is needed.

pub struct Scaling<T>(T,T,T);

struct SRT<T:VElem=f32>{
	scaling:Vec3<T>,rotation:Vec3<T>,translation:Vec3<T>
}
struct SQT<T:VElem=f32>{
	scaling:Vec3<T>,rotation:Quaternion<T>,translation:Vec3<T>
}

struct RotateX<T>(T);
struct RotateY<T>(T);
struct RotateZ<T>(T);

pub trait Axes3<V> {
	fn axisX(&self)->V;
	fn axisY(&self)->V;
	fn axisZ(&self)->V;
	fn matrix3(&self)->Matrix3<V>;
}

pub trait AxisW<V> {
	fn axisW(&self)->V;
	fn pos(&self)->V;
}
/*
impl<T:Float> IMatrix3<Vec3<T>> for RotateX<T> {
	fn matrix3(&self)->Matri3<Vec3<T>> {
		let angle=*self;
		let (s,c)=sin_cos(&angle);
		Matrix3::new(
			&vec3(one::<T>(),zero::<T>(),zero::<T>()),
			&vec3(zero::<T>(),c.clone(),s.clone()),
			&vec3(zero::<T>(),-s.clone(),c.clone()))
	}
	fn axisX(&self)->Vec3<T> {let m =self.matrix3(); m.ax}
	fn axisY(&self)->Vec3<T> {let m =self.matrix3(); m.ay}
	fn axisZ(&self)->Vec3<T> {let m =self.matrix3(); m.az}
}
impl<T:Float+Clone> IMatrix3<Vec3<T>> for RotateY<T> {
	fn matrix3(&self)->Matrix3<Vec3<T>> {
		let angle=*self;
		let (s,c)=sin_cos(&angle);
		Matrix3::new(
			&vec3(c.clone(),zero::<T>(),s.clone()),
			&vec3(zero::<T>(),one::<T>(),zero::<T>()),
			&vec3(-s.clone(),zero::<T>(),c.clone()))
	}
	fn axisX(&self)->Vec3<T> {let m =self.matrix3(); m.ax}
	fn axisY(&self)->Vec3<T> {let m =self.matrix3(); m.ay}
	fn axisZ(&self)->Vec3<T> {let m =self.matrix3(); m.az}
}
impl<T:Float+Clone> IMatrix3<Vec3<T>> for RotateZ<T> {
	fn matrix3(&self)->Matrix3<Vec3<T>> {
		let angle=self;
		let (s,c)=angle.sin_cos();
		Matrix3::new(
			&vec3(c.clone(),s.clone(),zero::<T>()),
			&vec3(-s.clone(),c.clone(),zero::<T>()),
			&vec3(zero::<T>(),zero::<T>(),one::<T>()))
	}
	fn axisX(&self)->Vec3<T> {let m =self.matrix3(); m.ax}
	fn axisY(&self)->Vec3<T> {let m =self.matrix3(); m.ay}
	fn axisZ(&self)->Vec3<T> {let m =self.matrix3(); m.az}
}
*/
// was Vec3<T>
impl<T:Float+Clone+Default,V:HasXYZ<Elem=T>> Axes3<V> for Scaling<T> {
	fn axisX(&self)->V	{V::from_xyz(self.0.clone(),zero::<T>(),zero::<T>())}
	fn axisY(&self)->V	{V::from_xyz(zero::<T>(),self.1.clone(),zero::<T>())}
	fn axisZ(&self)->V	{V::from_xyz(zero::<T>(),zero::<T>(),self.2.clone())}
	fn matrix3(&self)->Matrix3<V>{Matrix3::new(&self.axisX(),&self.axisY(),&self.axisZ())}
}
impl<T:Float+Clone+Default, V:HasXYZ<Elem=T>> AxisW<V> for Scaling<T> {
	fn pos(&self)->V	{V::from_xyz(zero::<T>(),zero::<T>(),zero::<T>())}
	fn axisW(&self)->V	{self.pos()}
}

//impl<T,V:VecOps<T>> IMatrix3<V> for Matrix3<V> {

impl<V:Clone> Axes3<V> for Matrix3<V> {
	fn axisX(&self)->V{self.ax.clone()}
	fn axisY(&self)->V{self.ay.clone()}
	fn axisZ(&self)->V{self.az.clone()}
	fn matrix3(&self)->Matrix3<V>{Matrix3::new(&self.axisX(),&self.axisY(),&self.axisZ())}
}

// Matrix axis accessors
//impl<T,V:VecOps<T>> IMatrix4<V> for Matrix4<V> {

impl<V:Clone> AxisW<V> for Matrix4<V>{
	fn axisW(&self)->V{self.aw.clone()}
	fn pos(&self)->V{self.aw.clone()}
}

//impl<T:Float,V:VecOps<T>> Matrix4<V> {
//}
//impl<T:Float=f32,V:VecOps<T> =Vec4<T> > Matrix4<V> {
//impl<V:VecOps<T> =Vec4<f32>,T:Float+Clone=f32 > Matrix4<V> 


impl<F:Float> Matrix4<Vec4<F>> where
{
	// transpose assuming vec4
	pub fn transpose4(&self) -> Matrix4<Vec4<F>> {
		Matrix4::new(
			&vec4(self.ax.vx(), self.ay.vx(), self.az.vx(), self.aw.vx()),
			&vec4(self.ax.vy(), self.ay.vy(), self.az.vy(), self.aw.vy()),
			&vec4(self.ax.vz(), self.ay.vz(), self.az.vz(), self.aw.vz()),
			&vec4(self.ax.vw(), self.ay.vw(), self.az.vw(), self.aw.vw())
		)
	}
}

impl<F:Float> Matrix3<Vec3<F>> where {
	pub fn transpose3(&self) -> Matrix3<Vec3<F>> {
		Matrix3::new(
			&vec3(self.ax.vx(), self.ay.vx(), self.az.vx()),
			&vec3(self.ax.vy(), self.ay.vy(), self.az.vy()),
			&vec3(self.ax.vz(), self.ay.vz(), self.az.vz())
		)
	}
}
impl<T:Float,V:VMath<Elem=T>> Matrix4<V> where
    //<V as Siblings>::V3 :VMath,
    //<<V as Siblings>::V3 as HasElem>::Elem :Float
{
	pub fn identity()->Matrix4<V>{ //todo-move to square impl Matrix4<Vec4,..>   Matrix3<Vec3.,..>
		Matrix4::new(
			&VecConsts::vaxis(0),
			&VecConsts::vaxis(1),
			&VecConsts::vaxis(2),
			&VecConsts::vaxis(3))
	}
	pub fn translate(trans:&V)->Matrix4<V>{
		Matrix4::new(
			&VecConsts::vaxis(0),
			&VecConsts::vaxis(1),
			&VecConsts::vaxis(2),
			&trans.clone())
	}
	pub fn look_along(pos:&V,fwd:&V,up_:&V)->Matrix4<V>{
		let up=up_.vnormalize();
		let az=fwd.vnormalize();
		let ax=az.vcross_norm(&up);
		let ay=ax.vcross_norm(&az);
		Matrix4::new(&ax,&ay,&az,pos)
	}
	pub fn look_at(pos:&V,target:&V,up:&V)->Matrix4<V> {
		Matrix4::look_along(pos,&target.vsub(pos),up)
	}
	pub fn orthonormalize_zyx(self)->Matrix4<V> {
		Matrix4::look_along(&self.aw,&self.az,&self.ay)
	}
	pub fn mul_point(&self,pt:&V)->V{
		self.aw.vmadd_x(&self.ax,pt).vmadd_y(&self.ay,pt).vmadd_z(&self.az,pt)
	}
	pub fn inv_mul_point(&self,pt:&V)->V{
		let ofs=pt.vsub(&self.aw);
		VecOps::vfrom_xyz_f(ofs.vdot(&self.ax),ofs.vdot(&self.ay),ofs.vdot(&self.az))
	}
    // reverse order useful for composing transformation reading from left to right
    pub fn pre_mul_matrix(&self,other:&Matrix4<V>)->Matrix4<V>{
        other.mul_matrix(self)
    }
}

/// Implementations relating to Vec3,Vec4 specifically, not just 'any vector type'.
impl<T:Float> Matrix4<Vec4<T>> {
    pub fn mul_vec3w0(&self,pt:&Vec3<T>)->Vec4<T>{
        self.ax.vscale(pt.x).vmadd(&self.ay,pt.y).vmadd(&self.az,pt.z)
    }
    pub fn mul_vec3w1(&self,pt:&Vec3<T>)->Vec4<T>{
        self.ax.vscale(pt.x).vmadd(&self.ay,pt.y).vmadd(&self.az,pt.z).vadd(&self.aw)
    }
    /// 'point' in context of 4d matrix is 3d vector with w=1
    pub fn mul_vec3_point(&self,pt:&Vec3<T>)->Vec4<T>{
        self.mul_vec3w1(pt)
    }
    /// 'axis' = 3d vector assuming w=0
    pub fn mul_axis(&self,pt:&Vec3<T>)->Vec3<T>{
        self.mul_vec3w0(pt).permute_xyz()
    }
    pub fn mul_vec4(&self,pt:&Vec4<T>)->Vec4<T>{
        self.ax.vscale(pt.x).vmadd(&self.ay,pt.y).vmadd(&self.az,pt.z).vmadd(&self.aw,pt.w)
    }
    // multiply with the x/y/z of a 4d point, but replace w=0
    pub fn mul_vec4w0(&self,pt:&Vec4<T>)->Vec4<T>{
        self.ax.vscale(pt.x).vmadd(&self.ay,pt.y).vmadd(&self.az,pt.z)
    }
    /// Multiply with the x/y/z of a 4d point, but replace w=1
    pub fn mul_vec4w1(&self,pt:&Vec4<T>)->Vec4<T>{
        self.aw.vmadd(&self.ax,pt.x).vmadd(&self.ay,pt.y).vmadd(&self.az,pt.z)
    }

    pub fn to_mat43(&self)->Matrix4<Vec3<T>>{
        Matrix4::new(
            &self.ax.permute_xyz(),
            &self.ay.permute_xyz(),
            &self.az.permute_xyz(),
            &self.aw.permute_xyz())
    }
    pub fn to_mat33(&self)->Matrix3<Vec3<T>>{
        Matrix3::new(
            &self.ax.permute_xyz(),
            &self.ay.permute_xyz(),
            &self.az.permute_xyz())
    }

    pub fn from_mat33_pos(axes:&Matrix3<Vec3<T>>,pos:&Vec3<T>)->Matrix4<Vec4<T>>{
        Matrix4::new(
            &axes.ax.permute_xyz0(),
            &axes.ay.permute_xyz0(),
            &axes.az.permute_xyz0(),
            &pos.permute_xyz1()
        )
    }

	pub fn inv_orthonormal_matrix(&self)->Matrix4<Vec4<T>>{

		let t=self.transpose4();
		let ax=t.ax.permute_xyz0();
		let ay=t.ay.permute_xyz0();
		let az=t.az.permute_xyz0();

		let invpos = Vec4{
            x:- self.ax.vdot( & self.aw),
            y:- self.ay.vdot(& self.aw),
            z:- self.az.vdot( & self.aw),
            w:One::one()
        };
		Matrix4::new(&ax,&ay,&az,&invpos)
	}

    pub fn inv_ortho_matrix(&self)->Matrix4<Vec4<T>>{

        let rscalexsq=self.ax.vsqr().recip();
        let rscaleysq=self.ay.vsqr().recip();
        let rscalezsq=self.az.vsqr().recip();

        let mscaled=Matrix4::new(
            &self.ax.vscale(rscalexsq),
            &self.ay.vscale(rscaleysq),
            &self.az.vscale(rscalezsq),
            &Vec4::zero()
        );

        let t = mscaled.transpose4();
        let ax = t.ax.permute_xyz0();
        let ay = t.ay.permute_xyz0();
        let az = t.az.permute_xyz0();

        let invpos = Vec4{
            x:- ax.vdot( & self.aw)* rscalexsq,
            y:- ay.vdot( & self.aw) * rscaleysq,
            z:- az.vdot( & self.aw) * rscalezsq,
            w:One::one()
        };
        Matrix4::new(&ax,&ay,&az,&invpos)
    }
}

/// Special case implementations for 4x3 matrices
impl<F:Float> Matrix4<Vec3<F>>{
	pub fn inv_orthonormal_matrix(&self)->Matrix4<Vec3<F>>{
		panic!();
		let t=self.transpose();
		let ax=t.ax.permute_xyz();
		let ay=t.ay.permute_xyz();
		let az=t.az.permute_xyz();
		let invpos = Vec3{
            x: - self.ax.vdot( &self.aw),
            y: - self.ay.vdot( & self.aw),
            z: - self.az.vdot( &self.aw)
        };
		Matrix4::new(&ax,&ay,&az,&invpos)
	}
	pub fn to_mat44(&self)->Matrix4<Vec4<F>> {
		Matrix4::new(
			&self.ax.permute_xyz0(),
			&self.ay.permute_xyz0(),
			&self.az.permute_xyz0(),
			&self.aw.permute_xyz1(),
		)
	}
}
impl<F:Float> Matrix3<Vec3<F>>{
	pub fn inv_orthonormal_matrix(&self)->Matrix3<Vec3<F>>{self.transpose()}
	pub fn to_mat43(&self,pos:&Vec3<F>)->Matrix4<Vec3<F>>{
		Matrix4::new(&self.ax,&self.ay,&self.az,pos)
	}
	pub fn to_mat44(&self)->Matrix4<Vec4<F>> {
		Matrix4::new(
			&self.ax.permute_xyz0(),
			&self.ay.permute_xyz0(),
			&self.az.permute_xyz0(),
			&Vec4::origin(),
		)
	}
    pub fn to_mat44_pos(&self,pos:&Vec3<F>)->Matrix4<Vec4<F>>{
        Matrix4::new(
            &self.ax.permute_xyz0(),
            &self.ay.permute_xyz0(),
            &self.az.permute_xyz0(),
            &pos.permute_xyz1(),
        )
    }
}

impl<V> Matrix4<V> {
	pub unsafe fn as_raw_ptr(&self)->*const V {&self.ax}
}

// matrix * matrix
impl<'l, T:Float> Mul<&'l Matrix4<Vec4<T>> > for &'l Matrix4<Vec4<T>> {
	type Output=Matrix4<Vec4<T>>;
	fn mul(self,rhs:&'l Matrix4<Vec4<T>>) -> Matrix4<Vec4<T>>{
		self.mul_matrix(rhs)
	}
}

// matrix * vector4
impl<'l, T:Float> Mul<&'l Vec4<T> > for &'l Matrix4<Vec4<T>> {
	type Output=Vec4<T>;
	fn mul(self,rhs:&'l Vec4<T>) -> Vec4<T>{
		self.mul_vec4(rhs)
	}
}
// matrix43 * vector3
impl<'l, T:Float> Mul<&'l Vec3<T> > for &'l Matrix4<Vec3<T>> {
	type Output=Vec3<T>;
	fn mul(self,rhs:&'l Vec3<T>) -> Vec3<T>{
		self.mul_point(rhs)
	}
}

pub fn identity<F:Float>()->Matrix4<Vec4<F>> {
	Matrix4::<Vec4<F>>::identity()
}

//impl<F:Num+Zero+One> Matrix4<Vec4<F>> {
pub fn projection<F:Float>(tan_half_fov:F, aspect:F, znear:F, zfar:F)->Matrix4<Vec4<F>> {
	let xymax=znear * tan_half_fov;
	let ymin=-xymax;
	let xmin=-xymax;
	let width=xymax-xmin;
	let height=xymax-ymin;

	let zero=zero::<F>();
	let one=one::<F>();

	let depth = zfar-znear;
	let q=-(zfar+znear)/depth;
	let two = one+one;
	let qn=-two*(zfar*znear)/depth;
	let w=two*znear/width;
	let w= w/aspect;
	let h=two*znear/ height;
	
	Matrix4::new(
		&vec4(w, zero, zero, zero),
		&vec4(zero, h, zero, zero),
		&vec4(zero, zero, q, -one),
		&vec4(zero, zero, qn, zero),
	)
}

pub fn view_xyz<F:Float>()->Matrix4<Vec4<F>>{
    let one=one::<F>(); let zero=zero::<F>();
    Matrix4{
        ax: vec4(one, zero, zero, zero),
        ay: vec4(zero, one, zero, zero),
        az: vec4(zero, zero, one, zero),
        aw: vec4(zero, zero, zero, one),
    }
}
pub fn inv_view_xyz<F:Float>()->Matrix4<Vec4<F>>{
    view_xyz()
}

pub fn view_xzy<F:Float>()->Matrix4<Vec4<F>>{
    let one=one::<F>(); let zero=zero::<F>();
    Matrix4::new(
        &vec4(one,  zero,   zero,   zero),
        &vec4(zero, zero,   one,   zero),
        &vec4(zero, -one,   zero,   zero),
        &vec4(zero, zero,   zero,   one),
    )
}
pub fn inv_view_xzy<F:Float>()->Matrix4<Vec4<F>>{
    let one=one::<F>(); let zero=zero::<F>();
    Matrix4::new(
        &vec4(one,  zero,   zero,   zero),
        &vec4(zero, zero,   -one,   zero),
        &vec4(zero, one,   zero,   zero),
        &vec4(zero, zero,   zero,   one),
    )
}
pub fn view_yzx<F:Float>()->Matrix4<Vec4<F>>{
let one=one::<F>(); let zero=zero::<F>();
Matrix4::new(
&vec4(zero, one,   zero,   zero),
&vec4(zero, zero,   one,   zero),
&vec4(one,  zero, zero,   zero),
&vec4(zero, zero,   zero,   one),
)
}

pub fn inv_view_yzx<F:Float>()->Matrix4<Vec4<F>>{
    let one=one::<F>(); let zero=zero::<F>();
    Matrix4::new(
        &vec4(zero,  zero,   one,   zero),
        &vec4(one, zero,   zero,   zero),
        &vec4(zero, one,   zero,   zero),
        &vec4(zero, zero,   zero,   one),
    )
}
pub fn scale_translate<F:Float>(scale:&Vec3<F>,trans:&Vec3<F>)->Matrix4<Vec4<F>>{
    let one=one::<F>(); let zero=zero::<F>();
    Matrix4::new(
        &vec4(scale.x,    zero,    zero,    zero  ),
        &vec4(zero,    scale.y,    zero,    zero),
        &vec4(zero,    zero,    scale.z,    zero  ),
        &vec4(trans.x, trans.y,   trans.z,    one  )
    )
}

pub fn rotate_x<F:Float>(a:F)->Matrix4<Vec4<F>> {
	let (s,c)=a.sin_cos(); let one=one::<F>(); let zero=zero::<F>();
	Matrix4::new(
		&vec4(one,	zero,	zero,	zero),
		&vec4(zero,	c,		s,	zero),
		&vec4(zero,	-s,		c,	zero),
		&vec4(zero,	zero,	zero,	one))
}
pub fn rotate_y<F:Float>(a:F)->Matrix4<Vec4<F>> {
	let (s,c)=a.sin_cos(); let one=one::<F>(); let zero=zero::<F>();
	Matrix4::new(
		&vec4(c,		zero,	s,	zero),
		&vec4(zero,	one,	zero,	zero),
		&vec4(-s,		zero,	c,	zero),
		&vec4(zero,	zero,	zero,	one))
}
pub fn rotate_z<F:Float>(a:F)->Matrix4<Vec4<F>> {
	let (s,c)=a.sin_cos(); let one=one::<F>(); let zero=zero::<F>();
	Matrix4::new(
		&vec4(c,		s,	zero,	zero),
		&vec4(-s,		c,	zero,	zero),
		&vec4(zero,	zero,	one,	zero),
		&vec4(zero,	zero,	zero,	one))
}
pub fn translate_xyz<F:Float>(x:F,y:F,z:F)->Matrix4<Vec4<F>> {
	let one=one::<F>(); let zero=zero::<F>();
	Matrix4::new(
		&vec4(one,	zero,	zero,	zero),
		&vec4(zero,	one,	zero,	zero),
		&vec4(zero,	zero,	one,	zero),
		&vec4(x,	y,	z,	one))
}
pub fn translate_vec4<F:Float>(trans:&Vec4<F>)->Matrix4<Vec4<F>> {
	let one=one::<F>(); let zero=zero::<F>();
	Matrix4::new(
		&vec4(one,	zero,	zero,	zero),
		&vec4(zero,	one,	zero,	zero),
		&vec4(zero,	zero,	one,	zero),
		trans)
}
pub fn translate<F:Float>(trans:&Vec3<F>)->Matrix4<Vec4<F>> {
	let one=one::<F>(); let zero=zero::<F>();
	Matrix4::new(
		&vec4(one,	zero,	zero,	zero),
		&vec4(zero,	one,	zero,	zero),
		&vec4(zero,	zero,	one,	zero),
		&vec4(trans.x,trans.y,trans.z, one))
}



pub fn projection_frustum<F:Float>(left:F,right:F, bottom:F, top:F, fov_radians:F, aspect:F, fnear:F, ffar:F)->Matrix4<Vec4<F>> {
	let one=one::<F>(); 
	let zero=zero::<F>();
    let two=one+one;
    let a=(right+left)/(right-left);
    let b=(top+bottom)/(top-bottom);
    let c=-(ffar+fnear)/(ffar-fnear);
    let d=-(two*ffar*fnear/(ffar-fnear));
	Matrix4::new(
		&vec4(two*fnear/(right-left), zero, zero, zero),
		&vec4(zero, two*fnear/(top-bottom), zero, zero),
		&vec4(a, b, c, -one),
		&vec4(zero, zero, d, zero),
	)
}
//pub fn camera_look_at<F:Float>(pos:Vec3f,)

// TODO - wrapper types to make a 'Vec4<Vec4<T>>' a semantic matrix.
// till then we rely on the 'Matrix4<Vec3>' for semantic '3 columns'.
// with the one wrapper type we should be able to simplify further.

pub trait Transpose {
	type Output;
	fn transpose(&self)->Self::Output;
}

impl<T:VElem> Transpose for Matrix4<Vec3<T>> {
	type Output = Matrix3<Vec4<T>>;
	fn transpose(&self)->Self::Output{
		Matrix3::new(
			&vec4(self.ax.x,	self.ay.x, self.az.x, self.aw.x),
			&vec4(self.ax.y,	self.ay.y, self.az.y, self.aw.y),
			&vec4(self.ax.z,	self.ay.z, self.az.z, self.aw.z),
		)
	}
}

impl<T:VElem,> Transpose for Matrix4<Vec4<T>> {
	type Output = Matrix4<Vec4<T>>;
	fn transpose(&self)->Self::Output{
        // TODO - SIMD impl in 4x4x32bit case
		Matrix4::new(
			&vec4(self.ax.x,	self.ay.x, self.az.x, self.aw.x),
			&vec4(self.ax.y,	self.ay.y, self.az.y, self.aw.y),
			&vec4(self.ax.z,	self.ay.z, self.az.z, self.aw.z),
			&vec4(self.ax.w,	self.ay.w, self.az.w, self.aw.w),
		)
	}
}

impl<T:VElem> Transpose for Matrix3<Vec4<T>> {
	type Output = Matrix4<Vec3<T>>;
	fn transpose(&self)->Self::Output{
		Matrix4::new(
			&vec3(self.ax.x,	self.ay.x, self.az.x),
			&vec3(self.ax.y,	self.ay.y, self.az.y),
			&vec3(self.ax.z,	self.ay.z, self.az.z),
			&vec3(self.ax.w,	self.ay.w, self.az.w),
		)
	}
}

impl<T:VElem> Transpose for Matrix3<Vec3<T>> {
	type Output = Matrix3<Vec3<T>>;
	fn transpose(&self)->Self::Output{
		Matrix3::new(
			&vec3(self.ax.x,	self.ay.x, self.az.x),
			&vec3(self.ax.y,	self.ay.y, self.az.y),
			&vec3(self.ax.z,	self.ay.z, self.az.z)
		)
	}
}

/// Experimental: Represent a Matrix as a wrapped type, similar to Point<V>, Normal<V>
/// such that you plug in Matrix<Vec4<Vec3>> for 4x3, Matrix 4x4 , etc.
/// would implement matrix multiply as m.scale_components_by_vec(v).sum_components() .. etc.
/// see haskell approach , e.g. implementing 'fmap','fold', etc
/// However the amount of nesting might become oppresive, hence worse error messages.
/// 'Matrix<Vec4<Vec4<T>>>  .. vs 'Matrix4<Vec4<T>>'

pub struct Matrix<V>(V);
pub type Mat4x4_<T> = Matrix<Vec4<Vec4<T>>>;
pub type Mat4_<ColV> = Matrix<Vec4<ColV>>;
pub type Mat3_<ColV> = Matrix<Vec3<ColV>>;
pub type Mat2_<ColV> = Matrix<Vec2<ColV>>;
pub type Mat1_<ColV> = Matrix<Vec1<ColV>>;

fn Mat4_<V:VElem>(ax:&V,ay:&V,az:&V,pos:&V)->Mat4_<V>{
	Matrix(vec4(ax.clone(),ay.clone(),az.clone(),pos.clone()))
}
fn Mat3_<V:VElem>(ax:&V,ay:&V,az:&V)->Mat3_<V>{
	Matrix(vec3(ax.clone(),ay.clone(),az.clone()))
}


/*
/// example of how the 'wrapped vector of vectors', see would work.
pub fn rotate_x_ng<F:Float+VElem>(a:F)->Matrix<Vec4<Vec4<F>>> {
	let (s,c)=a.sin_cos(); let one=one::<F>(); let zero=zero::<F>();
	Mat4_(
		&Vec4(one,	zero,	zero,	zero),
		&Vec4(zero,	c,		s,	zero),
		&Vec4(zero,	-s,		c,	zero),
		&Vec4(zero,	zero,	zero,	one))
}
*/

pub struct Mat4x4{}

impl Mat4x4 {
    pub fn mul(&self,other:&Self)->Self{
        unimplemented!()
    }
    pub fn identity()->Self{
        unimplemented ! ()
    }
}

/// for autocomplete experiment
pub struct Vector3<T>(pub T,pub T,pub T);
pub struct Matrix43<T>(pub Vector3<T>,pub Vector3<T>,pub Vector3<T>,pub Vector3<T>);
impl<T:Float> Matrix43<T> {
pub fn mul_point(&self,v:Vector3<T>)->Vector3<T>{
        unimplemented!();
    }
}
pub type Matrix43f= Matrix43<f32>;

fn foo(){
    let x:Mat44f;

}





