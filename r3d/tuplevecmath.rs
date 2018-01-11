use super::*;

pub fn v3cross<T:Num>((x0,y0,z0):(T,T,T),(x1,y1,z1):(T,T,T))->(T,T,T) {
	((y0*z1-z1*y0),(z0*x1-x0*z1),(x0*y1-y0*x1))
}

pub fn v3add<T:Num>((x0,y0,z0):(T,T,T),(x1,y1,z1):(T,T,T))->(T,T,T){
	(x0+x1,y0+y1,z0+z1)
}

pub fn v3div<T:Num>((x0,y0,z0):(T,T,T),(x1,y1,z1):(T,T,T))->(T,T,T){
	(x0/x1,y0/y1,z0/z1)
}

pub fn v3sub<T:Num>((x0,y0,z0):(T,T,T),(x1,y1,z1):(T,T,T))->(T,T,T){
	(x0-x1,y0-y1,z0-z1)
}
pub fn v3dot<T:Num>((x0,y0,z0):(T,T,T),(x1,y1,z1):(T,T,T))->T{
	x0*x1+y0*y1+z0*z1
}
pub fn v3mul<T:Num>((x0,y0,z0):(T,T,T),(x1,y1,z1):(T,T,T))->(T,T,T){
	(x0*x1,y0*y1,z0*z1)
}
pub fn v3scale<T:Num>((x0,y0,z0):(T,T,T),f:T)->(T,T,T){
	(x0*f,y0*f,z0*f)
}
pub fn v3mad<T:Num>(v0:(T,T,T),v1:(T,T,T),f:T)->(T,T,T){ v3add(v0,v3scale(v1,f))}

pub fn v3lerp<T:Num>(v0:(T,T,T),v1:(T,T,T),f:T)->(T,T,T){ v3add(v0.clone(),v3scale(v3sub(v1,v0.clone()),f))}

pub fn v3sqr<T:Num>(v0:(T,T,T))->T {v3dot(v0.clone(),v0.clone())}
pub fn v3length<T:Float>(v0:(T,T,T))->T { v3sqr(v0).sqrt()}
pub fn v3normalize<T:Float>(v0:(T,T,T))->(T,T,T) { v3scale(v0.clone(),v3sqr(v0.clone()).rsqrt())}
