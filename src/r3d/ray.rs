use super::*

pub struct Ray {
    pub start: Normal;
    pub dir: Vec3;
    pub tmax:f32;
}

pub struct RayHit{
    pub t:f32,
    pub point:Vec3;
    pub normal:Vec3;
}

pub struct RaySeg {
    pub ray:Ray,
    pub tmin:f32,
    pub tmax:f32
}

pub fn create(s:Vec3, e:Vec3)->Ray{
    let dir=e.vsub(&s);
    let len=dir.vlength();
    Ray{start:s, dir:dir.vscale(1.0f32/len)}
}

/// ray vs plane intersection
/// TODO ... not so sure about 'optionals' for maths case.
/// we may want to keep values in the vector pipeline,
/// can we look into solutions like optvec .. optfloat..?

impl Ray {
    fn point(&self, t:f32)->Vec3 {self.start.vmadd(self.dir,t)}
    fn intersect_plane(&self, p:&Plane) -> Option<RayHit> {
        let dp=r.dir;
        if  dp<epsilon {None}
        else{
            // r dot n = p   (r-center) dot n = 0
            //(a+td-center)dot n = 0
            // t= -(a-center) dot n / d dot n
            let t=-(self.start.vsub(&p.centre).vdot(&p.normal) / dp)
            Some(RayHit{t:t, point:_{r.point(t),normal:p.normal })
        }
    }
}



