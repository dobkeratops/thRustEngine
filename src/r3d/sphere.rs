use super::*;

struct Sphere{
    centre:Vec3,
    radius:f32
}

impl Collide<Sphere> for Sphere {

}
impl Collide<LineSeg> for Sphere{

}
impl Collide<Point> for Sphere{

}

