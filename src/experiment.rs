use super::*;

struct FooTest { foo:i32,b:f32}

impl r3d::vector::VecCmpOps for FooTest {
    type CmpOutput=();
    fn vmin(&self,b:&Self)->Self{unimplemented!()}
    fn vmax(&self,b:&Self)->Self{unimplemented!()}
    fn gt(&self,b:&Self)->Self::CmpOutput{unimplemented!()}
    fn lt(&self,b:&Self)->Self::CmpOutput{unimplemented!()}

}

// desktop main.


macro_rules! for_each_component {($x:ident)=>{
	$x!(hello)
	$x!(goodbye)
}}

macro_rules! try_it {($a:ident)=>{
	println!(stringify!($a));
}}
fn foo_bar(){println!("foo_bar\n")}


// zipWith::(a->b->c)  ->  s a  ->  s b  ->  s c
trait ZipWith<'a,'b,B,F> {
    type Other; type OtherElem; type Output; type Elem; type OutElem;
    fn zipWith(&'a self, b:&'b Self::Other,f:F)->Self::Output;
}

impl<'a,'b,A,B,C,F> ZipWith<'a,'b,B,F> for Vec<A> where F:Fn(&'a A,&'b B)->C,A:'a, B:'b{
    type Other=Vec<B>;
    type Elem=A;
    type OtherElem=B;
    type Output=Vec<C>;
    type OutElem=C;

    fn zipWith(&'a self,b:&'b Vec<B>,f:F)->Vec<C>
    //		where F:Fn(&A,&B)->C
    {
        self.iter().zip(b.iter()).map(|(x,y)|f(x,y)).collect::<Vec<_>>()
    }
}

struct ActorId(i32);

struct PointMass {
    id:ActorId,
    pos:Vec3,
    vec:Vec3,
    radius:f32,
    mass:f32,
}
struct Frame {
    id:ActorId,
    matrix:Mat43f,
    vel:Vec3f,
    angvel:Vec3f,
    flags:u32,		// universal flags
}

enum Flags {	// universal flags
    RENDERABLE=0x0001,			// compared to triggers etc
    COLLIDEABLE=0x0002,
    PLAYER1=0x0010,
    PLAYER=0x00f0,		// 4 bits ..16players
    EDIBLE=0x0100,
    DANGEROUS=0x0200,
}

// define the ecs
macro_rules! ecs_world {
	(
		{ $($event_name:ident $args:tt),* }
		$(	$component_name:ident {
				// [1] fields
				$($field_name:ident : $fty:ty = $field_init:expr)*
			}
			// event handlers
			[
				$($hname:ident => $b:block)*
			]
		)*
	)
	=>{
		//struct World
	}
}

// roll example
ecs_world!{
	{update(),activate(),render()}
	Entity {
		//fields
		matrix:Mat43f	=matrix::identiy()
		vel:Vec3f		=Vec3(0.0f32,0.0f32,0.0f32)
		angvel:Vec3f	=Vec3(0.0f32,0.0f32,0.0f32)
		radius:f32		=0.0f32
	}
	// event handlers
	[
		update => {
			println!("entity update {:?}",self.matrix);
		}
		activate => {
			println!("entity update {:?}",self.matrix);
		}
	]
}

class!{
    Foo(args) {
        //fields and initializers
    }
    window::Flow
}

/// macro for rolling boxed lambda
macro_rules! boxfn{

    // boxfn!{args=>expression}
    {$($arg:ident),* => $($a:stmt);* }
    => {Box::new(move |$($arg),*|{$($a);* })};

    // statement block with no arguments
    { $($a:stmt);* }
    => {Box::new(move ||{$($a);* })};

    //{$($args:ident),* => $e:block}
    //=> {Box::new(move |$($args),*| $e )}
}


fn foo()->Box<Fn(i32,i32)->i32> {
    let b=boxfn!{ println!("no arg lambda from this macro");let x=0;x};
    b();
    boxfn! {x,y => let z=x+y;println!("hi from macro'd lambda\n");z}
}


fn compare<'x>(a:&'x usize,b:&'x usize, c:&'x usize)->&'x usize{
    if *a==*b {b} else {c}
}

use r3d::vector::Min;
fn try_math(){
    {
        let (x,y,z)=(1,2,3);
        let p=compare(&x,&y,&z);
        let (f0,f1,f2)=(Foo(&x,&y),Foo(&x,&z),Foo(&y,&z));
        let p1=f0.compare2(&x,&y);
    }

    let a=Vec3(1.0f32,2.0f32,3.0f32);
    let b=Vec3(10.0f32,10.0f32,10.0f32);
    let bb=Vec3(5.0f32,10.0f32,-10.0f32);
    //	let c=vecmath::lerp_ref(&a,&b,0.5);
    let c=lerp((a,b),0.5f32);
    let d=a*b;
    let e=a/b;
    let v = (&a - &b).normalize();
    let a32= Vec3(1.0f32,1.0f32,1.0f32);
    //	let a64:Vec3<f64> = a32.into();

    println!("{:?}" , a.magnitude());
    println!("{:?}" , inv_lerp((10.0f32,20.0f32),15.0f32));

    println!("{:?}", v );
    println!("{:?}", a.x.min(a.y) );
    println!("{:?}", a.min(b) );
    println!("{:?}", 1.0f32.reciprocal());
    println!("{:?}", a.dot(b));
    println!("{:?}", a.normalize());
    println!("{:?}", a..b);
    println!("{:?}", a.pair_with(b).lerp_by(0.5));
    //	println!("{:?}", a>b);
    println!("{:?}", a.sub_norm(b) );
    let (norm,len)=a.normal_and_length();
    let n=a.triangle_normal(b,bb);
    let (para,perp)=a.para_perp_of(&b);


    let v0: vector::Point<Vec3<f32>> = vector::Point(Vec3(0.1f32, 0.2f32, 0.3f32));
    let v1: vector::Point<Vec3<f64>> = v0.into();
    let ar0 = vec![1,2,3,4];
    let f=FooTest{foo:10,b:0.0};

    let v3=Vec3(0.0f32,0.0f32,0.0f32);
}


