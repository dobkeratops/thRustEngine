use super::*;
//use ::r3d::draw;
use window as w;
use window::Window as ws;
type V3= (f32,f32,f32);

pub struct FlyMode{
    pos:V3,
    hdg:f32,
}

pub fn new<A:'static>()->sto<window::Window<A>>{
    sto::new(FlyMode{
        pos:(0.0f32,0.0f32,0.0f32),
        hdg:0.0f32
    })
}

trait Render{
    fn sphere(&self,a:&V3,r:f32,c:u32);
    fn line(&self,a:&V3,b:&V3,c:u32);
    fn arrow(&self,a:&V3,b:&V3,r:f32,c:u32);
}

impl<A:'static> window::Window<A> for FlyMode {
    fn on_key(&mut self, a:&mut A, kp:w::KeyAt,wc:&window::WinCursor)->window::Flow<A>{
        println!("{:?}",kp);
        match (kp.0,kp.2) {
            (window::WinKey::KeyCode('e'),KeyDown)=>Flow::Push(editor::make_editor_window::<A,editor::Scene>()),
            _=>Flow::Continue()
        }
    }
    fn update(&mut self,a:&mut A, dt:f32)->window::Flow<A> {
        let r = 30.0f32;

        let movespeed = 0.1f32;
        let rotspeed = 0.025f32;
        //let js = gtexi0_joystick;
        let (mut dx, mut dz) = (0.0f32, 0.0f32);//(js.0 .0, js.0 .1);

        let axiskeys = |a,b|{if w::key(a) { -1.0f32 } else if w::key(b) { 1.0f32 } else { 0.0f32 }};
        self.hdg += axiskeys('a', 'd') * rotspeed;
        dz += axiskeys('s', 'w') * movespeed;
        dx += axiskeys(',', '.') * movespeed;

        let (ax, ay) = (sin(self.hdg), cos(self.hdg));
        self.pos = (self.pos.0 + ax * dz + ay * dx,
                 self.pos.1 + ay * dz - ax * dx,
                 self.pos.2);

        //        let at = Vec3(0.0f32,0.0f32,0.0f32);
        window::Flow::Continue()
    }

    fn render(&self, a:&A, wc:&window::WinCursor) {

        let eye: Vec3 = self.pos.into();
        let (ax, ay) = (sin(self.hdg), cos(self.hdg));
        let at = Vec3(eye.x + ax, eye.y + ay, eye.z);
        let cam = Camera::look_along(
            &Frustum(1.0f32, 1.0f32, (0.1f32, 1000.0f32)),
            &Vec3(eye.x, eye.y, eye.z),
            &Vec3(ax, ay, 0.0f32),
            &Vec3(0.0f32, 0.0f32, 1.0f32));
        unsafe {
            glutSetCursor(GLUT_CURSOR_CROSSHAIR as i32);
            // g_ypos-=0.1f32;

            glSetMatrix(GL_PROJECTION, &cam.projection.ax.x);
            //glMatrixMode(GL_PROJECTION);
           // glLoadMatrixf(&cam.projection.ax.x);
            glSetMatrix(GL_MODELVIEW,&cam.view.ax.x);

			
            //    draw::grid_xz();
            draw::grid_xy_at((0.0f32, 0.0f32, -10.0f32));
            draw::grid_xy_at((0.0f32, 0.0f32, 10.0f32));
            draw::cuboid_aabb_at(&(0.0, 0.0, 0.0), &(1.0, 4.0, 9.0), 0xff00ff00);

            draw::string_at(&(0.0,0.0,0.0),0xff00ffff,"hello world");
            draw::main_mode_text("flymode- use WASD, < >");
        }
    }
}
// numVertex

fn render_scene(r:&Render){
}

