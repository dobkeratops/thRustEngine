use super::*;
pub use Flow::*;
use std::path;

// glut/whatever interaction, and handles states.

type ScreenPos=[i32;2];
static mut g_key:[bool;256]=[false;256];
static mut g_joystick:((f32,f32),u32)=((0.0f32,0.0f32),0);
static mut g_keypress_pos:[i32;2]=[0,0];
static mut g_keypress:Option<u8>=None;
static mut g_screensize:[i32;2]=[1024,1024];
pub fn keyboard(key:u8, x:i32,y:i32){
    unsafe{g_keypress=Some(key); g_key[key as usize]=true;g_keypress_pos=[x,y];}
}
pub fn keyboard_up(key:u8, x:i32,y:i32){
    unsafe{g_key[key as usize]=false;}
}
pub fn on_joystick(button:c_uint, dx:c_int,dy:c_int,dz:c_int){
    let s:f32=1.0f32/1024.0f32;
    unsafe{ g_joystick=((dx as f32 * s, dy as f32 * s),button as c_uint);}
    println!("JS:{:?} {:?},{:?},{:?}",button,dx,dy,dz);
}

// example,
//#[derive(Clone,Debug)]
pub enum Command {
    Start,
    Stop,
    Reset,
    Move((f32,f32,f32)),
    Create((f32,f32,f32),String),
}

pub enum Flow{
    Continue(),
    Redraw(),           /// if not animating, input responses can request redraw
    Info(String),       /// feedback text, otherwise 'continue'
    Push(Box<State>),
    Passthru(),           /// Send command to next, if an overlay.
    Pop(),
    SendToOwner(Command),
    SendToAll(Command),
    Replace(Box<State>),
    Overlay(Box<State>),    /// e.g. popup menu
    SetBackground(Box<State>),  /// equivalent to Replace(x)+Overlay(Self)
    Root(Box<State>),
    SwapWith(i32),      /// swap with relative indexed state
    Cycle(),            //
    Back(),            //
    Forward(),            //
    Toggle(),            // rotate top with forward stack
    Spawn(Box<State>), // multi-windowing
}

pub trait State {
    fn on_activate(&mut self)   {}
    fn on_deactivate(&mut self) {}
    fn render(&self,t:f32)      {}
    fn info(&self)->String      { String::new()}
    fn update(&mut self,dt:f32)->Flow{Flow::Continue()}   // controller access..
    fn on_keypress(&mut self,key:u8,xy:ScreenPos)->Flow{Flow::Passthru()}
    fn on_mousemove(&mut self,xy:ScreenPos)->Flow   {Flow::Passthru()}
    //fn on_drop(&mut self,f:path::Path,sp:ScreenPos)           {}
    fn command(&mut self, c:Command)->Flow{ Flow::Passthru() }
}

type WindowId=c_int;
//static mut g_windows:*mut Vec<(Box<Window>,WindowId)> ;

pub fn render_begin(){
    unsafe {
        glClearColor(0.5f32, 0.5f32, 0.5f32, 1.0f32);

        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    }
}
pub fn render_end(){
    unsafe {
        glFlush();
        glutSwapBuffers();
    }
}
type IsOverlay=bool;
type Windows=(Vec<(Box<State>,IsOverlay)>,WindowId);

pub fn render_null(){}

pub fn process_flow(next:Flow,wins:&mut Windows){
// process state flow.
    //assert!(unsafe {glutGetWindow()}==wins.1);

    let top=wins.0 .len();
    if top==0 {return}
    match next {
        //todo - spawn , new glut window.
        Flow::Pop() => { wins.0.pop(); }
        Flow::Push(w) => { wins.0.push((w, false)); }
        Flow::Replace(w) => { wins.0[top - 1] = (w, false); }
        Flow::Overlay(w) => { wins.0[top - 1] = (w, true); }
        Flow::SwapWith(i)=> wins.0 .swap((top-1) as usize,(top as i32 -1+i) as usize),
        Flow::SetBackground(w)=>{ // current becomes an overlay on 'w'
            wins.0.push((w,false));
            let l=wins.0 .len();
            wins.0 .swap(l-1,l-2);

            wins.0[l-2].1=true;
        }
        Flow::Root(w) => {
            wins.0 .truncate(0);//discard all
            wins.0 .push((w, false));
        }

        Flow::Cycle() => {
            for i in 1..top {
                wins.0.swap(i - 1, i);
            }
        }
        _ => () // default - continue
    }
}


pub fn render_and_update(wins:&mut Windows){
    unsafe{
        //render.
        let top=wins.0 .len();
        if top>0 {
            render_begin();

            let i=top-1;
            let win=&wins.0[i];
            // if it's an overlay , render previous first.
            if i>0 && win.1 {
                // todo- generalize, any number of overlays
                wins.0 [i-1] .0 .render(0.0f32);
            }
            win.0 .render(0.0f32);



            render_end();
        }


        //update
        let top=wins.0 .len();
        if top>0 {

            if let Some(k) = g_keypress {

                process_flow(
                    {let win = &mut wins.0[top - 1];
                        win.0 .on_keypress(k, g_keypress_pos)}
                    ,wins);
            }
        }
        let dt=1.0f32 / 60.0f32;
        let top=wins.0 .len();
        if top>0{
            // overlays still allow base to update.
            if top>1 && wins.0[top-1].1 {
                let flow=wins.0[top-1].0 .update(dt);
                process_flow(flow,wins);
            }
            let flow= {
                let l=wins .0 .len();
                let win = &mut wins.0[l - 1];
                win.0 .update(dt)
            };
            process_flow(flow,wins)
        }
    }
}

// you have to push an initial state, which is a window.
pub fn run_loop(mut w:Box<State>) {
    let mut wins:Windows=(vec![],0);
    push(&mut wins, w);

    unsafe {
        println!("window handler main");
        let mut argc:c_int=0;
        let argv=Vec::<*const c_char>::new();
        glutInit((&mut argc) as *mut c_int,0 as *const *const c_char );

        glutInitDisplayMode(GLUT_DEPTH | GLUT_DOUBLE | GLUT_RGBA);
        let win=glutCreateWindow(c_str("world main loop"));
        //		glewInit(); //TODO- where the hell is glewInit. -lGLEW isn't found
        glDrawBuffer(GL_BACK);
        glutReshapeWindow(g_screensize[0],g_screensize[1]);
        glutDisplayFunc(self::render_null as *const u8);
        glutKeyboardFunc(self::keyboard as *const u8);
        glutKeyboardUpFunc(self::keyboard_up as *const u8);
        glutIdleFunc(super::idle as *const u8);
        glEnable(GL_DEPTH_TEST);
        glutJoystickFunc(self::on_joystick as *const u8,16);

        glDrawBuffer(GL_BACK);
        glClear(GL_COLOR_BUFFER_BIT|GL_DEPTH_BUFFER_BIT);

        //glutMainLoop();
        while true {
            glutCheckLoop();
            render_and_update(&mut wins);
        }
    }
}

pub fn push(wins:&mut Windows, w:Box<State>){
    unsafe{
        wins.0 .push((w,false));
        wins.1 = glutGetWindow();
    }
}
pub fn key(k:char)->bool{
    unsafe{g_key[k as usize]}
}

//pub fn flow_continue()->Flow {Flow::Continue()}
//pub fn flow_replace<X:State>(mut x:Box<X>)->Flow where X:'static{ Flow::Replace(x as Box<window::State>) }
//pub fn flow_push<X:State>(mut x:Box<X>)->Flow where X:'static{ Flow::Push(x as Box<window::State>) }
//pub fn flow_root<X:State>(mut x:Box<X>)->Flow where X:'static{ Flow::Root(x as Box<window::State>) }
//pub fn flow_spawn<X:State>(mut x:Box<X>)->Flow where X:'static{ Flow::Spawn(x as Box<window::State>) }
