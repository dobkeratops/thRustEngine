use super::*;
pub use Flow::*;
use std::path;
use ::bsp::bspdraw;

//type sto<T>=Rc<RefCell<T>>;g_mo
pub type sto<T>=Box<T>; //shared trait object

//type sp<T> = Rc<RekeyfCell<T>>;
//fn new_sp<T>()

// glut/whatever interaction, and handles states.
pub type KeyCode=char;

#[derive(Debug,Clone,Copy)]
pub struct KeyAt(pub KeyCode,pub bool,pub ViewPos);
pub type PixelPos=(i32,i32);
pub type PixelVec=(i32,i32);
pub type PixelSize=(i32,i32);
pub type PixelRect=(PixelPos,PixelPos);
pub type ViewPos=(f32,f32);
pub fn screen_size()->PixelPos{ unsafe{g_screensize} }

// example,
//#[derive(Clone,Debug)]
pub enum Command {
    Start,
    Stop,
    Reset,
    Move((f32,f32,f32)),
    Create((f32,f32,f32),String),
}
/// UI events passed to window interfaces.
#[derive(Clone,Debug,Copy)]
pub enum Event {
    NoEvent,
    Activate(),
    Deactivate(),
    KeyEvent(KeyAt),
    MouseMove(ViewPos),
    MouseButton(MouseButtons,bool,ViewPos)
}
pub use self::Event::*;

/// returned value controls flow between states
#[repr(u32)]
pub enum Flow{
    Continue(),
    Redraw(),           /// if not animating, input responses can request redraw
    Info(String),       /// feedback text, otherwise 'continue'
    Push(sto<State>),
    Passthru(),           /// Send command to next, if an overlay.
    Pop(),
    SendToOwner(),
    SendToAll(),
    Replace(sto<State>),
    Overlay(sto<State>),    /// e.g. popup menu
    SetBackground(sto<State>),  /// equivalent to Replace(x)+Overlay(Self)
    SetRoot(sto<State>),
    SwapWith(i32),      /// swap with relative indexed state
    Cycle(),            //
    Back(),            //
    Forward(),            //
    Toggle(),            // rotate top with forward stack
    NewWindow(sto<State>), // multi-windowing
}
pub use self::Flow::*;

pub type KeyMappings = FnMut(KeyCode,&str, &mut FnMut()->Flow);
pub trait State {            //'C' the user defined commands it can respond to.
    fn on_activate(&mut self)   {}
    fn on_deactivate(&mut self) {}
    fn render(&self,t:f32)      {}  // todo render with presel
    fn info(&self)->String      { String::new()}
    fn update(&mut self,dt:f32)->Flow{Flow::Continue()}   // controller access..

    // iterate key mappings, along with functionality, to automate
    // rolling statusbar/tooltips/menu assignment

    fn key_mappings(&mut self, kmf:&mut KeyMappings){}

    fn on_mouse_move(&mut self,xy:ViewPos)->Flow {
        Passthru()
    }
    fn on_mouse_button(&mut self,mb:MouseButtons,s:bool,xy:ViewPos)->Flow {
        Passthru()
    }
    fn on_key(&mut self,kp:KeyAt)->Flow{
        Passthru()
    }
    //fn on_drop(&mut self,f:path::Path,sp:ScreenPos)           {}
    fn command(&mut self, c:Command)->Flow{ Flow::Passthru() }

    // enum of every event,
    // defaults to calling the fn's
    fn event(&mut self,e:Event)->Flow{
        match e{
            Activate()=>{self.on_activate();Continue()},
            Deactivate()=>{self.on_deactivate();Continue()},
            KeyEvent(k)=>self.on_key(k),
            MouseMove(delta)=>self.on_mouse_move(delta),
            MouseButton(mb,s,pos)=>self.on_mouse_button(mb,s,pos),
            _=>Continue(),
        }
    }
}


static mut g_key:[bool;256]=[false;256];
static mut g_mouse_button:i32=0;
static mut g_mouse_pos:PixelPos=(0,0);
static mut g_joystick:((f32,f32),u32)=((0.0f32,0.0f32),0);
static mut g_screensize:PixelPos=(1024,1024);
const  MaxEvent:usize=256;
static mut g_ui_event:[Event;MaxEvent]=[NoEvent;MaxEvent];
static mut g_head:i32=0;
static mut g_tail:i32=0;



fn reshape_func(x:i32,y:i32){
    println!("resizing..{:?} {:?}",x,y);
    unsafe{g_screensize=(x,y); glViewport(0,0,x,y);}

   // panic!();
}
fn keyboard_func_sub(key:u8,isdown:bool,x:i32,y:i32){
    unsafe{
        g_key[key as usize]=isdown;
        let kp=KeyAt(key as KeyCode, isdown, to_viewpos((x,y)));
        push_event(KeyEvent(kp));
    }
}

fn keyboard_func(key:u8, x:i32,y:i32){
    keyboard_func_sub(key,true,x,y);
}
fn special_func_sub(key:GLuint, isdown:bool,x:i32,y:i32){
    assert!((key as u32&0xff) == key as u32);
    unsafe{
        let kp=KeyAt(key as u8 as KeyCode,isdown, to_viewpos((x,y)));
        push_event(KeyEvent(kp));
        g_key[key as usize]=isdown;
    }
}
fn special_func(key:GLuint, x:i32,y:i32){
    special_func_sub(key,true,x,y);
}
fn special_up_func(key:GLuint, x:i32,y:i32){
    special_func_sub(key,false,x,y);
}

//pub fn modifiers()->GLuint{glutGetModifiers()}

fn keyboard_up_func(key:u8, x:i32,y:i32){
    keyboard_func_sub(key,false,x,y);
}
fn divf32(a:i32,b:i32)->f32{a as f32 / b as f32}
fn to_viewpos(s:PixelPos)->ViewPos{
    unsafe{
    (   divf32(s.0,g_screensize.0)*2.0f32-1.0f32,
        -(divf32(s.1,g_screensize.1)*2.0f32-1.0f32)
    )}
}
fn set_mouse_pos(x:i32,y:i32){unsafe{g_mouse_pos=(x,y)};}
fn push_event(e:Event){
    unsafe {
        let next = (g_head + 1) & ((MaxEvent - 1) as i32);
        if next != g_tail {
            g_ui_event[g_head as usize] = e;
            g_head=next;
        } else {
            println!("buffer full,lost event");
        }
    }
}
fn pop_event()->Option<Event>{
    unsafe {
        let next = (g_tail + 1) & ((MaxEvent - 1) as i32);
        if g_tail == g_head {
            None
        } else {
            let e = g_ui_event[g_tail as usize];
            g_tail = next;
            Some(e)
        }
    }
}

fn mouse_func(button:i32,state:i32,x:i32,y:i32){
    unsafe {
        if state!=0{g_mouse_button|=button} else {g_mouse_button&=!button};
    }
    push_event(MouseButton  (
        match button as u32{
            GLUT_LEFT_BUTTON    =>LeftButton,
            GLUT_RIGHT_BUTTON   =>RightButton,
            GLUT_MID_BUTTON =>MidButton,
            GLUT_WHEEL_UP   =>WheelUp,
            GLUT_WHEEL_DOWN =>WheelDown,
            _=>NoButton
        },
        match state as u32{GLUT_DOWN=>true,_=>false},
        to_viewpos((x,y))
    ));
    set_mouse_pos(x,y);
}
fn motion_func(x:i32,y:i32){ set_mouse_pos(x,y);}

fn joystick_func(button:c_uint, dx:c_int,dy:c_int,dz:c_int){
    let s:f32=1.0f32/1024.0f32;
    unsafe{ g_joystick=((dx as f32 * s, dy as f32 * s),button as c_uint);}
    println!("JS:{:?} {:?},{:?},{:?}",button,dx,dy,dz);
}

#[derive(Clone,Debug,Copy)]
#[repr(i32)]
pub enum MouseButtons{
    NoButton=0,LeftButton=0x0001,MidButton=0x0002,RightButton=0x0004,WheelUp=0x0008,WheelDown=0x0010,WheelLeft=0x0020,WheelRight=0x0040
}
pub use self::MouseButtons::*;


type WindowId=c_int;
//static mut g_windows:*mut Vec<(Box<Window>,WindowId)> ;

fn render_begin(){
    unsafe {
        glClearColor(0.5f32, 0.5f32, 0.5f32, 1.0f32);

        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
        glMatrixMode(GL_PROJECTION);
        glLoadIdentity();
        glMatrixMode(GL_MODELVIEW);
        glLoadIdentity();
    }
}
fn render_end(){
    unsafe {
        glFlush();
        glutSwapBuffers();
    }
}
type IsOverlay=bool;
type Windows=(Vec<(Box<State>,IsOverlay)>,WindowId);

fn render_null(){}

fn process_flow(next:Flow,wins:&mut Windows){
// process state flow.
    //assert!(unsafe {glutGetWindow()}==wins.1);

    let top=wins.0 .len();
    if top==0 {return}
    match next {
        //todo - spawn , new glut window.
        Pop() => { wins.0.pop(); }
        Push(w) => { wins.0.push((w, false)); }
        Replace(w) => { wins.0[top - 1] = (w, false); }
        Overlay(w) => { wins.0[top - 1] = (w, true); }
        SwapWith(i)=> wins.0 .swap((top-1) as usize,(top as i32 -1+i) as usize),
        SetBackground(w)=>{ // current becomes an overlay on 'w'
            wins.0.push((w,false));
            let l=wins.0 .len();
            wins.0 .swap(l-1,l-2);

            wins.0[l-2].1=true;
        }
        SetRoot(w) => {
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


fn render_and_update(wins:&mut Windows){
    unsafe{
        //render.
        let top=wins.0 .len();
        if top>0 {
            render_begin();
            let i = top - 1;
            {

                let win = &wins.0[i];
                // if it's an overlay , render previous first.
                if i > 0 && win.1 {
                    // todo- generalize, any number of overlays
                    wins.0[i - 1].0.render(0.0f32);
                }
                win.0.render(0.0f32);
                // check the keymappings,
            }
            {
                let mut y = 0.9f32;
                let win = &mut wins.0[i];
                win.0.key_mappings(&mut move |k, name, _| {
                    bspdraw::char_at(&(-0.95f32, y, 0.0f32), 0x00ff00, k);
                    bspdraw::string_at(&(-0.9f32, y, 0.0f32), 0x00ff00, name);
                    y -= 0.05f32;
                });
            }


            render_end();
        }


        // Update: process all the events.
        while let Some(e)=pop_event() {
            let top=wins.0 .len();
            if top>0 {
                process_flow(
                    {let win = &mut wins.0[top - 1];
                        win.0 .event(e)}
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

fn idle_func(){
    unsafe {glutPostRedisplay(); }
}


// you have to push an initial state, which is a window.
pub fn run_loop(mut w:sto<State>) {
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
        glutReshapeWindow(g_screensize.0,g_screensize.1);

        glutDisplayFunc(render_null as *const u8);
        glutKeyboardFunc(keyboard_func as *const u8);
        glutKeyboardUpFunc(keyboard_up_func as *const u8);
        glutMouseFunc(mouse_func as *const u8);
        glutReshapeFunc(reshape_func as *const u8);
        glutMotionFunc(motion_func as *const u8);
        glutIdleFunc(idle_func as *const u8);
        glutJoystickFunc(joystick_func as *const u8,16);
        glutSpecialFunc(special_func as *const u8);
        glutSpecialUpFunc(special_up_func as *const u8);
        glEnable(GL_DEPTH_TEST);

        glDrawBuffer(GL_BACK);
        glClear(GL_COLOR_BUFFER_BIT|GL_DEPTH_BUFFER_BIT);

        //glutMainLoop();
        while true {
            glutCheckLoop();
            render_and_update(&mut wins);
        }
    }
}

pub fn push(wins:&mut Windows, w:sto<State>){
    unsafe{
        wins.0 .push((w,false));
        wins.1 = glutGetWindow();
    }
}
pub fn key(k:char)->bool{
    unsafe{g_key[k as usize]}
}
