use super::*;
use std::path;

//type sto<T>=Rc<RefCell<T>>;g_mo
pub type sto<T>=Box<T>; //shared trait object
pub type Renderer=();
pub trait TextOutput {

}
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

pub type ViewController = State;
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
    None,
    Render(f32),
    Update(f32),
    Activate(),
    Deactivate(),
    Key(KeyAt),
    Move(ViewPos),
    Clicked(MouseButtons,ViewPos),
    TryBeginDrag(MouseButtons,ViewPos),
    Dragging(MouseButtons,ViewPos,ViewPos,DragMode),
    Dragged(MouseButtons,ViewPos,ViewPos,DragMode),
    Button(MouseButtons,bool,ViewPos),
	DropFile(&'static str)
}

pub type MouseAt=(MouseButtons,ViewPos);

#[repr(u32)]
#[derive(Clone,Debug,Copy)]
pub enum DragMode {
    None,
    Line,
    Rect,
    Freehand,
    Lasso,
    Circle,
    Default,
}
enum Drag {
    Line(ViewPos,ViewPos),
    Rect(ViewPos,ViewPos),
    Circle(ViewPos,ViewPos),
    FreeHand(Vec<ViewPos>),
    Lasso(Vec<ViewPos>)
}

/// returned value controls flow between states
#[repr(u32)]
pub enum Flow{
    Continue(),
    Redraw(),           /// if not animating, input responses can request redraw
    Info(String),       /// feedback text, otherwise 'continue'
    Push(sto<State>),
    PassThru(),           /// Send command to next, if an overlay.
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

pub type Rect=(ViewPos,ViewPos);

pub type KeyMappings = FnMut(KeyCode,&str, &mut FnMut()->Flow);
pub trait State {            //'C' the user defined commands it can respond to.
	fn name(&self)->&str		{"none"}
    fn on_activate(&mut self)   {}
    fn on_deactivate(&mut self) {}
    fn render(&self, _:&RC)      {}  // todo render with mousepos
    fn info(&self)->String      { String::new()}
    fn update(&mut self,dt:f32)->Flow {Flow::Continue()}   // controller access..

    // TODO: all this could be eliminated?
    // just use event() and clients switch.
    // no need to dispatch the fixed set
    // however, there's the nesting issue.

    // iterate key mappings, along with functionality, to automate
    // rolling statusbar/tooltips/menu assignment
    fn key_mappings(&mut self, kmf:&mut KeyMappings){}
    fn on_mouse_move(&mut self, xy:ViewPos)->Flow {
        // TODO - unsure if this is better dispatched by framework.
        /*
        if let Some(p)=unsafe{g_ldrag_start}{
                    self.on_ldragging(p ,xy)
        } else
        if let Some(p)=unsafe{g_rdrag_start}{
            self.on_rdragging(p ,xy)
        } else{
            self.on_passive_move(p,xy)
        }
        */
        Flow::PassThru()
    }
    // hooks for each specific dragable button state,
    // streamline rolling a tool
    fn on_passive_move(&mut self,pos:ViewPos)->Flow{Flow::PassThru()}
    fn on_ldragging(&mut self,start:ViewPos,pos:ViewPos)->Flow{Flow::PassThru()}
    fn on_mdragging(&mut self,start:ViewPos,pos:ViewPos)->Flow{Flow::PassThru()}
    fn on_rdragging(&mut self,start:ViewPos,pos:ViewPos)->Flow{Flow::PassThru()}
    fn on_wheel_up(&mut self,pos:ViewPos)->Flow{Flow::PassThru()}
    fn on_wheel_down(&mut self,pos:ViewPos)->Flow{Flow::PassThru()}
    // click - mouse pressed with no motion
    fn on_lclick(&mut self, p:ViewPos)->Flow{println!("lclick{:?}",p);Flow::PassThru()}
    fn on_rclick(&mut self, p:ViewPos)->Flow{println!("rclick{:?}",p);Flow::PassThru()}
    fn on_mclick(&mut self, p:ViewPos)->Flow{println!("mclick{:?}",p);Flow::PassThru()}

    fn on_mouse_dragged(&mut self, mb:MouseButtons, start:ViewPos, pos:ViewPos, mode:DragMode)->Flow{
        println!("dragged{:?} {:?} {:?}",start,pos,mode);
        Flow::PassThru()
    }

    fn on_mouse_dragging(&mut self, mb:MouseButtons, start:ViewPos, pos:ViewPos, mode:DragMode)->Flow{
        match mb{
            LeftButton => self.on_ldragging(start,pos),
            RightButton => self.on_ldragging(start,pos),
            MidButton => self.on_mdragging(start,pos),
            _=>Flow::PassThru()
        }
    }

    fn on_lbutton_down(&mut self, pos:ViewPos)->Flow{Flow::PassThru()}
    fn on_rbutton_down(&mut self, pos:ViewPos)->Flow{Flow::PassThru()}
    fn on_lbutton_up(&mut self, pos:ViewPos)->Flow{Flow::PassThru()}
    fn on_rbutton_up(&mut self, pos:ViewPos)->Flow{Flow::PassThru()}
    fn on_mbutton_down(&mut self, pos:ViewPos)->Flow{Flow::PassThru()}
    fn on_mbutton_up(&mut self, pos:ViewPos)->Flow{Flow::PassThru()}
    fn on_mouse_button(&mut self, mb:MouseButtons,s:bool,pos:ViewPos)->Flow {
        match (mb,s){
            (MouseButtons::Left,true)=>self.on_lbutton_down(pos),
            (MouseButtons::Left,false)=>self.on_lbutton_up(pos),
            (MouseButtons::Right,true)=>self.on_lbutton_down(pos),
            (MouseButtons::Right,false)=>self.on_rbutton_up(pos),
            (MouseButtons::Mid,true)=>self.on_mbutton_down(pos),
            (MouseButtons::Mid,false)=>self.on_mbutton_up(pos),
            (MouseButtons::WheelUp,true)=>self.on_wheel_up(pos),
            (MouseButtons::WheelDown,true)=>self.on_wheel_down(pos),
			_=>Flow::PassThru()
        }
    }
    fn try_drag(&self,mbpos:(MouseButtons,ViewPos))->DragMode{
        trace!();
        DragMode::Rect
    }
    fn on_click(&mut self, mbvpos:(MouseButtons,ViewPos))->Flow{
        // dispatch to specific button if so desired.
        let (mb,vpos)=mbvpos;
        match mb{
            MouseButtons::Left=>self.on_lclick(vpos),
            MouseButtons::Right=>self.on_rclick(vpos),
            MouseButtons::Mid=>self.on_mclick(vpos),
            _=>{warn!();Flow::Continue()}
        }
    }
    fn on_key_down(&mut self, k:KeyCode, pos:ViewPos)->Flow { Flow::PassThru() }
    fn on_key_up(&mut self, k:KeyCode, pos:ViewPos)->Flow { Flow::PassThru() }
    fn on_key(&mut self, k:KeyAt)->Flow{
        let KeyAt(kc,s,pos)=k;
        //default : route to seperate keydown,keyup
        match s{ true=>self.on_key_down(kc,pos),false=>self.on_key_up(kc,pos)}
    }
    //fn on_drop(&mut self,f:path::Path,sp:ScreenPos)           {}
    fn command(&mut self, c:Command)->Flow{ Flow::PassThru() }

    // enum of every event,
    // defaults to calling the fn's
    fn event(&mut self, e:Event)->Flow{
		use self::Event as Ev;
        match e{
            Ev::Update(dt)  =>{self.update(dt);Flow::Continue()},
            Ev::Render(t)   =>{
				self.render(&RC{rect:((0.0,0.0),(1.0,1.0)),mouse_pos:get_mouse_vpos(), t:0.0f32});
				Flow::Continue()
			},
            Ev::Activate()  =>{self.on_activate();Flow::Continue()},
            Ev::Deactivate()    =>{self.on_deactivate();Flow::Continue()},
            Ev::Key(k)     =>self.on_key(k),
            Ev::Move(pos)  =>self.on_mouse_move(pos),
            Ev::TryBeginDrag(mb,pos)=>{
                match get_dragmode(){
                    DragMode::None=>set_dragmode(self.try_drag((mb,pos))),
                    _=>{}
                };
                Flow::Continue()
            }
            Ev::Clicked(mb,pos)=>self.on_click((mb,pos)),
            Ev::Dragging(mb,start,current,dm) =>self.on_mouse_dragging(mb,start,current,dm),
            Ev::Dragged(mb,start,current,dm)  =>self.on_mouse_dragged(mb,start,current,dm),
            Ev::Button(mb,s,pos)           =>self.on_mouse_button(mb,s,pos),
            _               =>Flow::Continue(),
        }
    }
}
// actually we could make it a mutable trait object taking the message
// and sugar wrapper for


static mut g_key:[bool;256]=[false;256];
static mut g_mouse_button:u32=0;
static mut g_mouse_pos:PixelPos=(0,0);
static mut g_dragmode:DragMode=DragMode::None;
static mut g_ldrag_start:Option<PixelPos>=None;
static mut g_rdrag_start:Option<PixelPos>=None;
static mut g_mdrag_start:Option<PixelPos>=None;
static mut g_drag_points:*mut Vec<ViewPos>=0 as *mut Vec<ViewPos>;
static mut g_joystick:((f32,f32),u32)=((0.0f32,0.0f32),0);
static mut g_screensize:PixelPos=(1024,1024);
const  MaxEvent:usize=256;
static mut g_ui_event:[Event;MaxEvent]=[Event::None;MaxEvent];
static mut g_head:i32=0;
static mut g_tail:i32=0;


mod callbacks {
    use super::*;
    pub fn reshape_func(x: i32, y: i32) {
        println!("resizing..{:?} {:?}", x, y);
        unsafe {
            g_screensize = (x, y);
            glViewport(0, 0, x, y);
        }

        // panic!();
    }

    pub fn keyboard_func_sub(key: u8, isdown: bool, x: i32, y: i32) {
        unsafe {
            g_key[key as usize] = isdown;
            let kp = KeyAt(key as KeyCode, isdown, to_viewpos((x, y)));
            push_event(Event::Key(kp));
        }
    }

    pub fn keyboard_func(key: u8, x: i32, y: i32) {
        keyboard_func_sub(key, true, x, y);
    }

    pub fn special_func_sub(key: GLuint, isdown: bool, x: i32, y: i32) {
        assert!((key as u32 & 0xff) == key as u32);
        unsafe {
            let kp = KeyAt(key as u8 as KeyCode, isdown, to_viewpos((x, y)));
            push_event(Event::Key(kp));
            g_key[key as usize] = isdown;
        }
    }

    pub fn special_func(key: GLuint, x: i32, y: i32) {
        special_func_sub(key, true, x, y);
    }

    pub fn special_up_func(key: GLuint, x: i32, y: i32) {
        special_func_sub(key, false, x, y);
    }

    //pub fn modifiers()->GLuint{glutGetModifiers()}

    pub fn keyboard_up_func(key: u8, x: i32, y: i32) {
        keyboard_func_sub(key, false, x, y);
    }
    pub fn motion_func(x:i32,y:i32) {
        set_mouse_pos(x, y);
        let cvp = to_viewpos((x, y));
        unsafe {
            use self::MouseButtons as MB;
            let dm=g_dragmode;
            push_event(Event::TryBeginDrag(
                match(g_ldrag_start,g_rdrag_start,g_mdrag_start){
                    (Some(_),_,_)=>MB::Left,
                    (None,Some(_),_)=>MB::Right,
                    (None,None,Some(_))=>MB::Mid,
                    (None,None,None)=>{panic!()}
                },
                cvp));

            if let Some(op) = g_ldrag_start {
                push_event(Event::Dragging(MB::Left, to_viewpos(op), cvp,dm))
            }
            if let Some(op) = g_rdrag_start {
                push_event(Event::Dragging(MB::Right, to_viewpos(op), cvp,dm))
            }
            if let Some(op) = g_mdrag_start {
                push_event(Event::Dragging(MB::Mid, to_viewpos(op), cvp,dm))
            }
        }
    }

    pub fn passive_motion_func(x:i32,y:i32){
        set_mouse_pos(x,y);
        push_event(Event::Move(to_viewpos((x,y))));
    }
    pub fn render_null(){}
    pub fn idle_func(){
        unsafe {glutPostRedisplay(); }
    }

    pub fn joystick_func(button:c_uint, dx:c_int,dy:c_int,dz:c_int){
        let s:f32=1.0f32/1024.0f32;
        unsafe{ g_joystick=((dx as f32 * s, dy as f32 * s),button as c_uint);}
        println!("JS:{:?} {:?},{:?},{:?}",button,dx,dy,dz);
    }
    pub unsafe fn mouse_func(button:u32, state:u32, x:i32, y:i32){
        let pos=(x,y);
        let vpos = to_viewpos(pos);
        let oldbs=unsafe{g_mouse_button};
        println!("mouse event {:?} {:?}",state, (x,y));
        dump!(state,x,y);
        if state==GLUT_DOWN{g_mouse_button|=button} else {g_mouse_button&=!button};

        // raw up-down events pass to window,
        push_event(Event::Button  (
            match button as u32{
                GLUT_LEFT_BUTTON    =>MouseButtons::Left,
                GLUT_RIGHT_BUTTON   =>MouseButtons::Right,
                GLUT_MID_BUTTON =>MouseButtons::Mid,
                GLUT_WHEEL_UP   =>MouseButtons::WheelUp,
                GLUT_WHEEL_DOWN =>MouseButtons::WheelDown,
                _=>MouseButtons::None
            },
            match state as u32{GLUT_DOWN=>true,_=>false},
            to_viewpos((x,y))
        ));

        // Now decode a bit more to establish click vs drag.
        if state == GLUT_DOWN {
            // we have seperate l/m/r drags because drags may be combined
            // each button may be pressed or released at different times.
            match button {
                GLUT_LEFT_BUTTON =>g_ldrag_start=Some(pos),
                GLUT_RIGHT_BUTTON =>g_rdrag_start=Some(pos),
                GLUT_MID_BUTTON =>g_mdrag_start=Some(pos)
            }
        } else {
            // Process mouse release: it may be a click or drag
            let (oldpos,mb)=match button{
                GLUT_LEFT_BUTTON=>(&mut g_ldrag_start,MouseButtons::Left),
                GLUT_RIGHT_BUTTON=>(&mut g_rdrag_start,MouseButtons::Right),
                GLUT_MID_BUTTON=>(&mut g_mdrag_start,MouseButtons::Mid),
            };
            if drag_mdist(oldpos.unwrap(),pos)==0{
                push_event(Event::Clicked(mb,vpos))
            } else{
                push_event(Event::Dragged(mb,to_viewpos(oldpos.unwrap()),vpos,g_dragmode))
            }
            // clear the old position.
            *oldpos=Option::None;
            set_dragmode(DragMode::None);
        }

        set_mouse_pos(x,y);
    }
}
fn drag_mdist(a:PixelPos,b:PixelPos)->i32{
    let dx=b.0-a.0;
    let dy=b.1-a.1;
    dx.abs()+dy.abs()
}
fn divf32(a:i32,b:i32)->f32{a as f32 / b as f32}
fn to_viewpos(s:PixelPos)->ViewPos{
    unsafe{
    (   divf32(s.0,g_screensize.0)*2.0f32-1.0f32,
        -(divf32(s.1,g_screensize.1)*2.0f32-1.0f32)
    )}
}
fn set_mouse_pos(x:i32,y:i32){unsafe{g_mouse_pos=(x,y)};}
fn get_mouse_vpos()->ViewPos{unsafe{to_viewpos(g_mouse_pos)}}
fn get_mouse_ppos()->PixelPos{unsafe{g_mouse_pos}}
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


#[derive(Clone,Debug,Copy)]
#[repr(u32)]
pub enum MouseButtons{
    None=0,Left=0x0001,Mid=0x0002,Right=0x0004,WheelUp=0x0008,WheelDown=0x0010,WheelLeft=0x0020,WheelRight=0x0040
}


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


fn process_flow(next:Flow,wins:&mut Windows){
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
        Flow::SetRoot(w) => {
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
fn from_tuple_z((x,y):(f32,f32),z:f32)->Vec3{Vec3(x,y,z)}

pub struct RC {
    rect:Rect,
	mouse_pos:ViewPos,
	t:f32
}
fn set_dragmode(d:DragMode){
    unsafe{ g_dragmode=d;}
}
fn get_dragmode()->DragMode{
    unsafe{g_dragmode}
}
unsafe fn get_drag_points()->&'static mut Vec<ViewPos>{
    if g_drag_points==0 as *mut _{
        let mut f=Box::new( Vec::<ViewPos>::new());
        g_drag_points = &mut *f as _;
        std::mem::forget(f);
    }
    &mut *g_drag_points as _
}

unsafe fn render_drag_overlay(){
    match g_dragmode{
        DragMode::None=>return,
        _=>{},
    }
    let cvp=get_mouse_vpos();

    // clear all rendering states
    draw::identity();
    let sv=from_tuple_z(to_viewpos(g_ldrag_start.unwrap()),0.0);
    let ev=from_tuple_z(cvp.into(),0.0);
    match g_dragmode{
        DragMode::Line=>{
            draw::line(sv, ev);
        },
        DragMode::Lasso=>{
            draw::lines_xy(get_drag_points(), 0.0f32, false);
        }
        DragMode::Freehand=>{
            draw::lines_xy(get_drag_points(), 0.0f32, true);

        }
        DragMode::Default=> {
            draw::line(sv, ev);
            draw::rect_corners_xy(sv, ev, 0.1);
        }
        DragMode::Rect=>{
            draw::rect_outline(sv, ev);
        },
        DragMode::Circle=>{
            draw::circle_xy(sv, (ev-sv).vmagnitude() );
        },
        _=>{},
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
				let rc=RC{rect:((0.0,0.0),(1.0,1.0)),mouse_pos:get_mouse_vpos(),t:0.0f32};
                let win = &wins.0[i];
                // if it's an overlay , render previous first.
                if i > 0 && win.1 {
                    // todo- generalize, any number of overlays
                    wins.0[i - 1].0.render(&rc);
                }
                win.0.render(&rc);
                // check the keymappings,
            }
            {
                let mut y = 0.9f32;
                let win = &mut wins.0[i];
                win.0.key_mappings(&mut move |k, name, _| {
                    draw::char_at(&(-0.95f32, y, 0.0f32), 0x00ff00, k);
                    draw::string_at(&(-0.9f32, y, 0.0f32), 0x00ff00, name);
                    y -= 0.05f32;
                });
            }

            render_drag_overlay();

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

        // glut callback malarchy
        glutDisplayFunc(callbacks::render_null as *const u8);
        glutKeyboardFunc(callbacks::keyboard_func as *const u8);
        glutKeyboardUpFunc(callbacks::keyboard_up_func as *const u8);
        glutMouseFunc(callbacks::mouse_func as *const u8);
        glutReshapeFunc(callbacks::reshape_func as *const u8);
        glutMotionFunc(callbacks::motion_func as *const u8);
        glutPassiveMotionFunc(callbacks::passive_motion_func as *const u8);
        glutIdleFunc(callbacks::idle_func as *const u8);
        glutJoystickFunc(callbacks::joystick_func as *const u8,16);
        glutSpecialFunc(callbacks::special_func as *const u8);
        glutSpecialUpFunc(callbacks::special_up_func as *const u8);
        glEnable(GL_DEPTH_TEST);

        glDrawBuffer(GL_BACK);
        glClear(GL_COLOR_BUFFER_BIT|GL_DEPTH_BUFFER_BIT);

        //glutMainLoop();
        while true {
            glutCheckLoop();//draw

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
