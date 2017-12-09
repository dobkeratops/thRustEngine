use super::*;
use std::path;


//type sto<T>=Rc<RefCell<T>>;g_mo
pub type sto<T>=Box<T>; //might be shared trait object
pub type Renderer=();
pub trait TextOutput {

}
//type sp<T> = Rc<RekeyfCell<T>>;
//fn new_sp<T>()

// glut/whatever interaction, and handles states.
pub type KeyCode=char;
pub use self::State as Window; // rename imminent

#[derive(Debug,Clone,Copy)]
pub enum KeyTransition{
    KeyDown,KeyUp
}
pub use self::KeyTransition::*;

pub type Modifiers=u32;
#[derive(Debug,Clone,Copy)]
pub struct KeyAt(pub KeyCode,pub Modifiers, pub KeyTransition,pub ViewPos);
pub type PixelPos=(i32,i32);
pub type PixelVec=(i32,i32);
pub type PixelSize=(i32,i32);
pub type PixelRect=(PixelPos,PixelPos);
pub type ViewPos=(f32,f32);
pub fn screen_size()->PixelPos{ unsafe{g_screensize} }

pub type ViewController<A> = State<A>;
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

static mut g_firstdrag:bool=false;
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
pub enum Flow<A>{
    Continue(),
    Redraw(),           /// if not animating, input responses can request redraw
    Info(String),       /// feedback text, otherwise 'continue'
    Push(sto<State<A>>),
    PassThru(),           /// Send command to next, if an overlay.
    Pop(),
    SendToOwner(),
    SendToAll(),
    Replace(sto<State<A>>),
    Overlay(sto<State<A>>),    /// e.g. popup menu
    SetBackground(sto<State<A>>),  /// equivalent to Replace(x)+Overlay(Self)
    SetRoot(sto<State<A>>),
    SwapWith(i32),      /// swap with relative indexed state
    Cycle(),            //
    Back(),            //
    Forward(),            //
    Toggle(),            // rotate top with forward stack
    NewWindow(sto<State<A>>), // multi-windowing
}

pub type Rect=(ViewPos,ViewPos);

pub enum ForeachResult{
    Continue,
    Quit,
}


pub type KeyMappings<A> = FnMut(KeyCode,&str, &mut FnMut()->Flow<A>);
pub trait State<A> {            //'C' the user defined commands it can respond to.

    fn ask_size(&self)->Option<ViewPos> { None }
	fn name(&self)->&str		{"none"}
    fn on_activate(&mut self, app:&mut A)   {}
    fn on_deactivate(&mut self, app:&mut A) {}
    fn render(&self,a:&A, _:&RC)      {}  // todo render with mousepos
    fn info(&self)->String      { String::new()}
    fn update(&mut self,app:&mut A,dt:f32)->Flow<A> {Flow::Continue()}   // controller access..

    //todo on_enter/on_exit


    // TODO: all this could be eliminated?
    // just use event() and clients switch.
    // no need to dispatch the fixed set
    // however, there's the nesting issue.

    // iterate key mappings, along with functionality, to automate
    // rolling statusbar/tooltips/menu assignment
    fn key_mappings(&mut self,app:&A, kmf:&mut KeyMappings<A>){}
    fn on_mouse_move(&mut self, app:&mut A,opos:ViewPos, xy:ViewPos)->Flow<A> {
        // TODO - unsure if this is better dispatched by framework.

        if let Some(ds)=unsafe{g_ldrag_start} {
            unsafe {

                if g_firstdrag {
                    println!("first drag begin\n");
                    self.on_ldrag_begin(app, to_viewpos(ds), xy);
                    g_firstdrag = false;
                }
            }
            self.on_ldragging(app, to_viewpos(ds) ,xy)
        } else
        if let Some(ds)=unsafe{g_rdrag_start}{
            unsafe {
                if g_firstdrag {
                    self.on_rdrag_begin(app, to_viewpos(ds), xy);
                    unsafe { g_firstdrag = false; }
                }
            }
            self.on_rdragging(app, to_viewpos(ds) ,xy)
        } else{
            self.on_passive_move(app,opos,xy)
        }
    }
    // hooks for each specific dragable button state,
    // streamline rolling a tool
    fn on_passive_move(&mut self,app:&mut A, opos:ViewPos, pos:ViewPos)->Flow<A>{Flow::PassThru()}
    // first frame of dragging might have special behaviour; by default, just issue normal move
    fn on_ldrag_begin(&mut self,app:&mut A, start:ViewPos, pos:ViewPos)->Flow<A>{Flow::PassThru()}
    fn on_mdrag_begin(&mut self,app:&mut A, start:ViewPos, pos:ViewPos)->Flow<A>{Flow::PassThru()}
    fn on_rdrag_begin(&mut self,app:&mut A, start:ViewPos, pos:ViewPos)->Flow<A>{Flow::PassThru()}
    fn on_ldrag_end(&mut self, app:&mut A, start:ViewPos,end:ViewPos)->Flow<A>{Flow::PassThru()}
    fn on_mdrag_end(&mut self, app:&mut A, start:ViewPos,end:ViewPos)->Flow<A>{Flow::PassThru()}
    fn on_rdrag_end(&mut self, app:&mut A, start:ViewPos,end:ViewPos)->Flow<A>{Flow::PassThru()}
    fn on_ldragging(&mut self,app:&mut A,start:ViewPos,pos:ViewPos)->Flow<A>{Flow::PassThru()}
    fn on_mdragging(&mut self,app:&mut A,start:ViewPos,pos:ViewPos)->Flow<A>{Flow::PassThru()}
    fn on_rdragging(&mut self,app:&mut A,start:ViewPos,pos:ViewPos)->Flow<A>{Flow::PassThru()}
    fn on_wheel_up(&mut self, app:&mut A,pos:ViewPos)->Flow<A>{Flow::PassThru()}
    fn on_wheel_down(&mut self, app:&mut A,pos:ViewPos)->Flow<A>{Flow::PassThru()}
    // click - mouse pressed with no motion
    fn on_lclick(&mut self,app:&mut A, p:ViewPos)->Flow<A>{println!("lclick{:?}",p);Flow::PassThru()}
    fn on_rclick(&mut self,app:&mut A, p:ViewPos)->Flow<A>{println!("rclick{:?}",p);Flow::PassThru()}
    fn on_mclick(&mut self,app:&mut A, p:ViewPos)->Flow<A>{println!("mclick{:?}",p);Flow::PassThru()}

    fn on_mouse_dragged(&mut self, app:&mut A, mb:MouseButtons, start:ViewPos, pos:ViewPos, mode:DragMode)->Flow<A>{
        println!("dragged{:?} {:?} {:?}",start,pos,mode);
        match mb {
            LeftButton=>self.on_ldrag_end(app, start, pos),
            MiddleButton=>self.on_mdrag_end(app, start, pos),
            RightButton=>self.on_rdrag_end(app, start, pos),


            _=>Flow::PassThru()
        }
    }

    fn on_mouse_dragging(&mut self, app:&mut A, mb:MouseButtons, start:ViewPos, pos:ViewPos, mode:DragMode)->Flow<A>{

        match mb{
            LeftButton =>{
                if unsafe{g_firstdrag} {
                    println!("first drag begin\n");
                    self.on_ldrag_begin(app, start, pos);
                    unsafe{g_firstdrag = false;}
                }
                self.on_ldragging(app,start,pos)
            },
            RightButton => self.on_ldragging(app,start,pos),
            MidButton => self.on_mdragging(app,start,pos),
            _=>Flow::PassThru()
        }
    }

    fn on_lbutton_down(&mut self,app:&mut A, pos:ViewPos)->Flow<A>{Flow::PassThru()}
    fn on_rbutton_down(&mut self,app:&mut A,  pos:ViewPos)->Flow<A>{Flow::PassThru()}
    fn on_lbutton_up(&mut self,app:&mut A,  pos:ViewPos)->Flow<A>{
        match unsafe{g_ldrag_start} {
            Some(prev)=>{println!("lbutton up (dragged)");}
            None=>{println!("lbutton up (no drag)");}
        }
        Flow::PassThru()
    }
    fn on_rbutton_up(&mut self,app:&mut A,  pos:ViewPos)->Flow<A>{Flow::PassThru()}
    fn on_mbutton_down(&mut self,app:&mut A,  pos:ViewPos)->Flow<A>{Flow::PassThru()}
    fn on_mbutton_up(&mut self,app:&mut A,  pos:ViewPos)->Flow<A>{Flow::PassThru()}
    fn on_mouse_button(&mut self,app:&mut A,  mb:MouseButtons,s:bool,pos:ViewPos)->Flow<A> {
        match (mb,s){
            (MouseButtons::Left,true)=>self.on_lbutton_down(app,pos),
            (MouseButtons::Left,false)=>self.on_lbutton_up(app,pos),
            (MouseButtons::Right,true)=>self.on_lbutton_down(app,pos),
            (MouseButtons::Right,false)=>self.on_rbutton_up(app,pos),
            (MouseButtons::Mid,true)=>self.on_mbutton_down(app,pos),
            (MouseButtons::Mid,false)=>self.on_mbutton_up(app,pos),
            (MouseButtons::WheelUp,true)=>self.on_wheel_up(app,pos),
            (MouseButtons::WheelDown,true)=>self.on_wheel_down(app,pos),
			_=>Flow::PassThru()
        }
    }
    fn try_drag(&self, app:&A,mbpos:(MouseButtons,ViewPos))->DragMode{
        trace!();
        DragMode::Rect
    }
    fn on_click(&mut self, app:&mut A, mbvpos:(MouseButtons,ViewPos))->Flow<A>{
        // dispatch to specific button if so desired.
        let (mb,vpos)=mbvpos;
        match mb{
            MouseButtons::Left=>self.on_lclick(app,vpos),
            MouseButtons::Right=>self.on_rclick(app,vpos),
            MouseButtons::Mid=>self.on_mclick(app,vpos),
            _=>{warn!();Flow::Continue()}
        }
    }
    fn on_key_down(&mut self,app:&mut A,  k:KeyCode, modifiers:u32, pos:ViewPos)->Flow<A> { Flow::PassThru() }
    fn on_key_up(&mut self, app:&mut A,k:KeyCode, pos:ViewPos)->Flow<A> { Flow::PassThru() }
    fn on_key(&mut self,app:&mut A,k:KeyAt)->Flow<A>{
        let KeyAt(keycode,modk,state,pos)=k;
        //default : route to seperate keydown,keyup
        match state{
            KeyDown=>self.on_key_down(app,keycode,modk,pos),
            KeyUp=>self.on_key_up(app,keycode,pos)}
    }
    fn on_drop(&mut self, f:&str, sp:ViewPos)           {}
    fn command( &mut self, app:&mut A,c:Command)->Flow<A>{ Flow::PassThru() }

    // enum of every event,
    // defaults to calling the fn's
    fn event_dispatch(&mut self, app:&mut A, e:Event)->Flow<A>{
        match e{
            Event::Update(dt)  =>{self.update(app,dt);Flow::Continue()},
            Event::Render(t)   =>{
                self.render(app,
					&RC{
						rect:((0.0,0.0),(1.0,1.0)),
						mouse_old_pos:get_mouse_ovpos(),
						mouse_pos:get_mouse_vpos(), 
						t:0.0f32});
                Flow::Continue()
            },
            Event::Activate()  =>{self.on_activate(app);Flow::Continue()},
            Event::Deactivate()    =>{self.on_deactivate(app);Flow::Continue()},
            Event::Key(k)     =>self.on_key(app,k),
            Event::Move(pos)  =>self.on_mouse_move(app,get_mouse_ovpos(), pos),
            Event::TryBeginDrag(mb,pos)=>{
                match get_dragmode(){
                    DragMode::None=>set_dragmode(self.try_drag(app,(mb,pos))),
                    _=>{}
                };
                Flow::Continue()
            }
            Event::Clicked(mb,pos)=>self.on_click(app,(mb,pos)),
            Event::Dragging(mb,start,current,dm) =>self.on_mouse_dragging(app,mb,start,current,dm),
            Event::Dragged(mb,start,current,dm)  =>self.on_mouse_dragged(app,mb,start,current,dm),
            Event::Button(mb,s,pos)           =>self.on_mouse_button(app,mb,s,pos),
            _               =>Flow::Continue(),
        }
    }
    // main event dispatch: override to iter sub
    fn event(&mut self,app:&mut A, e:Event, r:Rect)->Flow<A>{
		use self::Event as Ev;
        self.event_dispatch(app,e)
    }
    //fn foreach_child(&self, r:Option<Rect>, f:&FnMut(&State<A>,Option<Rect>)->ForeachResult);
    //fn foreach_child_mut(&mut self, r:Option<Rect>, f:&FnMut(&mut State<A>,Option<Rect>)->ForeachResult);

}
// actually we could make it a mutable trait object taking the message
// and sugar wrapper for


// implement this to open a 'gridview' on something
// texture-palette, toolbox etc.
type Index=i32;


pub trait WindowContainer<A> {
	fn sub_win_count(&self)->usize;
    fn sub_win_get(&self,i:usize)->Option<&window::State<A>>;
    fn sub_win_set(&mut self, i:usize, optbox<window::State<A>>);
	fn sub_win_push(&mut self, optbox<window::State<A>>);
    fn sub_win_take(&self,i:usize)->optbox<window::State<A>>;
}

//impl<A> HasCells<A> for WindowContainer<A> {
//}
/*
pub trait HasCells<A> {
	fn render_cell(&self, i:usize, r:RenderContext);
	fn num_cells(&self)->usize;
}

pub trait TabView<A> : WindowContainer<A>{
}


pub trait GridView<A> : HasCells<A>{
    fn select(item:Index);
    fn scroll(&mut self, dxy:(i32,i32));
    fn grid_size(&self,r:Rect)->(Index,Index);
}
*/
/*
impl<T:A> window::State<A> for T:GridView<A> {
    // render grid..
    // click down , ..
    //event -
}
*/
/*
inheritance:-
 class window
   class gridView : window
     class mygridview..

rustic:-
impl window for T:gridview
impl gridview for mygridview  // gets 'window'

*/


static mut g_key:[bool;256]=[false;256];
static mut g_mouse_button:u32=0;
static mut g_mouse_pos:PixelPos=(0,0);
static mut g_mouse_opos:PixelPos=(0,0);
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
pub const CTRL:u32=0x0001;
pub const SHIFT:u32=0x0002;
pub const ALT:u32=0x0004;

fn get_modifiers()->u32{
	unsafe {let modk:u32=glutGetModifiers() as u32;	
	let ret=0u32
	|if modk & GLUT_ACTIVE_SHIFT!=0{SHIFT}else{0}
	|if modk & GLUT_ACTIVE_CTRL!=0{CTRL}else{0}
	|if modk & GLUT_ACTIVE_ALT!=0{ALT}else{0};
	
	dump!(modk,ret);
	ret}
}
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
            let kp = KeyAt(key as KeyCode,get_modifiers(), if isdown{KeyDown}else{KeyUp}, to_viewpos((x, y)));
            push_event(Event::Key(kp));
        }
    }

    pub fn keyboard_func(key: u8, x: i32, y: i32) {
        keyboard_func_sub(key, true, x, y);
    }

    pub fn special_func_sub(key: GLuint, isdown: bool, x: i32, y: i32) {
        assert!((key as u32 & 0xff) == key as u32);
        unsafe {
            let kp = KeyAt(key as u8 as KeyCode,get_modifiers(), if isdown{KeyDown}else{KeyUp}, to_viewpos((x, y)));
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
        //println!("mouse event {:?} {:?} fd={:?}",state, (x,y),unsafe{g_firstdrag});
        //dump!(state,x,y);
        if state==GLUT_DOWN{
            g_mouse_button|=button;
            unsafe{
                g_firstdrag=true;
            }
        }
            else {g_mouse_button&=!button};

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
fn set_mouse_pos(x:i32,y:i32){unsafe{g_mouse_opos=g_mouse_pos; g_mouse_pos=(x,y)};}
fn get_mouse_vpos()->ViewPos{unsafe{to_viewpos(g_mouse_pos)}}
fn get_mouse_ovpos()->ViewPos{unsafe{to_viewpos(g_mouse_opos)}}
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
type Windows<A>=(Vec<(Box<State<A>>,IsOverlay)>,WindowId);


fn process_flow<APP>(next:Flow<APP>,wins:&mut Windows<APP>){
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
    pub rect:Rect,
	pub mouse_old_pos:ViewPos,
	pub mouse_pos:ViewPos,
	pub t:f32
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
fn render_and_update<APP>(wins:&mut Windows<APP>,a:&mut APP){
    let whole_rect = ((0.0,0.0),(1.0,1.0));
    unsafe{
        //render.
        let top=wins.0 .len();
        if top>0 {
            render_begin();
            let i = top - 1;
            {
				let rc=RC{rect:((0.0,0.0),(1.0,1.0)),mouse_pos:get_mouse_vpos(),mouse_old_pos:get_mouse_ovpos(),t:0.0f32};
                let win = &wins.0[i];
                // if it's an overlay , render previous first.
                if i > 0 && win.1 {
                    // todo- generalize, any number of overlays
                    wins.0[i - 1].0.render(a,&rc);
                }
                win.0.render(a,&rc);
                // check the keymappings,
            }
            {
                let mut y = 0.9f32;
                let win = &mut wins.0[i];
                win.0.key_mappings(a,&mut move |k, name, _| {
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
                        win.0 .event(a,e,whole_rect)}
                    ,wins);
            }
        }
        let dt=1.0f32 / 60.0f32;
        let top=wins.0 .len();
        if top>0{
            // overlays still allow base to update.
            if top>1 && wins.0[top-1].1 {
                let flow=wins.0[top-1].0 .update(a,dt);
                process_flow(flow,wins);
            }
            let flow= {
                let l=wins .0 .len();
                let win = &mut wins.0[l - 1];
                win.0 .update(a,dt)
            };
            process_flow(flow,wins)
        }
    }
}


// you have to push an initial state, which is a window.
pub fn run_loop<APP>(mut w:sto<State<APP>>, app:&mut APP) {
    let mut wins:Windows<APP>=(vec![],0);
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
        loop {
            glutCheckLoop();//draw

            render_and_update(&mut wins,app);
        }
    }
}

pub fn push<A>(wins:&mut Windows<A>, w:sto<State<A>>){
    unsafe{
        wins.0 .push((w,false));
        wins.1 = glutGetWindow();
    }
}
pub fn key(k:char)->bool{
    unsafe{g_key[k as usize]}
}
