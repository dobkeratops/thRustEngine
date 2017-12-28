use super::*;
use r3d::*;
use std::path;
// todo - rename APP as owner
// application owns frame window
// frame window owns panes
// etc
//
// Window Framework and a few (hopefully generally useful) windows.
// glRasterPos3f
//type sto<T>=Rc<RefCell<T>>;g_mo
pub type sto<T>=Box<T>; //might be shared trait object
pub type Renderer=();
pub trait TextOutput {

}
//type sp<T> = Rc<RekeyfCell<T>>;
//fn new_sp<T>()

// glut/whatever interaction, and handles states.
pub type KeyCode=char;

#[derive(Debug,Clone,Copy)]
pub enum KeyTransition{
    KeyDown,KeyUp
}
pub use self::KeyTransition::*;

#[derive(Debug,Clone,Copy)]
pub enum WinKey{
    KeyCode(char),KeyF1,KeyF2,KeyF3,KeyF4,KeyF5,KeyF6,KeyF7,KeyF8,KeyF9,
    KeyF10,KeyF11,KeyF12, KeyCursorLeft,KeyCursorUp,KeyCursorDown,KeyCursorRight, KeyPageUp,KeyPageDown,KeyHome,KeyEnd,KeyInsert
}

pub type Modifiers=u32;
#[derive(Debug,Clone,Copy)]
pub struct KeyAt(pub WinKey,pub Modifiers, pub KeyTransition,pub ScreenPos);
impl KeyAt{ pub fn pos(&self)->ScreenPos{self.3}}
pub type PixelPosi=(i32,i32);
pub type PixelPosf=(f32,f32);
pub type PixelVec=(i32,i32);
pub type PixelSizei=(i32,i32);
pub type PixelSizef=(f32,f32);
pub type PixelRectf=(PixelPosf,PixelPosf);
pub type PixelRecti=(PixelPosi,PixelPosi);
pub type ScreenPos=Vec2f;
#[derive(Debug,Clone,Copy)]
pub struct ScrPos(pub f32,pub f32);

impl HasElem for ScrPos{
	type Elem=f32;
}
impl HasXY for ScrPos{
    fn x(&self)->Self::Elem{self.0}
    fn y(&self)->Self::Elem{self.1}
    fn from_xy(x:Self::Elem, y:Self::Elem)->Self{ScrPos(x,y)}
}
fn foo(){
    let a=ScrPos(1.0,2.0);
    let b=v2add(&a,&a);
    //let b=v2add(&a);
}


pub type ScreenRect=Extents<ScreenPos>;
pub fn screen_sizei()->PixelPosi{ unsafe{g_screen_pixel_sizei} }
pub fn screen_sizef()->PixelPosf{ unsafe{g_screen_pixel_sizef} }

pub type ViewController<A> = Window<A>;
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
    Move(ScreenPos),
    Clicked(MouseButtons,ScreenPos),
    TryBeginDrag(MouseButtons,ScreenPos),
    Dragging(MouseButtons,ScreenPos,ScreenPos,DragMode),
    Dragged(MouseButtons,ScreenPos,ScreenPos,DragMode),
    Button(MouseButtons,bool,ScreenPos),
	DropFile(&'static str)
}
impl Event{
    fn pos(&self)->Option<ScreenPos>{
        match self{
            &Event::None=>None,
            &Event::Render(f32)=>None,
            &Event::Update(f32)=>None,
            &Event::Activate()=>None,
            &Event::Deactivate()=>None,
            &Event::Key(kc)=>Some(kc.pos()),
            &Event::Move(pos)=>Some(pos),
            &Event::Clicked(_,pos)=>Some(pos),
            &Event::TryBeginDrag(_,pos)=>Some(pos),
            &Event::Dragging(_,_,pos,_)=>Some(pos),
            &Event::Dragged(_,_,pos,_)=>Some(pos),
            &Event::Button(_,_,pos)=>Some(pos),
            &Event::DropFile(_)=>None,//TODO - dropfile needs pos!
        }
    }
}

pub type MouseAt=(MouseButtons,ScreenPos);

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
    Line(ScreenPos,ScreenPos),
    Rect(ScreenPos,ScreenPos),
    Circle(ScreenPos,ScreenPos),
    FreeHand(Vec<ScreenPos>),
    Lasso(Vec<ScreenPos>)
}

/// returned value controls flow between states
#[repr(u32)]
pub enum Flow<A>{
    Continue(),
    Redraw(),           /// if not animating, input responses can request redraw
    Info(String),       /// feedback text, otherwise 'continue'
    Push(sto<Window<A>>),
    PassThru(),           /// Send command to next, if an overlay.
    Pop(),
    SendToOwner(),
    SendToAll(),
    Replace(sto<Window<A>>),
    Overlay(sto<Window<A>>),    /// e.g. popup menu
    SetBackground(sto<Window<A>>),  /// equivalent to Replace(x)+Overlay(Self)
    SetRoot(sto<Window<A>>),
    SwapWith(i32),      /// swap with relative indexed state
    Cycle(),            //
    Back(),            //
    Forward(),            //
    Toggle(),            // rotate top with forward stack
    NewWindow(sto<Window<A>>), // multi-windowing
}

fn split_x(r:&Rect,f0:f32,f1:f32)->Rect {
    let size=r.size();
    Extents(&Vec2(r.min.x+size.x*f0, r.min.y),&Vec2(r.min.x+size.x*f1, r.max.y))
}
fn split_y(r:&Rect,f0:f32,f1:f32)->Rect {
    let size=r.size();
    Extents(&Vec2(r.min.x, r.min.y+size.y*f0),&Vec2(r.max.x, r.min.y+size.y*f1))
}

pub enum ForeachResult{
    Continue,
    Quit,
}

pub struct Dragging{
    pub start:ScreenPos,
    pub delta:ScreenPos,
    pub pos:ScreenPos,
}
pub type KeyMappings<A> = FnMut(KeyCode,&str, &mut FnMut()->Flow<A>);
pub trait Window<A> {            //'C' the user defined commands it can respond to.

    fn ask_size(&self)->Option<ScreenPos> { None }
	fn name(&self)->&str		{"none"}
    fn on_activate(&mut self, app:&mut A)   {}
    fn on_deactivate(&mut self, app:&mut A) {}
    fn render(&self,a:&A, _:&WinCursor)      {}  // todo render with mousepos
    fn info(&self)->String      { String::new()}
    fn update(&mut self,app:&mut A,dt:f32)->Flow<A> {Flow::Continue()}   // controller access..

    //todo on_enter/on_exit


    // TODO: all this could be eliminated?
    // just use event() and clients switch.
    // no need to dispatch the fixed set
    // however, there's the nesting issue.

    // iterate key mappings, along with functionality, to automate
    // rolling statusbar/tooltips/menu assignment
    fn key_mappings(&mut self,owner:&mut A, kmf:&mut KeyMappings<A>){}
    fn on_mouse_move(&mut self, owner:&mut A, wc:&WinCursor)->Flow<A> {
        // TODO - unsure if this is better dispatched by framework.
        let delta=v2sub(&wc.pos, &wc.old_pos);

        if let Some(ds)=unsafe{g_ldrag_start} {
            let d=Dragging{
                start: to_screenpos(ds),
                delta: delta,
                pos:wc.pos
            };

            unsafe {

                if g_firstdrag {
                    println!("first drag begin\n");
                    self.on_ldrag_begin(owner, wc);
                    g_firstdrag = false;
                }
            }
            self.on_ldragging(owner, &d, wc)
        } else
        if let Some(ds)=unsafe{g_rdrag_start}{
            let d=Dragging{
                start:to_screenpos(ds),
                delta: delta,
                pos: wc.pos
            };
            unsafe {
                if g_firstdrag {
                    self.on_rdrag_begin(owner, wc);
                    unsafe { g_firstdrag = false; }
                }
            }
            self.on_rdragging(owner, &d,wc)
        } else{
            self.on_passive_move(owner,wc)
        }
    }
    // hooks for each specific dragable button state,
    // streamline rolling a tool
    fn on_passive_move(&mut self,owner:&mut A, _:&WinCursor)->Flow<A>{Flow::PassThru()}
    // first frame of dragging might have special behaviour; by default, just issue normal move
    fn on_ldrag_begin(&mut self,owner:&mut A, _:&WinCursor)->Flow<A>{Flow::PassThru()}
    fn on_mdrag_begin(&mut self,owner:&mut A, _:&WinCursor)->Flow<A>{Flow::PassThru()}
    fn on_rdrag_begin(&mut self,owner:&mut A, _:&WinCursor)->Flow<A>{Flow::PassThru()}
    fn on_ldrag_end(&mut self, owner:&mut A, _:&WinCursor)->Flow<A>{Flow::PassThru()}
    fn on_mdrag_end(&mut self, owner:&mut A, _:&WinCursor)->Flow<A>{Flow::PassThru()}
    fn on_rdrag_end(&mut self, owner:&mut A, _:&WinCursor)->Flow<A>{Flow::PassThru()}
    fn on_ldragging(&mut self,owner:&mut A,d:&Dragging,w:&WinCursor)->Flow<A>{Flow::PassThru()}
    fn on_mdragging(&mut self,owner:&mut A,d:&Dragging,w:&WinCursor)->Flow<A>{Flow::PassThru()}
    fn on_rdragging(&mut self,owner:&mut A,d:&Dragging,w:&WinCursor)->Flow<A>{Flow::PassThru()}
    fn on_wheel_up(&mut self, owner:&mut A,_:&WinCursor)->Flow<A>{Flow::PassThru()}
    fn on_wheel_down(&mut self, owner:&mut A,_:&WinCursor)->Flow<A>{Flow::PassThru()}
    // click - mouse pressed with no motion
    fn on_lclick(&mut self,owner:&mut A, w:&WinCursor)->Flow<A>{println!("lclick{:?}",w);Flow::PassThru()}
    fn on_rclick(&mut self,owner:&mut A, w:&WinCursor)->Flow<A>{println!("rclick{:?}",w);Flow::PassThru()}
    fn on_mclick(&mut self,owner:&mut A, w:&WinCursor)->Flow<A>{println!("mclick{:?}",w);Flow::PassThru()}

    fn on_mouse_dragged(&mut self, owner:&mut A, mb:MouseButtons, w:&WinCursor, mode:DragMode)->Flow<A>{
        println!("dragged{:?} {:?} {:?}",w.drag_start,w.pos,mode);
        match mb {
            LeftButton=>self.on_ldrag_end(owner, w),
            MiddleButton=>self.on_mdrag_end(owner, w),
            RightButton=>self.on_rdrag_end(owner, w),


            _=>Flow::PassThru()
        }
    }

    fn on_mouse_dragging(&mut self, owner:&mut A, mb:MouseButtons, w:&WinCursor, mode:DragMode)->Flow<A>{

        let d=Dragging{
            start:w.drag_start.unwrap(),
            delta: v2sub(&to_screenpos(unsafe{g_mouse_pos}),&to_screenpos(unsafe{g_mouse_opos})),
            pos:w.pos
        };
        match mb{
            LeftButton =>{
                if unsafe{g_firstdrag} {
                    println!("firthist drag begin\n");
                    self.on_ldrag_begin(owner, w);
                    unsafe{g_firstdrag = false;}
                }
                self.on_ldragging(owner,&d,w)
            },
            RightButton => self.on_ldragging(owner,&d,w),
            MidButton => self.on_mdragging(owner,&d,w),
            _=>Flow::PassThru()
        }
    }

    fn on_lbutton_down(&mut self,owner:&mut A, w:&WinCursor)->Flow<A>{Flow::PassThru()}
    fn on_rbutton_down(&mut self,owner:&mut A,  w:&WinCursor)->Flow<A>{Flow::PassThru()}
    fn on_lbutton_up(&mut self,owner:&mut A, w:&WinCursor)->Flow<A>{
        match unsafe{g_ldrag_start} {
            Some(prev)=>{println!("lbutton up (dragged)");}
            None=>{println!("lbutton up (no drag)");}
        }
        Flow::PassThru()
    }
    fn on_rbutton_up(&mut self,owner:&mut A,  _:&WinCursor)->Flow<A>{Flow::PassThru()}
    fn on_mbutton_down(&mut self,owner:&mut A,  _:&WinCursor)->Flow<A>{Flow::PassThru()}
    fn on_mbutton_up(&mut self,owner:&mut A,  _:&WinCursor)->Flow<A>{Flow::PassThru()}
    fn on_mouse_button(&mut self,owner:&mut A,  mb:MouseButtons,s:bool, w:&WinCursor)->Flow<A> {
        match (mb,s){
            (MouseButtons::Left,true)=>self.on_lbutton_down(owner,w),
            (MouseButtons::Left,false)=>self.on_lbutton_up(owner,w),
            (MouseButtons::Right,true)=>self.on_lbutton_down(owner,w),
            (MouseButtons::Right,false)=>self.on_rbutton_up(owner,w),
            (MouseButtons::Mid,true)=>self.on_mbutton_down(owner,w),
            (MouseButtons::Mid,false)=>self.on_mbutton_up(owner,w),
            (MouseButtons::WheelUp,true)=>self.on_wheel_up(owner,w),
            (MouseButtons::WheelDown,true)=>self.on_wheel_down(owner,w),
			_=>Flow::PassThru()
        }
    }
    fn try_drag(&self, owner:&A, mbpos:MouseButtons, w:&WinCursor)->DragMode{
        trace!();
        DragMode::Rect
    }
    fn on_click(&mut self, owner:&mut A, mb:MouseButtons, w:&WinCursor)->Flow<A>{
        // dispatch to specific button if so desired.
        match mb{
            MouseButtons::Left=>self.on_lclick(owner,w),
            MouseButtons::Right=>self.on_rclick(owner,w),
            MouseButtons::Mid=>self.on_mclick(owner,w),
            _=>{warn!();Flow::Continue()}
        }
    }
    fn on_key_down(&mut self,owner:&mut A,  k:window::WinKey, modifiers:u32, w:&WinCursor)->Flow<A> { Flow::PassThru() }
    fn on_key_up(&mut self, owner:&mut A,k:window::WinKey, w:&WinCursor)->Flow<A> { Flow::PassThru() }
    fn on_key(&mut self,owner:&mut A,k:KeyAt,w:&WinCursor)->Flow<A>{
        let KeyAt(keycode,modk,state,pos)=k;
        //default : route to seperate keydown,keyup
        match state{
            KeyDown=>self.on_key_down(owner,keycode,modk,w),
            KeyUp=>self.on_key_up(owner,keycode,w)}
    }
    fn on_drop(&mut self, f:&str, w:&WinCursor)           {}
    fn command( &mut self, owner:&mut A,c:Command)->Flow<A>{ Flow::PassThru() }

    // enum of every event,
    // defaults to calling the fn's
    fn event_dispatch(&mut self, owner:&mut A, e:Event,r:&ScreenRect)->Flow<A>{
        match e{
            Event::Update(dt)  =>{self.update(owner,dt);Flow::Continue()},
            Event::Render(t)   =>{
                self.render(owner,
                    &WinCursor
					{
						rect:r.clone(),
						old_pos: get_mouse_ovpos(),
						pos: get_mouse_vpos(),
                        drag_start:get_drag_start(),
						aspect_ratio:get_aspect_ratio(),
						t:0.0f32});
                Flow::Continue()
            },
            Event::Activate()  =>{self.on_activate(owner);Flow::Continue()},
            Event::Deactivate()    =>{self.on_deactivate(owner);Flow::Continue()},
            Event::Key(k)     =>self.on_key(owner,k,&mkwc(r,&get_mouse_vpos())),
            Event::Move(pos)  =>self.on_mouse_move(owner,&mkwcm(r,&pos,&get_mouse_ovpos())),
            Event::TryBeginDrag(mb,pos)=>{
                match get_dragmode(){
                    DragMode::None=>set_dragmode(self.try_drag(owner,mb,&mkwc(r,&pos))),
                    _=>{}
                };
                Flow::Continue()
            }
            Event::Clicked(mb,pos)=>self.on_click(owner,mb, &mkwc(r,&pos)),

            Event::Dragging(mb,start,current,dmode) =>
                self.on_mouse_dragging(owner,mb,&mkwcd(r,&current,&start),dmode),

            Event::Dragged(mb,start,current,dmode)  =>
                self.on_mouse_dragged(owner,mb,&mkwcd(r,&current,&start),dmode),
            Event::Button(mb,s,pos)           =>self.on_mouse_button(owner,mb,s,&mkwc(r,&pos)),
            _               =>Flow::Continue(),
        }
    }
    // main event dispatch: override to iter sub
    // TODO is this pointless forwarding to event_dispatch - should it just be one
    fn event(&mut self,owner:&mut A, e:Event, r:&ScreenRect)->Flow<A>{
		use self::Event as Ev;
        self.event_dispatch(owner,e,r)
    }
    //fn foreach_child(&self, r:Option<Rect>, f:&FnMut(&Window<A>,Option<Rect>)->ForeachResult);
    //fn foreach_child_mut(&mut self, r:Option<Rect>, f:&FnMut(&mut Window<A>,Option<Rect>)->ForeachResult);

}
// actually we could make it a mutable trait object taking the message
// and sugar wrapper for


// implement this to open a 'gridview' on something
// texture-palette, toolbox etc.
type Idx=i32;


pub trait WindowContainer<A> {
	fn sub_win_count(&self)->usize;
    fn sub_win_get(&self,i:usize)->Option<&window::Window<A>>;
    fn sub_win_set(&mut self, i:usize, optbox<window::Window<A>>);
	fn sub_win_push(&mut self, optbox<window::Window<A>>);
    fn sub_win_take(&self,i:usize)->optbox<window::Window<A>>;
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
impl<T:A> window::Window<A> for T:GridView<A> {
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
static mut g_mouse_pos:PixelPosi=(0,0);
static mut g_mouse_opos:PixelPosi=(0,0);
static mut g_dragmode:DragMode=DragMode::None;
static mut g_ldrag_start:Option<PixelPosi>=None;
static mut g_rdrag_start:Option<PixelPosi>=None;
static mut g_mdrag_start:Option<PixelPosi>=None;
static mut g_drag_points:*mut Vec<ScreenPos>=0 as *mut Vec<ScreenPos>;
static mut g_joystick:((f32,f32),u32)=((0.0f32,0.0f32),0);
static mut g_screen_pixel_sizei:PixelPosi=(960,480);
static mut g_screen_pixel_sizef:PixelPosf=(960.0f32,480.0f32);
const  MaxEvent:usize=256;
static mut g_aspect_ratio:f32=1.0f32;
fn get_aspect_ratio()->f32{unsafe{return g_aspect_ratio}}
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
	
	ret}
}
fn xlat_special_key(skey:GLuint)->WinKey{
    match skey{
        GLUT_KEY_LEFT=>WinKey::KeyCursorLeft,
        GLUT_KEY_RIGHT=>WinKey::KeyCursorRight,
        GLUT_KEY_UP=>WinKey::KeyCursorUp,
        GLUT_KEY_DOWN=>WinKey::KeyCursorDown,
        GLUT_KEY_INSERT=>WinKey::KeyInsert,
        GLUT_KEY_HOME=>WinKey::KeyHome,
        GLUT_KEY_END=>WinKey::KeyEnd,
        GLUT_KEY_PAGE_UP=>WinKey::KeyPageUp,
        GLUT_KEY_PAGE_DOWN=>WinKey::KeyPageDown,
        GLUT_KEY_F1=>WinKey::KeyF1,
        GLUT_KEY_F2=>WinKey::KeyF2,
        GLUT_KEY_F3=>WinKey::KeyF3,
        GLUT_KEY_F4=>WinKey::KeyF4,
        GLUT_KEY_F5=>WinKey::KeyF5,
        GLUT_KEY_F6=>WinKey::KeyF6,
        GLUT_KEY_F7=>WinKey::KeyF7,
        GLUT_KEY_F8=>WinKey::KeyF8,
        GLUT_KEY_F9=>WinKey::KeyF9,
        GLUT_KEY_F10=>WinKey::KeyF10,
        GLUT_KEY_F11=>WinKey::KeyF11,
        GLUT_KEY_F12=>WinKey::KeyF12,

        _=>WinKey::KeyCode(0 as char),
    }
}
mod callbacks {
    use super::*;
    pub fn reshape_func(x: i32, y: i32) {
        println!("resizing..{:?} {:?}", x, y);
        unsafe {
            let fx=x as f32;let fy=y as f32;
            g_screen_pixel_sizei = (x, y);
            g_screen_pixel_sizef = (fx,fy);
            g_aspect_ratio = fx/fy;
            glViewport(0, 0, x, y);
        }

        // panic!();
    }

    pub fn keyboard_func_sub(key: u8, isdown: bool, x: i32, y: i32) {
        unsafe {
            g_key[key as usize] = isdown;
            let kp = KeyAt(WinKey::KeyCode(key as KeyCode),get_modifiers(), if isdown{KeyDown}else{KeyUp}, to_screenpos((x, y)));
            push_event(Event::Key(kp));
        }
    }

    pub fn keyboard_func(key: u8, x: i32, y: i32) {
        keyboard_func_sub(key, true, x, y);
    }

    pub fn special_func_sub(key: GLuint, isdown: bool, x: i32, y: i32) {
        assert!((key as u32 & 0xff) == key as u32);
        unsafe {
            println!("special key {:?}",key);
            let kp = KeyAt(xlat_special_key(key),get_modifiers(), if isdown{KeyDown}else{KeyUp}, to_screenpos((x, y)));
            push_event(Event::Key(kp));
            //g_key[key as usize] = isdown; /// specialkeys do not
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
        let cvp = to_screenpos((x, y));
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
                push_event(Event::Dragging(MB::Left, to_screenpos(op), cvp,dm))
            }
            if let Some(op) = g_rdrag_start {
                push_event(Event::Dragging(MB::Right, to_screenpos(op), cvp,dm))
            }
            if let Some(op) = g_mdrag_start {
                push_event(Event::Dragging(MB::Mid, to_screenpos(op), cvp,dm))
            }
        }
    }

    pub fn passive_motion_func(x:i32,y:i32){
        set_mouse_pos(x,y);
        push_event(Event::Move(to_screenpos((x,y))));
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
        let posi=(x,y);
        let vpos = to_screenpos(posi);
        let oldbs=unsafe{g_mouse_button};
        //println!("mouse event {:?} {:?} fd={:?}",state, (x,y),unsafe{g_firstdrag});
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
            to_screenpos((x,y))
        ));

        // Now decode a bit more to establish click vs drag.
        if state == GLUT_DOWN {
            // we have seperate l/m/r drags because drags may be combined
            // each button may be pressed or released at different times.
            match button {
                GLUT_LEFT_BUTTON =>g_ldrag_start=Some(posi),
                GLUT_RIGHT_BUTTON =>g_rdrag_start=Some(posi),
                GLUT_MID_BUTTON =>g_mdrag_start=Some(posi)
            }
        } else {
            // Process mouse release: it may be a click or drag
            let (oldpos,mb)=match button{
                GLUT_LEFT_BUTTON=>(&mut g_ldrag_start,MouseButtons::Left),
                GLUT_RIGHT_BUTTON=>(&mut g_rdrag_start,MouseButtons::Right),
                GLUT_MID_BUTTON=>(&mut g_mdrag_start,MouseButtons::Mid),
            };
            if drag_mdist(oldpos.unwrap(),posi)==0{
                push_event(Event::Clicked(mb,vpos))
            } else{
                push_event(Event::Dragged(mb,to_screenpos(oldpos.unwrap()),vpos,g_dragmode))
            }
            // clear the old position.
            *oldpos=Option::None;
            set_dragmode(DragMode::None);
        }

        set_mouse_pos(x,y);
    }


}
fn drag_mdist(a:PixelPosi,b:PixelPosi)->i32{
    let dx=b.0-a.0;
    let dy=b.1-a.1;
    dx.abs()+dy.abs()
}
fn drag_dist(a:PixelPosf,b:PixelPosf)->f32{
    let dx=b.0-a.0;
    let dy=b.1-a.1;
    sqrt(dx*dx+dy*dy)
}
fn divf32(a:i32,b:i32)->f32{a as f32 / b as f32}
fn to_screenpos(s:PixelPosi)->ScreenPos{
    unsafe{
    Vec2(   divf32(s.0,g_screen_pixel_sizei.0)*2.0f32-1.0f32,
        -(divf32(s.1,g_screen_pixel_sizei.1)*2.0f32-1.0f32)
    )}
}
fn set_mouse_pos(x:i32,y:i32){unsafe{g_mouse_opos=g_mouse_pos; g_mouse_pos=(x,y)};}
fn get_mouse_vpos()->ScreenPos{unsafe{to_screenpos(g_mouse_pos)}}
fn get_mouse_ovpos()->ScreenPos{unsafe{to_screenpos(g_mouse_opos)}}
fn get_mouse_ppos()->PixelPosi{unsafe{g_mouse_pos}}
fn get_drag_start()->Option<ScreenPos>{unsafe{
    if let Some(ipos)=g_ldrag_start { Some(to_screenpos(ipos))} else {None}
}}
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
        gl_scissor_s(&Extents(&Vec2(-1.0f32,-1.0f32),&Vec2(1.0f32,1.0f32)));
        //glScissor(0,0,g_screen_pixel_sizei.0,g_screen_pixel_sizei.1);
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
type Windows<A>=(Vec<(Box<Window<A>>,IsOverlay)>,WindowId);

#[cfg(target_os = "emscripten")]
fn glutGetWindow()->i32{0}


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
fn from_tuple_z((x,y):(f32,f32),z:f32)->Vec3f{Vec3(x,y,z)}

/// window and cursor state, passed down because the framework calculates sizes.
#[derive(Clone,Debug)]
pub struct WinCursor {
    pub rect:ScreenRect,
	pub aspect_ratio:f32,
    pub drag_start:Option<ScreenPos>,
	pub old_pos:ScreenPos,
	pub pos:ScreenPos,
    pub t:f32,  //TODO - time ticks
}
fn mkwc(r:&ScreenRect,p:&ScreenPos)->WinCursor{
    WinCursor{
        rect:r.clone(), pos:p.clone(), old_pos:p.clone(), drag_start:None, t:0.0f32, aspect_ratio:get_aspect_ratio()
    }
}
// arg swap because it's like a default ('none')
fn mkwcd(r:&ScreenRect,p:&ScreenPos,dragstart:&ScreenPos)->WinCursor{
    WinCursor{
        rect:r.clone(), pos:p.clone(), old_pos:p.clone(), drag_start:Some(dragstart.clone()),t:0.0f32, aspect_ratio:get_aspect_ratio()
    }
}
fn mkwcm(r:&ScreenRect,p:&ScreenPos,opos:&ScreenPos)->WinCursor{
    WinCursor{
        rect:r.clone(), pos:p.clone(), old_pos:opos.clone(), drag_start:None,t:0.0f32, aspect_ratio:get_aspect_ratio()
    }
}
static g_drag_color:u32=0xffffffff;
fn set_dragmode(d:DragMode){
    unsafe{ g_dragmode=d;}
}
fn get_dragmode()->DragMode{
    unsafe{g_dragmode}
}
unsafe fn get_drag_points()->&'static mut Vec<ScreenPos>{
    if g_drag_points==0 as *mut _{
        let mut f=Box::new( Vec::<ScreenPos>::new());
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
    let sv=to_screenpos(g_ldrag_start.unwrap()).to_vec3_z(0.0);
    let ev=Vec3(cvp.x,cvp.y,0.0);//from_tuple_z(cvp.into(),0.0);
    match g_dragmode{
        DragMode::Line=>{
            draw::line(&sv, &ev);
        },
        DragMode::Lasso=>{
            draw::lines_xy(get_drag_points(), 0.0f32, false);
        }
        DragMode::Freehand=>{
            draw::lines_xy(get_drag_points(), 0.0f32, true);

        }

        DragMode::Default=> {
            draw::line(&sv, &ev);
            draw::rect_corners_xy(&sv, &ev, 0.1, g_drag_color);
        }
        DragMode::Rect=>{
            draw::rect_outline(&sv, &ev);
        },
        DragMode::Circle=>{
            draw::circle_xy(&sv, (ev-sv).vmagnitude() );
        },
        _=>{},
    }
}
fn render_and_update(wins:&mut Windows<()>, a:&mut ()){
    let whole_rect = Extents(&Vec2(-1.0,-1.0),&Vec2(1.0,1.0));
    unsafe{
        //render.
        let top=wins.0 .len();
        if top>0 {
            render_begin();
            let i = top - 1;
            {
				let wc=WinCursor{
                    rect:Extents(&Vec2(-1.0,-1.0),&Vec2(1.0,1.0)),
                    old_pos:get_mouse_ovpos(),
                    pos:get_mouse_vpos(),
                    drag_start:None,
					aspect_ratio:get_aspect_ratio(),
                    t:0.0f32
                };
                let win = &wins.0[i];
                // if it's an overlay , render previous first.
                if i > 0 && win.1 {
                    // todo- generalize, any number of overlays
                    wins.0[i - 1].0.render(a,&wc);
                }
                win.0.render(a,&wc);
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
                        win.0 .event(a,e,&whole_rect)}
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
#[cfg(target_os = "emscripten")]
unsafe fn glutCheckLoop(){
}
#[cfg(target_os = "linux")]
unsafe fn glutCheckLoop(){
	glutMainLoopEvent();
}

#[cfg(target_os = "emscripten")]
unsafe fn extra_callbacks(){
}

#[cfg(not(target_os = "emscripten"))]
unsafe fn extra_callbacks(){
	// joystick on linux broken :(
	#[cfg(not(target_os = "linux"))]
        glutJoystickFunc(callbacks::joystick_func as *const u8,16);
}


#[cfg(not(target_os = "emscripten"))]
fn run_main_loop(){
}
#[cfg(target_os = "emscripten")]
fn run_main_loop(){
}


#[cfg(target_os = "emscripten")]
extern { pub fn emscripten_set_main_loop(_:*const u8,framerate:i32,one:i32);}

static mut g_windows:Option<Windows<()>>=None;
static mut g_wins:*mut Windows<()> =0 as *mut _;
static mut g_app_ptr:usize=0;
fn wins()->&'static mut Windows<()>{unsafe {
	return mem::transmute::<*mut Windows<()>,&'static mut Windows<()>>(g_wins);
}}


#[cfg(target_os = "emscripten")]
unsafe fn em_main_loop_body<APP>(){
	println!("emscripten main loop invokation..");
	render_and_update(
		&mut*g_wins,&mut ()
);
}
#[cfg(target_os = "emscripten")]
unsafe fn mock_main_loop(){
	//let winptr=(g_wins_ptr as *mut Windows<T>) as &mut Windows<T>;

	render_and_update(&mut*g_wins,&mut ());
	glFlush();
	glutSwapBuffers();	
 //	render_and_update(wins(),&mut ());
}

// you have to push an initial state, which is a window.
pub fn run_loop(mut swins:Vec<sto<Window<()>>>, app:&mut ()) {
    println!("run loop..");
	
	let mut _wins = Box::new((vec![],0));
	unsafe{
		g_wins=(&mut *_wins) as *mut Windows<()>;
		for w in swins{push(wins(), w)};
		std::mem::forget(_wins);
	}
    unsafe {
		// todo , better place to set initial dimensions
		g_aspect_ratio=g_screen_pixel_sizef.0/g_screen_pixel_sizef.1;
        println!("window handler main");
        let mut argc:c_int=0;
        let argv=Vec::<*const c_char>::new();
        glutInit((&mut argc) as *mut c_int,0 as *const *const c_char );
        glutInitDisplayMode(GLUT_DEPTH | GLUT_DOUBLE | GLUT_RGBA);
        let win=glutCreateWindow(c_str("world main loop\0"));
		println!("{:?},{:?}",glutGet(GLUT_WINDOW_WIDTH),
			glutGet(GLUT_WINDOW_HEIGHT));
		//return;
		glutReshapeWindow(g_screen_pixel_sizei.0,g_screen_pixel_sizei.1);
		glViewport(0,0,g_screen_pixel_sizei.0,g_screen_pixel_sizei.1);
        //		glewInit(); //TODO- where the hell is glewInit. -lGLEW isn't found
		#[cfg(not(target_os = "emscripten"))]
		{
			glDrawBuffer(GL_BACK);
		}

        // glut callback malarchy
        glutDisplayFunc(callbacks::render_null as *const u8);
        glutKeyboardFunc(callbacks::keyboard_func as *const u8);
        glutKeyboardUpFunc(callbacks::keyboard_up_func as *const u8);
        glutMouseFunc(callbacks::mouse_func as *const u8);
        glutReshapeFunc(callbacks::reshape_func as *const u8);
        glutMotionFunc(callbacks::motion_func as *const u8);
        glutPassiveMotionFunc(callbacks::passive_motion_func as *const u8);
        glutIdleFunc(callbacks::idle_func as *const u8);

		extra_callbacks();
        glutSpecialFunc(callbacks::special_func as *const u8);
        glutSpecialUpFunc(callbacks::special_up_func as *const u8);
        glEnable(GL_DEPTH_TEST);

		glClearColor(0.5f32,0.5f32,0.5f32,1.0f32);
        glClear(GL_COLOR_BUFFER_BIT|GL_DEPTH_BUFFER_BIT);
/*
		#[cfg(all(target_os = "emscripten",shadertest))]
		{
			println!("emscripten shadertest minmial test");
        glDisable(GL_DEPTH_TEST);
			::minimal_shader::mainr();
			return;
		}		
*/

		#[cfg(target_os = "emscripten")]
		{
			println!("emscripten set main loop");
			//g_test_texture[0]=emscripten_run_script_int(cstr!("load_texture_url(\"https://upload.wikimedia.org/wikipedia/commons/1/17/Begur_Sa_Tuna_02_JMM.JPG\");\0"))as GLuint;
			//g_test_texture[1]=emscripten_run_script_int(cstr!("load_texture_url(\"https://upload.wikimedia.org/wikipedia/commons/6/6e/Oblast_mezi_Libeňským_mostem_a_Negrelliho_viaduktem_%2817%29.jpg\");\0")) as GLuint;
			//println!("tex1 tex 2 {}{}",g_test_texture[0],g_test_texture[1]);

			emscripten_set_main_loop(mock_main_loop as *const u8,0,1);
			return;
		}

        //glutMainLoop();
		println!("enter main loop");
		#[cfg(not(target_os = "emscripten"))]

        loop {
            glutCheckLoop();//draw

            unsafe {render_and_update(wins(),app);}
        }
		#[cfg(target_os = "emscripten")]
		unsafe {
			render_and_update(wins(), app);

			println!("init emscripten main loop");
			emscripten_set_main_loop(mock_main_loop as *const u8,60,1);
		}
    }
}

pub fn push<A>(wins:&mut Windows<A>, w:sto<Window<A>>){
    unsafe{
        wins.0 .push((w,false));
        wins.1 = glutGetWindow();
    }
}
pub fn key(k:char)->bool{
    unsafe{g_key[k as usize]}
}

/// Editor: handles tool switching and forwards events to the tool.
impl<O,E> Window<O> for MainWindow<E> {
    fn event(&mut self,owner:&mut O, e:Event, r:&Rect)->Flow<O>{
        use self::Event as Ev;
        //self.event_dispatch(owner,e)

        // todo - respond to flow ..
        // replacement of the frame etc should be possible.
        let sw=&mut*self.subwindow;
        sw.event(&mut self.content,e,r);
        Flow::Continue()
    }
    fn render(&self,a:&O, wc:&WinCursor){
		unsafe{
			glActiveTexture(GL_TEXTURE1+0);
			glBindTexture(GL_TEXTURE_2D, 0);
			glDisable(GL_TEXTURE_2D);
			glActiveTexture(GL_TEXTURE0+0);
			glBindTexture(GL_TEXTURE_2D, 0);
			glDisable(GL_TEXTURE_2D);
		}
        gl_scissor_s(&wc.rect);
        self.subwindow.render(&self.content,wc);
    }
}
pub fn gl_scissor_s(sr:&ScreenRect) {
    //let w = sr.1 .0 - sr.0 .0;
    //let h = sr.1 .1 - sr.0 .1;
    let size=sr.size();
    unsafe {
        glEnable(GL_SCISSOR_TEST);
        glScissor(
        ((sr.min .x + 1.0f32) * g_screen_pixel_sizef.0 * 0.5f32) as GLint,
        ((sr.min .y + 1.0f32) * g_screen_pixel_sizef.1 * 0.5f32) as GLint,
        (size.x * g_screen_pixel_sizef.0 * 0.5f32) as GLsizei,
        (size.y * g_screen_pixel_sizef.1 * 0.5f32) as GLsizei);
    }
}
pub struct MainWindow<T> {
    pub content:T,
    pub subwindow: Box<Window<T>>,
}
pub fn MainWindow<T,OWNER>(content:T, rootframe: Box<Window<T>>)->Box<Window<OWNER>>
where OWNER:'static, T:'static
{
    Box::new(MainWindow{content:content, subwindow:rootframe})
}

#[derive(Clone,Copy,Debug)]
enum SplitMode{RelDynamic,RelHoriz,RelVert,AbsLeft,AbsRight,AbsUp,AbsDown}
impl Default for SplitMode{ fn default()->SplitMode{SplitMode::RelDynamic}}

struct Split<OWNER>{
    proportion:f32,
    mode:SplitMode,
    subwin:[Box<Window<OWNER>>;2]
}
const  split_epsilon:f32=0.005f32;// todo - pixel split
impl<OWNER> Split<OWNER>{
    fn calc_rects(&self,parentRect:&Rect)->[Rect;2]{
        let f0=0.0;
        let f1=self.proportion-split_epsilon;
        let f2=self.proportion+split_epsilon;
        let f3=1.0;

        let split_horiz=||{
            [split_x(parentRect, f0, f1),
                split_x(parentRect, f2, f3)]
        };
        let split_vert=||{
            [split_y(parentRect, f0, f1),
                split_y(parentRect, f2, f3)]
        };
        let size=parentRect.size();
        match self.mode {
            SplitMode::RelVert=>split_vert(),
            SplitMode::RelHoriz=>split_horiz(),
            _=>if size.x*get_aspect_ratio()>=size.y {split_horiz()} else {split_vert()}
        }
    }
    fn calc_rect(&self, parentRect:&Rect, i:usize)->Rect{
        // todo , less stupidly
        let rects=self.calc_rects(parentRect);
        rects[i].clone()
    }
}

pub fn SplitHoriz<OWNER:'static>(a:Box<Window<OWNER>>,b:Box<Window<OWNER>>)->Box<Window<OWNER>>{
    Box::new(Split::<OWNER>{
        mode:SplitMode::RelHoriz,
        proportion:0.5f32,
        subwin:[a,b],
    })
}
// default split type is dynamic - checks if wider or taller..
pub fn Split<OWNER:'static>(prop:f32,a:Box<Window<OWNER>>,b:Box<Window<OWNER>>)->Box<Window<OWNER>>{
    Box::new(Split::<OWNER>{
        mode:SplitMode::RelDynamic,
        proportion:prop,
        subwin:[a,b],
    })
}
macro_rules! windbg{
    ($($e:expr),*)=>{}
}

impl<OWNER> Window<OWNER> for Split<OWNER>{

    fn event(&mut self, owner:&mut OWNER, e:Event, r:&ScreenRect)->Flow<OWNER>{
        //todo- dispatch event per side..
        // todo - this is actually BSP split?!
        let rects=self.calc_rects(&r);
        match e.pos(){
            // non-spatial event - dispatch to all
            None=>{self.subwin[1].event(owner,e,r); self.subwin[0].event(owner,e,r)},
            // spatial event: dispatch to the sub-window enclosing it.
            Some(pos)=>{
                windbg!("spatial event {:?}, pos={:?}", e, pos);
                let mut ret:Flow<OWNER>=Flow::Continue();
                for (i,subr) in rects.iter().enumerate(){
                    if v2is_inside(&pos, (&subr.min,&subr.max)){
                        //println!("dispatch {:?} to win {:?}",e,i);
                        ret=self.subwin[i].event(owner,e, subr);
                    }
                }
                ret
            }
        }

//        self.first.event(owner,e,r)
    }
    fn render(&self, a:&OWNER, wc:&WinCursor){
        let rects = self.calc_rects(&wc.rect);
        let mut subwin0=wc.clone(); let mut subwin1=wc.clone();
        subwin0.rect=rects[0].clone(); subwin1.rect=rects[1].clone();
        gl_scissor_s(&subwin0.rect);
        self.subwin[0].render(a, &subwin0);
        gl_scissor_s(&subwin1.rect);
        self.subwin[1].render(a, &subwin1);
    }
}

