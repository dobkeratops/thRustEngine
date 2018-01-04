use super::*;
use r3d::vector::Project;
use window::*;//{KeyAt,Flow,Event,KeyMappings,MouseButtons,ViewPos,RC,DragMode,MouseClick};
use window::KeyTransition::*;
use std::marker::PhantomData;
pub mod edscene;
pub use self::edscene::*; //TODO remove when properly decoupled.

/*
             MTool   EdScene
+-Editor--                   +
|        ToolBar
  Viewport  Viewport
          +          Palette
  ViewPort  Viewport
      Palette                |
+                      ------+

Viewport -  xy.. 3d/iso..  schematic
EdScene -
           ApplyRect(Rect,Command)
           Transform(dx,)
*/


type VertexIndex=usize;


enum View{ ViewXY,ViewXZ,ViewYZ,View}

pub struct Cam{
    pos:V3,zoom:f32,
}

// todo - parameterize 'scene'
// Editor<Scene>, MTool<Scene>

// editor app, data parts.
// TODO - is the editor the app?
// should we just pass ownership of the state into the editor when we 'fire it up' ?

// wrapped op: user defined ops alongside inbuilt 'cut-copy-paste' commands, framework manages copy-buffer
enum EditorOp<T:Editable>{
    Op(Box<Operation<T>>),      // modifies actual scene
    Delete(ScreenPos),          // related to cut-copy-paste buffer.
    Cut(ScreenPos),
    Copy(ScreenPos),
    Paste(ScreenPos),             // cut-copy-paste differ in that they need access to copy-buffer.
    // TODO- cycle multi-copy-buffer, visualize that.
}

pub struct Editor<T:Editable> {             // a type of frame window.
    scene:T,    // Current view of whats' being edited
    clipboard:T,
    operations:Vec<EditorOp<T>>, // undo stack of operations generating 'scene'
	redo_stack:Vec<EditorOp<T>>,
    transient_op:optbox<Operation<T>>,// not part of the doc yet
    transient_scene:Option<T>,
    interest_point    :Vec3f,
    zoom:f32,
    tool    :Box<Tool<T>>,             // current tool.
    last_tool    :optbox<Tool<T>>,     // saved for 'last tool toggle'
    saved_tool    :optbox<Tool<T>>, // saved for 'sticky-keys' mode
    presel: ToolPresel,
    dragstart:Option<ScreenPos>,
    drag:ToolDrag,
}

// main window for an application.
// holds the application and the 'main frame' view
// see borrowing issues if you just embed the window in the app.. fail


impl<T:Editable> Editor<T> {
    fn e_scene<'a>(&'a self)->&'a T{
        match self.transient_scene{
            Some(ref s)=>s,
            None=>&self.scene
        }
    }
    fn e_scene_mut<'a>(&'a mut self)->&'a mut T{
        match self.transient_scene{
            Some(ref mut s)=>s,
            None=>&mut self.scene
        }
    }
    fn e_action(&mut self, a:Action<T>) {
        match a {
            Action::None => {},
            Action::SetTool(t) => self.e_set_tool(t),
            Action::DoOperation(op) => self.operations.push(EditorOp::Op(op)),
        }
    }
    fn e_push_tool(&mut self, newtool: Box<Tool<T>>) {
        assert!(!self.saved_tool.is_some());
        self.saved_tool = Some(std::mem::replace(&mut self.tool, newtool));
    }
    fn e_pop_tool(&mut self) {
        assert!(self.saved_tool.is_some());
        self.tool=std::mem::replace(&mut self.saved_tool, None).unwrap();
    }
    fn e_set_tool(&mut self, newtool: Box<Tool<T>>){
		self.tool.tool_deactivate();
        self.last_tool= Some(std::mem::replace(&mut self.tool, newtool));   // cache the last tool.
		self.tool.tool_activate();
    }
    fn e_swap_last_tool(&mut self){
        if let Some(x)= std::mem::replace(&mut self.last_tool, None) {
            self.e_set_tool(x);
        }else{
            println!("swap last tool - none");
        }
    }
    // todo - caching of 'cuts' that are pasted.
	fn e_cut(&mut self,spos:ScreenPos){ self.e_push_op(EditorOp::Cut(spos)) }
	fn e_copy(&mut self,spos:ScreenPos){ self.e_push_op(EditorOp::Copy(spos)) }
	fn e_paste(&mut self,spos:ScreenPos){ self.e_push_op(EditorOp::Paste(spos)) } // paste knows.
    fn e_delete(&mut self,spos:ScreenPos){ self.e_push_op(EditorOp::Delete(spos)) } // paste knows.
	fn e_cancel(&mut self){
		self.tool.tool_cancel();
		self.drag=ToolDrag::None;
		self.presel=ToolPresel::None;
	}
	fn e_undo(&mut self){
		self.e_cancel();
		println!("undo");
		if let Some(op)=self.operations.pop(){
			// todo: cache copies logarithmically
			self.redo_stack.push(op);
			self.e_recompute_scene();
		}
	}
	fn e_redo(&mut self){
		self.e_cancel();
		println!("redo");
		if let Some(op)=self.redo_stack.pop(){
			self.operations.push(op);
			self.e_recompute_scene();
		}
	}
}

/*{
vertices: vec ! [( - 0.5f32, 0.0f32, 0.5f32), (0.5f32, - 0.5f32, 0.5f32),
( - 1.0, - 1.0, 0.0), (1.0, 1.0, 0.0)],
vertex_tags: vec ! [],
edges: vec ! [[0, 1], [2, 3]],
},
*/
pub fn make_editor_window<A:'static,T:Editable+'static>() -> sto<Window<A>> {
		let views3=true;
		println!("make editor window");
        window::MainWindow(
            Editor::<T> {
                operations: Vec::new(),
                scene: T::default(),
                clipboard:T::default(),
                transient_op: None,
                transient_scene: None,
                redo_stack: Vec::new(),
                tool: T::default_tool(),
                saved_tool: None,
                last_tool: None,
                drag: ToolDrag::None,
                presel: ToolPresel::None,
                interest_point: v3zero(),
                zoom:1.0f32,
                dragstart: None,
            },
			// 3 views is predominantly topdown with the other 2 axes slightly minor.  2/4 views would be all equal
            window::Split::<Editor<T>>(
				if views3 {0.66f32} else {0.5f32},
                Box::new(SpatialViewPane{
                    view:ViewMode::XY,cam: Cam { pos: (0.0, 0.0, 0.0), zoom: 1.0 },phantom:PhantomData,
                }),
				if views3 {window::Split::<Editor<T>>(
					0.5f32,
	                Box::new(SpatialViewPane{
		                view:ViewMode::XZ, cam: Cam { pos: (0.0, 0.0, 0.0), zoom: 1.0 },phantom:PhantomData,
			        }),
	                Box::new(SpatialViewPane{
		                view:ViewMode::YZ, cam: Cam { pos: (0.0, 0.0, 0.0), zoom: 1.0 },phantom:PhantomData,
			        }),
				)} else{
                Box::new(SpatialViewPane{
	                view:ViewMode::XZ, cam: Cam { pos: (0.0, 0.0, 0.0), zoom: 1.0 },phantom:PhantomData,
		        })
				}
  
          )
        )
}

// todo:
// is the creation of another object for
// the 'tool mode' un-necassery?
// shouldn't it be possible with a window overlay?
// doing this seems more direct
// also allows l/m/r sepereate

/// Tool, dispatches subset of events to work on the scene
/// TODO should all windows/events take a target like this anyway

//editor operation sits on a modifier stack
// these should produce an inspectable graph of operations
pub trait Operation<T:Editable> {
    fn op_name(&self)->String{String::from("operation")}
    fn op_dump(&self){}
    // todo - show UI - tweakable parameters.
    //fn num_params();
    //fn foreach_param((name:string,value:f32));
    fn op_apply(&self, s:&mut T);
    // todo - how to do this without earlier knowledge of the traits
    // e.g. combine many selection operations, combine many transformations, etc.
    // could we say 'this is a combinable Vector operation; this is a combinable Set operation'?
    fn op_can_collapse_with<'e>(&self, other:&'e Operation<T>)->bool {false}
    fn op_collapse_with<'e>(&self, other:&'e Operation<T>)->optbox<Operation<T>> {None}
    //todo - dependancy graph sorting..
}

pub enum ViewMode{
    XY,XZ,YZ,Perspective
}
// view pane in some sort of holder that gets size
pub struct SpatialViewPane<T> {
    view: ViewMode,
    cam:   Cam,
    phantom:PhantomData<T>,
}
impl<T:Editable> SpatialViewPane<T> {
    fn render(ed: &Editor<T>, rc: &Rect) {}

    fn matrix_world_to_eye(&self, ed: &Editor<T>) -> Mat44f {
        // todo: ed -> 'Camera' or something like that
        // it would be nice to hae a better way to specifiy synced/free viewports.
        let mcentre=matrix::translate(&ed.interest_point.vneg());
        match self.view {
            ViewMode::XY => matrix::inv_view_xyz().mul_matrix(&mcentre),
            ViewMode::XZ => matrix::inv_view_xzy().mul_matrix(&mcentre),
            ViewMode::YZ => matrix::inv_view_yzx().mul_matrix(&mcentre),
            _ => unimplemented!()
        }
    }
    fn matrix_eye_to_world(&self, ed:&Editor<T>)->Mat44f{
        let mcentre=matrix::translate(&ed.interest_point);
        match self.view{
            ViewMode::XY=>mcentre.mul_matrix(&matrix::view_xyz()),
            ViewMode::XZ=>mcentre.mul_matrix(&matrix::view_xzy()),
            ViewMode::YZ=>mcentre.mul_matrix(&matrix::view_yzx()),
            _=>unimplemented!()
        }

    }
    // assumes projection centred.
    fn matrix_eye_to_viewport(&self,ed:&Editor<T>, r:&Rect)->Mat44f {
        // scale -1 to 1 range into <whatever rect we were given>
        // and transalte across.
        let centre=r.centre();
        let scale=self.viewport_scale(ed,r);
        matrix::scale_translate(&vec3(scale,scale,1.0f32),&vec3(centre.x,centre.y,0.0f32))
    }

    fn matrix_viewport_to_eye(&self, ed:&Editor<T>, r:&Rect)->Mat44f{
        let centre=r.centre();
        let inv_scale=1.0f32/self.viewport_scale(ed,r);
        matrix::scale_translate(&vec3(inv_scale,inv_scale,1.0f32),&vec3(-centre.x*inv_scale,-centre.y*inv_scale,0.0f32))

    }
    // why scissoring, why not just set glViewport to the damn window?
    fn matrix_world_to_viewport(&self, ed:&Editor<T>, r:&Rect)->Mat44f{
        let m=self.matrix_eye_to_viewport(ed,r).mul_matrix(&self.matrix_world_to_eye(ed));
        m
    }
    fn matrix_viewport_to_world(&self,ed:&Editor<T>,r:&Rect)->Mat44f{
        self.matrix_eye_to_world(ed).mul_matrix(&self.matrix_viewport_to_eye(ed,r))
    }

    fn viewport_scale(&self,ed:&Editor<T>,r:&Rect)->f32{
        let dst_size=r.size();//v2sub(&r.1,&r.0);
        let src_size=vec2(2.0f32,2.0f32);
        let scalex=dst_size.x/src_size.x;
        let scaley=dst_size.y/src_size.y;
        let scale=max(scalex,scaley);// you get weird skewing the other way
        scale*ed.zoom
    }


    fn view_cursor_scene_sub(&self,ed:&Editor<T>, w:&WinCursor)->ViewCursorSceneS {
        let mat_w_to_s=self.matrix_world_to_viewport(ed,&w.rect);
        ViewCursorSceneS {
            drag_start: w.drag_start,
            old_pos: w.old_pos,
            pos: w.pos,
            rect: w.rect.clone(),
            world_to_screen: mat_w_to_s.clone(),
            screen_to_world: self.matrix_viewport_to_world(ed,&w.rect),
            interest_point: ed.interest_point,
            screen_interest_point: mat_w_to_s.mul_vec3w1(&ed.interest_point).project_to_vec3()
        }
    }

    // inverse.. are they really? need to assume a 'z' and so on..
}

//
type SceneViewPos<'e,SCENE/*:Editable*/>=(&'e SCENE, &'e ScreenPos);

pub trait Tool<T:Editable>{
	// why the prefixing? - easier with grep/simple autocomplete.
	// we still get polymorphism (there are many 'tool_activate..' implementations)
    fn tool_activate(&self){}
    fn tool_deactivate(&self){}
    fn tool_preselection(&self, e:ViewCursorScene<T>)->ToolPresel; // common computation between highlight & operation
    fn tool_lclick(&mut self, p:&ToolPresel, e:ViewCursorScene<T>)->optbox<Operation<T>>{return None;}
    fn tool_mclick(&mut self, p:&ToolPresel, e:ViewCursorScene<T>)->optbox<Operation<T>>{return None;}
    fn tool_rclick(&mut self, p:&ToolPresel, e:ViewCursorScene<T>)->optbox<Operation<T>>{return None;}
    fn tool_drag_end(&mut self, d:&ToolDrag, e:ViewCursorScene<T>)->optbox<Operation<T>>{ return None;}
    fn tool_drag(&mut self, d:&ToolDrag, e:ViewCursorScene<T>)->optbox<Operation<T>>{None}// TODO - return transient Operation..
    fn tool_drag_begin(&mut self, p:&ToolPresel, e:ViewCursorScene<T> )->ToolDrag{ToolDrag::None}
    fn tool_render_passive(&self, p:&ToolPresel, e:ViewCursorScene<T>){}
    fn tool_render_drag(&self, /*p:&Self::Presel, */d:&ToolDrag,e:ViewCursorScene<T>){}
	// TODO - this needs to return something to have purpose
    fn tool_passive_move(&self,e:ViewCursorScene<T>){}
    fn tool_cancel(&mut self){println!("cancel")}
    //fn try_drag(&self, e:SceneView, mbpos:(MouseButtons,ScreenPos))->DragMode{DragMode::Rect }
}


// operations that any 'Editable' must have
#[derive(Debug,Clone)]
pub struct OpCut<T:Editable>{at:ScreenPos, phantom:PhantomData<T>}

#[derive(Debug,Clone)]
pub struct OpCopy<T:Editable>{at:ScreenPos, phantom: PhantomData<T>}

#[derive(Debug,Clone)]
pub struct OpPaste<T:Editable>{at:ScreenPos, phantom: PhantomData<T>}



#[derive(Default,Debug,Clone)]
pub struct ComposedOp<A,B>(A,B);


impl<T:Editable, A:Operation<T>,B:Operation<T>> Operation<T> for ComposedOp<A,B>{
    fn op_apply(&self,ns:&mut T){
        self.0.op_apply(ns);
        self.1.op_apply(ns);
    }
}


// Possible actions in the drawing tool based on cursor location
#[derive(Debug,Clone)]
pub enum ToolPresel{
    None,
    PickPoint(VertexIndex),
    MakePoint(V3),
    ConnectLine(VertexIndex,VertexIndex),
    DrawLine(VertexIndex,V3),
    // TODO: split-edge..
}

impl Default for ToolPresel{ fn default()->Self{ToolPresel::None}}

#[derive(Debug,Clone,PartialEq,Copy)]
pub enum SelectMode {
	Select,Deselect,Invert
}

impl Default for SelectMode {
	fn default()->SelectMode{SelectMode::Select}
}

#[derive(Debug,Clone,PartialEq)]
pub enum ToolDrag{
    None,
    MovePoint(VertexIndex),
	Rect(SelectMode),
}
impl Default for ToolDrag{ fn default()->Self{ToolDrag::None}}


/// box a struct and associate with a vtable, infering types from context.
/// TODO - needs the 'Unsized<TO>' fro nightly
//fn new_object<S:Clone,TO:?Sized>(data:S)->TO{
//    Box::new(data.clone()) as _
//}

type ZoomFactor=f32;
type InterestPoint=Vec3f;

struct App{}

type Win = window::Window<()>;

/*
trait SubWin<E>{

}

// this can be general purpose..

// editor view pane
struct ViewPane(ViewMode);
impl State<Editor> for ViewPane{}
impl State<editor::App> for Split<Editor>{}


pub enum ViewMode{XY,XZ,YZ,Perspective}
fn create_4views()->Box<Win>{
    Box::new(Split::<Editor>(
        0.5f32,
        SplitMode::RelDynamic,
        Box::new(Split::<Editor>(
            0.5,
            SplitMode::RelDynamic,
            Box::new(ViewPane(ViewMode::XY)),
            Box::new(ViewPane(ViewMode::XZ))
        )),
        Box::new(Split::<Editor>(
            0.5,
            SplitMode::RelDynamic,
            Box::new(ViewPane(ViewMode::XY)),
            Box::new(ViewPane(ViewMode::XZ))
        ))
    ))
}
*/
static g_snap_radius:f32=0.015f32;
static g_color_feedback:u32=0xff0000ff;
static g_color_selected:u32=0xff0000ff;
static g_color_highlight:u32=0xff0000ff;
static g_color_wireframe:u32=0xffc0c0c0;

// defered 'action' to take, needed for command invocations
// to happen safely
pub enum Action<T:Editable>{
    None,
    SetTool(Box<Tool<T>>),
    DoOperation(Box<Operation<T>>),
}

pub trait Editable : Sized+Clone+Default+'static{
    fn default_tool()->Box<Tool<Self>>;
    fn edscn_key(&self, ed:&Editor<Self>, k:&KeyAt)->Action<Self>;
    fn scn_render(&self, m:&Mat44);
    fn copy(&self, pos:&ScreenPos, clipboard:&mut Self);
    fn paste(&mut self, pos:&ScreenPos, clipboard:&Self);
    fn delete(&mut self);// 'cut'=copy + delete.
}

impl<T:Editable> EditorOp<T>{
    fn wo_apply(&self, s:&mut T, clipboard:&mut T) {
        match self {
            &EditorOp::Op(ref o) => o.op_apply(s),
            &EditorOp::Delete(ref pos) => {s.delete()},
            &EditorOp::Cut(ref pos) => {*clipboard=T::default();s.copy(pos, clipboard); s.delete()},
            &EditorOp::Copy(ref pos) => {*clipboard=T::default();s.copy(pos, clipboard)},
            &EditorOp::Paste(ref pos) => s.paste(pos, clipboard),
        }
    }
}


// editor working on a specific type of document, with an undo stack.
impl<T:Editable> Editor<T> {
    fn e_clear_clipboard(&mut self){ self.clipboard=T::default();}

    fn e_push_op(&mut self, wo:EditorOp<T>){
        wo.wo_apply(&mut self.scene,&mut self.clipboard);
        self.operations.push(wo);
            //o.op_dump();
    }
    fn e_push_operation(&mut self, op:Box<Operation<T>>){
        op.op_apply(&mut self.scene);
        self.operations.push(EditorOp::Op(op));
    }
    fn e_push_op_maybe(&mut self, op:optbox<Operation<T>>){
        if let Some(o)=op{ self.e_push_operation(o);}
    }
    fn e_transient_op(&mut self, oo:optbox<Operation<T>>){
        //match self.app.doc.transient_op{
          //  Some(ref op)=>{self.app.transient_scene}
        //}
        //println!("todo only recalc last!");
//        self.recompute_scene();
		if let Some(ref op)=oo{
			let mut scn=self.scene.clone();
			op.op_apply(&mut scn);
			self.transient_scene=Some(scn);
		} else{ self.transient_scene=None;}
        self.transient_op=oo;
    }

    fn e_recompute_scene(&mut self){
        // todo - logarithmic caching spacing eg [0       n/2      n-n/4 n-n/8 n-1 n]
        let mut s=T::default();
        for o in self.operations.iter() {
            o.wo_apply(&mut self.scene, &mut self.clipboard);
        }
        self.scene=s;
        match self.transient_op{
            Some(ref op)=>{
                let mut scn=self.scene.clone();
                op.op_apply(&mut scn);
                self.transient_scene=Some(scn);
            },
            None=>self.transient_scene=None
        }
    }
}
/*
fn view_cursor_scene<'e,SCENE:Editable>(ed:&'e Editor<SCENE>, s:&'e SCENE, w:&WinCursor)->ViewCursorScene<'e,SCENE> {
    ViewCursorScene {
        scene: s,
        drag_start: w.drag_start,
        old_pos: w.old_pos,
        pos: w.pos,
        rect: w.rect,
        world_to_screen: matrix::identity(),
        screen_to_world: matrix::identity(),
        interest_point: ed.interest_point
    }
}
*/
// everything passed into Tools for interface to scene
pub struct ViewCursorSceneS{
//    scene:&'e SCENE,
    drag_start:Option<ScreenPos>,   // copied from 'wincursor'.
    old_pos:ScreenPos,
    pos:ScreenPos,
    rect:ScreenRect,
    //todo: cursor ray in world space.
    world_to_screen:Mat44,
    screen_to_world:Mat44,  // incorrect but we hack it for 2d views
    // 3d views need a complete different idea
    interest_point:Vec3f,
    screen_interest_point:Vec3f, //IP transformed into screenspace inc depth.
}
impl ViewCursorSceneS{
    pub fn world_to_viewport(&self,v:&Vec3f)->Vec3f{
        self.world_to_screen.mul_vec3_point(v).project_to_vec3()
    }
    pub fn viewport_to_world(&self,v:&Vec3f)->Vec3f{
        self.screen_to_world.mul_vec3_point(v).project_to_vec3()
    }
}
pub type ViewCursorScene<'e, SCENE/*:Editable*/> = (&'e SCENE, &'e ViewCursorSceneS);

fn unwrap_ref_or<'q,T>(a:&'q Option<T>,b:&'q T)->&'q T{
    if let Some(ref x)=*a{x} else {b}
}
//type A=();
impl<T:Editable> Window<Editor<T>> for SpatialViewPane<T> {
    fn key_mappings(&mut self, ed:&mut Editor<T>, f:&mut KeyMappings<Editor<T>>){
        f('\x1b',"cancel",  &mut ||{ed.e_cancel();Flow::Continue()});
        f('q',"back",  &mut ||Flow::Pop());
        f('1',"foo",  &mut ||Flow::Pop());
        f('2',"bar", &mut ||Flow::Pop());
    }
    // universal keys, also consults the Scene's key preferences.
    // todo - less messy..
    fn on_key(&mut self, ed:&mut Editor<T>, k:KeyAt,wc:&WinCursor)->Flow<Editor<T>> {
        if k.1 == window::CTRL { println!("ctrl"); }
        let vpos = k.pos();
        //todo -we want plain chars really
        {
            let keyaction=ed.e_scene().edscn_key(ed, &k);
            ed.e_action(keyaction);
        }
        let move_step=0.2f32/ed.zoom;
        let zoom_step=1.2f32;
        use window::WinKey;
        let vcs=self.view_cursor_scene_sub(ed,wc);
        let mut view_delta:Vec3=zero();
		let mk_move_step=|s,mods|{if mods&window::CTRL!=0{s*0.2f32}else if mods&window::SHIFT!=0{s*5.0f32}else{s}};
        match (k.0, k.1,k.2) {
//            ('s',KeyDown)=>self.set_tool(SelectTool()),
			//ctrl-z
			(WinKey::KeyCode('\u{1a}'),window::CTRL, KeyDown)=>ed.e_undo(),
			(WinKey::KeyCode('\u{19}'),window::CTRL, KeyDown)=>ed.e_redo(),
			(WinKey::KeyCode('\u{18}'),window::CTRL, KeyDown)=>ed.e_cut(vpos),
			(WinKey::KeyCode('\u{3}'),window::CTRL, KeyDown)=>ed.e_copy(vpos),
			(WinKey::KeyCode('\u{16}'),window::CTRL, KeyDown)=>ed.e_paste(vpos),
            (WinKey::KeyCode('='),0, KeyDown)=>{ed.zoom*=zoom_step;dump!(ed.zoom)},
            (WinKey::KeyCode('-'),0, KeyDown)=>{ed.zoom/=zoom_step;dump!(ed.zoom)},
            (WinKey::KeyCode('x'),0, KeyDown)=>ed.e_swap_last_tool(),

            (WinKey::KeyCursorLeft, 0, KeyDown)=>view_delta.x-=move_step,
            (WinKey::KeyCursorRight,0, KeyDown)=>view_delta.x+=move_step,
            (WinKey::KeyCursorUp,0, KeyDown)=>view_delta.y+=move_step,
            (WinKey::KeyCursorDown,0, KeyDown)=>view_delta.y-=move_step,
            (WinKey::KeyPageUp,mods, KeyDown)=>view_delta.z+=mk_move_step(move_step,mods),  // move perp to curr screen.
            (WinKey::KeyPageDown,mods, KeyDown)=>view_delta.z-=mk_move_step(move_step,mods),

            //seditor::make_editor_window::<(),editor::Scene>()
            (WinKey::KeyCode('\x1b'),0,KeyDown)=>{return Flow::Pop()},
            _=>{println!("keycode:{:?}, mods{:?}",k.0,k.1);
                ()}
        };
        let wdelta = vcs.screen_to_world.ax.vscale(view_delta.x).vmadd(&vcs.screen_to_world.ay,view_delta.y).vmadd(&vcs.screen_to_world.az,view_delta.z).to_vec3();
        ed.interest_point.vassign_add(&wdelta);
        Flow::Continue()
    }
    fn render(&self,ed:&Editor<T>, wc:&WinCursor){
			

        draw::rect_outline_v2(&wc.rect.min, &wc.rect.max, if v2is_inside(&wc.pos, (&wc.rect.min, &wc.rect.max)){0xa0a0a0}else{0x909090});

        let scn=ed.e_scene();
        let mat=self.matrix_world_to_viewport(ed, &wc.rect);
        scn.scn_render(&mat);
        //for t in self.vertex_tags(){

        //}
        let vcs=self.view_cursor_scene_sub(ed,wc);
        //println!("todo pass the matrix in here");
		match ed.dragstart{
	        Some(vs)=>ed.tool.tool_render_drag(&ed.drag, (scn,&vcs)),
//                (vs.0,vs.1), (&ed.e_scene(),&wc.pos),  wc.rect),
			_=>ed.tool.tool_render_passive(&ed.presel, (scn,&vcs))//ed.e_scene(),  rc),
		}
        draw::main_mode_text("lmb-draw rmb-cancel");
    }
    fn on_passive_move(&mut self,ed:&mut Editor<T>, wc:&WinCursor)->Flow<Editor<T>> {
        let vcs=self.view_cursor_scene_sub(ed,wc);
        ed.presel=ed.tool.tool_preselection((&ed.scene,&vcs));
//            &
//            (&ed.scene/*not transient*/,&pos));
        ed.tool.tool_passive_move((&ed.scene, &vcs));
        Flow::Continue()
    }
    fn on_ldrag_begin(&mut self,ed:&mut Editor<T>, wc:&WinCursor)->Flow<Editor<T>>{
        let vcs=self.view_cursor_scene_sub(ed,wc);
		let drag=
			ed.tool.tool_drag_begin(&ed.presel, (&ed.scene,&vcs));
        ed.drag=drag;
		ed.dragstart=Some(wc.drag_start.unwrap());
        Flow::Continue()
    }

    fn on_ldragging(&mut self, ed:&mut Editor<T>, d:&window::Dragging,wc:&WinCursor)->Flow<Editor<T>> {
        // todo - where is the transient
        println!("where is the transient scene?");
        let transient_op={
            let vcs = self.view_cursor_scene_sub(ed,wc);
            if let ToolDrag::None = ed.drag {None} else {
                // todo - solution is holder for transient+scene
                let s=unwrap_ref_or(&ed.transient_scene,&ed.scene);
                    //ed.e_scene();
                ed.tool.tool_drag(&ed.drag, (s,&vcs))
            }
        };
        ed.e_transient_op(transient_op);
        Flow::Continue()
    }

    fn on_ldrag_end(&mut self, ed:&mut Editor<T>, wc:&WinCursor)->Flow<Editor<T>> {
        println!("editor ldrag end");
        ed.e_transient_op(None);
        let vcs:ViewCursorSceneS = self.view_cursor_scene_sub(ed,wc);
        let op={
            let s=unwrap_ref_or(&ed.transient_scene,&ed.scene);
            ed.tool.tool_drag_end(&ed.drag, (s,&vcs))
        };
        ed.e_push_op_maybe(op);
		ed.dragstart=None;
        Flow::Continue()
    }

//    fn on_rdragging(&mut self, a:&mut A, startpos:ScreenPos, pos:ScreenPos )->Flow<A> {
//        self.tool.drag(startpos, (&self.app.scene,&pos));
//            Flow::Continue()
//    }
    fn on_lclick(&mut self, ed:&mut Editor<T>, wc:&WinCursor)->Flow<Editor<T>>{
        println!("editor onclick");
        let vcs=self.view_cursor_scene_sub(ed,wc);
        let op=ed.tool.tool_lclick(&ed.presel, (&ed.scene,&vcs));
        ed.e_push_op_maybe(op);
        Flow::Continue()
    }
    fn on_rclick(&mut self, ed:&mut Editor<T>, wc:&WinCursor)->Flow<Editor<T>> {
        println!("editor onrclick");
        let vcs=self.view_cursor_scene_sub(ed,wc);
        let op = ed.tool.tool_rclick(&ed.presel, (&ed.scene,&vcs));
        ed.e_push_op_maybe(op);
        Flow::Continue()
    }

    fn try_drag(&self, a:&Editor<T>, mb:MouseButtons, wc:&WinCursor)->DragMode{
        //self.tool.try_drag(&self.app.scene,  (mb,pos))
        DragMode::None
    }
/*
    fn event(&mut self, e: Event)->Flow{
        println!("event:");dump!(e);
        match e{
            Event::Button(MouseButtons::Left,true,(x,y))=>{
                self.state=EState::LastPoint(ve);
            },
            Event::Button(MouseButtons::Right,true,(x,y))=> {
                self.state=EState::None;
            },
            _=>{}
        }
		Flow::Continue()
    }
    */
}








