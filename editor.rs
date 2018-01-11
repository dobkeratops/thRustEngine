use super::*;
use vector::Project;
use window::*;//{KeyAt,Flow,Event,KeyMappings,MouseButtons,ViewPos,RC,DragMode,MouseClick};
use window::KeyTransition::*;
use std::marker::PhantomData;

#[derive(Debug,Clone,PartialEq,Copy)]
pub enum BoolOp {Set,Clear,Invert}
impl BoolOp{pub fn apply(self,x:bool)->bool{match self{
	BoolOp::Set=>true,	BoolOp::Clear=>false,	BoolOp::Invert=>!x
}}}

/// defered action to take, part of editor<->editable communication
pub enum Action<D:Doc>{
    SetTool(Box<Tool<D>>),
	PushTool(Box<Tool<D>>),
	PopTool(),
    DoOperation(Box<Operation<D>>),
}

/// Trait for editable datastructures, e.g. scene for 3d editor, etc.
/// 'editable' could be implemented for components to make dedicated editors,
/// e.g. mesh ,scene, materials could all be different editables describing their own
pub trait Doc : Sized+Clone+Default+'static{
    fn doc_default_tool()->Box<Tool<Self>>;
    fn doc_key(&self, ed:&Editor<Self>, k:&KeyAt)->Option<Action<Self>>;
    fn doc_render(&self, proj_view_matrix:&Mat44);
    // handlers for common commands all editables should support
    fn doc_copy(&self, pos:&ScreenPos)->Self;
    fn doc_paste(&mut self, pos:&ScreenPos, clipboard:&Self);
    fn doc_delete(&mut self);
    fn doc_cut(&mut self,pos:&ScreenPos){self.doc_copy(pos); self.doc_delete();}
    fn doc_select_all(&mut self, sm:BoolOp);    // all editors should have select none command
    fn doc_cancel(&mut self);                   // escape key should do something sane
    fn doc_dump(&self);
}

pub trait Operation<D:Doc> {
    fn op_name(&self)->String{String::from("operation")}
    fn op_dump(&self){}
    // todo - show UI - tweakable parameters.
    //fn num_params();
    //fn foreach_param((name:string,value:f32));
    fn op_apply(&self, s:&mut D);
    // todo - how to do this without earlier knowledge of the traits
    // e.g. combine many selection operations, combine many transformations, etc.
    // could we say 'this is a combinable Vector operation; this is a combinable Set operation'?
    fn op_can_collapse_with<'e>(&self, other:&'e Operation<D>)->bool {false}
    fn op_collapse_with<'e>(&self, other:&'e Operation<D>)->optbox<Operation<D>> {None}
    //todo - dependancy graph sorting..
}

type SceneViewPos<'e,SCENE/*:Editable*/>=(&'e SCENE, &'e ScreenPos);

pub trait Tool<D:Doc>{
	// why the prefixing? - easier with grep/simple autocomplete.
	// we still get polymorphism (there are many 'tool_activate..' implementations)
    fn tool_name(&self)->&'static str{"un-named tool"}
    fn tool_activate(&mut self){}
    fn tool_deactivate(&mut self){}
    fn tool_passive_move(&mut self, e:ViewCursorScene<D>); // common computation between highlight & operation
    fn tool_lclick(&mut self, e:ViewCursorScene<D>)->optbox<Operation<D>>{return None;}
    fn tool_mclick(&mut self, e:ViewCursorScene<D>)->optbox<Operation<D>>{return None;}
    fn tool_rclick(&mut self, e:ViewCursorScene<D>)->optbox<Operation<D>>{return None;}

    fn tool_drag_begin(&mut self, e:ViewCursorScene<D> );
    fn tool_drag(&mut self, e:ViewCursorScene<D>)->optbox<Operation<D>>{None}// TODO - return transient Operation..
    fn tool_drag_end(&mut self, e:ViewCursorScene<D>)->optbox<Operation<D>>{ return None;}
    fn tool_render(&self, e:ViewCursorScene<D>){}
    fn tool_cancel(&mut self){println!("cancel")}
}
pub type PTool<D>=Box<Tool<D>>;
// wrapped op: user defined ops alongside inbuilt 'cut-copy-paste' commands, framework manages copy-buffer
enum EditorOp<D:Doc>{
    Op(Box<Operation<D>>),      // modifies actual scene
    Delete(ScreenPos),          // related to cut-copy-paste buffer.
    Cut(ScreenPos),
    Copy(ScreenPos),
    Paste(ScreenPos),             // cut-copy-paste differ in that they need access to copy-buffer.
    SelectAll(BoolOp),
    Cancel()
    // TODO- cycle multi-copy-buffer, visualize that.
}

pub struct Editor<D:Doc> {             // a type of frame window.
    scene:D,    // Current view of whats' being edited
    clipboard:D,
    operations:Vec<EditorOp<D>>, // undo stack of operations generating 'scene'
	redo_stack:Vec<EditorOp<D>>,
    transient_op:optbox<Operation<D>>,// not part of the doc yet
    transient_scene:Option<D>,
    interest_point    :Vec3f,
    zoom:f32,
    tool    :Box<Tool<D>>,             // current tool.
    last_tool    :optbox<Tool<D>>,     // saved for 'last tool toggle'
    saved_tool    :optbox<Tool<D>>, // saved for 'sticky-keys' mode
}
pub enum ViewMode{
    XY,XZ,YZ,Perspective
}
// view pane in some sort of holder that gets size
pub struct SpatialViewPane<D> {
    view: ViewMode,
    cam:   Cam,
    phantom:PhantomData<D>,
}
// everything passed into Tools for interface to scene
pub struct ViewCursorSceneS{
//    scene:&'e SCENE,
	pub drag_start:Option<ScreenPos>,   // copied from 'wincursor'.
    pub old_pos:ScreenPos,
    pub pos:ScreenPos,
    pub rect:ScreenRect,
    //todo: cursor ray in world space.
    pub world_to_screen:Mat44,
    pub screen_to_world:Mat44,  // incorrect but we hack it for 2d views
    // 3d views need a complete different idea
    pub interest_point:Vec3f,
    pub screen_interest_point:Vec3f, //IP transformed into screenspace inc depth.
}

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




enum View{ ViewXY,ViewXZ,ViewYZ,View}

pub struct Cam{
    pos:V3,zoom:f32,
}

// todo - parameterize 'scene'
// Editor<Scene>, MTool<Scene>

// editor app, data parts.
// TODO - is the editor the app?
// should we just pass ownership of the state into the editor when we 'fire it up' ?


impl<D:Doc> EditorOp<D>{
    pub fn eop_dump(&self){
        match self{
            &EditorOp::Op(ref op)=>op.op_dump(),
            &EditorOp::Delete(_)=>println!("op_delete"),
            &EditorOp::Cut(_)=>println!("op_cut"),
            &EditorOp::Copy(_)=>println!("op_copy"),
            &EditorOp::Paste(_)=>println!("op_paste"),
            &EditorOp::SelectAll(_)=>println!("op_select"),
            &EditorOp::Cancel()=>println!("op_cancel"),
        }
    }
}


// main window for an application.
// holds the application and the 'main frame' view
// see borrowing issues if you just embed the window in the app.. fail


impl<D:Doc> Editor<D> {
    fn ed_scene<'a>(&'a self)->&'a D{
        match self.transient_scene{
            Some(ref s)=>s,
            None=>&self.scene
        }
    }
    fn ed_scene_mut<'a>(&'a mut self)->&'a mut D{
        match self.transient_scene{
            Some(ref mut s)=>s,
            None=>&mut self.scene
        }
    }
    fn ed_action(&mut self, a:Action<D>) {
        match a {
            Action::SetTool(t) => self.ed_set_tool(t),
            Action::PushTool(t) => self.ed_push_tool(t),
			Action::PopTool()=> self.ed_pop_tool(),
            Action::DoOperation(op) => self.ed_push_operation(op),
        }
    }
    fn ed_push_tool(&mut self, newtool: PTool<D>) {
        assert!(!self.saved_tool.is_some());
        self.saved_tool = Some(std::mem::replace(&mut self.tool, newtool));
    }
    fn ed_pop_tool(&mut self) {
        assert!(self.saved_tool.is_some());
        self.tool=std::mem::replace(&mut self.saved_tool, None).unwrap();
    }
    fn ed_set_tool(&mut self, newtool: PTool<D>){
		self.tool.tool_deactivate();
        self.last_tool= Some(std::mem::replace(&mut self.tool, newtool));   // cache the last tool.
		self.tool.tool_activate();
    }
    fn ed_swap_last_tool(&mut self){
        if let Some(x)= std::mem::replace(&mut self.last_tool, None) {
            self.ed_set_tool(x);
        }else{
            println!("swap last tool - none");
        }
    }
    // todo - caching of 'cuts' that are pasted.
	fn ed_cut(&mut self,spos:ScreenPos){ self.ed_push_op(EditorOp::Cut(spos)) }
	fn ed_copy(&mut self,spos:ScreenPos){ self.ed_push_op(EditorOp::Copy(spos)) }
	fn ed_paste(&mut self,spos:ScreenPos){ self.ed_push_op(EditorOp::Paste(spos)) } // paste knows.
    fn ed_delete(&mut self,spos:ScreenPos){ self.ed_push_op(EditorOp::Delete(spos)) } // paste knows.
	fn ed_cancel(&mut self){
		self.tool.tool_cancel();
	}
	fn ed_undo(&mut self){
		self.ed_cancel();
		println!("undo: ops={}",self.operations.len());

		if let Some(op)=self.operations.pop(){
            println!("popped operation"); op.eop_dump();
			// todo: cache copies logarithmically
			self.redo_stack.push(op);
			self.ed_recompute_scene();
            self.scene.doc_dump();
		}
	}
	fn ed_redo(&mut self){
		self.ed_cancel();
		println!("redo");
		if let Some(op)=self.redo_stack.pop(){
			self.operations.push(op);
			self.ed_recompute_scene();
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
pub fn make_editor_window<A:'static,D:Doc+'static>() -> sto<Window<A>> {
		let views3=true;
		println!("make editor window");
        window::MainWindow(
            Editor::<D> {
                operations: Vec::new(),
                scene: D::default(),
                clipboard:D::default(),
                transient_op: None,
                transient_scene: None,
                redo_stack: Vec::new(),
                tool: D::doc_default_tool(),
                saved_tool: None,
                last_tool: None,
                interest_point: v3zero(),
                zoom:1.0f32,
            },
			// 3 views is predominantly topdown with the other 2 axes slightly minor.  2/4 views would be all equal
            window::Split::<Editor<D>>(
				if views3 {0.66f32} else {0.5f32},
                Box::new(SpatialViewPane{
                    view:ViewMode::XY,cam: Cam { pos: (0.0, 0.0, 0.0), zoom: 1.0 },phantom:PhantomData,
                }),
				if views3 {window::Split::<Editor<D>>(
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

impl<D:Doc> SpatialViewPane<D> {
    fn render(ed: &Editor<D>, rc: &Rect) {}

    fn matrix_world_to_eye(&self, ed: &Editor<D>) -> Mat44f {
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
    fn matrix_eye_to_world(&self, ed:&Editor<D>)->Mat44f{
        let mcentre=matrix::translate(&ed.interest_point);
        match self.view{
            ViewMode::XY=>mcentre.mul_matrix(&matrix::view_xyz()),
            ViewMode::XZ=>mcentre.mul_matrix(&matrix::view_xzy()),
            ViewMode::YZ=>mcentre.mul_matrix(&matrix::view_yzx()),
            _=>unimplemented!()
        }

    }
    // assumes projection centred.
    fn matrix_eye_to_viewport(&self,ed:&Editor<D>, r:&Rect)->Mat44f {
        // scale -1 to 1 range into <whatever rect we were given>
        // and transalte across.
        let centre=r.centre();
        let scale=self.viewport_scale(ed,r);
        matrix::scale_translate(&vec3(scale,scale,1.0f32),&vec3(centre.x,centre.y,0.0f32))
    }

    fn matrix_viewport_to_eye(&self, ed:&Editor<D>, r:&Rect)->Mat44f{
        let centre=r.centre();
        let inv_scale=1.0f32/self.viewport_scale(ed,r);
        matrix::scale_translate(&vec3(inv_scale,inv_scale,1.0f32),&vec3(-centre.x*inv_scale,-centre.y*inv_scale,0.0f32))

    }
    // why scissoring, why not just set glViewport to the damn window?
    fn matrix_world_to_viewport(&self, ed:&Editor<D>, r:&Rect)->Mat44f{
        let m=self.matrix_eye_to_viewport(ed,r).mul_matrix(&self.matrix_world_to_eye(ed));
        m
    }
    fn matrix_viewport_to_world(&self,ed:&Editor<D>,r:&Rect)->Mat44f{
        self.matrix_eye_to_world(ed).mul_matrix(&self.matrix_viewport_to_eye(ed,r))
    }

    fn viewport_scale(&self,ed:&Editor<D>,r:&Rect)->f32{
        let dst_size=r.size();//v2sub(&r.1,&r.0);
        let src_size=vec2(2.0f32,2.0f32);
        let scalex=dst_size.x/src_size.x;
        let scaley=dst_size.y/src_size.y;
        let scale=max(scalex,scaley);// you get weird skewing the other way
        scale*ed.zoom
    }


    fn view_cursor_scene_sub(&self,ed:&Editor<D>, w:&WinCursor)->ViewCursorSceneS {
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
// operations that any 'Editable' must have
#[derive(Debug,Clone)]
pub struct OpCut<D:Doc>{at:ScreenPos, phantom:PhantomData<D>}

#[derive(Debug,Clone)]
pub struct OpCopy<D:Doc>{at:ScreenPos, phantom: PhantomData<D>}

#[derive(Debug,Clone)]
pub struct OpPaste<D:Doc>{at:ScreenPos, phantom: PhantomData<D>}



#[derive(Default,Debug,Clone)]
pub struct ComposedOp<A,B>(pub A,pub B);


impl<D:Doc, A:Operation<D>,B:Operation<D>> Operation<D> for ComposedOp<A,B>{
    fn op_apply(&self,ns:&mut D){
        self.0.op_apply(ns);
        self.1.op_apply(ns);
    }
}

impl Default for BoolOp {
	fn default()->Self{BoolOp::Clear}
}


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
pub static g_snap_radius:f32=0.015f32;
pub static g_color_feedback:u32=0xff0000ff;
pub static g_color_selected:u32=0xff0000ff;
pub static g_color_highlight:u32=0xff0000ff;
pub static g_color_wireframe:u32=0xffc0c0c0;



impl<D:Doc> EditorOp<D>{
    fn wo_apply(&self, s:&mut D, clipboard:&mut D) {
        match self {
            &EditorOp::Op(ref o) => o.op_apply(s),
            &EditorOp::Delete(ref pos) => {s.doc_delete()},
            &EditorOp::Cut(ref pos) => {*clipboard=s.doc_copy(pos); s.doc_delete()},
            &EditorOp::Copy(ref pos) => { *clipboard = s.doc_copy(pos)},
            &EditorOp::Paste(ref pos) => s.doc_paste(pos, clipboard),
            &EditorOp::SelectAll(sm) => s.doc_select_all(sm),
			&EditorOp::Cancel()=>s.doc_cancel(),
        }
    }
}
// editor working on a specific type of document, with an undo stack.
impl<D:Doc> Editor<D> {
    fn ed_clear_clipboard(&mut self){ self.clipboard=D::default();}

    fn ed_push_op(&mut self, wo:EditorOp<D>){
        wo.wo_apply(&mut self.scene,&mut self.clipboard);
        self.operations.push(wo);
            //o.op_dump();
    }
    fn ed_push_operation(&mut self, op:Box<Operation<D>>){
        op.op_apply(&mut self.scene);
        self.operations.push(EditorOp::Op(op));
    }
    fn ed_push_op_maybe(&mut self, op:optbox<Operation<D>>){
        if let Some(o)=op{ self.ed_push_operation(o);}
    }
    fn ed_transient_op(&mut self, oo:optbox<Operation<D>>){
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

    fn ed_recompute_scene(&mut self){
        println!("ed_recompute_scene ops={}",self.operations.len());
        // todo - logarithmic caching spacing eg [0       n/2      n-n/4 n-n/8 n-1 n]
        let mut s=D::default();
        for o in self.operations.iter() {
            o.wo_apply(&mut s, &mut self.clipboard);
            self.scene.doc_dump();
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
impl<D:Doc> Window<Editor<D>> for SpatialViewPane<D> {
    fn win_key_mappings(&mut self, ed:&mut Editor<D>, f:&mut KeyMappings<Editor<D>>){
        f('\x1b',"cancel",  &mut ||{ed.ed_cancel();Flow::Continue()});
        f('q',"back",  &mut ||Flow::Pop());
        f('1',"foo",  &mut ||Flow::Pop());
        f('2',"bar", &mut ||Flow::Pop());
    }
    // universal keys, also consults the Scene's key preferences.
    // todo - less messy..
    fn win_key(&mut self, ed:&mut Editor<D>, k:KeyAt,wc:&WinCursor)->Flow<Editor<D>> {
        if k.1 == window::CTRL { println!("ctrl"); }
        let vpos = k.pos();
        //todo -we want plain chars really
        if let Some(keyaction)=ed.ed_scene().doc_key(ed, &k){
			ed.ed_action(keyaction);
			return Flow::Continue();
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
			(WinKey::KeyCode('\u{1a}'),window::CTRL, KeyDown)=>ed.ed_undo(),
			(WinKey::KeyCode('\u{19}'),window::CTRL, KeyDown)=>ed.ed_redo(),
			(WinKey::KeyCode('\u{18}'),window::CTRL, KeyDown)=>ed.ed_cut(vpos),
			(WinKey::KeyCode('\u{3}'),window::CTRL, KeyDown)=>ed.ed_copy(vpos),
			(WinKey::KeyCode('\u{16}'),window::CTRL, KeyDown)=>ed.ed_paste(vpos),
            (WinKey::KeyCode('='),0, KeyDown)=>{ed.zoom*=zoom_step;dump!(ed.zoom)},
            (WinKey::KeyCode('-'),0, KeyDown)=>{ed.zoom/=zoom_step;dump!(ed.zoom)},
            (WinKey::KeyCode('x'),0, KeyDown)=>ed.ed_swap_last_tool(),

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
    fn win_render(&self,ed:&Editor<D>, wc:&WinCursor){
			

        draw::rect_outline_v2(&wc.rect.min, &wc.rect.max, if v2is_inside(&wc.pos, (&wc.rect.min, &wc.rect.max)){0xa0a0a0}else{0x909090});

        let scn=ed.ed_scene();
        let mat=self.matrix_world_to_viewport(ed, &wc.rect);
        scn.doc_render(&mat);
        //for t in self.vertex_tags(){

        //}
        let vcs=self.view_cursor_scene_sub(ed,wc);
		ed.tool.tool_render((scn,&vcs));
        draw::main_mode_text("lmb-draw rmb-cancel");
    }
    fn win_passive_move(&mut self,ed:&mut Editor<D>, wc:&WinCursor)->Flow<Editor<D>> {
        let vcs=self.view_cursor_scene_sub(ed,wc);
        ed.tool.tool_passive_move((&ed.scene,&vcs));
        Flow::Continue()
    }
    fn win_ldrag_begin(&mut self,ed:&mut Editor<D>, wc:&WinCursor)->Flow<Editor<D>>{
        let vcs=self.view_cursor_scene_sub(ed,wc);
		ed.tool.tool_drag_begin((&ed.scene,&vcs));
        Flow::Continue()
    }

    fn win_ldragging(&mut self, ed:&mut Editor<D>, d:&window::Dragging,wc:&WinCursor)->Flow<Editor<D>> {
        // todo - where is the transient
        println!("where is the transient scene?");
        let transient_op={
            let vcs = self.view_cursor_scene_sub(ed,wc);
            let s=unwrap_ref_or(&ed.transient_scene,&ed.scene);
            ed.tool.tool_drag((s,&vcs))
         
        };
        ed.ed_transient_op(transient_op);
        Flow::Continue()
    }

    fn win_ldrag_end(&mut self, ed:&mut Editor<D>, wc:&WinCursor)->Flow<Editor<D>> {
        println!("editor ldrag end");
        ed.ed_transient_op(None);
        let vcs:ViewCursorSceneS = self.view_cursor_scene_sub(ed,wc);
        let op={
            let s=unwrap_ref_or(&ed.transient_scene,&ed.scene);
            ed.tool.tool_drag_end((s,&vcs))
        };
        ed.ed_push_op_maybe(op);
        Flow::Continue()
    }

//    fn win_rdragging(&mut self, a:&mut A, startpos:ScreenPos, pos:ScreenPos )->Flow<A> {
//        self.tool.drag(startpos, (&self.app.scene,&pos));
//            Flow::Continue()
//    }
    fn win_lclick(&mut self, ed:&mut Editor<D>, wc:&WinCursor)->Flow<Editor<D>>{
        println!("editor onclick");
        let vcs=self.view_cursor_scene_sub(ed,wc);
        let op=ed.tool.tool_lclick((&ed.scene,&vcs));
        ed.ed_push_op_maybe(op);
        Flow::Continue()
    }
    fn win_rclick(&mut self, ed:&mut Editor<D>, wc:&WinCursor)->Flow<Editor<D>> {
        println!("editor onrclick");
        let vcs=self.view_cursor_scene_sub(ed,wc);
        let op = ed.tool.tool_rclick((&ed.scene,&vcs));
        ed.ed_push_op_maybe(op);
        Flow::Continue()
    }
}








