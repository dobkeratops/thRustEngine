use super::*;
use window::*;//{KeyAt,Flow,Event,KeyMappings,MouseButtons,ViewPos,RC,DragMode,MouseClick};
type V3=(f32,f32,f32);
use window::KeyTransition::*;

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

struct Cam{
    pos:V3,zoom:f32,
}

type VertexIndex=usize;

#[derive(Default,Clone)]
pub struct SceneGeom {
    vertices:   Vec<(V3)>,
    edges:      Vec<[VertexIndex;2]>,
}
impl SceneGeom{
	fn clear(&mut self){
		self.vertices=Vec::new();
		self.edges=Vec::new();
	}
}
#[derive(Default,Clone)]
pub struct Scene{
	main:SceneGeom,
    vertex_tags:Vec<bool>,
	copy_buffer:SceneGeom
}

type SceneSelection = Vec<bool>;

#[derive(Default)]
pub struct Doc {
    operators:vecbox<Operation>,
}


impl Scene{
    // todo - vertex tag iterator.
    fn clear_vertex_tags(&mut self){
        for t in self.vertex_tags.iter_mut(){*t=false;}
    }
    fn vertex_tags_invert_all(&mut self){
        for t in self.vertex_tags.iter_mut(){*t^=true;}
    }
    fn vertex_tags_set_all(&mut self){
        for t in self.vertex_tags.iter_mut(){*t=true;}
    }
	fn vertex_tag_invert(&mut self, i:VertexIndex){
		self.vertex_tags[i]^=true;
	}
	fn vertex_tag_change(&mut self, i:VertexIndex, mode:SelectMode){
		let tag=self.vertex_tags[i];
		self.vertex_tags[i]=match mode{ 
			SelectMode::Invert=>tag^true,
			SelectMode::Select=>true,
			SelectMode::Deselect=>false
		}
	}
    fn create_vertex(&mut self, pos:V3,tag:bool){
        self.main.vertices.push(pos);
        self.vertex_tags.push(tag);

    }

}

// todo - parameterize 'scene'
// Editor<Scene>, MTool<Scene>

// editor app, data parts.
// TODO - is the editor the app?
// should we just pass ownership of the state into the editor when we 'fire it up' ?

pub struct Editor {
    doc:Doc,					// todo - doc:Undoable<Scene>, tied to 'Operation' type.
	redo_stack:vecbox<Operation>,
    scene:Scene,    // cached generated
    transient_op:optbox<Operation>,// not part of the doc yet
    transient_scene:Option<Scene>,

    tool    :Box<Tool>,
    presel: ToolPresel,
    dragstart:Option<V2>,
    drag:ToolDrag,
    cam:   Cam,

    saved_tool    :optbox<Tool>, // todo.. push em in a vector surely
}
impl Editor {
    fn get_scene<'a>(&'a self)->&'a Scene{
        match self.transient_scene{
            Some(ref s)=>s,
            None=>&self.scene
        }
    }
    fn get_scene_mut<'a>(&'a mut self)->&'a mut Scene{
        match self.transient_scene{
            Some(ref mut s)=>s,
            None=>&mut self.scene
        }
    }

    fn push_tool(&mut self, newtool: Box<Tool>) {
        assert!(!self.saved_tool.is_some());
        self.saved_tool = Some(std::mem::replace(&mut self.tool, newtool));
    }
    fn pop_tool(&mut self) {
        assert!(self.saved_tool.is_some());
        self.tool=std::mem::replace(&mut self.saved_tool, None).unwrap();
    }
    fn set_tool(&mut self, newtool: Box<Tool>){
		self.tool.tool_deactivate();
        self.tool = newtool;
		self.tool.tool_activate();
    }
	fn cut(&mut self,vp:ViewPos){ self.push_op(Some( Box::new(OpCut()) as _)) }
	fn copy(&mut self,vp:ViewPos){ self.push_op(Some(Box::new(OpCopy()) as _)) }
	fn paste(&mut self,vp:ViewPos){ self.push_op(Some(Box::new(OpPaste(vp)) as _)) } // paste knows.
	fn cancel(&mut self){
		self.tool.tool_cancel();
		self.drag=ToolDrag::None;
		self.presel=ToolPresel::None;
	}
	fn undo(&mut self){
		self.cancel();
		println!("undo");
		if let Some(op)=self.doc.operators.pop(){
			// todo: cache copies logarithmically
			self.redo_stack.push(op);
			self.recompute_scene();
		}
	}
	fn redo(&mut self){
		self.cancel();
		println!("redo");
		if let Some(op)=self.redo_stack.pop(){
			self.doc.operators.push(op);
			self.recompute_scene();
		}
	}
}

impl Scene{
    fn add_vertex(&mut self,p:V3)->VertexIndex{
        let vi=self.main.vertices.len();
        self.main.vertices.push(p);
        vi as VertexIndex
    }
    fn add_edge(&mut self,a:VertexIndex,b:VertexIndex){
        self.main.edges.push([a,b]);
    }
}
/*{
vertices: vec ! [( - 0.5f32, 0.0f32, 0.5f32), (0.5f32, - 0.5f32, 0.5f32),
( - 1.0, - 1.0, 0.0), (1.0, 1.0, 0.0)],
vertex_tags: vec ! [],
edges: vec ! [[0, 1], [2, 3]],
},
*/
pub fn new<A>() -> sto<Window<A>> {
    Box::new(
        Editor {
            doc:Doc::default(),
            scene: Scene::default(),
            transient_op: None,
            transient_scene: None,
			redo_stack:Vec::new(),
			cam: Cam { pos: (0.0, 0.0, 0.0), zoom: 1.0 },
            tool: Box::new(DrawTool::default()) as Box<Tool>,
            drag:ToolDrag::None,
            presel:ToolPresel::None,
            dragstart:None,
            saved_tool: None,
            //subwin:create_4views(),
        }
    ) as Box<Window<A>>
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
pub trait Operation {
    fn op_name(&self)->String{String::from("operation")}
    fn op_dump(&self){}
    // todo - show UI - tweakable parameters.
    //fn num_params();
    //fn foreach_param((name:string,value:f32));
    fn op_apply(&self, s:&mut Scene){}
    fn op_can_collapse_with<'e>(&self, other:&'e Operation)->bool {false}
    fn op_collapse_with<'e>(&self, other:&'e Operation)->optbox<Operation> {None}
}


//
type SceneViewPos<'e>=(&'e Scene, &'e ViewPos);

pub trait Tool{
	// why the prefixing? - easier with grep/simple autocomplete.
	// we still get polymorphism (there are many 'tool_activate..' implementations)
    fn tool_activate(&mut self){}
    fn tool_deactivate(&mut self){}
    fn tool_preselection(&self, e:SceneViewPos)->ToolPresel; // common computation between highlight & operation
    fn tool_lclick(&mut self, p:&ToolPresel, e:SceneViewPos)->optbox<Operation>{return None;}
    fn tool_mclick(&mut self, p:&ToolPresel, e:SceneViewPos)->optbox<Operation>{return None;}
    fn tool_rclick(&mut self, p:&ToolPresel, e:SceneViewPos)->optbox<Operation>{return None;}
    fn tool_drag_end(&mut self, d:&ToolDrag, start:ViewPos, e:SceneViewPos)->optbox<Operation>{ return None;}
    fn tool_drag(&mut self, d:&ToolDrag, opos:ViewPos, e:SceneViewPos)->optbox<Operation>{None}// TODO - return transient Operation..
    fn tool_drag_begin(&self, p:&ToolPresel, e:SceneViewPos )->ToolDrag{ToolDrag::None}
    fn tool_render_passive(&self, p:&ToolPresel, e:&Scene, rc:&RC){}
    fn tool_render_drag(&self, /*p:&Self::Presel, */d:&ToolDrag, opos:ViewPos,e:SceneViewPos, rc:&RC){}
	// TODO - this needs to return something to have purpose
    fn tool_passive_move(&self, opos:ViewPos, e:SceneViewPos){}
    fn tool_cancel(&mut self){println!("cancel")}
    //fn try_drag(&self, e:SceneView, mbpos:(MouseButtons,ViewPos))->DragMode{DragMode::Rect }
}

#[derive(Default,Debug,Clone)]
struct DrawTool{last_point:Option<VertexIndex>}

#[derive(Default,Debug,Clone)]
struct AddPoint(V3);

#[derive(Default,Debug,Clone)]
struct ToggleSelPoint(VertexIndex);

#[derive(Default,Debug,Clone)]
struct SingleSelPoint(VertexIndex);

#[derive(Default,Debug,Clone)]
struct ConnectLine(VertexIndex,VertexIndex);

#[derive(Default,Debug,Clone)]
struct DrawLine(VertexIndex,V3);

#[derive(Default,Debug,Clone)]
struct OpCut();
#[derive(Default,Debug,Clone)]
struct OpCopy();

#[derive(Default,Debug,Clone)]
struct OpPaste(ViewPos);

#[derive(Default,Debug,Clone)]
struct MovePointBy(VertexIndex,V3);

impl Scene {
    fn pick_point_at(&self, pos:&V3,maxr:f32)->Option<VertexIndex>{
        let mut maxr2 = sqr(maxr);
        let mut besti=None;
        for (vti,&vt) in self.main.vertices.iter().enumerate(){
            let r2=v3dist_squared(pos,&vt);
            if r2<maxr2{besti=Some(vti as VertexIndex); maxr2=r2;}
        }
        return besti;
    }
}
#[derive(Default,Debug,Clone)]
struct Translate{
    delta:V3
}
#[derive(Default,Debug,Clone)]
struct ComposedOp<A,B>(A,B);

#[derive(Default,Debug,Clone)]
struct SelectPoints(SelectMode,Vec<VertexIndex>);

// copy_to scene ->copybuffer
impl Operation for OpCopy{
	fn op_apply(&self, s:&mut Scene){
		s.copy_buffer.clear();
		let null_vertex=(-(1 as isize)) as VertexIndex;	// todo , will they?
		let mut vertex_xlat:Vec<VertexIndex>=vec![null_vertex; s.main.vertices.len()];

		// every selected vertex is shoved across,
		for (i,v) in s.main.vertices.iter().enumerate(){
			if s.vertex_tags[i]{
				vertex_xlat[i]=s.copy_buffer.vertices.len();
				s.copy_buffer.vertices.push(*v);
			}
		}
		
		// every primitive entirely selected is shoved across.
		println!("OpCopy-  todo!!");
	}
}

// copy_to copybuffer->scene
impl Operation for OpPaste{
	fn op_apply(&self, s:&mut Scene){
		println!("OpPaste-  todo!!");
	}
}

// copy_to scene->copybuffer, delete.
impl Operation for OpCut{
	fn op_apply(&self, s:&mut Scene){
		println!("OpCut-  todo!!");
	}
}

impl Operation for SingleSelPoint{
    fn op_apply(&self,ns:&mut Scene){
        ns.clear_vertex_tags();
        ns.vertex_tags[self.0]=true;
    }
}

impl Operation for ToggleSelPoint{
    fn op_apply(&self,ns:&mut Scene){
		ns.vertex_tag_invert(self.0);
    }
}
impl Operation for SelectPoints {
    fn op_apply(&self,ns:&mut Scene){
		for &v in self.1.iter() {
			ns.vertex_tag_change(v,self.0);
		}
	}	
	fn op_dump(&self){
		println!("select points[{:?}]",self.0);
	}
}
impl<A:Operation,B:Operation> Operation for ComposedOp<A,B>{
    fn op_apply(&self,ns:&mut Scene){
        self.0.op_apply(ns);
        self.1.op_apply(ns);
    }
}

impl Operation for DrawLine{
    fn op_apply(&self,ns:&mut Scene){
        ns.clear_vertex_tags();
        ns.create_vertex(self.1, true); // create a vertex and tag it
        ns.main.edges.push([self.0, (ns.main.vertices.len() as  VertexIndex)-1]);
    }
    fn op_dump(&self){println!("{:?}",self); }
}
impl Operation for ConnectLine{
    //dump!("CONNECT LINE:",self);
    fn op_apply(&self,ns:&mut Scene){
        ns.clear_vertex_tags();
        ns.main.edges.push([self.0, self.1]);
        ns.vertex_tags[self.1]=true;
    }
    fn op_dump(&self){println!("{:?}",self); }
}
impl Operation for AddPoint {
    fn op_apply(&self,ns:&mut Scene){
        ns.clear_vertex_tags();
        ns.main.vertices.push(self.0);
        ns.vertex_tags.push(true);
    }
    fn op_dump(&self){println!("{:?}",self); }
}
impl Operation for Translate {
    fn op_apply(&self,ns:&mut Scene){
        for (i,vt) in ns.main.vertices.iter_mut().enumerate(){ if ns.vertex_tags[i]{v3addto(vt, &self.delta);}}
    }
    fn op_dump(&self){println!("{:?}",self);}
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


impl Tool for DrawTool {

    fn tool_preselection(&self, (scn,vp): SceneViewPos) -> ToolPresel {
        let newpt=v3fromv2(vp,0.0f32);
        let picked_point=scn.pick_point_at(&(vp.0,vp.1,0.0f32),g_snap_radius);
        // now here's where rust shines..
        let ret=match (self.last_point, picked_point) {
            (None,None)         =>ToolPresel::MakePoint(newpt),
            (None,Some(ei))     =>ToolPresel::PickPoint(ei),
            (Some(si),None)     =>ToolPresel::DrawLine(si, newpt),
            (Some(si),Some(ei)) =>ToolPresel::ConnectLine(si,ei)
        };
        //dump!(ret);
        ret
	}

    fn tool_render_passive(&self, p:&ToolPresel,s:&Scene,rc:&RC)
    {
        match p{
            &ToolPresel::MakePoint(ref newpt)=>
                draw::circle_fill_xy_c(newpt,g_snap_radius,g_color_feedback,),
            &ToolPresel::PickPoint(ei)=>
                draw::circle_fill_xy_c(&s.main.vertices[ei as usize],g_snap_radius,g_color_feedback),
            &ToolPresel::ConnectLine(si,ei)=>
                draw::line_c(&s.main.vertices[si as usize],&s.main.vertices[ei  as usize], g_color_feedback),
            &ToolPresel::DrawLine(si,ref newpt)=>
                draw::line_c(&s.main.vertices[si  as usize],newpt,g_color_feedback),
            _=>{},
        }
    }

    fn tool_render_drag(&self, d:&ToolDrag, dragstart:ViewPos, (s,vpos):SceneViewPos,rc:&RC){
		match d {
			&ToolDrag::Rect(_)=>{
				draw::rect_outline_v2(dragstart,*vpos,g_color_feedback)	
			},
			_=>{}
		}
    }
    fn tool_drag(&mut self, d:&ToolDrag, opos:ViewPos, (s,vpos):SceneViewPos) ->optbox<Operation>{
        // shows what drag-end would produce, as a transient state.
        self.tool_drag_end(d,opos,(s,vpos))
    }

	// todo - should the drag itself be an object that completes itself?
	// we have the logic for a single mode split between several places.
	// you could have IDrag { start, render, end}
    fn tool_drag_end(&mut self, d:&ToolDrag, dragstart:ViewPos,(s,vp):SceneViewPos)->optbox<Operation>{

        match d {
            &ToolDrag::MovePoint(ref vti)=>Some(Box::new(
                ComposedOp(
                    SingleSelPoint(*vti),
                    Translate{delta:v3fromv2(&v2sub(vp, &dragstart),0.0f32)}
                )
            )),
			&ToolDrag::Rect(ref mode)=>{
				let pts=s.get_vertices_in_rect((dragstart,*vp));
				Some(Box::new(SelectPoints(SelectMode::Invert,pts)))
			},
            _ =>{None}

        }
    }

    fn tool_lclick(&mut self, p:&ToolPresel, (s,vp): SceneViewPos) -> optbox<Operation> {
        println!("drawtool lclick , make a addpoint op");
        match p {
            &ToolPresel::MakePoint(newpt) =>{
                self.last_point =Some(s.main.vertices.len() as VertexIndex); println!("vertices.len()={:?}",s.main.vertices.len());Some(Box::new(AddPoint(newpt)) as _)
            },
            &ToolPresel::PickPoint(pti)     =>{
                self.last_point=Some(pti);Some(Box::new(ToggleSelPoint(pti)) as _)
            },
            &ToolPresel::ConnectLine(si,ei) =>{
                self.last_point=Some(ei);Some(Box::new(ConnectLine(si,ei)) as _)
            },
            &ToolPresel::DrawLine(si,newpt) =>{
                self.last_point=Some(s.main.vertices.len() as VertexIndex);Some(Box::new(DrawLine(si,newpt)) as _)
            },
            _ => None
        }
    }
	
    fn tool_rclick(&mut self, p:&ToolPresel, (s,vp):SceneViewPos)->optbox<Operation>{
        println!("rclick");
        self.tool_cancel();
        None
    }

    fn tool_drag_begin(&self, p:&ToolPresel, (s,vp):SceneViewPos)->ToolDrag{
        if let &ToolPresel::PickPoint(pti)=p{
	        println!("drag start-movepoint");
            ToolDrag::MovePoint(pti)
        } else {
            ToolDrag::Rect(SelectMode::Invert)
        }
    }

    fn tool_cancel(&mut self){self.last_point=None;}

    /*    fn lclick(&mut self, scn:&mut Scene, a:ViewPos){
            let ve=scn.add_vertex((a.0,a.1,0.0f32));
            match scn.state{
                EState::LastPoint(vs)=> scn.add_edge(ve,vs),
                _=>{}
            }
            scn.state=EState::LastPoint(ve);
        }
        fn rclick(&mut self, scn:&mut Scene, a:ViewPos){
            scn.state=EState::None;
        }
        fn try_drag(&self, e:&Scene, (mb,pos):MouseAt)->DragMode{
            DragMode::Line
        }
        */
}


/// box a struct and associate with a vtable, infering types from context.
/// TODO - needs the 'Unsized<TO>' fro nightly
//fn new_object<S:Clone,TO:?Sized>(data:S)->TO{
//    Box::new(data.clone()) as _
//}

type ZoomFactor=f32;
type InterestPoint=Vec3f;
enum SplitMode{RelHoriz,RelVert,RelDynamic,AbsLeft,AbsRight,AbsUp,AbsDown}

struct App{}
/*
type Win = window::Window<editor::App>;

trait SubWin<E>{

}

// this can be general purpose..
struct Split<OWNER>(f32, SplitMode, Box<window::Window<OWNER>>, Box<window::Window<OWNER>>);

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

impl Editor {
    fn push_op(&mut self, oo:optbox<Operation>){
        if let Some(o)=oo{
            o.op_dump();
			o.op_apply(&mut self.scene);
            self.doc.operators.push(o);
//            self.recompute_scene();
        }
    }
    fn transient_op(&mut self, oo:optbox<Operation>){
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

    fn recompute_scene(&mut self){
        // todo - logarithmic caching spacing eg [0       n/2      n-n/4 n-n/8 n-1 n]
        let mut s=Scene::default();
        for o in self.doc.operators.iter() {
            o.op_apply(&mut s);
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

/// Editor: handles tool switching and forwards events to the tool.
impl<A> Window<A> for Editor{

    fn key_mappings(&mut self, a:&A, f:&mut KeyMappings<A>){
        f('\x1b',"cancel",  &mut ||{self.cancel();Flow::Continue()});
        f('q',"back",  &mut ||Flow::Pop());
        f('1',"foo",  &mut ||Flow::Pop());
        f('2',"bar", &mut ||Flow::Pop());
    }
    fn on_key(&mut self, a:&mut A, k:KeyAt)->Flow<A>{
		if k.1==window::CTRL{println!("ctrl");}
		let vpos=k.3;
//todo -we want plain chars really
        match (k.0, k.1,k.2) {
//            ('s',KeyDown)=>self.set_tool(SelectTool()),
            ('d',0,KeyDown)=>self.set_tool(Box::new(DrawTool::default()) as Box<Tool>),
			//ctrl-z
			('\u{1a}',window::CTRL, KeyDown)=>self.undo(),
			('\u{19}',window::CTRL, KeyDown)=>self.redo(),
			('\u{18}',window::CTRL, KeyDown)=>self.cut(vpos),
			('\u{3}',window::CTRL, KeyDown)=>self.copy(vpos),
			('\u{16}',window::CTRL, KeyDown)=>self.paste(vpos),
            //s
            ('\x1b',0,KeyDown)=>{return Flow::Pop()},
            _=>()
        };
        Flow::Continue()
    }
    fn render(&self,a:&A, rc:&RC){
        let scn=&self.get_scene();
        for line in scn.main.edges.iter(){
            draw::line_c(&scn.main.vertices[line[0] as usize], &scn.main.vertices[line[1] as usize], g_color_wireframe)
        }
        for (i,v) in scn.main.vertices.iter().enumerate(){
        //    draw::vertex(v,2,0xff00ff00);
            draw::circle_fill_xy_c(&v,g_snap_radius*0.5f32,
                if scn.vertex_tags[i]{g_color_selected}else{g_color_wireframe}
            );
        }
        //for t in self.vertex_tags(){

        //}
		match self.dragstart{
	        Some(vs)=>self.tool.tool_render_drag(&self.drag, (vs.0,vs.1), (&self.get_scene(),&rc.mouse_pos),  rc),
			_=>self.tool.tool_render_passive(&self.presel, self.get_scene(),  rc),
		}
        draw::main_mode_text("lmb-draw rmb-cancel");
    }
    fn on_passive_move(&mut self,a:&mut A, opos:ViewPos, pos:ViewPos)->Flow<A> {
        self.presel=self.tool.tool_preselection((&self.scene/*not transient*/,&pos));
        self.tool.tool_passive_move(opos,(self.get_scene(),&pos));
        Flow::Continue()
    }
    fn on_ldrag_begin(&mut self,a:&mut A, startpos:ViewPos, pos:ViewPos)->Flow<A>{
		let drag=
			self.tool.tool_drag_begin(&self.presel, (self.get_scene(),&pos));
        self.drag=drag;
		self.dragstart=Some(startpos);
        Flow::Continue()
    }
    fn on_ldragging(&mut self, a:&mut A, startpos:ViewPos, pos:ViewPos )->Flow<A> {
		if let ToolDrag::None=self.drag{
		} else{
	        let transient_op=self.tool.tool_drag(&self.drag, startpos, (&self.scene/* not transient*/,&pos));
		    self.transient_op(transient_op);
		}
        Flow::Continue()
    }

    fn on_ldrag_end(&mut self, a:&mut A, startpos:ViewPos, pos:ViewPos)->Flow<A>{
        println!("editor ldrag end");
        self.transient_op(None);
        let op=self.tool.tool_drag_end(&self.drag, startpos,(&self.scene,&pos));
		if let Some(ref o)=op{o.op_dump();}
        self.push_op(op);
		self.dragstart=None;
        Flow::Continue()
    }

//    fn on_rdragging(&mut self, a:&mut A, startpos:ViewPos, pos:ViewPos )->Flow<A> {
//        self.tool.drag(startpos, (&self.app.scene,&pos));
//            Flow::Continue()
//    }
    fn on_lclick(&mut self, a:&mut A, vpos:ViewPos)->Flow<A>{
        println!("editor onclick");
        let op=self.tool.tool_lclick(&self.presel, (&self.scene,&vpos));
        self.push_op(op);
        Flow::Continue()
    }
    fn on_rclick(&mut self, a:&mut A, vpos:ViewPos)->Flow<A> {
        println!("editor onrclick");
        let op = self.tool.tool_rclick(&self.presel, (&self.scene, &vpos));
        self.push_op(op);
        Flow::Continue()
    }

    fn try_drag(&self, a:&A, (mb,pos):MouseAt )->DragMode{
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

impl Scene {
	fn get_vertices_in_rect(&self,(start,end):(ViewPos,ViewPos))->Vec<VertexIndex>
	{
		let mut ret=Vec::new();	
		let minv=v2min(&start,&end);
		let maxv=v2max(&start,&end);
		
		for (i,v) in self.main.vertices.iter().enumerate(){
			if inrange(v.0, (minv.0, maxv.0)) &&
				inrange(v.1, (minv.1,maxv.1)){
				ret.push(i)
			}
		}
		ret
	}
}	








