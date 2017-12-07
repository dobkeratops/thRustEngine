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

type VertexIndex=i32;

#[derive(Default)]
pub struct Scene{
    vertices:   Vec<(V3)>,
    edges:      Vec<[VertexIndex;2]>,
    vertex_tags:Vec<usize>,
}
#[derive(Default)]
pub struct Doc {
    operators:vecbox<Operation>,
    scene:Scene,
}

pub struct App {
    doc:Doc,
    scene:Scene,    // cached generated
    cam:   Cam,
}

// routes commands to editable scene.
enum SelectCommand{ Select,Deselect,Toggle }

impl Scene{
}

// todo - parameterize 'scene'
// Editor<Scene>, MTool<Scene>

// editor app, data parts.
// TODO - is the editor the app?
// should we just pass ownership of the state into the editor when we 'fire it up' ?

pub struct Editor {
    app     :App,
    tool    :Box<ITool>,
    saved_tool    :optbox<ITool>, // todo.. push em in a vector surely
}
impl Editor {
    fn push_tool(&mut self, newtool: Box<ITool>) {
        assert!(!self.saved_tool.is_some());
        self.saved_tool = Some(std::mem::replace(&mut self.tool, newtool));
    }
    fn pop_tool(&mut self) {
        assert!(self.saved_tool.is_some());
        self.tool=std::mem::replace(&mut self.saved_tool, None).unwrap();
    }
    fn set_tool(&mut self, newtool: Box<ITool>){
        self.tool = newtool;
    }
}

impl Scene{
    fn add_vertex(&mut self,p:V3)->VertexIndex{
        let vi=self.vertices.len();
        self.vertices.push(p);
        vi as VertexIndex
    }
    fn add_edge(&mut self,a:VertexIndex,b:VertexIndex){
        self.edges.push([a,b]);
    }
}

pub fn new<A>() -> sto<Window<A>> {
    Box::new(
        Editor {
            app:App{
                doc:Doc::default(),
                scene: Scene {
                    vertices: vec ! [( - 0.5f32, 0.0f32, 0.5f32), (0.5f32, - 0.5f32, 0.5f32),
                    ( - 1.0, - 1.0, 0.0), (1.0, 1.0, 0.0)],
                    vertex_tags: vec ! [],
                    edges: vec ! [[0, 1], [2, 3]],
                },
                cam: Cam { pos: (0.0, 0.0, 0.0), zoom: 1.0 },
            },
            tool: GetITool::<DrawTool>(),
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
    fn name(&self)->String;
    fn dump(&self);
    // todo - show UI - tweakable parameters.
    //fn num_params();
    //fn foreach_param((name:string,value:f32));
    fn apply(&self, &Scene)->Scene;
    fn can_collapse_with<'e>(&self, other:&'e Operation)->bool {false}
    fn collapse_with<'e>(&self, other:&'e Operation)->optbox<Operation> {None}
}

//
type SceneViewPos<'e>=(&'e Scene, &'e ViewPos);

pub trait Tool{
    type Presel;        // information about where the cursor is and what would happen if you click *now*
    type Drag;
    fn new()->Self;
    fn activate(&mut self){}
    fn deactivate(&mut self){}
    fn compute_preselection(&self, e:SceneViewPos)->Self::Presel; // common computation between highlight & operation
    fn lclick(&mut self, p:&Self::Presel, e:SceneViewPos)->optbox<Operation>{return None;}
    fn rclick(&mut self, p:&Self::Presel, e:SceneViewPos)->optbox<Operation>{return None;}
    fn drag_end(&mut self, p:&Self::Presel, d:&Self::Drag, e:SceneViewPos, drag:&Self::Drag)->optbox<Operation>{ return None;}
    fn drag_begin(&mut self, p:&Self::Presel, e:SceneViewPos )->optbox<Self::Drag>{None}
    fn render(&self, p:&Self::Presel, e:&Scene, rc:&RC){}
    fn render_drag(&self, p:&Self::Presel, d:&Self::Drag, e:SceneViewPos, rc:&RC){}
    fn mouse_move(&mut self, opos:ViewPos, e:SceneViewPos){}
    fn cancel(&mut self){}
    //fn try_drag(&self, e:SceneView, mbpos:(MouseButtons,ViewPos))->DragMode{DragMode::Rect }
}
// wrapper for the tool handles Drag,Presel.
//those can't be managed by the app because those types are specific to the tool!

struct ToolWrapper<TOOL : Tool> {
    tool:TOOL,
    presel:Option<TOOL::Presel>,
    drag:Option<TOOL::Drag>,
}

trait ITool {
    fn activate(&mut self);
    fn deactivate(&mut self);
    fn lclick(&mut self, e:SceneViewPos)->optbox<Operation>;
    fn rclick(&mut self, e:SceneViewPos)->optbox<Operation>;
    fn mouse_move(&mut self, opos:ViewPos, e:SceneViewPos);
    fn render(&self, s:&Scene, r:&RC);
}

impl<TOOL : Tool> ITool for ToolWrapper<TOOL> {
    fn activate(&mut self)  {self.tool.activate();}
    fn deactivate(&mut self){ self.tool.deactivate();}
    fn mouse_move(&mut self, opos:ViewPos, e:SceneViewPos){
        self.presel=Some(self.tool.compute_preselection(e));
        self.tool.mouse_move(opos,e);
    }
    fn  render(&self, e:&Scene, rc:&RC){
        if let Some(ref p)=self.presel { self.tool.render(p,e,rc);}
    }
    fn lclick(&mut self, e:SceneViewPos)->optbox<Operation>{
        if let Some(ref p)=self.presel {self.tool.lclick(p,e)}
        else {None}
    }
    fn rclick(&mut self, e:SceneViewPos)->optbox<Operation>{
        if let Some(ref p)=self.presel {self.tool.lclick(p,e)}
            else {None}
    }
}

fn GetITool<TOOL:Tool+'static>()->Box<ITool>{
    Box::new(ToolWrapper::<TOOL>{drag:None, presel:None, tool:TOOL::new()}) as Box<ITool>
}

/*
impl Tool for SelectTool{
    fn lclick(&mut self,scn:&mut Scene, a:ViewPos){
        let ve=scn.add_vertex((a.0,a.1,0.0f32));
        match scn.state{
            EState::LastPoint(vs)=> scn.add_edge(ve,vs),
            _=>{}
        }
        scn.state=EState::LastPoint(ve);
    }
    fn rclick(&mut self,scn:&mut Scene, a:ViewPos){
        scn.state=EState::None;
    }
    fn try_drag(&self,e:&Scene, (mb,pos):MouseAt)->DragMode{
        DragMode::Rect
    }
}
*/

#[derive(Default)]
struct DrawTool();

impl Tool for DrawTool{
    type Presel = ();
    type Drag = ();
    fn new()->Self {DrawTool()}
    fn compute_preselection(&self, e:SceneViewPos)->Self::Presel{
        ()
    }


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

type SubWin = window::Window<editor::App>;
type BSubWin = Box<SubWin>;

type ZoomFactor=f32;
type InterestPoint=Vec3f;
enum SplitMode{RelHoriz,RelVert,RelDynamic,AbsLeft,AbsRight,AbsUp,AbsDown}

struct Split(f32, SplitMode, BSubWin, BSubWin);
//fn Split(

struct ViewPane(ViewMode);

pub enum ViewMode{XY,XZ,YZ,Perspective}

/*
fn create_4views()->BSubWin{

    Box::new(
            Box::new(
                Split(
                    0.5,
                    SplitMode::RelDynamic,
                    Box::new(
                        Split(0.5, SplitMode::RelDynamic,
                            Box::new(ViewPane(ViewMode::XY)) as _,
                            Box::new(ViewPane(ViewMode::XZ)) as _
                        )
                    ) as _,
                    Box::new(
                        Split(0.5, SplitMode::RelDynamic,
                            Box::new(ViewPane(ViewMode::YZ)) as _,
                            Box::new(ViewPane(ViewMode::Perspective)) as _
                        )
                    ) as _,
                )
            ) as _
    ) as _

}
*/
/// Editor: handles tool switching and forwards events to the tool.
impl<A> Window<A> for Editor{

    fn key_mappings(&self, a:&A, f:&mut KeyMappings<A>){
        f('\x1b',"back",  &mut ||Flow::Pop());
        f('1',"foo",  &mut ||Flow::Pop());
        f('2',"bar", &mut ||Flow::Pop());
    }
    fn on_key(&mut self,a:&mut A, p:KeyAt)->Flow<A>{

        match (p.0,p.1) {
//            ('s',KeyDown)=>self.set_tool(SelectTool()),
            ('d',KeyDown)=>self.set_tool(GetITool::<DrawTool>()),
            //s
            ('\x1b',KeyDown)=>{return Flow::Pop()},
            _=>()
        };
        Flow::Continue()
    }
    fn render(&self,a:&A, rc:&RC){
        let scn=&self.app.scene;
        for line in scn.edges.iter(){
            draw::line_c(&scn.vertices[line[0] as usize], &scn.vertices[line[1] as usize], 0xff00ff00)
        }
        for v in scn.vertices.iter(){
        //    draw::vertex(v,2,0xff00ff00);
        }
        //for t in self.vertex_tags(){

        //}
        self.tool.render(&self.app.scene,rc);
        draw::main_mode_text("lmb-draw rmb-cancel");
    }

    fn on_lclick(&mut self,a:&mut A, p:ViewPos)->Flow<A>{
        self.tool.lclick((&self.app.scene,&p));
        Flow::Continue()
    }
    fn on_rclick(&mut self,a:&mut A, vp:ViewPos)->Flow<A>{
        self.tool.rclick((&self.app.scene,&vp));
        Flow::Continue()
    }

    fn try_drag(&self,a:&A,(mb,pos):MouseAt)->DragMode{
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








