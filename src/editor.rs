use super::*;
//use window as w;
use window::{Flow,Event,MouseButtons};
type V3=(f32,f32,f32);

struct Cam{
    pos:V3,zoom:f32,
}

type VertexIndex=i32;
///
pub struct Editor {
    vertices:Vec<(V3)>,
    edges:Vec<[VertexIndex;2]>,
    cam:Cam,
    tool:Tool,
    state:EState
}

impl Editor{
    fn add_vertex(&mut self,p:V3)->VertexIndex{
        let vi=self.vertices.len();
        self.vertices.push(p);
        vi as VertexIndex
    }
    fn add_edge(&mut self,a:VertexIndex,b:VertexIndex){
        self.edges.push([a,b]);
    }
}

pub fn new() -> sto<window::State> {
    new!(Editor{
        vertices: vec![(-0.5f32,0.0f32,0.5f32),(0.5f32,-0.5f32,0.5f32),
                    (-1.0,-1.0,0.0),(1.0,1.0,0.0)],
        edges:vec![[0,1],[2,3]],
        cam:Cam{pos:(0.0,0.0,0.0),zoom:1.0},
        tool:Tool::DrawLine,
        state:EState::None,
    }=>window::State)
}

type DragStart=window::KeyAt;
type AfterDrag=Tool;

#[derive(Clone,Debug,Copy)]
enum Tool {
    Camera,
    Move,
    Select,
    DrawLine,
    Place,
}
impl Default for Tool{fn default()->Self{Tool::DrawLine}}

#[derive(Clone,Debug,Copy)]
enum EState{
    None,
    LastPoint(i32),
    Move(DragStart),
    Line(DragStart),
    Rect(DragStart),
}
impl Default for EState{fn default()->Self{EState::None}}

impl window::State for Editor{

    fn key_mappings(&mut self,f:&mut window::KeyMappings){
        f('\x1b',"back",  &mut ||Flow::Pop());
        f('1',"foo",  &mut ||Flow::Pop());
        f('2',"bar", &mut ||Flow::Pop());
    }
    fn on_key(&mut self,p:window::KeyAt)->Flow{

        match (p.0,p.1) {
            //s
            ('\x1b',true)=>Flow::Pop(),
            _=>Flow::Continue()
        }
    }
    fn render(&self,rc:&window::RenderContext){

        for l in self.edges.iter(){
            draw::line_c(&self.vertices[l[0] as usize], &self.vertices[l[1] as usize], 0xff00ff00)
        }
        draw::main_mode_text("lmb-draw rmb-cancel");
    }
    fn event(&mut self, e: Event)->Flow{
        println!("event:");dump!(e);
        match e{
            Event::Button(MouseButtons::Left,true,(x,y))=>{
                let ve=self.add_vertex((x,y,0.0f32));
                match self.state{
                    EState::LastPoint(vs)=> self.add_edge(ve,vs),
                    _=>{}
                }
                self.state=EState::LastPoint(ve);
            },
            Event::Button(MouseButtons::Right,true,(x,y))=> {
                self.state=EState::None;
            },
            _=>{}
        }
		Flow::Continue()
    }
}






