use super::*;
use ::bsp::bspdraw as draw;
use window as w;
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
    state:ES
}

///todo: grow a MeshEditor interface.
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

pub fn new() -> sto<w::State> {
    new!(Editor{
        vertices: vec![(-0.5f32,0.0f32,0.5f32),(0.5f32,-0.5f32,0.5f32),
                    (-1.0,-1.0,0.0),(1.0,1.0,0.0)],
        edges:vec![[0,1],[2,3]],
        cam:Cam{pos:(0.0,0.0,0.0),zoom:1.0},
        tool:Tool::DrawLine,
        state:ES::None,
    }=>w::State)
}

type DragStart=w::KeyAt;
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
enum ES{
    None,
    LastPoint(i32),
    Move(DragStart),
    Line(DragStart),
    Rect(DragStart),
}
impl Default for ES{fn default()->Self{ES::None}}

impl w::State for Editor{
    fn key_mappings(&mut self,f:&mut w::KeyMappings){
        f('\x1b',"back",  &mut ||Flow::Pop());
        f('1',"add obj",  &mut ||Flow::Pop());
        f('2',"add trigger", &mut ||Flow::Pop());
    }

    fn on_key(&mut self,p:w::KeyAt)->w::Flow{
        match (p.0,p.1) {
            //s
            ('\x1b',true)=>w::Pop(),
            _=>w::Continue()
        }
    }
    fn render(&self,_:f32){
        for l in self.edges.iter(){
            draw::line(&self.vertices[l[0] as usize], &self.vertices[l[1] as usize], 0xff00ff00)
        }
        draw::main_mode_text("editor");
    }
    fn event(&mut self, e:w::Event)->w::Flow{
        println!("event:");dump!(e);
        match e{
            w::MouseButton(w::LeftButton,true,(x,y))=>{
                let ve=self.add_vertex((x,y,0.0f32));
                match self.state{
                    ES::LastPoint(vs)=> self.add_edge(ve,vs),
                    _=>{}
                }
                self.state=ES::LastPoint(ve);
            },
            w::MouseButton(w::RightButton,true,(x,y))=> {
                self.state=ES::None;
            },
            _=>{}
        }

        w::Continue()
    }
}

/*
trait Pane {
    fn size(&self)->w::PixelSize;

}
*/

// map view may rend




