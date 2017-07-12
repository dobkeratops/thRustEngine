use super::*;
use ::bsp::bspdraw as draw;
type V3=(f32,f32,f32);

pub struct Editor {
    lines:Vec<(V3,V3)>,
    tool:Tool,
    drag:Drag
}

pub fn new() -> sto<window::State> {
    new!(Editor{
        lines: vec![],
        tool:Tool::DrawLine,
        drag:Drag::None,
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
enum Drag{
    None,
    Move(DragStart),
    Line(DragStart),
    Rect(DragStart),
}
impl Default for Drag{fn default()->Self{Drag::None}}

impl window::State for editor::Editor{
    fn key_mappings(&mut self,f:&mut window::KeyMappings){
        f('\x1b',"back",  &mut ||Flow::Pop());
        f('1',"add obj",  &mut ||Flow::Pop());
        f('2',"add trigger", &mut ||Flow::Pop());
    }

    fn on_key_down(&mut self,p:window::KeyAt)->window::Flow{
        match p.0 {
            //s
            '\x1b'=>Flow::Pop(),
            _=>Flow::Continue()
        }
    }
    fn render(&self,_:f32){
        for l in self.lines.iter(){
            draw::line(&l.0, &l.1, 0xff00ff00)
        }
        ::bsp::bspdraw::main_mode_text("editor");
    }
}

// map view may rend




