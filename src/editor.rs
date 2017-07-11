use super::*;
use ::bsp::bspdraw as draw;
type V3=(f32,f32,f32);


pub struct Editor {
    lines:Vec<(V3,V3)>,
}

pub fn new() -> Box<window::State> {
    Box::new(Editor{
        lines: vec![],
    }) as Box<window::State>
}

impl window::State for editor::Editor{
    fn on_keypress(&mut self,key:u8,xy:[i32;2])->window::Flow{
        match key{
            //s
            27=>Flow::Pop(),
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

