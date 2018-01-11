use super::*;

struct s{

}



pub fn new<A>()->Box<Window<A>> {
    Box::new(s{

    }) as _
}

impl<A> Window<A> for s{

    fn win_render(&self, a:&A, wc:&window::WinCursor){
        //draw::line( (1.0,2.0,1.0), (0.5,0.5,0.5), )
        unsafe {
            draw::clear(0x808080);
            draw::main_mode_text("hello world");
            draw::line(&vec3(-1.0,0.0,0.5),&vec3(1.0,0.0,0.5));
            draw::line(&vec3(-1.0,0.0,-0.5),&vec3(1.0,0.0,-0.5));
            glFlush();
        }
    }
}


/*
pub fn new()-> ~Window {
    let x=~Test{

    }
    x as ~Window;
}
*/
