use super::*;

pub static g_minimal_vs:&'static str="
//#version 120
    attribute vec4 position;    
	uniform mat4 uMatProj;
	uniform mat4 uMatModelView;
    void main()                  
    {                     
		vec4 vpos=vec4(position.xyz,1.0);
		vec4 epos=uMatProj*uMatModelView*vpos;       
       gl_Position = epos;  
    }                           
\0"; 
static g_minimal_fs:&'static str ="
//#version 120
    precision mediump float;
    void main()                                  
    {                                            
      gl_FragColor = vec4 (1.0, 1.0, 1.0, 1.0 );
    }                               
\0";

pub fn mainr(){
	unsafe{subr()};
}
pub static mut g_vao:GLuint=0;
pub static mut g_vbo:GLuint=0;
pub static mut g_sp:GLuint=0;

unsafe fn subr(){
	trace!();
	SDL_Init(SDL_INIT_VIDEO);
    SDL_GL_SetAttribute(SDL_GL_CONTEXT_MAJOR_VERSION, 2);
    SDL_GL_SetAttribute(SDL_GL_CONTEXT_MINOR_VERSION, 0);

	SDL_GL_SetAttribute( SDL_GL_RED_SIZE, 5 );
    SDL_GL_SetAttribute( SDL_GL_GREEN_SIZE, 5 );
    SDL_GL_SetAttribute( SDL_GL_BLUE_SIZE, 5 );
    SDL_GL_SetAttribute( SDL_GL_DEPTH_SIZE, 24 );
    SDL_GL_SetAttribute( SDL_GL_DOUBLEBUFFER, 1 );
	trace!();

	let w=512; let h=384;
		SDL_SetVideoMode(w,h,16,SDL_OPENGL);

	trace!();
	let mut vao:GLuint=0;
	let mut vao1:GLuint=0;
	glGenVertexArrays(1,&mut vao);
	glGenVertexArrays(1,&mut vao1);
	dump!(vao);
	glBindVertexArray(vao1);	
	glBindVertexArray(vao);	
	let mut vbo:GLuint=0;
	trace!();
	glGenBuffers(1,&mut vbo);
	let vertices=vec![0.0f32,0.5f32,0.0f32, 0.5f32,-0.5f32,0.0f32, -0.5f32,-0.5f32,0.0f32];
	glBindBuffer(GL_ARRAY_BUFFER,vbo);
	glBufferData(GL_ARRAY_BUFFER, 3*3 *4, 
		&vertices[0] as *const _  as *const c_void,
		GL_STATIC_DRAW);
	let vs=glCreateShader(GL_VERTEX_SHADER);
	let vss=[c_str(g_minimal_vs)];
	trace!();
	gl_verify!{glShaderSource(vs,1,&vss as *const *const c_char,0 as *const c_int);}
	gl_verify!{glCompileShader(vs);}
	let fss=[c_str(g_minimal_fs)];
	let fs=glCreateShader(GL_FRAGMENT_SHADER);	
	dump!(vs,fs);
	gl_verify!{glShaderSource(fs, 1,&fss as *const *const c_char, 0 as *const c_int);}
	gl_verify!{glCompileShader(fs);}
	trace!();
	let sp=glCreateProgram();
	gl_verify!{glAttachShader(sp,vs);}
	gl_verify!{glAttachShader(sp,fs);}
	gl_verify!{glLinkProgram(sp);}
	gl_verify!{glUseProgram(sp);}
	g_vao=vao; g_vbo=vbo; g_sp=sp;

	trace!();
	let a_pos=glGetAttribLocation(sp,c_str("position\0"));
	dump!(a_pos);
		glViewport(0,0,w,h);
		glClearColor(0.0,1.0,0.0,0.0);
		glClear(GL_COLOR_BUFFER_BIT);
		//glDrawArrays(GL_TRIANGLES,0,3);
        SDL_GL_SwapBuffers();
		glClearColor(0.0,1.0,1.0,0.0);
		glClear(GL_COLOR_BUFFER_BIT);
		//glDrawArrays(GL_TRIANGLES,0,3);
        SDL_GL_SwapBuffers();
	trace!();
	dump!(g_shader_program);
	mainloop();
	mainloop();
	mainloop();
	mainloop();
	emscripten::emscripten_set_main_loop(mainloop as *const u8, 0,1);	
}

pub  fn mainloop(){
	unsafe{
		glClearColor(0.5,0.4,0.6,0.0);
		glEnable(GL_DEPTH_TEST);
		glClear(GL_COLOR_BUFFER_BIT|GL_DEPTH_BUFFER_BIT);

		glUseProgram(g_sp);		
		glBindVertexArray(g_vao);	
		glBindBuffer(GL_ARRAY_BUFFER,g_vbo);
		let a_pos=glGetAttribLocation(g_sp,c_str("position\0"));
		gl_verify!{glEnableVertexAttribArray(a_pos as u32);};
		gl_verify!{glVertexAttribPointer(a_pos as u32, 3, GL_FLOAT,GL_FALSE,3*4,0 as *const c_void);}

			let mp= glGetUniformLocation(	g_sp,c_str("uMatProj\0"));
			let mmv= glGetUniformLocation(	g_sp,c_str("uMatModelView\0"));
		//dump!(mp,mmv);
		let imat:[f32;16]=
			[	1.0,0.0,0.0,0.0,
				0.0,1.0,0.0,0.0,
				0.0,0.0,1.0,0.0,
				0.0,0.0,0.0,1.0
			];
		glUniformMatrix4fv(mp,1,GL_FALSE,&imat[0]);
		glUniformMatrix4fv(mmv,1,GL_FALSE,&imat[0]);
				
		glDrawArrays(GL_LINE_STRIP,0,3);
		glBindVertexArray(0);	
		glBindBuffer(GL_ARRAY_BUFFER,0);	

		::render_no_swap(0);
		glFlush();
        SDL_GL_SwapBuffers();
	}
}
