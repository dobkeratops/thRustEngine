//pub use std::libc::*;
//use r3d::gl_constants::*;
//use r3d::gl_h_consts::*;
//use r3d::glut_h_consts::*;
use std::os::raw::*;
pub type int=i32;
pub type uint=u32;


pub type GLenum=uint;
pub type GLboolean=u8;
pub type GLchar = c_char;
pub type GLvoid = c_void;
pub type GLbyte = i8;
pub type GLshort = i16;
pub type GLushort = u16;
pub type GLint= c_int;
pub type GLubyte= u8;
pub type GLuint= c_uint;
pub type GLsizei= c_int;
pub type GLfloat= f32;
pub type GLclampf= f32;
pub type GLdouble = f64;
pub type GLclampd=f64;	/* double precision float in [0,1] */

pub type RustTempCFunc=* const u8;
pub type PGLchar = *const GLchar;
pub type PGLvoid = *const GLvoid;
pub type PGLenum = *const GLenum;
pub type PGLfloat = *const GLfloat;
pub type PGLint = *const GLint;
pub type PGLuint = *const GLuint;
pub type PGLsizei = *const GLsizei;
extern
{
	pub fn hello_from_c(a:c_int,b:c_int);
	
	pub fn run_glut();
	pub fn setup_sub();
	pub fn displayCall();
	pub fn displaySub(txt:*const c_char);
	
	pub fn glewInit();
	
	pub fn glCompileShader(shader:GLuint);
	pub fn glShaderSource(shader:GLuint, count:GLsizei, string:*const *const GLchar, length:*const GLint);
	pub fn glGetShaderInfoLog(shader:GLuint, maxLength:GLsizei, length:*const GLsizei, infoLog:*const GLchar);
	pub fn glTexParameterf(target:GLenum,pname:GLenum,param:GLfloat);
	pub fn glTexParameteri(target:GLenum,pname:GLenum,param:GLint);

	pub fn glVertexAttribPointer(index:GLuint, size:GLint, _type:GLenum, normalized:GLboolean,stride:GLsizei, pointer:PGLvoid);
	pub fn glVertexAttribIPointer(index:GLuint, size:GLint, _type:GLenum, stride:GLsizei, pointer:PGLvoid);
	pub fn glVertexAttribLPointer(index:GLuint, size:GLint, _type:GLenum, stride:GLsizei, pointer:PGLvoid);
	pub fn glColorPointer(size:GLint, _type:GLenum, stride:GLsizei, pointer:PGLvoid);
	pub fn glVertexPointer(size:GLint, _type:GLenum, stride:GLsizei, pointer:PGLvoid);
	pub fn glNormalPointer(_type:GLenum, stride:GLsizei, pointer:PGLvoid);
	pub fn glTexCoordPointer(size:GLint, _type:GLenum, stride:GLsizei, pointer:PGLvoid);

	
	pub fn glCreateShader(shaderType:GLenum)->GLuint;
	pub fn glCreateProgram()->GLuint;
	pub fn glBindAttribLocation(prog:GLuint, index:GLuint, name:PGLchar);
	pub fn glAttachShader(program:GLuint, shader:GLuint);
	pub fn glLinkProgram(program:GLuint);
	pub fn glBindBuffer(target:GLenum, buffer:GLuint);
	pub fn glEnableVertexAttribArray(index:GLuint);
	pub fn glActiveTexture(texture:GLenum);
	pub fn glDrawElements(mode:GLenum, count:GLsizei, _type:GLenum, indices:PGLvoid);
	pub fn glDrawArrays(mode:GLenum,first:GLsizei, count:GLsizei,);
	pub fn glUseProgram(prog:GLuint);
	pub fn glGetActiveAttrib(program:GLuint, index:GLuint, bufsize:PGLsizei, length:PGLsizei, size:GLint, _type:PGLenum, name:PGLchar);
	pub fn glGetActiveUniform(program:GLuint, index:GLuint, bufsize:GLsizei, length:PGLsizei, size:PGLint, _type:PGLenum, name:PGLchar); 
	pub fn glGetAttachedShaders(program:GLuint, maxCount:GLsizei, count:GLsizei, shaders:GLuint);
	pub fn glGetAttribLocation(prog:GLuint, name:PGLchar)->GLint;
	pub fn glGetUniformfv(prog:GLuint, location:GLint, params:PGLfloat);
	pub fn glGetUniformiv(prog:GLuint, location:GLint, params:PGLint);
	
	pub fn glGetUniformLocation(prog:GLuint, name:PGLchar)->GLint;
	
	pub fn glUniform1f(location:GLint, v0:GLfloat);
	pub fn glUniform2f(location:GLint, v0:GLfloat, v1:GLfloat);
	pub fn glUniform3f(location:GLint, v0:GLfloat, v1:GLfloat, v2:GLfloat);
	pub fn glUniform4f(location:GLint, v0:GLfloat, v1:GLfloat, v2:GLfloat, v3:GLfloat);
	pub fn glUniform1i(location:GLint, v0:GLint);
	pub fn glUniform2i(location:GLint, v0:GLint, v1:GLint);
	pub fn glUniform3i(location:GLint, v0:GLint, v1:GLint, v2:GLint);
	pub fn glUniform4i(location:GLint, v0:GLint, v1:GLint, v2:GLint, v3:GLint);

	pub fn glUniform1fv(location:GLint, count:GLsizei, v0:PGLfloat);
	pub fn glUniform2fv(location:GLint, count:GLsizei, v0:PGLfloat);
	pub fn glUniform3fv(location:GLint, count:GLsizei, v0:PGLfloat);
	pub fn glUniform4fv(location:GLint, count:GLsizei, v0:PGLfloat);
	pub fn glUniform1iv(location:GLint, count:GLsizei, v0:PGLint);
	pub fn glUniform2iv(location:GLint, count:GLsizei, v0:PGLint);
	pub fn glUniform3iv(location:GLint, count:GLsizei, v0:PGLint);
	pub fn glUniform4iv(location:GLint, count:GLsizei, v0:PGLint);

	pub fn glUniformMatrix2fv(location:GLint, count:GLsizei,transpose:GLboolean, v0:PGLfloat);
	pub fn glUniformMatrix3fv(location:GLint, count:GLsizei,transpose:GLboolean, v0:PGLfloat);
	pub fn glUniformMatrix4fv(location:GLint, count:GLsizei,transpose:GLboolean, v0:PGLfloat);
	pub fn glUniformMatrix4fvARB(location:GLint, count:GLsizei,transpose:GLboolean, v0:PGLfloat);

	pub fn glGetProgramiv(program:GLuint, pname:GLenum, params:PGLint);
	pub fn glGetProgramInfoLog(program:GLuint, maxlength:GLsizei, length:PGLsizei, infoLog:PGLchar);
	pub fn glGetShaderiv(shader:GLuint, pname:GLenum, params:PGLint);
	pub fn glIsProgram(program:GLuint)->GLboolean;
	pub fn glShaderBinary(n:GLsizei, shaers:PGLuint, binary_format:GLenum, binary:*const c_void, length:PGLsizei);
	pub fn glIsShader(program:GLuint)->GLboolean;
	pub fn glDeleteProgram(program:GLuint);
	pub fn glDeleteShader(program:GLuint);
	
	pub fn glGenBuffers(num:GLsizei, buffers:*mut GLuint);
	pub fn glDeleteBuffers(num:GLsizei, buffers:PGLuint);
	pub fn glIsBuffer(buffer:GLuint)->GLboolean;

	pub fn glLoadIdentity();
	pub fn glMatrixMode(e:GLenum);
	pub fn glClear(e:GLenum);
	pub fn glLoadMatrixf(f:&GLfloat);

	pub fn glClearColor(r:GLfloat,g:GLfloat,b:GLfloat,a:GLfloat);
	pub fn glAlphaFunc( func:GLenum , _ref:GLclampf );
	pub fn glBlendFunc( sfactor:GLenum, dfactor:GLenum );
	pub fn glLogicOp( opcode:GLenum );
	pub fn glCullFace( mode:GLenum );
	pub fn glDrawBuffer( mode:GLenum  );
	pub fn glReadBuffer( mode:GLenum  );
	pub fn glEnable(e:GLenum);
	pub fn glDisable(e:GLenum);
	pub fn glEnableClientState( cap:GLenum );  /* 1.1 */
	pub fn glDisableClientState( cap:GLenum );  /* 1.1 */
	pub fn glGetFloatv( pname:GLenum, params:*mut GLfloat );

    pub fn glViewport(_:GLint,_:GLint,_:GLint,_:GLint);
    pub fn glScissor(	x:GLint, y:GLint, width:GLsizei, height:GLsizei);

pub fn glGetIntegerv( pname:GLenum, params:* mut GLint );
	pub fn glRenderMode( mode:GLenum )->GLint;
	pub fn glGetError()->GLenum;
	pub fn glGetString( name:GLenum  )->*const GLubyte;
	pub fn glFinish( );
	pub fn glFlush(  );
	pub fn glHint( target:GLenum, mode:GLenum );
	pub fn glClearDepth(  depth:GLclampd );
	pub fn glDepthFunc(  func:GLenum );
	pub fn glDepthMask( flag:GLboolean  );
	pub fn glDepthRange( near_val:GLclampd , far_v:GLclampd );

	pub fn glRasterPos2f( x:GLfloat, y:GLfloat );
	pub fn glRasterPos2i( x:GLint, y:GLint );
    pub fn glRasterPos3f(_:f32,_:f32,_:f32);


	pub fn glOrtho(x0:GLdouble,y0:GLdouble,z0:GLdouble,x1:GLdouble,y1:GLdouble,z1:GLdouble);
	pub fn gluLookAt(posx:GLdouble,posy:GLdouble,posz:GLdouble,atx:GLdouble,aty:GLdouble,atz:GLfloat, upx:GLdouble,upy:GLdouble,upz:GLdouble);
	pub fn glRotatef(angle:GLfloat,axisx:GLfloat,axisy:GLfloat,axisz:GLfloat);	
	pub fn glTranslatef(x:GLfloat,y:GLfloat,z:GLfloat);
	pub fn glScalef(x:GLfloat,y:GLfloat,z:GLfloat);
	
	pub fn gluPerspective(fovy:GLdouble, aspect:GLdouble, znear:GLdouble, zfar:GLdouble);

	pub fn glGenTextures( n:GLsizei, textures:*mut GLuint );

	pub fn glDeleteTextures( n:GLsizei, textures: PGLuint);

	pub fn glBindTexture( target:GLenum , texture:GLuint );
	

	pub fn glTexImage1D( target:GLenum, level:GLint,
                                    internalFormat:GLint ,
                                    width:GLsizei, border:GLint ,
                                    format:GLenum, _type:GLenum,
                                    pixels: PGLvoid );

	pub fn glTexImage2D( target:GLenum, level:GLint,
                                    internalFormat:GLint ,
                                    width:GLsizei, height:GLsizei ,
                                    border:GLint , format:GLenum , _type:GLenum,
                                    pixels: *const u8 );

	pub fn glGetTexImage( target:GLenum, level:GLint,
                                     format:GLenum, _type:GLenum,
                                     pixels:*mut GLvoid );

	pub fn glDrawPixels( width:GLsizei, height:GLsizei,
                                    format:GLenum, _type:GLenum,
                                    pixels:PGLvoid );
	pub fn glReadPixels( x:GLint,y: GLint,
                                    width:GLsizei, height:GLsizei,
                                    format:GLenum, _type:GLenum,
                                    pixels:*mut GLvoid );

	pub fn glTexEnvf( target:GLenum, pname:GLenum , param:GLfloat );
	pub fn glTexEnvi( target:GLenum, pname:GLenum, param :GLint );

/* 1.1 functions */

	pub fn glBegin(mode:GLenum);
	pub fn glEnd();
	pub fn glColor3f(r:GLfloat,g:GLfloat,b:GLfloat);
	pub fn glColor4f(r:GLfloat,g:GLfloat,b:GLfloat,a:GLfloat);

    pub fn glVertex2f(x:GLfloat,y:GLfloat);
    pub fn glVertex2d(x:GLdouble,y:GLdouble);
    pub fn glVertex2i(x:GLint,y:GLint);
    pub fn glVertex2s(x:GLshort,y:GLshort);

    pub fn glVertex3f(x:GLfloat,y:GLfloat,z:GLfloat);
    pub fn glVertex3d(x:GLdouble,y:GLdouble,z:GLdouble);
    pub fn glVertex3i(x:GLint,y:GLint,z:GLint);
    pub fn glVertex3s(x:GLshort,y:GLshort,z:GLshort);

    pub fn glVertex4f(x:GLfloat,y:GLfloat,z:GLfloat,w:GLfloat);
    pub fn glVertex4d(x:GLdouble,y:GLdouble,z:GLdouble,w:GLdouble);
    pub fn glVertex4i(x:GLint,y:GLint,z:GLint,w:GLint);
    pub fn glVertex4s(x:GLshort,y:GLshort,z:GLshort,w:GLshort);
	pub fn glNormal3f(x:GLfloat,y:GLfloat,z:GLfloat);
    pub fn glVertex2fv(v:*const GLfloat);
    pub fn glVertex3fv(v:*const GLfloat);
    pub fn glVertex4fv(v:*const GLfloat);
    pub fn glVertex2dv(v:*const GLdouble);
    pub fn glVertex3dv(v:*const GLdouble);
    pub fn glVertex4dv(v:*const GLdouble);


    pub fn glTexCoord2f( s:GLfloat , t:GLfloat );

	pub fn glutInit(argc:*mut c_int,argc:*const *const c_char);
	pub fn glutInitDisplayMode(mode:GLenum);
	pub fn glutCheckLoop();

	pub fn glutCreateWindow(x:*const c_char)->c_int;
	pub fn glutCreateSubWindow(x:*const c_char)->c_int;
	pub fn glutDestroyWindow(x:c_int);
	pub fn glutSetWindow(win:c_int);
	pub fn glutGetWindow()->c_int;
	pub fn glutGet(_:GLuint)->c_int;
	pub fn glutGetModifiers()->c_int;
	pub fn glutSetWindowTitle(x:*const c_char);
	pub fn glutSetIconTitle(x:*const c_char);
	pub fn glutReshapeWindow(x:GLint, y:GLint);
	pub fn glutPositionWindow(x:c_int,y:c_int);
	pub fn glutIconifyWindow();
	pub fn glutShowWindow();
	pub fn glutHideWindow();
	pub fn glutPushWindow();
	pub fn glutPopWindow();
	pub fn glutFullScreen();
	pub fn glutPostRedisplay();

	pub fn glBufferData(target:GLenum, size:GLsizei, data:PGLvoid, usage:GLenum);
    pub fn glutBitmapCharacter(_:*const c_void, _:c_char)->();
	pub static mut glutStrokeMonoRoman:&'static u8;
	pub static mut glutBitmap8By13:&'static u8;

	pub fn glutPostWindowRedisplay(  window:c_int );
    pub fn glutSwapBuffers();
    pub fn glutMainLoopEvent(); /* for user loop to poll messages*/
    pub fn glutMainLoop(); /* for user loop to poll messages*/

/*
 * Mouse cursor functions, see freeglut_cursor.c
 */
	pub fn glutWarpPointer( x:c_int,y:c_int );
	pub fn glutSetCursor( cursor:c_int);


/*
 * Global callback functions, see freeglut_callbacks.c
 */
	pub fn glutTimerFunc( time_val:c_uint, f:RustTempCFunc/*&fn( v:c_int )*/,  data:c_int );
	pub fn glutIdleFunc(f:RustTempCFunc);// f:&fn() );
	pub fn glutDisplayFunc(f:RustTempCFunc);// f:&fn() );

	pub fn glutGameModeString(s:*const c_char );
	pub fn glutEnterGameMode( );
	pub fn glutLeaveGameMode( );
	pub fn glutGameModeGet( query:GLenum  );

	pub fn glutInitWindowPosition(x:GLint,y:GLint);
	pub fn glutInitWindowSize(x:GLint,y:GLint);

	pub fn glutSetKeyRepeat( repeatMode:c_int);

	pub fn glutMouseFunc(f:RustTempCFunc);//f:f:extern "C" fn(button:c_int, state:c_int,x:c_int, y:c_int));
	pub fn glutMotionFunc(f:RustTempCFunc);//f:f:extern "C" fn(x:c_int, y:c_int));
	pub fn glutPassiveMotionFunc(f:RustTempCFunc);//f:extern "C" fn(x:c_int, y:c_int));
	pub fn glutEntryFunc(f:RustTempCFunc);//f:f:extern "C" fn (e:c_int ) );
	pub fn glutKeyboardFunc(f:RustTempCFunc);//f:extern "C" fn(key:c_uchar,x:c_int, y:c_int));
	pub fn glutKeyboardUpFunc(f:RustTempCFunc);//f:extern "C" fn(button:c_uchar, x:c_int, y:c_int));
	pub fn glutSpecialFunc(f:RustTempCFunc);//f:f:extern "C" fn(button:c_int,x:c_int, y:c_int));
	pub fn glutSpecialUpFunc(f:RustTempCFunc);//f:f:extern "C" fn(button:c_int,x:c_int, y:c_int));
	pub fn glutReshapeFunc(f:RustTempCFunc);//f:f:extern "C" fn(x:c_int,y:c_int));
	pub fn glutTabletMotionFunc(f:extern "C" fn( x:c_int, y:c_int ) );
	pub fn glutTabletButtonFunc(f:extern "C" fn( button:c_int, state:c_int, x:c_int, y:c_int ) );
	pub fn glutJoystickFunc(f:RustTempCFunc,i:c_int);
	//pub fn glewInit();

/*
	Text functions
*/
	pub fn glutStrokeCharacter(c:*const c_void, c:c_char);

}
pub fn glSetMatrix(e:GLenum, f:&GLfloat){ unsafe{glMatrixMode(e);glLoadMatrixf(f);}}
pub fn glSetMatrixIdentity(e:GLenum){ unsafe{glMatrixMode(e);glLoadIdentity();}}
