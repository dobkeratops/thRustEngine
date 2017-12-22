use super::*;
pub const SDL_INIT_TIMER:u32=          0x00000001;
pub const SDL_INIT_AUDIO:u32=          0x00000010;
pub const SDL_INIT_VIDEO:u32=          0x00000020;  /**< SDL_INIT_VIDEO implies SDL_INIT_EVENTS */
pub const SDL_INIT_JOYSTICK:u32=       0x00000200;  /**< SDL_INIT_JOYSTICK implies SDL_INIT_EVENTS */
pub const SDL_INIT_HAPTIC:u32=         0x00001000;
pub const SDL_INIT_GAMECONTROLLER:u32= 0x00002000;  /**< SDL_INIT_GAMECONTROLLER implies SDL_INIT_JOYSTICK */
pub const SDL_INIT_EVENTS:u32=         0x00004000;
pub const SDL_INIT_NOPARACHUTE:u32=    0x00100000;  /**< compatibility; this flag is ignored. */
pub const SDL_INIT_EVERYTHING:u32=

SDL_INIT_TIMER | SDL_INIT_AUDIO | SDL_INIT_VIDEO | SDL_INIT_EVENTS |
SDL_INIT_JOYSTICK | SDL_INIT_HAPTIC | SDL_INIT_GAMECONTROLLER ;
/* @} */
pub const SDL_OPENGL:u32=0x4000000;

/**
 *  This function initializes  the subsystems specified by \c flags
 */

pub const SDL_WINDOW_OPENGL:u32 = 0x00000002;             /**< window usable with OpenGL context */

#[repr(u32)]
pub enum  SDL_GLattr
{
SDL_GL_RED_SIZE,
SDL_GL_GREEN_SIZE,
SDL_GL_BLUE_SIZE,
SDL_GL_ALPHA_SIZE,
SDL_GL_BUFFER_SIZE,
SDL_GL_DOUBLEBUFFER,
SDL_GL_DEPTH_SIZE,
SDL_GL_STENCIL_SIZE,
SDL_GL_ACCUM_RED_SIZE,
SDL_GL_ACCUM_GREEN_SIZE,
SDL_GL_ACCUM_BLUE_SIZE,
SDL_GL_ACCUM_ALPHA_SIZE,
SDL_GL_STEREO,
SDL_GL_MULTISAMPLEBUFFERS,
SDL_GL_MULTISAMPLESAMPLES,
SDL_GL_ACCELERATED_VISUAL,
SDL_GL_RETAINED_BACKING,
SDL_GL_CONTEXT_MAJOR_VERSION,
SDL_GL_CONTEXT_MINOR_VERSION,
SDL_GL_CONTEXT_EGL,
SDL_GL_CONTEXT_FLAGS,
SDL_GL_CONTEXT_PROFILE_MASK,
SDL_GL_SHARE_WITH_CURRENT_CONTEXT,
SDL_GL_FRAMEBUFFER_SRGB_CAPABLE,
SDL_GL_CONTEXT_RELEASE_BEHAVIOR
}
pub use SDL_GLattr::*;
pub type SDL_WindowPtr=*const u8;
pub type SDL_WindowSurfacePtr=*const u8;
#[repr(C)]
pub struct SDL_Event{
    padding:[u8;56],
}
#[repr(C)]
pub struct SDL_Surface{}
extern "C"{
    pub fn SDL_Init(_:u32)->isize;
    pub fn SDL_CreateWindow(_:*const u8,_:i32,_:i32,_:i32,_:i32,_:u32)->SDL_WindowPtr;
	pub fn SDL_SetVideoMode(width:i32, height:i32, bpp:i32, flags:u32)->*mut SDL_Surface;
    pub fn SDL_GL_SwapWindow(_:SDL_WindowPtr);
	pub fn SDL_GL_SwapBuffers();
    pub fn SDL_PollEvent(_:*mut SDL_Event)->bool;
    pub fn SDL_GL_SetAttribute(_:SDL_GLattr,_:isize);
    pub fn SDL_GetWindowSurface(_:SDL_WindowPtr)->SDL_WindowSurfacePtr;
    pub fn SDL_UpdateWindowSurface(_:SDL_WindowPtr);
}

