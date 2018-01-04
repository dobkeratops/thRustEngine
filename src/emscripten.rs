// emscripten networking functions?
// unsafe C api port

use os::raw;
use raw::{c_int,c_uint,c_void,c_char,c_ulong};
use ffi::{CString,CStr};
use ffi::CString as c_str;
use collections::{HashSet,HashMap};
use r3d::*;
extern {
	pub fn emscripten_run_script(script:*const c_char);
	pub fn emscripten_run_script_int(script:*const c_char)->c_int;
	pub fn emscripten_run_script_string(script:*const c_char)->*mut c_char;
	pub fn emscripten_async_run_script(script:*const c_char,millis:i32);
	pub fn emscripten_async_load_script(script:*const c_char, f_oncomplete:*const u8,f_onerror:*const u8);

	pub fn emscripten_set_main_loop_timing(mode:isize, value:isize)->isize;
	pub fn emscripten_get_main_loop_timing(mode:&isize, timing:&isize);
	pub fn emscripten_set_main_loop_arg(func:*const u8, arg:*mut c_void, fps:i32, simulate_infinite_loop:i32);
	pub fn emscripten_pause_main_loop();
	pub fn emscripten_resume_main_loop();
	pub fn emscripten_cancel_main_loop();
	pub fn emscripten_fetch_attr_init(fetch_attr:&mut fetch_attr_t);
	pub fn emscripten_async_call(func:*const u8, arg:*mut c_void, millis:isize);
	pub fn emscripten_get_device_pixel_ratio()->f64;
	pub fn emscripten_hide_mouse();
	pub fn emscripten_set_canvas_size(width:isize, height:isize);
	pub fn emscripten_get_canvas_size(width:&mut isize, height:&mut isize,isFullscreen:&mut isize);

	pub fn emscripten_wget(url:*const c_char, file:*const c_char);
	pub fn emscripten_async_wget(url:*const c_char, file:*const c_char, onload:em_str_callback_func, onerror:em_str_callback_func);


	pub fn emscripten_fetch(fetch_attr:&mut fetch_attr_t, url:*const c_char)->*mut fetch_t;
	pub fn emscripten_fetch_wait(fetch:&mut fetch_t, timeoutMSecs:f64)->EMSCRIPTEN_RESULT ;
	pub fn  emscripten_fetch_close(fetch:&mut fetch_t)->EMSCRIPTEN_RESULT;
}
#[cfg(target_os="emscripten")]
extern
{
    pub fn emscripten_set_main_loop(func:*const u8, fps:i32, simulate_infinite_loop:i32);

}
type em_callback_func=&'static extern  fn();
type em_arg_callback_func=&'static extern fn(*mut c_void);
type em_str_callback_func=&'static extern fn(*const c_char);
type em_async_wget2_onload_func=&'static extern  fn(usize, *mut c_void, *const c_char);
type em_async_wget2_onstatus_func=&'static extern  fn(usize,*mut c_void, isize);
type em_async_wget_onload_func=&'static extern  fn(*mut c_void, *mut c_void,isize);
type EMSCRIPTEN_RESULT=isize;
const EMSCRIPTEN_RESULT_SUCCESS:EMSCRIPTEN_RESULT=0;
const EMSCRIPTEN_RESULT_DEFERRED:EMSCRIPTEN_RESULT=1;
const EMSCRIPTEN_RESULT_NOT_SUPPORTED:EMSCRIPTEN_RESULT=-1;
const EMSCRIPTEN_RESULT_FAILED_NOT_DEFERRED:EMSCRIPTEN_RESULT=-2;
const EMSCRIPTEN_RESULT_INVALID_TARGET:EMSCRIPTEN_RESULT=-3;

extern{ fn printf(_:*const c_char);}
// slow interface to JS, string malarchy
#[cfg(target_os = "emscripten")]
pub fn run_script(s:&str){
	unsafe{printf(c_str(s));}
	unsafe{emscripten_run_script(c_str(s))}
}
#[cfg(target_os = "emscripten")]
pub fn run_script_int(s:&str)->c_int{
	unsafe{emscripten_run_script_int(c_str(s))}
}
#[cfg(target_os = "emscripten")]
pub fn run_script_string(s:&str)->String {
	unsafe{let ret=emscripten_run_script_string(c_str(s));
		let cs=CString::from_raw(ret);
		match cs.into_string(){
			Result::Ok(rs)=>rs,
			Result::Err(_)=>String::new()
		}
	}
}

#[cfg(target_os = "emscripten")]
pub fn alert(s:&str){
	let tmp=format!("alert(\"{}\")\0",s);
	unsafe {emscripten_run_script(c_str(tmp.as_str()));}
}
#[cfg(target_os = "emscripten")]
pub fn confirm(s:&str)->bool{
	let tmp=format!("confirm(\"{}\")\0",s);
	unsafe {emscripten_run_script_int(c_str(tmp.as_str())) as isize!=0}
}
#[cfg(target_os = "emscripten")]
pub fn prompt(s:&str)->String{
	let tmp=format!("prompt(\"{}\")\0",s);
	let r=unsafe {emscripten_run_script_string(c_str(tmp.as_str()))};
	let mut rtn=String::new();
	let mut x:isize=0;
	unsafe {while *r.offset(x)!=0{
		rtn.push(*r.offset(x) as u8 as char);
		x+=1;
	}}
	rtn
}

#[cfg(not(target_os = "emscripten"))]
pub fn alert(s:&str){ println!("{}",s);}
#[cfg(not(target_os = "emscripten"))]
pub fn confirm(s:&str)->bool{ unimplemented!()}
#[cfg(not(target_os = "emscripten"))]
pub fn prompt(s:&str)->String{ unimplemented!()}

type EMuint=u32;
type EM_BOOL=bool;
const EMSCRIPTEN_FETCH_LOAD_TO_MEMORY:EMuint=0x0001;
const EMSCRIPTEN_FETCH_STREAM_DATA:EMuint=0x0002;

const EMSCRIPTEN_FETCH_PERSIST_FILE:EMuint= 4;

const EMSCRIPTEN_FETCH_APPEND:EMuint=8;

const EMSCRIPTEN_FETCH_REPLACE:EMuint= 16;

const EMSCRIPTEN_FETCH_NO_DOWNLOAD:EMuint= 32;

const EMSCRIPTEN_FETCH_SYNCHRONOUS:EMuint= 64;

const EMSCRIPTEN_FETCH_WAITABLE:EMuint= 128;

//struct emscripten_fetch_t;


// Specifies the parameters for a newly initiated fetch operation.
#[repr(C)]
pub struct fetch_attr_t
{
	// 'POST', 'GET', etc.
	pub requestMethod:[c_char;32],

	// Custom data that can be tagged along the process.
	pub userData:*mut c_void,

	pub onsuccess:*const extern fn(&'static fetch_t),
	pub onerror:*const extern fn(&'static fetch_t),
	pub onprogress:*const extern fn(&'static fetch_t),
//	void (*onsuccess)(struct emscripten_fetch_t *fetch);
//	void (*onerror)(struct emscripten_fetch_t *fetch);
//	void (*onprogress)(struct emscripten_fetch_t *fetch);

	// EMSCRIPTEN_FETCH_* attributes
	pub attributes:u32,

	// Specifies the amount of time the request can take before failing due to a timeout.
//	unsigned long timeoutMSecs;
	pub timeoutMSecs:c_ulong,

	// Indicates whether cross-site access control requests should be made using credentials.
	pub withCredentials:EM_BOOL,

	//const char *destinationPath;
	pub destinationPath:*const c_char,	// ewww

	// Specifies the authentication username to use for the request, if necessary.
	// Note that this struct does not contain space to hold this string, it only carries a pointer.
	// Calling emscripten_fetch() will make an internal copy of this string.
	pub userName:*const c_char,	// ewww

	// Specifies the authentication username to use for the request, if necessary.
	// Note that this struct does not contain space to hold this string, it only carries a pointer.
	// Calling emscripten_fetch() will make an internal copy of this string.
	pub password:*const c_char,

	// Points to an array of strings to pass custom headers to the request. This array takes the form
	// {"key1", "value1", "key2", "value2", "key3", "value3", ..., 0 }; Note especially that the array
	// needs to be terminated with a null pointer.
//	const char * const *requestHeaders;
	pub requestHeaders:*const *const c_char,//null terminated array of string pointers..ghastly

	// Pass a custom MIME type here to force the browser to treat the received data with the given type.
	//const char *overriddenMimeType;
	pub overriddenMimeType:*const c_char,

	// If non-zero, specifies a pointer to the data that is to be passed as the body (payload) of the request
	// that is being performed. Leave as zero if no request body needs to be sent.
	// The memory pointed to by this field is provided by the user, and needs to be valid only until the call to
	// emscripten_fetch() returns.
//	const char *requestData;
	pub requestData:*const c_char,

	// Specifies the length of the buffer pointed by 'requestData'. Leave as 0 if no request body needs to be sent.
	pub requestDataSize:usize,
}

#[repr(C)]
pub struct fetch_t
{
	// Unique identifier for this fetch in progress.
	//unsigned int id;
	pub id:c_uint,

	// Custom data that can be tagged along the process.
	//void *userData;
	pub userData:*mut c_void,

	// The remote URL that is being downloaded.
	//const char *url;
	pub url:*const c_char,

	// In onsuccess() handler:
	//   - If the EMSCRIPTEN_FETCH_LOAD_TO_MEMORY attribute was specified for the transfer, this points to the
	//     body of the downloaded data. Otherwise this will be null.
	// In onprogress() handler:
	//   - If the EMSCRIPTEN_FETCH_STREAM_DATA attribute was specified for the transfer, this points to a partial
	//     chunk of bytes related to the transfer. Otherwise this will be null.
	// The data buffer provided here has identical lifetime with the emscripten_fetch_t object itself, and is freed by
	// calling emscripten_fetch_close() on the emscripten_fetch_t pointer.
	//const char *data;
	pub data:*const c_char,

	// Specifies the length of the above data block in bytes. When the download finishes, this field will be valid even if
	// EMSCRIPTEN_FETCH_LOAD_TO_MEMORY was not specified.
	//uint64_t numBytes;
	pub numBytes:u64,

	// If EMSCRIPTEN_FETCH_STREAM_DATA is being performed, this indicates the byte offset from the start of the stream
	// that the data block specifies. (for onprogress() streaming XHR transfer, the number of bytes downloaded so far before this chunk)
	//uint64_t dataOffset;
	pub dataOffset:u64,

	// Specifies the total number of bytes that the response body will be.
	// Note: This field may be zero, if the server does not report the Content-Length field.
	//uint64_t totalBytes;
	pub totalBytes:u64,

	// Specifies the readyState of the XHR request:
	// 0: UNSENT: request not sent yet
	// 1: OPENED: emscripten_fetch has been called.
	// 2: HEADERS_RECEIVED: emscripten_fetch has been called, and headers and status are available.
	// 3: LOADING: download in progress.
	// 4: DONE: download finished.
	// See https://developer.mozilla.org/en-US/docs/Web/API/XMLHttpRequest/readyState
	//unsigned short readyState;
	pub readyState:u16,

	// Specifies the status code of the response.
	//unsigned short status;
	pub status:u16,

	// Specifies a human-readable form of the status code.
	//char statusText[64];
	pub statusText:[c_char;64],

	//uint32_t __proxyState;
	pub __proxyState:u32,

	// For internal use only.
	pub __attributes:fetch_attr_t,
}

#[cfg(not(target_os="emscripten"))]
pub fn emscripten_set_main_loop(func:*const u8, fps:i32, simulate_infinite_loop:i32){
    println!("emscripten main loop ");
}





