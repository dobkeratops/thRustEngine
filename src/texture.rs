use super::*;

pub static mut g_textures:[GLuint;5]=[0;5];
pub type TextureIndex=usize;

#[cfg(target_os="emscripten")]
pub fn load_texture(filename:&str)->GLuint{
	return 0;
}

#[cfg(not(target_os="emscripten"))]
pub fn load_texture(filename:&str)->GLuint{
	use image::*;
	use std::io::prelude::*;
	use std::fs::File;

	let mut data=Vec::<u8>::new();
	if let Ok(mut f)=File::open(filename){
		println!("opened {}",filename);
		if let Err(_)=f.read_to_end(&mut data){
			println!("error in reading");
			return 0;
		}
	}
	else {
		println!("could not open {}",filename);
		return 0;
	}
	
	println!("loaded {} bytes from {}",data.len(),filename);
	let imr=image::load_from_memory(&data);

	match imr{
		Err(x)=>{println!("failed to init {}",filename); return 0;},
		Ok(mut dimg)=>{
			let (mut usize,mut vsize)=dimg.dimensions();
			let mut usize1=1; let mut vsize1=1;
			while usize1<usize{usize1*=2;}
			while vsize1<vsize{vsize1*=2;}
			if !(usize1==usize && vsize1==vsize){
				println!("scaling to {}x{}",usize1,vsize1);
				dimg=dimg.resize(usize1,vsize1,FilterType::Gaussian);
			}
			if let DynamicImage::ImageRgb8(img)=dimg{
				let (mut usize,mut vsize)=img.dimensions();
				let mut usize1=1; let mut vsize1=1;
				while usize1<usize{usize1*=2;}
				while vsize1<vsize{vsize1*=2;}
				println!("loaded rgb image {}x{}",usize,vsize);
	//			let bfr=img.into_raw();
				let mut texid:GLuint=0;
				let fmt=GL_RGB;
				let mut my=Vec::<u8>::new();
				let ustep =usize/16;
				let vstep=vsize/16;
				for j in 0..vsize/vstep{
					for i in 0..usize/ustep{
						let p=img.get_pixel(i*ustep as u32,j*vstep as u32);
						my.push(p.data[0]);
						my.push(p.data[1]);
						my.push(p.data[2]);
						print!("{}",if p.data[1]>128{if p.data[1]>192{"O"}else{"o"}}else{if p.data[1]>64{"."}else{" "}});
					}
					print!("\n");
				}
				unsafe {
					glGenTextures(1,&mut texid);
					glBindTexture(GL_TEXTURE_2D,texid);
					glTexEnvi( GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_MODULATE as GLint);
					glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR as GLint);
					glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR as GLint);
					glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT as GLint);
					glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT as GLint);
					let bfr=img.into_vec();
					glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB as GLint, usize as GLint,vsize as GLint, 0, fmt, GL_UNSIGNED_BYTE, (&bfr[0]));
					return texid;
				}
			} else{
				println!("not rgb image, not supported");
				return 0;
			}
		},
		Ok(image::DynamicImage::ImageRgba8(img))=>{
//			return create_tex(img,GL_RGBA)
			return 0;
		},
		_=>{println!("image not handled");return 0;}
	}
}


pub unsafe fn create_texture(filename:String)->GLuint {
	return g_textures[0]
}

pub fn	create_textures() {
//	static_assert(sizeof(GLuint)==sizeof(int));
	// hardcoded test pattern
	unsafe {
		glGenTextures(1,&mut g_textures[0]);
		glBindTexture(GL_TEXTURE_2D,g_textures[0]);
		glTexEnvi( GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_MODULATE as GLint);
		glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR as GLint);
		glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR as GLint);
		glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT as GLint);
		glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT as GLint);

		let	(usize,vsize)=(256,256);
		let buffer:Vec<u32> = vec_from_fn(usize*vsize,&|index|{
			let (i,j)=div_rem(index,usize);
			(i+j*256+255*256*256) as u32
		});
//		for i in 0 as GLint..8 as GLint {
		glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB as GLint, usize as GLint,vsize as GLint, 0, GL_RGB, GL_UNSIGNED_BYTE, &buffer[0] as *const _ as _);
//		}
		glBindTexture(GL_TEXTURE_2D,0);
	
		#[cfg(emscripten)]
		{
			println!("bypass texture load");
			for i in 1..5 {g_textures[i]=g_textures[0];}				
		}
		#[cfg(not(emscripten))]
		{
			println!("texture loading");
			g_textures[1] = load_texture("data/mossy_rock.jpg");
			g_textures[2] = load_texture("data/stone.jpg");
			g_textures[3] = load_texture("data/metal.jpg");
			g_textures[4] = load_texture("data/grass7.png");
		}
		println!("texture load done");
	}
}

pub fn load_file(fname:&str)->Vec<u8>{
	let mut buffer=Vec::<u8>::new();
	let f=File::open(fname).unwrap().read_to_end(&mut buffer);
	buffer
}
// todo - async..
fn create_texture_from_url(url:&str,waiting_color:u32)->GLuint{
	// todo - make a tmp filename hash
	unsafe {
		emscripten::emscripten_wget(c_str(url),c_str("tmp_image1.dat\0"));
		let buffer=load_file("tmp_image1.dat\0");
		let mut texname:GLuint=0;
		glGenTextures(1,&mut texname);
		glBindTexture(GL_TEXTURE_2D,texname);
		dump!(texname);

		glTexEnvi( GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_MODULATE as GLint);
		glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR as GLint);
		glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR as GLint);
		glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT as GLint);
		glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT as GLint);

		let	(usize,vsize)=(16,16);
		let buffer:Vec<u32> = vec_from_fn(usize*vsize,&|index|{
			waiting_color
		});
		for i in 0 as GLint..8 as GLint {
			glTexImage2D(GL_TEXTURE_2D, i, GL_RGB as GLint, usize as GLint,vsize as GLint, 0, GL_RGB, GL_UNSIGNED_BYTE, &buffer[0] as *const _ as _);
		}
		texname
	}
}
