#platforms: OSX,Linux
ifeq ($(shell uname),Darwin)
    CFLAGS += -I/Library/Frameworks/SDL2.framework/Headers -ISystem/Library/frameworks/OpenGL.framework/OpenGL -I/usr/local/include -I/usr/include -I/opt/X11/include
	LFLAGS += -F/Library/Frameworks -framework Cocoa -framework SDL2 -framework OpenGL -L/usr/local/lib -L/opt/x11/lib
	#-Z force-overflow-checks=off
	RUSTFLAGS = -l framework=OpenGL -l framework=glut    -L/usr/local/lib -L/opt/x11/lib
else
	CFLAGS += $(shell sdl2-config --cflags) $(shell pkg-config --cflags pango)
	LFLAGS += $(shell sdl2-config --libs)
	LFLAGS += -lGL -lglut $(shell pkg-config --libs pango)
	RUSTFLAGS = -lGL -lglut
endif

#-l framework=SDL2

SOURCES:=$(shell find . -name "*.rs")
# main.rs r3d/vector.rs window.rs r3d/matrix.rs r3d/quaternion.rs r3d/meshbuilder.rs r3d/draw.rs r3d/render.rs bsp/mod.rs bsp/bspdraw.rs editor.rs

main: desktop asmjs

asmjs: env
	rustc --version
	rustc main.rs --target=asmjs-unknown-emscripten

env: 
	source ~/.cargo/env



desktop : $(SOURCES)
	rustc main.rs $(RUSTFLAGS) 
#	rustc shadertest.rs  --emit obj -o shadertest.o
#	g++ shadertest.o cstuff.o  $(LFLAGS) -lGLU -lXext -lglut -lGL -o shadertest 
#7	rustdoc main.rs



debug : $(SOURCES)
	rustc main.rs $(RUSTFLAGS)
	./main

libhello.a : cstuff.c
	gcc $(CFLAGS) cstuff.c -c -std=c99
	ar rcs libcstuff.a cstuff.o


#walter@vaio:~/rustwork$ grep "#define" /usr/include/GL/gl.h | sed 's/#define\s*\([a-zA-Z_0-9]*\)\s*\([a-zA-Z0-9_]*\)/pub static \1:uint=\2;/'^Cwalter@vaio:~/rustwork$ 

clean:
	rm ./*.o
	rm ./*.a
	rm ./main


