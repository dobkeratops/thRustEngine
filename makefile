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



run_cargo: $(SOURCES) link_data
	RUST_BACKTRACE=1 cargo run
	echo "ok; (also see targets asmjs shadertest)"

asmjs_shadertest: $(SOURCES)
	cargo build --target=asmjs-unknown-emscripten
	cp ../target/asmjs-unknown-emscripten/debug/rustv.js ./shadertest.js



shadertest: $(SOURCES)
	rustc main.rs $(RUSTFLAGS) --cfg shadertest -o shadertest

main: desktop asmjs shadertest
	echo "built all versions"

link_data:
	if [! -d data];then ln -s ../data data;fi

asmjs: $(SOURCES)
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



clean:
	rm ./*.o
	rm ./*.a
	rm ./main


