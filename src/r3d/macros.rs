
use std::ops::*;

/// macro to roll list of 'classes' with various extra stuff..
///
/// - constructor functions allowing calculation of fields from args
/// - a single 'world' object holding a vec of each class type,
/// - an enum for refering to them
/// - gathers 'documentation strings'
/// (TODO - intent is to assist rolling UI .. hence storing of UI strings

macro_rules! def_classes{
    // todo -update/render args
    {
        $struct_holding_all:ident
        {$($all_method_names:ident),*}   // list of all method names

    $(
        $class_name:ident
        ( $($ctr_arg:ident :$ctr_ty:ty = $ctr_arg_default:expr  => $ctr_arg_doc:expr),* )  // constructor args
        =>[$class_doc:expr]  //UI info for the class itself
        {
            // struct fields
            $( $field:ident : $f_type:ty=$f_init:expr =>$field_doc:expr,)*   // struct fields and initializer expressions (using ctr args)
        }
        {
            $(
                $method_name:ident $method_body:block

            )*
        }),*

    }


    =>
    {
        //[1] roll structs
        #[derive(Debug,Clone)]
        $(struct $class_name {
            $($field : $f_type),*
        })*

        //[2] roll constructor functions
        $(
            fn $class_name( $($ctr_arg:$ctr_ty),*  )->$class_name{
                $class_name{
                    $($field: $f_init),*
                }
            }
        )*

        //[3] roll 'method' functions
        $(
            impl $class_name {
                $(fn $method_name(&self) $method_body   )*

                fn get_enum(&self)->EnumOfClasses{ EnumOfClasses::$class_name }
            }

        )*

        //$(def_fn!{$name $($e);*})*

        // roll a world structure holding a vec per class
        struct $struct_holding_all{
            $( $class_name : Vec<$class_name>),*
        }
        // roll calls to all ..
        impl $struct_holding_all{
            fn update(&self) {
                $(for x in self.$class_name.iter(){
                    x.update();
                })*
            }
        }

        //[4] Roll documentation string of each class
        fn get_all_docs()->Vec<&'static str> { vec![ $($class_doc),+  ]}


        //[5] roll an enum holding an item per class,
        enum EnumOfClasses{
            $($class_name),*
        }

    }
}

def_classes!{
    World{update,render}
    Foo( a:i32=0 =>"",b:i32=0 =>"") =>["info for foo"]{
        x:i32=a+b =>"x position",
        y:i32=a-b =>"y position",
    }{
        update{ println!("foo.update()") }
        render{}
    },

    Bar()=>["bar"]{


    }{
        update{}
    },

    Baz()=>["baz"]{


    }{
        update{}
    }
}

/*
TODO: macro for rolling shader node functions
- UI items, and code to generate the shader snippet
*/

/*
fn main() {
    let w:World = World{Foo: vec![], Bar: vec![], Baz: vec![],};
    for b in w.Bar { };
    println!("class docs:{:?}", get_all_docs());
    let x=Foo(10,20);
    x.update();
    println!("created by ctr {:?}", x);
}
*/