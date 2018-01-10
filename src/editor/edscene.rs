use super::*;

pub type Edge=[VertexIndex;2];
pub type Vertex=V3;
#[derive(Default,Clone)]
pub struct Scene{   // todo - switch to TriMesh hrc and tags as seperate entity
    vertices:   Vec<Vertex>,
    edges:      Vec<Edge>,
    vertex_tags:Vec<bool>,
}

type SceneSelection = Vec<bool>;

trait SpatialScene{
    fn ss_full_extents(&self)->Option<Extents<Vec3f>>;
    fn ss_selection_extents(&self)->Option<Extents<Vec3f>>;
}

impl SpatialScene for Scene {
    fn ss_full_extents(&self)->Option<Extents<Vec3f>>{
        None
    }
    fn ss_selection_extents(&self)->Option<Extents<Vec3f>>{
        None
    }
}

struct TreeNode<Common>{
    node:Common,            // every node has this embedded, e.g. "name","matrix",..
    children:Vec<TreeNode<Common>>
}
struct SRT{
    pos:Vec3f, rotation:Vec3f, scaling:Vec3f,
}
struct Info{
    tag:bool,
    name:String,
    // SRT values forms the transform
    transform:SRT,
}

struct Vertices<V=Vec3f> {
    // tags, transform, ..
    vertices:Vec<V>,
    tags:Vec<bool>
}


pub enum RefOrBox<'e,T:'e>{
    Ref(&'e T),Box(Box<T>)
}

/// TODO - generator of initial mesh from parameters
/// parameters to be editable after in opstack
mod primitive{
    struct Sphere(f32,(i32,i32));
    struct Torus(f32,f32,(i32,i32));
    struct Cone((f32,f32),i32,i32);
    struct Cylinder((f32,f32),(i32,i32));
    struct Cuboid((f32,f32,f32));
    struct Grid((f32,f32),(i32,i32));
}

impl Scene{
    // todo - vertex tag iterator.
    fn clear_vertex_tags(&mut self){
        for t in self.vertex_tags.iter_mut(){*t=false;}
    }
    fn vertex_tags_invert_all(&mut self){
        for t in self.vertex_tags.iter_mut(){*t^=true;}
    }
    fn vertex_tags_set_all(&mut self){
        for t in self.vertex_tags.iter_mut(){*t=true;}
    }
    fn vertex_tag_invert(&mut self, i:VertexIndex){
        self.vertex_tags[i]^=true;
    }
    fn vertex_tag_change(&mut self, i:VertexIndex, mode:SelectMode){
        let tag=self.vertex_tags[i];
        self.vertex_tags[i]=match mode{
            SelectMode::Invert=>tag^true,
            SelectMode::Select=>true,
            SelectMode::Deselect=>false
        }
    }
    fn create_vertex(&mut self, pos:V3,tag:bool){
        self.vertices.push(pos);
        self.vertex_tags.push(tag);

    }
    pub fn dump(&self){
        dump!(self.vertices.len());
    }
}

impl Scene{
    fn add_vertex(&mut self,p:V3)->VertexIndex{
        let vi=self.vertices.len();
        self.vertices.push(p);
        vi as VertexIndex
    }
    fn add_edge(&mut self,a:VertexIndex,b:VertexIndex){
        self.edges.push([a,b]);
    }
}

#[derive(Default,Debug,Clone)]
pub struct DrawTool{last_point:Option<VertexIndex>}

#[derive(Default,Debug,Clone)]
struct AddPoint(V3);

#[derive(Default,Debug,Clone)]
struct ToggleSelPoint(VertexIndex);

#[derive(Default,Debug,Clone)]
struct SingleSelPoint(VertexIndex);

#[derive(Default,Debug,Clone)]
struct ConnectLine(VertexIndex,VertexIndex);

#[derive(Default,Debug,Clone)]
struct DrawLine(VertexIndex,V3);

impl Scene {
    pub fn pick_point_at(&self, vcs:&ViewCursorSceneS, maxr:f32)->Option<VertexIndex>{
        let mut maxr2 = sqr(maxr);
        let mut besti=None;
        // iterate the vertices in reverse to tend to pick the newly created ones first..
        for (vti,&vt) in self.vertices.iter().enumerate().rev(){

            let mut spos=vcs.world_to_viewport(&vt.to_vec3());
            spos.z=zero();
            let r2=v3dist_squared(&vec3(vcs.pos.x,vcs.pos.y,zero()),&spos);
            if r2<maxr2{besti=Some(vti as VertexIndex); maxr2=r2;println!("picked pt\n")}
        }
        return besti;
    }
}

#[derive(Default,Debug,Clone)]
struct Translate{
    delta:V3
}

#[derive(Default,Debug,Clone)]
struct MovePointBy(VertexIndex,V3);

#[derive(Default,Debug,Clone)]
struct SelectPoints{mode:SelectMode,points:Vec<VertexIndex>}



impl Operation<Scene> for SingleSelPoint{
    fn op_apply(&self,ns:&mut Scene){
        ns.clear_vertex_tags();
        ns.vertex_tags[self.0]=true;
    }
}

impl Operation<Scene> for ToggleSelPoint{
    fn op_apply(&self,scn:&mut Scene){
        scn.vertex_tag_invert(self.0);
    }
}
impl Operation<Scene> for SelectPoints {
    fn op_apply(&self,ns:&mut Scene){
        for &v in self.points.iter() {
            ns.vertex_tag_change(v,self.mode);
        }
    }
    fn op_dump(&self){
        println!("select points[{:?}]",self.points);
    }
}

#[derive(Default,Debug,Clone)]
pub struct Subdivide{}
impl Operation<Scene> for Subdivide {
    fn op_apply(&self,msh:&mut Scene){
        println!("subdivide op");
        let mut new_edges=vec![];
        //todo,it could consume
        for edge in msh.edges.iter(){
            if msh.is_edge_tagged(edge){
                let ev0=edge[0];let ev1=edge[1];
                let new_vti=msh.vertices.len();
                let newvt=msh.vertices[ev0].vlerp(&msh.vertices[ev1],0.5f32);
                msh.vertices.push(newvt);
                new_edges.push([ev0,new_vti]);
                new_edges.push([new_vti,ev1]);
                msh.vertex_tags.push(true);
            } else {new_edges.push(edge.clone())}
        }
        // todo: combined filter_and_unordered_append
        //msh.edges.append(&mut new_edges);
        msh.edges=new_edges;
        //(&mut new_edges);
    }
}
/// in progress: macro to formalise the pattern of Operations , and their triggering from UI
macro_rules! operations{
    {$($hotkey:expr=>$opname:ident($($argname:ident : $argtype:ty = $argdefault:expr),*)$opbody:block)*}
    =>{
        $(
            // instantiate an object to hold the parameter block
            #[derive(Default,Debug,Clone)]
            pub struct $opname{
                pub $($argname:$argtype),*
            }
            impl $opname{
                pub fn new()->Self{
                    $argname:$argdefault
                }
                // todo: call with all operation args as params..
                pub fn op_apply(&self, d:&mut Scene) $opbody
            }
        )*
    }
}
#[derive(Default,Debug,Clone)]
pub struct Merge{}
impl Operation<Scene> for Merge {
    fn op_apply(&self, d: &mut Scene) {
        let mut sumpt:Vertex=zero();
        let mut total=0;
        let mut vt_xlat=vec![];
        vt_xlat.resize(d.vertices.len(),(-1isize) as usize);
        let mut mergevt:Option<VertexIndex>=None; // this will be pushed
        let vt_xlat:Vec<_> = (0..d.vertices.len()).map(|vti|
            if d.vertex_tags[vti] {
                // allocate the first vertex index as the merged point,
                if !mergevt.is_some(){mergevt=Some(vti)};
                sumpt.vassign_add(&d.vertices[vti]);total+=1;
                mergevt.unwrap()
            }  else {
                vti
            }
        ).collect();

        match mergevt {
            Some(merged_vertex)=>{
                d.vertices[merged_vertex]=sumpt.vscale(1.0/(total as f32));
                d.translate_vertex_indices(&vt_xlat);
                d.clear_vertex_tags();
                d.vertex_tags[mergevt.unwrap()]=true;
            }
            None=>{}
        }
    }
}


#[derive(Default,Debug,Clone)]
pub struct Extrude{}
impl Operation<Scene> for Extrude {
    fn op_apply(&self, d:&mut Scene){
        let mut vtmap = HashMap::<VertexIndex,VertexIndex>::new();
        let mut new_vertices=Vec::<Vertex>::new();
        let mut new_edges=vec![];
        // every extruded edge is duplicated
        let mut new_vertex=d.vertices.len();
        for i in 0.. d.vertices.len(){
            if d.vertex_tags[i]{
                new_edges.push([i,new_vertex]);
                vtmap.insert(i,new_vertex); new_vertex+=1;
                new_vertices.push(d.vertices[i].clone());
            }

        }
        // for every edge, extrude a polygon.. TODO
        // for the moment we just create a wireframe around where the poly would be.
        // if we go the half-edge route..
        for edge in d.edges.iter(){
            if d.is_edge_tagged(edge){
                new_edges.push([vtmap[&edge[0]],vtmap[&edge[1]]]);
            }
        }
        d.clear_vertex_tags();
        d.edges.append(&mut new_edges);
        d.vertices.append(&mut new_vertices);

        for (_,_) in vtmap.iter(){
            d.vertex_tags.push(true);
        }
    }
}

impl Operation<Scene> for DrawLine{
    fn op_apply(&self,ns:&mut Scene){
        ns.clear_vertex_tags();
        ns.create_vertex(self.1, true); // create a vertex and tag it
        ns.edges.push([self.0, (ns.vertices.len() as  VertexIndex)-1]);
    }
    fn op_dump(&self){println!("{:?}",self); }
}
impl Operation<Scene> for ConnectLine{
    //dump!("CONNECT LINE:",self);
    fn op_apply(&self,ns:&mut Scene){
        if !ns.edge_exists(self.0,self.1) {
            ns.clear_vertex_tags();
            ns.edges.push([self.0, self.1]);
            ns.vertex_tags[self.1] = true;
        }
    }
    fn op_dump(&self){println!("{:?}",self); }
}
impl Operation<Scene> for AddPoint {
    fn op_apply(&self,ns:&mut Scene){
        ns.clear_vertex_tags();
        ns.vertices.push(self.0);
        ns.vertex_tags.push(true);
    }
    fn op_dump(&self){println!("{:?}",self); }
}

impl Operation<Scene> for Translate {
    fn op_apply(&self,ns:&mut Scene){
        for (i,vt) in ns.vertices.iter_mut().enumerate(){ if ns.vertex_tags[i]{v3addto(vt, &self.delta);}}
    }
    fn op_dump(&self){println!("{:?}",self);}
}

static mut g_last_point:Option<VertexIndex>=None;

pub fn crosshair_feedback(pos:&Vec3f){
	draw::crosshair_xy(pos,g_snap_radius*0.5,g_snap_radius,g_color_feedback);
}

fn get_last_point(s:&Tool<Scene>)->Option<VertexIndex>{ unsafe{g_last_point}}

impl DrawTool{
	pub fn set_last_point(&mut self, o:Option<VertexIndex>){
	    println!("TODO fix last point setting, we had borrow issues");
		unsafe{g_last_point=o;}
	}
}


impl Tool<Scene> for DrawTool {

    fn tool_preselection(&self, (scene, e):ViewCursorScene<Scene>) -> ToolPresel {
        let newpt=e.viewport_to_world(&vec3(e.pos.x,e.pos.y,0.0f32));
        let viewpt_recovered=e.world_to_viewport(&newpt);
        //dump!(e.pos,newpt,viewpt_recovered);
        let picked_point=scene.pick_point_at(e,g_snap_radius);
        // now here's where rust shines..
        let ret=match (get_last_point(self), picked_point) {
            (None,None)         =>ToolPresel::MakePoint(newpt.to_tuple()),
            (None,Some(ei))     =>ToolPresel::PickPoint(ei),
            (Some(si),None)     =>ToolPresel::DrawLine(si, newpt.to_tuple()),
            (Some(si),Some(ei)) =>ToolPresel::ConnectLine(si,ei)
        };
        //dump!(ret);
        ret
    }

    fn tool_render_passive(&self, p:&ToolPresel, (s,e):ViewCursorScene<Scene>)
    {
        //let s=e.scene;
        match p{
            &ToolPresel::MakePoint(ref newpt)=>
				crosshair_feedback(&e.world_to_viewport(&newpt.to_vec3())),
            &ToolPresel::PickPoint(ei)=>
				crosshair_feedback(&e.world_to_viewport(&s.vertices[ei as usize].to_vec3())),
            &ToolPresel::ConnectLine(si,ei)=>
                draw::line_c(&e.world_to_viewport(&s.vertices[si as usize].to_vec3()),&e.world_to_viewport(&s.vertices[ei  as usize].to_vec3()), g_color_feedback),
            &ToolPresel::DrawLine(si,ref newpt)=>{
				crosshair_feedback(&e.world_to_viewport(&newpt.to_vec3()));
                draw::line_c(&e.world_to_viewport(&s.vertices[si  as usize].to_vec3()),&e.world_to_viewport(&newpt.to_vec3()),g_color_feedback)
			},
            _=>{},
        }
    }

    fn tool_render_drag(&self, d:&ToolDrag, (s,e):ViewCursorScene<Scene>){
        if !e.drag_start.is_some(){return;};
        match d {
            &ToolDrag::Rect(_)=>{
                draw::rect_outline_v2(&e.drag_start.unwrap(),&e.pos,g_color_feedback)
            },
            _=>{}
        }
    }
    fn tool_drag(&mut self, d:&ToolDrag, se:ViewCursorScene<Scene>) ->optbox<Operation<Scene>>{
        let (s,e)=se;
        // shows what drag-end would produce, as a transient state.
        // TODO.. suspicious of a function that just calls another
        // is it because this *might* yield operations?
        self.tool_drag_end(d,se)
    }
    fn tool_passive_move(&self, (s,e):ViewCursorScene<Scene>){

    }

    // todo - should the drag itself be an object that completes itself?
    // we have the logic for a single mode split between several places.
    // you could have IDrag { start, render, end}
    fn tool_drag_end(&mut self, d:&ToolDrag, (s, e):ViewCursorScene<Scene>)->optbox<Operation<Scene>>{
        if !e.drag_start.is_some(){return None;}
        let screen_delta=v2sub(&e.pos, &e.drag_start.unwrap());
        let world_delta=e.screen_to_world.mul_vec3w0(&vec3(screen_delta.x, screen_delta.y,0.0f32)).to_vec3();
//        let s=e.scene;
        match d {
            &ToolDrag::MovePoint(ref vti)=>Some(Box::new(
                ComposedOp(
                    SingleSelPoint(*vti),
                    Translate{delta:world_delta.to_tuple()}
                )
            )),
            &ToolDrag::Rect(ref mode)=>{
                let pts=s.get_vertices_in_rect(&Extents(&e.drag_start.unwrap(),&e.pos));
                Some(Box::new(SelectPoints{mode:SelectMode::Invert,points:pts}))
            },
            _ =>{None}

        }
    }

    fn tool_lclick(&mut self, p:&ToolPresel,(s,e):ViewCursorScene<Scene>) -> optbox<Operation<Scene>> {
        println!("drawtool lclick , make a addpoint op");
		// returns an operation
        match p {
            &ToolPresel::MakePoint(newpt) =>{
                self.set_last_point(Some(s.vertices.len() as VertexIndex)); println!("vertices.len()={:?}",s.vertices.len());Some(Box::new(AddPoint(newpt)) as _)
            },
            &ToolPresel::PickPoint(pti)     =>{
                self.set_last_point(Some(pti));Some(Box::new(ToggleSelPoint(pti)))
            },
            &ToolPresel::ConnectLine(si,ei) =>{
                self.set_last_point(None);Some(Box::new(ConnectLine(si,ei)))
            },
            &ToolPresel::DrawLine(si,newpt) =>{
                self.set_last_point(Some(s.vertices.len() as VertexIndex));Some(Box::new(DrawLine(si,newpt)) )
            },
            _ => None
        }
    }
    fn tool_rclick(&mut self, p:&ToolPresel, (s,e):(&Scene,&ViewCursorSceneS))->optbox<Operation<Scene>>{
        println!("rclick");
        self.tool_cancel();
        None
    }

    fn tool_drag_begin(&mut self, p:&ToolPresel,(s,e):(&Scene,&ViewCursorSceneS))->ToolDrag{
        if let &ToolPresel::PickPoint(pti)=p{
            println!("drag start-movepoint");
            ToolDrag::MovePoint(pti)
        } else {
            ToolDrag::Rect(SelectMode::Invert)
        }
    }

    fn tool_cancel(&mut self){self.set_last_point(None)}

    /*    fn lclick(&mut self, scn:&mut Scene, a:ScreenPos){
            let ve=scn.add_vertex((a.0,a.1,0.0f32));
            match scn.state{
                EState::LastPoint(vs)=> scn.add_edge(ve,vs),
                _=>{}
            }
            scn.state=EState::LastPoint(ve);
        }
        fn rclick(&mut self, scn:&mut Scene, a:ScreenPos){
            scn.state=EState::None;
        }
        fn try_drag(&self, e:&Scene, (mb,pos):MouseAt)->DragMode{
            DragMode::Line
        }
        */
}


impl Editable for Scene {
    fn default_tool()->Box<Tool<Scene>>{ Box::new(DrawTool::default()) }

    fn edscn_key(&self, ed:&Editor<Scene>, k:&KeyAt)->Action<Scene>{
        match (k.0, k.1, k.2) {
            (WinKey::KeyCode('d'),0,KeyDown)=>Action::SetTool(Box::new(DrawTool::default())),
            //(WinKey::KeyCode('m'),0,KeyDown)=>Action::SetTool(Box::new(DrawTool::default())),
            (WinKey::KeyCode('s'),0,KeyDown)=>Action::DoOperation(Box::new(Subdivide{})),
            (WinKey::KeyCode('e'),0,KeyDown)=>Action::DoOperation(Box::new(Extrude{})),
            (WinKey::KeyCode('m'),0,KeyDown)=>Action::DoOperation(Box::new(Merge{})),

            _=>Action::None,
        }
    }

    fn scn_render(&self, mat: &Mat44) {
        //let ofs = (rc.rect.0.0 + 1.0f32, rc.rect.0.1 + 1.0f32, 0.0f32);
        let ofs=(0.0,0.0,0.0);//vzero();

        // linedraw with transformation matrix..
        draw::set_matrix(0, &mat);

		unsafe {
			if g_test_texture[0]!=0{
/*				glActiveTexture(GL_TEXTURE0);
				glEnable(GL_TEXTURE_2D);
				glBindTexture(GL_TEXTURE_2D,g_test_texture[0]);
				glBegin(GL_TRIANGLE_STRIP);
				let z=0.99f32;
				let x0=-0.5f32;
				let x1=0.5f32;
				let y0=-0.5f32;
				let y1=0.5f32;
				

				glColor4f(1.0,1.0,1.0,1.0);
				glTexCoord2f(0.0f32, 1.0f32);
				glVertex3f(x1,y0,z);

				glColor4f(1.0,1.0,1.0,1.0);
				glTexCoord2f(1.0f32, 1.0f32);
				glVertex3f(x0,y0,z);

				glColor4f(1.0,1.0,1.0,1.0);
				glTexCoord2f(0.0f32, 0.0f32);
				glVertex3f(x1,y1,z);

				glColor4f(1.0,1.0,1.0,1.0);
				glTexCoord2f(1.0f32, 0.0f32);
				glVertex3f(x0,y1,z);
				glEnd();
				glDisable(GL_TEXTURE_2D);
*/
			
				draw::set_texture(0,g_test_texture[0]);
				draw::rect_tex(&vec2(-0.5f32,-0.5f32),&vec2(0.5f32,0.5f32),0.99f32 );
				draw::set_texture(0,0);
			}
		}

        for line in self.edges.iter() {
            draw::line_c(&v3add(&self.vertices[line[0] as usize], &ofs), &v3add(&self.vertices[line[1] as usize], &ofs), g_color_wireframe)
        }
        draw::set_matrix(0, &matrix::identity());
/*
        // old linedraw
        for line in self.edges.iter() {
            draw::line_c(&v3add(&self.vertices[line[0] as usize], &ofs), &v3add(&self.vertices[line[1] as usize], &ofs), g_color_wireframe)
        }
*/
        let m1=mat.transpose4();
        for (i, v) in self.vertices.iter().enumerate() {
            let epos=mat.mul_vec3_point(&v.to_vec3());
            let spos=epos.project_to_vec3();
            //    draw::vertex(v,2,0xff00ff00);
            draw::circle_fill_xy_c(&spos, g_snap_radius * 0.5f32,
                                   if self.vertex_tags[i] { g_color_selected } else { g_color_wireframe }
            );
        }
    }

    fn copy(&self, pos:&ScreenPos)->Self{
        let mut clipboard=Self::default();
        let null_vertex=(-(1 as isize)) as VertexIndex;	// todo , will they?
        let mut vertex_xlat:Vec<VertexIndex>=vec![null_vertex; self.vertices.len()];

        // every selected vertex is shoved across,
        for (i,v) in self.vertices.iter().enumerate() {
            if self.vertex_tags[i] {
                vertex_xlat[i] = clipboard.vertices.len();
                clipboard.vertices.push(*v);
            }
        }
        return clipboard;
    }
    //'delete selection', whatever that maybe
    //todo - mode filters ? (vertex,edge,..)
    fn delete(&mut self){
        // delete
        let mut new_edges=vec![];
        // TODO: filter_unordered(..)
        for edge in self.edges.iter(){
            if !self.is_edge_tagged(edge){new_edges.push(edge.clone())}
        }
        self.edges=new_edges;
        // remove unused vertices..
        // renumber vertices..
    }
    fn paste(&mut self, pos:&ScreenPos, clipboard:&Self){

    }
    fn dump(&self){ println!("edscene: vertices={} edges={}",self.vertices.len(),self.edges.len());}
}

impl Scene {
    fn is_edge_tagged(&self,e:&Edge)->bool {
        self.vertex_tags[e[0]] && self.vertex_tags[e[1]]
    }
    fn get_vertices_in_rect(&self,r:&ScreenRect)->Vec<VertexIndex>
    {
        let mut ret=Vec::new();
        let minv=v2min(&r.min,&r.max);
        let maxv=v2max(&r.min,&r.max);

        for (i,v) in self.vertices.iter().enumerate(){
            if inrange(v.0, (r.min.x, r.max.x)) &&
                inrange(v.1, (r.min.y,r.max.y)){
                ret.push(i)
            }
        }
        ret
    }
    fn edge_exists(&self,sv:VertexIndex,ev:VertexIndex)->bool{
        for edge in self.edges.iter() {
            if edge[0]==sv && edge[1]==ev || edge[0]==ev && edge[1]==sv{return true;}
        }
        return false;
    }
    fn translate_vertex_indices(&mut self, xlat:&Vec<VertexIndex>) {
        for edge in self.edges.iter_mut() {
            edge[0]=xlat[edge[0]];
            edge[1]=xlat[edge[1]];
        }

    }
    //fn delete_unused(&mut self){
      //  let mut tags:Vec<bool>=vec![false;self.vertices.len()];
        //for edge in self.edges.iter_mut() {tags[edge[0]]=true; tags[edge[1]]=true;}

    //}
    //vtc       0  1  2  3  4  5  6
    //xlat      0  1  2  2  4  2  6
    //inv_xlat  0  1  2  -1 4  -1 6
    //          0  1  2  6  4
    /*
    fn translate_vertex_indices_with_redundant_elim(&mut self, mut xlat:Vec<VertexIndex>){
        assert!(xlat.len()==self.vertices.len());
        let nonei=(-1isize) as usize;
        let mut inv_xlat:Vec<VertexIndex>=Vec::new();
        inv_xlat.resize(xlat.len(),nonei);
        // make the 'scatter' equivalent of this 'gather' index table

        let mut lastvt=self.vertices.len()-1;

        for i in 0..self.vertices.len(){inv_xlat[xlat[i]]=xlat[i];}
        // new_xlat is a gather,
        let new_xlat:Vec<_>=(0..self.vertices.len()).map(|i|
            if inv_xlat[i]!=nonei{
                xlat[i]
            }else{
                while inv_xlat[lastvt]==nonei{lastvt-=1};
                xlat[]
            }
        ).take(lastvt+1).collect();
        // now scatter
        for i in range 0..
        dump!(xlat);dump!(inv_xlat);dump!(new_xlat);
        self.translate_vertex_indices(&new_xlat);
        println!("merge: vertices in={} vertices out={}",self.vertices.len(),lastvt);
        self.vertices.resize(lastvt+1,zero());
        self.vertex_tags.resize(lastvt+1,false);
    }*/
}
