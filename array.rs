/*
#![feature(collections_range)]
#![feature(drain_filter)]
#![feature(slice_rsplit)]
#![feature(slice_get_slice)]
#![feature(vec_resize_default)] 
#![feature(vec_remove_item)]
#![feature(collections_range)] 
#![feature(slice_rotate)]
#![feature(swap_with_slice)]
*/
foo bar baz
use collections::range::RangeArgument;
use std::cmp::Ordering;
use std::borrow::Borrow;
use std::vec::{Drain,Splice,DrainFilter};
use std::ops::{Deref,DerefMut,Index,IndexMut};
use std::slice::{Iter,IterMut,Windows,Chunks,ChunksMut,Split,SplitMut,RSplit,RSplitMut,RSplitN,RSplitNMut,SplitN,SplitNMut,SliceIndex};
use std::marker::PhantomData; // this sucks!

//todo, how to handle 'enumerate'.
// would we have to impl 'my_enumerate' or something?

// wrapper for Vec<T> with indexing defaulting to i32
// todo , real vector impl, with smallvec  stuff

pub trait IndexTrait  {	// TODO - would be better to use official from/into, but it doesn't let us impl
	fn my_from(x:usize)->Self;
	fn my_into(self)->usize;
}
impl IndexTrait for i32{
	fn my_from(x:usize)->Self{x as Self}
	fn my_into(self)->usize{self as usize}
}


// grrr. can't impl theirs this way round?!
//trait MyInto {
//}

//TODO - wrapper or macro to roll a 'strongly typed index'
// e.g. I32<Polygon>

/*
impl Into<usize> for i32{
	fn into(self)->usize{ self as usize }
}
impl Into<usize> for u32{
	fn into(self)->usize{ self as usize }
}
impl Into<usize> for i16{
	fn into(self)->usize{ self as usize }
}
impl Into<usize> for u32{
	fn into(self)->usize{ self as usize }
}
impl Into<usize> for i8{
	fn into(self)->usize{ self as usize }
}
impl Into<usize> for u8{
	fn into(self)->usize{ self as usize }
}
impl Into<usize> for isize{
	fn into(self)->usize{ self as usize }
}
*/



#[derive(Debug)]
pub struct Array<T,I=i32>(pub Vec<T>,PhantomData<I>);

// my array helper fn's
impl<T:Clone,I:IndexTrait+Clone> Array<T,I>{
	/// TODO - better name. preserves ordering of vec![v;count].
	pub fn from_val_n(val:T, n:i32)->Self{
		let v=vec![val; n as usize];
		Array(v,PhantomData)
	}
	pub fn from_fn<F:Fn(I)->T>(count:I,f:F)->Self{
		let mut v=Vec::new();
		v.reserve(count.clone().my_into());
		for x in 0..count.my_into() {v.push(f(I::my_from(x)))}
		Array(v,PhantomData)
	}
	pub fn map<B,F:Fn(&T)->B>(&self,f:F)->Array<B,I>{
		let mut out=Array::<B,I>::new();
		out.reserve(self.len());
		for x in self.iter(){
			out.push(f(x))
		}
		out
	}
}

impl<T,I:IndexTrait+Clone> Array<T,I>{
	pub fn num_elems(&self)->i32{ self.0.len() as i32} // TODO - figure out generic int
	pub fn new()->Self{ Array(Vec::new(),PhantomData) }
	pub fn reserve(&mut self, additional: I){
		self.0.reserve(additional.my_into());
	}	
	pub fn push(&mut self,val:T){self.0.push(val)}
	pub fn shrink_to_fit(&mut self){self.0.shrink_to_fit()}
	pub fn truncate(&mut self, len: I){
		self.0.truncate(len.my_into());
	}
	pub fn as_slice(&self) -> &[T]{
		self.0.as_slice()
	}
	pub fn as_mut_slice(&mut self) -> &mut [T]{
		self.0.as_mut_slice()
	}
	pub fn swap_remove(&mut self, index: I) -> T{
		self.0.swap_remove(index.my_into())
	}
	pub fn insert(&mut self, index: I, element: T){
		self.0.insert(index.my_into(),element)
	}
	pub fn remove(&mut self, index: I) -> T{
		self.0.remove(index.my_into())
	}
	// aka filter in place
	pub fn retain<F:FnMut(&T)->bool>(&mut self, f: F) {
		self.0.retain(f)
	}
	pub fn dedup_by_key<F:FnMut(&mut T)->K, K:PartialEq<K>>(&mut self, key: F) {
		self.0.dedup_by_key(key)
	}
	pub fn dedup_by<F:FnMut(&mut T,&mut T)->bool>(&mut self, same_bucket: F) {
		self.0.dedup_by(same_bucket)
	}
	#[cfg(nightly_vector)]
	pub fn place_back(&mut self) -> PlaceBack<T>{
		self.0.place_back()
	}
	pub fn pop(&mut self) -> Option<T>{
		self.0.pop()
	}
	pub fn append(&mut self, other: &mut Vec<T>){
		self.0.append(other)
	}
	#[cfg(UseRangeArgument)]
	pub fn drain<R:RangeArgument<I>>(&mut self, range: R) -> Drain<T> 
	{
		self.0.drain(range)
	}
	pub fn clear(&mut self){
		self.0.clear()
	}
//	pub fn len(&self)->I{
//		self.0.len() as Index
//	}
//	pub fn is_empty(&self)->bool{ self.0.is_empty()}
	pub fn split_off(&mut self,at:I)->Array<T>{
		Array(self.0.split_off(at.my_into()),PhantomData)
	}
}
impl<T:Clone,I:IndexTrait> Array<T,I>{
	pub fn resize(&mut self, new_len:I, value:T){
		self.0.resize(new_len.my_into(),value)
	}
	pub fn extend_from_slice(&mut self, other:&[T]){
		self.0.extend_from_slice(other)
	}
}

impl<T:Default,I:IndexTrait> Array<T,I>{
	pub fn resize_default(&mut self, new_len:I){
		self.0.resize_default(new_len.my_into())
	}
}

impl<T:PartialEq<T>,I:IndexTrait> Array<T,I>{
	pub fn dedup(&mut self){
		self.0.dedup()
	}
	pub fn remove_item(&mut self, item:&T)->Option<T>{
		self.0.remove_item(item)
	}
}

impl<T,INDEX:IndexTrait> Array<T,INDEX>{
	/// TODO - figure out how to convert RangeArguemnt indices
	pub fn splice<I:IntoIterator<Item=T>,R:RangeArgument<usize>>(&mut self, range:R, replace_with:I)-> Splice<<I as IntoIterator>::IntoIter>
	{
		self.0.splice(range,replace_with)
	}
	pub fn drain_filter<F:FnMut(&mut T)->bool>(&mut self, filter: F) -> DrainFilter<T, F> {
		self.0.drain_filter(filter)
	}
}

impl<T,INDEX:IndexTrait> Deref for Array<T,INDEX>{
	type Target=[T];
	fn deref(&self)->&Self::Target { self.0.deref() }
}

impl<T,INDEX:IndexTrait> Array<T,INDEX>{
	fn len(&self)->INDEX{INDEX::my_from(self.0.len())}
	fn is_empty(&self)->bool{self.0.is_empty()}
	fn first(&self)->Option<&T>{self.0.first()}
	fn first_mut(&mut self)->Option<&mut T>{self.0.first_mut()}
	fn split_first(&self)->Option<(&T,&[T])>{self.0.split_first()}
	fn split_first_mut(&mut self)->Option<(&mut T, &mut [T])>{ self.0.split_first_mut() }
	fn split_last(&self)->Option<(&T,&[T])>{self.0.split_last()}
	fn split_last_mut(&mut self)->Option<(&mut T, &mut[T])>{self.0.split_last_mut()}
	fn last(&self)->Option<&T>{self.0.last()}
	fn last_mut(&mut self)->Option<&mut T>{self.0.last_mut()}
	fn get<I>(&self, index:I)->Option<&<I as SliceIndex<[T]> >::Output>
		where I:SliceIndex<[T]>
	{
		self.0.get(index)
	}
	fn get_mut<I>(&mut self, index:I)->Option<&mut <I as SliceIndex<[T]>>::Output>
		where I:SliceIndex<[T]>
	{
		self.0.get_mut(index)
	}
	unsafe fn get_unchecked<I>(&self, index: I) -> &<I as SliceIndex<[T]>>::Output 
where
    I: SliceIndex<[T]> {self.0.get_unchecked(index)}
unsafe fn get_unchecked_mut<I>(
	    &mut self, 
		index: I
	) -> &mut <I as SliceIndex<[T]>>::Output 
	where
		I: SliceIndex<[T]>{
		self.0.get_unchecked_mut(index)
	}
	fn as_ptr(&self)->*const T{self.0.as_ptr()}
	fn as_mut_ptr(&mut self)->*mut T{self.0.as_mut_ptr()}
	fn swap(&mut self, a:INDEX,b:INDEX){
		self.0.swap(a.my_into(),b.my_into())
	}
	fn reverse(&mut self){self.0.reverse()}
	fn iter(&self)->Iter<T>{self.0.iter()}
	fn iter_mut(&mut self)->IterMut<T>{self.0.iter_mut()}
	fn windows(&self,size:INDEX)->Windows<T>{self.0.windows(size.my_into())}
	fn chunks(&self,chunk_size:INDEX)->Chunks<T>{self.0.chunks(chunk_size.my_into())}
	
	fn chunks_mut(&mut self,chunk_size:INDEX)->ChunksMut<T>{self.0.chunks_mut(chunk_size.my_into())}
	fn split_at(&self, mid: INDEX) -> (&[T], &[T]){
		self.0.split_at(mid.my_into())
	}
	fn split_at_mut(&mut self, mid: INDEX) -> (&mut [T], &mut [T]){
		self.0.split_at_mut(mid.my_into())
	}
	fn split<F>(&self, pred: F) -> Split<T, F> 
		where F:FnMut(&T)->bool
	{
		self.0.split(pred)
	}
	fn split_mut<F>(&mut self, pred: F) -> SplitMut<T, F> 
		where F: FnMut(&T) -> bool
	{
		self.0.split_mut(pred)
	}
	fn rsplit<F>(&self, pred: F) -> RSplit<T, F> 
		where F: FnMut(&T) -> bool, 
	{
		self.0.rsplit(pred)
	}
	fn rsplit_mut<F>(&mut self, pred: F) -> RSplitMut<T, F>
		where F: FnMut(&T) -> bool	
	{
		self.0.rsplit_mut(pred)
	}
	fn splitn<F>(&self, n: INDEX, pred: F) -> SplitN<T, F> 
		where	F: FnMut(&T) -> bool
	{
		self.0.splitn(n.my_into(),pred)
	}
	fn splitn_mut<F>(&mut self, n: INDEX, pred: F) -> SplitNMut<T, F> 
		where F: FnMut(&T) -> bool
	{
		self.0.splitn_mut(n.my_into(),pred)
	}
	fn rsplitn<F>(&self, n: INDEX, pred: F) -> RSplitN<T, F> 
	where F: FnMut(&T) -> bool{
		self.0.rsplitn(n.my_into(),pred)
	}
	fn rsplitn_mut<F>(&mut self, n: INDEX, pred: F) -> RSplitNMut<T, F> 
where
    F: FnMut(&T) -> bool{
		self.0.rsplitn_mut(n.my_into(),pred)
	}
	fn contains(&self, x: &T) -> bool 
where
    T: PartialEq<T>{
		self.0.contains(x)
	}
	fn starts_with(&self, needle: &[T]) -> bool 
where
    T: PartialEq<T>{
		self.0.starts_with(needle)
	}
	fn ends_with(&self, needle: &[T]) -> bool 
where
    T: PartialEq<T>{
		self.0.ends_with(needle)
	}
	fn binary_search(&self, a: &T) -> Result<INDEX, INDEX> 
where
    T: Ord{
		match self.0.binary_search(a){
			Ok(x)=>Ok(INDEX::my_from(x)),
			Err(x)=>Err(INDEX::my_from(x))
		}
	}
	fn binary_search_by<'a, F>(&'a self, f: F) -> Result<INDEX, INDEX> 
		where F: FnMut(&'a T) -> Ordering{
		match self.0.binary_search_by(f){
			Ok(x)=>Ok(INDEX::my_from(x)),
			Err(x)=>Err(INDEX::my_from(x))
		}
	}
	fn binary_search_by_key<'a, B, F>(&'a self, b: &B, f: F) -> Result<INDEX, INDEX> 
	where
		B: Ord,
	    F: FnMut(&'a T) -> B,
		T: Ord
	{
		match self.0.binary_search_by_key(b,f){
			Ok(x)=>Ok(INDEX::my_from(x)),
			Err(x)=>Err(INDEX::my_from(x))
		}
	}
	fn sort(&mut self) where T:Ord{
		self.0.sort()
	}
	fn sort_by<F>(&mut self,f:F) where F:FnMut(&T,&T)->Ordering{
		self.0.sort_by(f)
	}
	fn sort_by_key<F,B>(&mut self,f:F) where B:Ord,F:FnMut(&T)->B{
		self.0.sort_by_key(f)
	}
	fn sort_unstable(&mut self)where T:Ord{self.0.sort_unstable()}
	
	fn sort_unstable_by<F>(&mut self,f:F)where T:Ord,F:FnMut(&T,&T)->Ordering{self.0.sort_unstable_by(f)}

	fn sort_unstable_by_key<B:Ord,F>(&mut self,f:F)where T:Ord,F:FnMut(&T)->B{self.0.sort_unstable_by_key(f)}
	fn rotate(&mut self,mid:INDEX){
		self.0.rotate(mid.my_into())
	}
	fn clone_from_slice(&mut self, src:&[T]) where T:Clone{
		self.0.clone_from_slice(src)
	}
	fn copy_from_slice(&mut self, src:&[T]) where T:Copy{
		self.0.copy_from_slice(src)
	}
	fn swap_with_slice(&mut self, src:&mut[T]){
		self.0.swap_with_slice(src)
	}
	fn to_vec(&self)->Array<T> where T:Clone{
		Array(self.0.to_vec(),PhantomData)
	}
	
}


impl<T,INDEX:IndexTrait> Index<INDEX> for Array<T,INDEX>{
	type Output=T;
	fn index(&self,i:INDEX)->&T{
		&self.0.index(i.my_into())
	}
}
impl<T,INDEX:IndexTrait> IndexMut<INDEX> for Array<T,INDEX>{
	fn index_mut(&mut self,i:INDEX)->&mut T{
		self.0.index_mut(i.my_into())
	}
}
impl<T:Clone,INDEX:IndexTrait> Clone for Array<T,INDEX>{
	fn clone(&self)->Self{
		Array(self.0.clone(),PhantomData)
	}
	fn clone_from(&mut self, other:&Self){
		self.0.clone_from(&other.0);
		self.1.clone_from(&other.1);
	}
	
}
impl<T,INDEX:IndexTrait> Default for Array<T,INDEX>{
	fn default()->Self{
		Array(Vec::<T>::default(),PhantomData)
	}
}

impl<T,INDEX:IndexTrait> Borrow<[T]> for Array<T,INDEX>{
	fn borrow(&self) -> &[T]{
		self.0.borrow()
	}
}

impl<T,INDEX:IndexTrait> AsRef<[T]> for Array<T,INDEX>{
	fn as_ref(&self)->&[T]{
		self.0.as_ref()
	}
} 
impl<T,INDEX:IndexTrait> AsRef<Array<T,INDEX>> for Array<T,INDEX>{
	fn as_ref(&self)->&Self{
		self
	}
} 

