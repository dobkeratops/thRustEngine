use super::*

pub struct MeshBuilder {
	vertices:Vec<(Point3,Color)>
}


  	            Conflicting draws

                  less verbose
               fewer entities to name & define    		 	  
    function   	   	------->   	  
    per trait                     big traits
      	   	   	   <--------
               easier to organize
               (seperate out & regroup,)
               (reducing dependancies
                per implementation)
	   			 
       no               ||       Garbage
     runtime/GC         ||     Collected
                        ||
                        || 	     D 	   	   	   	
         C              ||	   	   	   	  native 
        C++             || 	 Go    	ML?	   	   	 
                        ||------------------ 
      Rust              ||    java C#     JIT
     (contender)        || 	  				 
                        ||
						||
						||
				
				
