Output from Running Host Memory Test

ccc_v1_w_DN6k_191794@runweb21:~/rbirdman_EN605.617.FA/module4$ make host                                                                                                                                
nvcc host_memory.cu -std=c++11 -L /usr/local/cuda/lib -lcudart -o host_memory                                                                                                                           
nvcc warning : The 'compute_20', 'sm_20', and 'sm_21' architectures are deprecated, and may be removed in a future release (Use -Wno-deprecated-gpu-targets to suppress warning).                       
host_memory.cu: In function ‘int main()’:                                                                                                                                                               
host_memory.cu:50:44: warning: format ‘%d’ expects argument of type ‘int’, but argument 2 has type ‘long unsigned int’ [-Wformat=]                                                                      
     printf("Byte size: %d\n", N * sizeof(float));                                                                                                                                                      
                                            ^                                                                                                                                                           
ccc_v1_w_DN6k_191794@runweb21:~/rbirdman_EN605.617.FA/module4$ ./host_memory                                                                                                                            
Byte size: 4000                                                                                                                                                                                         
Allocate pinned memory Execution time = 2447376 microseconds.                                                                                                                                           
Array Initialization Execution time = 5 microseconds.                                                                                                                                                   
memcpy to device Execution time = 47 microseconds.                                                                                                                                                      
memcpy to host Execution time = 24 microseconds.                                                                                                                                                        
Verification Execution time = 17 microseconds.                                                                                                                                                          
Max error: 0.000000                                                                                                                                                                                     
Free pinned memory Execution time = 292 microseconds.                                                                                                                                                   
Program Execution Execution time = 2448232 microseconds.       



Output from Running Global Memory Test
ccc_v1_w_DN6k_191794@runweb21:~/rbirdman_EN605.617.FA/module4$ make global                                                                                                                              
nvcc global_memory.cu -std=c++11 -lcudart -o global_memory                                                                                                                                              
nvcc warning : The 'compute_20', 'sm_20', and 'sm_21' architectures are deprecat                                                                                                                        
global_memory.cu(172): warning: type qualifier on a reference type is not allowe                                                                                                                        
                                                                                                                                                                                                        
global_memory.cu(173): warning: type qualifier on a reference type is not allowe                                                                                                                        
                                                                                                                                                                                                        
global_memory.cu(447): warning: variable "non_interleaved_host_dest_ptr" is used                                                                                                                        
                                                                                                                                                                                                        
global_memory.cu(447): warning: variable "non_interleaved_host_src_ptr" is used                                                                                                                         
                                                                                                                                                                                                        
global_memory.cu(172): warning: type qualifier on a reference type is not allowe                                                                                                                        
                                                                                                                                                                                                        
global_memory.cu(173): warning: type qualifier on a reference type is not allowe                                                                                                                        
                                                                                                                                                                                                        
global_memory.cu(447): warning: variable "non_interleaved_host_dest_ptr" is used                                                                                                                        
                                                                                                                                                                                                        
global_memory.cu(447): warning: variable "non_interleaved_host_src_ptr" is used                                                                                                                         
                                                                                                                                                                                                        
ccc_v1_w_DN6k_191794@runweb21:~/rbirdman_EN605.617.FA/module4$ ./global_memory                                                                                                                          
Testing num elements: 8192                                                                                                                                                                              
Testing iterations: 1000                                                                                                                                                                                
gpu non interleaved duration: 8369.484375ms 



Full output report (Vocareum dumped the output from the other run)

ccc_v1_w_DN6k_191794@runweb21:~/rbirdman_EN605.617.FA/module4$ ./global_memory                                                                                                                          
Testing num elements: 8192                                                                                                                                                                              
Testing iterations: 1000                                                                                                                                                                                
cpu interleaved duration: 64.396385ms                                                                                                                                                                   
cpu non-interleaved duration: 76.609535ms                                                                                                                                                               
gpu interleaved duration: 2.070144ms                                                                                                                                                                    
gpu non interleaved duration: 8369.899414ms 