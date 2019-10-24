
// A kernel to fill an image with the beautiful XOR fractal:
//default xorfractal dx=0i
//default xorfractal dy=0i
//default xorfractal u =0f


__kernel void xorfractal  (__write_only DTYPE_IMAGE_OUT_3D dst, 
                                        int       dx, 
                                        int       dy, 
                                        float     u 
                          )
{
	int x = get_global_id(0); 
	int y = get_global_id(1);
	int z = get_global_id(2);
	
	WRITE_IMAGE_3D (dst, (int4)(x, y, z, 0), u*((x+dx)^((y+dy)+1)^(z+2)));
}


// A kernel to fill an image with a xor fractal filled sphere:
//default xorsphere cx=0i
//default xorsphere cy=0i
//default xorsphere cz=0i
//default xorsphere r =80f
__kernel void xorsphere   (__write_only DTYPE_IMAGE_OUT_3D dst, 
                                        int       cx, 
                                        int       cy,
                                        int       cz,  
                                        float     r 
                          )
{
  const int width  = get_image_width(dst);
  const int height = get_image_height(dst);
  const int depth  = get_image_depth(dst);
  
  float4 dim = (float4){width,height,depth,1};
  
  int x = get_global_id(0); 
  int y = get_global_id(1);
  int z = get_global_id(2);
  
  float4 pos = (float4){x,y,z,0};
  
  float4 cen = (float4){cx,cy,cz,0};
  
  float d = fast_length((pos-cen)/dim);
  
  float value = (float)( (x^y^z)*((d<r)?1:0) );
  
  WRITE_IMAGE_3D (dst, (int4){x,y,z,0}, value);
}


// A kernel to fill an image with a uniform filled sphere:
//default sphere cx=0i
//default sphere cy=0i
//default sphere cz=0i
//default sphere r =80f
__kernel void sphere   (__write_only DTYPE_IMAGE_OUT_3D dst, 
                                        int       cx, 
                                        int       cy,
                                        int       cz,  
                                        float     r 
                          )
{
  const int width  = get_image_width(dst);
  const int height = get_image_height(dst);
  const int depth  = get_image_depth(dst);
  
  float4 dim = (float4){width,height,depth,1};
  
  int x = get_global_id(0); 
  int y = get_global_id(1);
  int z = get_global_id(2);
  
  float4 pos = (float4){x,y,z,0};
  
  float4 cen = (float4){cx,cy,cz,0};
  
  float d = fast_length((pos-cen)/dim);
  
  float value = (float)((d<r)?1:0);
  
  WRITE_IMAGE_3D (dst, (int4){x,y,z,0}, value);
}

// A kernel to fill an image with a line:
//default aline a=0i
//default aline b=0i
//default aline c=0i
//default aline d=1i
//default aline r=0.1f
__kernel void aline   (__write_only DTYPE_IMAGE_OUT_3D dst, 
                                   int       a, 
                                   int       b,
                                   int       c,
                                   int       d,  
                                   float     r 
                     )
{
  const int width  = get_image_width(dst);
  const int height = get_image_height(dst);
  const int depth  = get_image_depth(dst);
  
  const int x = get_global_id(0); 
  const int y = get_global_id(1);
  const int z = get_global_id(2);
  
  float4 pos = (float4){x,y,z,0};
  
  float4 vec = (float4){a,b,c,d};
  
  float dist = fabs(dot(pos,vec));
  
  float value = (float)((dist<r)?1:0);
  
  WRITE_IMAGE_3D (dst, (int4){x,y,z,0}, value);
}

