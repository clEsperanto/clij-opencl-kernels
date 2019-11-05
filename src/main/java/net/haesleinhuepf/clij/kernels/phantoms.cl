
// A kernel to fill an image with the beautiful XOR fractal:
//default xorfractal dx=0i
//default xorfractal dy=0i
//default xorfractal u =0f


__kernel void xorfractal_2d(DTYPE_IMAGE_OUT_2D dst, 
                                        int       dx, 
                                        int       dy, 
                                        float     u 
                          )
{
	int x = get_global_id(0); 
	int y = get_global_id(1);
        float value = u*((x+dx)^((y+dy)+1)^(2));
	
	WRITE_IMAGE_2D (dst, (int2){x, y}, CONVERT_DTYPE_OUT(value));
}


__kernel void xorfractal_3d(DTYPE_IMAGE_OUT_3D dst, 
                                        int       dx, 
                                        int       dy, 
                                        float     u 
                          )
{
	int x = get_global_id(0); 
	int y = get_global_id(1);
	int z = get_global_id(2);
        float value = u*((x+dx)^((y+dy)+1)^(z+2));
	
	WRITE_IMAGE_3D (dst, (int4){x, y, z, 0}, CONVERT_DTYPE_OUT(value));
}


// A kernel to fill an image with a xor fractal filled sphere:
//default xorsphere cx=0i
//default xorsphere cy=0i
//default xorsphere r =80f
__kernel void xorsphere_2d(DTYPE_IMAGE_OUT_2D dst, 
                                        int       cx, 
                                        int       cy,
                                        float     r 
                          )
{
  const int width  = get_image_width(dst);
  const int height = get_image_height(dst);
  
  float2 dim = (float2){width,height};
  
  int x = get_global_id(0); 
  int y = get_global_id(1);
  
  float2 pos = (float2){x,y};
  
  float2 cen = (float2){cx,cy};
  
  float d = fast_length((pos-cen)/dim);
  
  float value = (float)( (x^y)*((d<r)?1:0) );
  
  WRITE_IMAGE_2D (dst, (int2){x,y}, CONVERT_DTYPE_OUT(value));
}



// A kernel to fill an image with a xor fractal filled sphere:
//default xorsphere cx=0i
//default xorsphere cy=0i
//default xorsphere cz=0i
//default xorsphere r =80f
__kernel void xorsphere_3d(DTYPE_IMAGE_OUT_3D dst, 
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
__kernel void sphere_2d(DTYPE_IMAGE_OUT_2D dst, 
                                        int       cx, 
                                        int       cy, 
                                        float     r 
                          )
{
  const int width  = get_image_width(dst);
  const int height = get_image_height(dst);
  
  float2 dim = (float2){width,height};
  
  int x = get_global_id(0); 
  int y = get_global_id(1);
  
  float2 pos = (float2){x,y};
  
  float2 cen = (float2){cx,cy};
  
  float d = fast_length((pos-cen)/dim);
  
  float value = (float)((d<r)?1:0);
  
  WRITE_IMAGE_2D (dst, (int2){x,y}, CONVERT_DTYPE_OUT(value));
}



// A kernel to fill an image with a uniform filled sphere:
//default sphere cx=0i
//default sphere cy=0i
//default sphere cz=0i
//default sphere r =80f
__kernel void sphere_3d(DTYPE_IMAGE_OUT_3D dst, 
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
  
  WRITE_IMAGE_3D (dst, (int4){x,y,z,0}, CONVERT_DTYPE_OUT(value));
}
