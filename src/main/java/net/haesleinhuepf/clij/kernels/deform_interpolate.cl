
#ifndef SAMPLER_FILTER
#define SAMPLER_FILTER CLK_FILTER_LINEAR
#endif

#ifndef SAMPLER_ADDRESS
#define SAMPLER_ADDRESS CLK_ADDRESS_CLAMP
#endif

__kernel void deform_3d_interpolate(DTYPE_IMAGE_IN_3D src,
                        DTYPE_IMAGE_IN_3D vectorX,
                        DTYPE_IMAGE_IN_3D vectorY,
                        DTYPE_IMAGE_IN_3D vectorZ,
	      			 DTYPE_IMAGE_OUT_3D dst)
{
  const sampler_t sampler = CLK_NORMALIZED_COORDS_TRUE|
      SAMPLER_ADDRESS |	SAMPLER_FILTER;

  uint i = get_global_id(0);
  uint j = get_global_id(1);
  uint k = get_global_id(2);

  uint Nx = GET_IMAGE_WIDTH(src);
  uint Ny = GET_IMAGE_HEIGHT(src);
  uint Nz = GET_IMAGE_DEPTH(src);

  float x = i+0.5f;
  float y = j+0.5f;
  float z = k+0.5f;

  int4 pos = (int4){i, j, k,0};

  float x2 = x + (float)(READ_IMAGE_3D(vectorX, sampler, pos).x);
  float y2 = y + (float)(READ_IMAGE_3D(vectorY, sampler, pos).x);
  float z2 = z + (float)(READ_IMAGE_3D(vectorZ, sampler, pos).x);


  //int4 coord_norm = (int4)(x2 * GET_IMAGE_WIDTH(input) / GET_IMAGE_WIDTH(output),y2 * GET_IMAGE_HEIGHT(input) / GET_IMAGE_HEIGHT(output), z2  * GET_IMAGE_DEPTH(input) / GET_IMAGE_DEPTH(output),0.f);
  float4 coord_norm = (float4)(x2 / Nx, y2 / Ny, z2 / Nz,0.f);



  float pix = 0;
  if (x2 >= 0 && y2 >= 0 && z2 >= 0 &&
      x2 < GET_IMAGE_WIDTH(src) && y2 < GET_IMAGE_HEIGHT(src) && z2 < GET_IMAGE_DEPTH(src)
  ) {
    pix = (float)(READ_IMAGE_3D(src, sampler, coord_norm).x);
  }

  WRITE_IMAGE_3D(dst, pos, CONVERT_DTYPE_OUT(pix));
}


__kernel void deform_2d_interpolate(DTYPE_IMAGE_IN_2D src,
                        DTYPE_IMAGE_IN_2D vectorX,
                        DTYPE_IMAGE_IN_2D vectorY,
 	      			 DTYPE_IMAGE_OUT_2D dst)
{
  const sampler_t sampler = CLK_NORMALIZED_COORDS_TRUE|
      SAMPLER_ADDRESS |	SAMPLER_FILTER;

  uint i = get_global_id(0);
  uint j = get_global_id(1);

  uint Nx = GET_IMAGE_WIDTH(src);
  uint Ny = GET_IMAGE_HEIGHT(src);

  float x = i+0.5f;
  float y = j+0.5f;


  int2 pos = (int2){i, j};

  float x2 = x + (float)(READ_IMAGE_2D(vectorX, sampler, pos).x);
  float y2 = y + (float)(READ_IMAGE_2D(vectorY, sampler, pos).x);


  float2 coord_norm = (float2)(x2 / Nx, y2 / Ny);

  float pix = 0;
  if (x2 >= 0 && y2 >= 0 &&
      x2 < GET_IMAGE_WIDTH(src) && y2 < GET_IMAGE_HEIGHT(src)
  ) {
    pix = (float)(READ_IMAGE_2D(src, sampler, coord_norm).x);
  }


  WRITE_IMAGE_2D(dst, pos, CONVERT_DTYPE_OUT(pix));
}