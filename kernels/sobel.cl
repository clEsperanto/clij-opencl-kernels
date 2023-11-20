__constant sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP_TO_EDGE | CLK_FILTER_NEAREST;

__kernel void sobel(
    IMAGE_src_TYPE  src,
    IMAGE_dst_TYPE  dst
)
{
  const int x = get_global_id(0);
  const int y = get_global_id(1);
  const int z = get_global_id(2);
  
  const POS_src_TYPE pos = POS_src_INSTANCE(x,y,z,0);

  const float hx[3] = {0.25, 0.5, 0.25};
  const float hy[3] = {0.25, 0.5, 0.25};
  const float hz[3] = {0.25, 0.5, 0.25};

  const float hpx[3] = {1, 0, -1};
  const float hpy[3] = {1, 0, -1};
  const float hpz[3] = {1, 0, -1};

  int4 r = (int4){0,0,0,0};
  if (GET_IMAGE_DEPTH(src)  > 1) { r.z = 2; }
  if (GET_IMAGE_HEIGHT(src) > 1) { r.y = 2; }
  if (GET_IMAGE_WIDTH(src)  > 1) { r.x = 2; }

  float gy[3][3][3];
  float gx[3][3][3];
  float gz[3][3][3];

  /*build the kernels i.e. h'_x(x,y,z)=h'(x)h(y)h(z)=gx(x,y,z)*/
  for(int m=0; m<=r.x; ++m) {
    for(int n=0; n<=r.y; ++n) {
      for(int k=0; k<=r.z; ++k) {
  	    gx[m][n][k] = hpx[m] * hy[n]  * hz[k];
  	    gy[m][n][k] = hx[m]  * hpy[n] * hz[k];
  	    gz[m][n][k] = hx[m]  * hy[n]  * hpz[k];
      }
    }
  }

  float sum_x=0, sum_y=0, sum_z=0;
  for(int m=0; m<=r.x; ++m) {
      for(int n=0; n<=r.y; ++n) {
          for(int k=0; k<=r.z; ++k) {
              if (GET_IMAGE_WIDTH(src)  > 1) { 
                sum_x += gx[m][n][k] * (float) READ_IMAGE(src, sampler, pos + POS_src_INSTANCE(m-1,n-1,k-1,0)).x;
              }
              if (GET_IMAGE_HEIGHT(src) > 1) { 
                sum_y += gy[m][n][k] * (float) READ_IMAGE(src, sampler, pos + POS_src_INSTANCE(m-1,n-1,k-1,0)).x;
              }
              if (GET_IMAGE_DEPTH(src)  > 1) { 
                sum_z += gz[m][n][k] * (float) READ_IMAGE(src, sampler, pos + POS_src_INSTANCE(m-1,n-1,k-1,0)).x;
              }
          }
      }
  }
  const float result = sqrt(sum_x * sum_x + sum_y * sum_y + sum_z * sum_z);
  
  WRITE_IMAGE(dst, POS_dst_INSTANCE(x,y,z,0), CONVERT_dst_PIXEL_TYPE(result));
}
