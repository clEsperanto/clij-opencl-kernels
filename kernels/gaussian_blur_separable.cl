// Adapted from Uwe Schmidt, https://github.com/ClearControl/fastfuse/blob/master/src/main/java/fastfuse/tasks/kernels/blur.cl

__constant sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP_TO_EDGE | CLK_FILTER_NEAREST;

__kernel void gaussian_blur_separable(
    IMAGE_src_TYPE  src,
    IMAGE_dst_TYPE  dst, 
    const int       dim, 
    const int       N,
    const float     s
)
{
  const int x = get_global_id(0);
  const int y = get_global_id(1);
  const int z = get_global_id(2);

  const POS_src_TYPE coord = POS_src_INSTANCE(x,y,z,0);
  const POS_src_TYPE dir   = POS_src_INSTANCE(dim==0,dim==1,dim==2,0);

  const int   center = (int) (N - 1) / 2;
  const float norm   = -2 * s * s;

  float res = 0;
  float hsum = 0;
  for (int v = -center; v <= center; ++v) {
    const float h = exp( (v * v) / norm );
    res += h * (float) READ_IMAGE(src, sampler, coord + v * dir).x;
    hsum += h;
  }
  
  WRITE_IMAGE(dst, POS_dst_INSTANCE(x,y,z,0), CONVERT_dst_PIXEL_TYPE(res / hsum));
}
