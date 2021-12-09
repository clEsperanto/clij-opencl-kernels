__constant sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP_TO_EDGE | CLK_FILTER_NEAREST;

__kernel void maximum_separable(
    IMAGE_src_TYPE  src,
    IMAGE_dst_TYPE  dst,
    const int       dim,
    const int       N,
    const float     s
)
{
  const int i = get_global_id(0);
  const int j = get_global_id(1);
  const int k = get_global_id(2);

  const POS_src_TYPE coord = POS_src_INSTANCE(i,j,k,0);
  const POS_src_TYPE dir   = POS_src_INSTANCE(dim==0,dim==1,dim==2,0);

  const int center = (int) (N-1) / 2;

  float res = (float) READ_IMAGE(src, sampler, coord).x;
  for (int v = -center; v <= center; ++v) {
    res = max(res, (float) READ_IMAGE(src, sampler, coord + v * dir).x);
  }

  WRITE_IMAGE(dst, POS_dst_INSTANCE(i,j,k,0), CONVERT_dst_PIXEL_TYPE(res));
}