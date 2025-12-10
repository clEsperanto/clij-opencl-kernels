__constant sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP_TO_EDGE | CLK_FILTER_NEAREST;

__kernel void laplace_box(
    IMAGE_src_TYPE  src,
    IMAGE_dst_TYPE  dst
)
{
  const int x = get_global_id(0);
  const int y = get_global_id(1);
  const int z = get_global_id(2);

  const int4 r = (int4){
    (GET_IMAGE_WIDTH(src) > 1), 
    (GET_IMAGE_HEIGHT(src) > 1), 
    (GET_IMAGE_DEPTH(src) > 1), 
    0};

  const POS_src_TYPE pos = POS_src_INSTANCE(x,y,z,0);
  const float norm = pow(3.0f, (int)(r.x + r.y + r.z)) - 1;

  float result = 0;
  for (int dz = -r.z; dz <= r.z; ++dz) {
    for (int dy = -r.y; dy <= r.y; ++dy) {
      for (int dx = -r.x; dx <= r.x; ++dx) {
        const int is_center = (dx == 0 && dy == 0 && dz == 0);
        const float weight = is_center ? norm : -1.0f;
        const POS_src_TYPE offset = is_center ? POS_src_INSTANCE(0,0,0,0) : POS_src_INSTANCE(dx,dy,dz,0);
        result += (float) READ_IMAGE(src, sampler, pos + offset).x * weight;
      }
    }
  }

  WRITE_IMAGE(dst, POS_dst_INSTANCE(x,y,z,0), CONVERT_dst_PIXEL_TYPE(result));
}
