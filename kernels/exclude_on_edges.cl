__constant sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_NONE | CLK_FILTER_NEAREST;

__kernel void exclude_on_edges_x(
    IMAGE_src_TYPE  src,
    IMAGE_dst_TYPE  dst
)
{
  int x = get_global_id(0);
  const int y = get_global_id(1);
  const int z = get_global_id(2);
  const int width = GET_IMAGE_WIDTH(src);

  x = 0;
  POS_src_TYPE pos = POS_src_INSTANCE(x, y, z, 0);
  int index = READ_IMAGE(src, sampler, pos).x;
  if (index > 0) {
    WRITE_IMAGE(dst, POS_dst_INSTANCE(index, 0, 0, 0), 0);
  }
  x = width - 1;
  pos = POS_src_INSTANCE(x, y, z, 0);
  index = READ_IMAGE(src, sampler, pos).x;
  if (index > 0) {
    WRITE_IMAGE(dst, POS_dst_INSTANCE(index, 0, 0, 0), 0);
  }
}


__kernel void exclude_on_edges_y(
    IMAGE_src_TYPE  src,
    IMAGE_dst_TYPE  dst
)
{
  const int x = get_global_id(0);
  int y = get_global_id(1);
  const int z = get_global_id(2);
  const int height = GET_IMAGE_HEIGHT(src);

  y = 0;
  POS_src_TYPE pos = POS_src_INSTANCE(x, y, z, 0);
  int index = READ_IMAGE(src, sampler, pos).x;
  if (index > 0) {
    WRITE_IMAGE(dst, POS_dst_INSTANCE(index, 0, 0, 0), 0);
  }
  y = height - 1;
  pos = POS_src_INSTANCE(x, y, z, 0);
  index = READ_IMAGE(src, sampler, pos).x;
  if (index > 0) {
    WRITE_IMAGE(dst, POS_dst_INSTANCE(index, 0, 0, 0), 0);
  }
}


__kernel void exclude_on_edges_z(
    IMAGE_src_TYPE  src,
    IMAGE_dst_TYPE  dst
)
{
  const int x = get_global_id(0);
  const int y = get_global_id(1);
  int z = get_global_id(2);
  const int depth = GET_IMAGE_DEPTH(src);

  z = 0;
  POS_src_TYPE pos = POS_src_INSTANCE(x, y, z, 0);
  int index = READ_IMAGE(src, sampler, pos).x;
  if (index > 0) {
    WRITE_IMAGE(dst, POS_dst_INSTANCE(index, 0, 0, 0), 0);
  }
  z = depth - 1;
  pos = POS_src_INSTANCE(x, y, z, 0);
  index = READ_IMAGE(src, sampler, pos).x;
  if (index > 0) {
    WRITE_IMAGE(dst, POS_dst_INSTANCE(index, 0, 0, 0), 0);
  }
}