__constant sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_NONE | CLK_FILTER_NEAREST;

#if EXCLUDE_AXIS == 0
  // X axis - iterate over Y and Z
  #define AXIS_SIZE(img) GET_IMAGE_WIDTH(img)
  #define IDX1_COORD get_global_id(1)
  #define IDX2_COORD get_global_id(2)
  #define POS_EDGE(edge_val, y, z) POS_src_INSTANCE(edge_val, y, z, 0)
#elif EXCLUDE_AXIS == 1
  // Y axis - iterate over X and Z
  #define AXIS_SIZE(img) GET_IMAGE_HEIGHT(img)
  #define IDX1_COORD get_global_id(0)
  #define IDX2_COORD get_global_id(2)
  #define POS_EDGE(edge_val, x, z) POS_src_INSTANCE(x, edge_val, z, 0)
#elif EXCLUDE_AXIS == 2
  // Z axis - iterate over X and Y
  #define AXIS_SIZE(img) GET_IMAGE_DEPTH(img)
  #define IDX1_COORD get_global_id(0)
  #define IDX2_COORD get_global_id(1)
  #define POS_EDGE(edge_val, x, y) POS_src_INSTANCE(x, y, edge_val, 0)
#else
  #error "EXCLUDE_AXIS must be defined as 0 (X), 1 (Y), or 2 (Z)"
#endif

__kernel void exclude_on_edges(
    IMAGE_src_TYPE  src,
    IMAGE_dst_TYPE  dst
)
{
  const int a = IDX1_COORD;
  const int b = IDX2_COORD;
  const int axis_size = AXIS_SIZE(src);

  // Check first edge
  int edge_val = 0;
  POS_src_TYPE pos = POS_EDGE(edge_val, a, b);
  int index = READ_IMAGE(src, sampler, pos).x;
  if (index > 0) {
    WRITE_IMAGE(dst, POS_dst_INSTANCE(index, 0, 0, 0), 0);
  }
  
  // Check last edge
  edge_val = axis_size - 1;
  pos = POS_EDGE(edge_val, IDX1_COORD, IDX2_COORD);
  index = READ_IMAGE(src, sampler, pos).x;
  if (index > 0) {
    WRITE_IMAGE(dst, POS_dst_INSTANCE(index, 0, 0, 0), 0);
  }
}