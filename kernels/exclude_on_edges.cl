__constant sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_NONE | CLK_FILTER_NEAREST;

// Define which axis to exclude on (0=X, 1=Y, 2=Z)
// This should be defined at compile time via -DEXCLUDE_AXIS=0, 1, or 2

#if EXCLUDE_AXIS == 0
  // X axis
  #define AXIS_SIZE(img) GET_IMAGE_WIDTH(img)
  #define IDX0_COORD_UNUSED x
  #define IDX1_COORD y
  #define IDX2_COORD z
  #define POS_EDGE(edge_val, y, z) POS_src_INSTANCE(edge_val, y, z, 0)
#elif EXCLUDE_AXIS == 1
  // Y axis
  #define AXIS_SIZE(img) GET_IMAGE_HEIGHT(img)
  #define IDX0_COORD_UNUSED y
  #define IDX1_COORD x
  #define IDX2_COORD z
  #define POS_EDGE(edge_val, x, z) POS_src_INSTANCE(x, edge_val, z, 0)
#elif EXCLUDE_AXIS == 2
  // Z axis
  #define AXIS_SIZE(img) GET_IMAGE_DEPTH(img)
  #define IDX0_COORD_UNUSED z
  #define IDX1_COORD x
  #define IDX2_COORD y
  #define POS_EDGE(edge_val, x, y) POS_src_INSTANCE(x, y, edge_val, 0)
#else
  #error "EXCLUDE_AXIS must be defined as 0 (X), 1 (Y), or 2 (Z)"
#endif

__kernel void exclude_on_edges(
    IMAGE_src_TYPE  src,
    IMAGE_dst_TYPE  dst
)
{
  int IDX0_COORD_UNUSED = get_global_id(0);
  const int IDX1_COORD = get_global_id(1);
  const int IDX2_COORD = get_global_id(2);
  const int axis_size = AXIS_SIZE(src);

  // Check first edge
  int edge_val = 0;
  POS_src_TYPE pos = POS_EDGE(edge_val, IDX1_COORD, IDX2_COORD);
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