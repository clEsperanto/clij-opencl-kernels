__constant sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP_TO_EDGE | CLK_FILTER_NEAREST;

// Define which projection axis to use (0=X, 1=Y, 2=Z)
// This should be defined at compile time via -DPROJECTION_AXIS=0, 1, or 2

#if PROJECTION_AXIS == 0
  // X projection
  #define AXIS_SIZE(img) GET_IMAGE_WIDTH(img)
  #define IDX0_COORD z
  #define IDX1_COORD y
  #define LOOP_COORD x
  #define INITIAL_POS POS_src_INSTANCE(0, y, z, 0)
  #define LOOP_POS POS_src_INSTANCE(x, y, z, 0)
  #define OUTPUT_POS POS_dst_INSTANCE(z, y, 0, 0)
#elif PROJECTION_AXIS == 1
  // Y projection
  #define AXIS_SIZE(img) GET_IMAGE_HEIGHT(img)
  #define IDX0_COORD x
  #define IDX1_COORD z
  #define LOOP_COORD y
  #define INITIAL_POS POS_src_INSTANCE(x, 0, z, 0)
  #define LOOP_POS POS_src_INSTANCE(x, y, z, 0)
  #define OUTPUT_POS POS_dst_INSTANCE(x, z, 0, 0)
#elif PROJECTION_AXIS == 2
  // Z projection
  #define AXIS_SIZE(img) GET_IMAGE_DEPTH(img)
  #define IDX0_COORD x
  #define IDX1_COORD y
  #define LOOP_COORD z
  #define INITIAL_POS POS_src_INSTANCE(x, y, 0, 0)
  #define LOOP_POS POS_src_INSTANCE(x, y, z, 0)
  #define OUTPUT_POS POS_dst_INSTANCE(x, y, 0, 0)

#else
  #error "PROJECTION_AXIS must be defined as 0 (X), 1 (Y), or 2 (Z)"
#endif

__kernel void maximum_projection(
    IMAGE_src_TYPE  src,
    IMAGE_dst_TYPE  dst
) 
{
  const int IDX0_COORD = get_global_id(0);
  const int IDX1_COORD = get_global_id(1);
  
  IMAGE_src_PIXEL_TYPE maximum = READ_IMAGE(src, sampler, INITIAL_POS).x;
  for (int LOOP_COORD = 1; LOOP_COORD < AXIS_SIZE(src); ++LOOP_COORD) {
    const IMAGE_src_PIXEL_TYPE value = READ_IMAGE(src, sampler, LOOP_POS).x;
    maximum = (maximum > value) ? maximum : value;
  }
  
  WRITE_IMAGE(dst, OUTPUT_POS, CONVERT_dst_PIXEL_TYPE(maximum));
}
