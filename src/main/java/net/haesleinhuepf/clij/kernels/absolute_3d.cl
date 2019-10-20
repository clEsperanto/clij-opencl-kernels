__constant sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP_TO_EDGE | CLK_FILTER_NEAREST;

__kernel void absolute_3d(DTYPE_IMAGE_IN_3D  src,
                          DTYPE_IMAGE_OUT_3D  dst
                     )
{
  const int x = get_global_id(0);
  const int y = get_global_id(1);
  const int z = get_global_id(2);

  const int4 pos = (int4){x,y,z,0};

  float value = READ_IMAGE_3D(src, sampler, pos).x;
  if ( value < 0 ) {
    value = -1 * value;
  }

  WRITE_IMAGE_3D (dst, pos, CONVERT_DTYPE_OUT(value));
}