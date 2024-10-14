__constant sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP_TO_EDGE | CLK_FILTER_NEAREST;

__kernel void detect_maxima(
    IMAGE_src_TYPE  src,
    IMAGE_dst_TYPE  dst
)
{ 
  const int x = get_global_id(0);
  const int y = get_global_id(1);
  const int z = get_global_id(2);

  int4 radius = (int4){0,0,0,0};
  if (GET_IMAGE_WIDTH(src)  > 1) { radius.x = 1; }
  if (GET_IMAGE_HEIGHT(src) > 1) { radius.y = 1; }
  if (GET_IMAGE_DEPTH(src)  > 1) { radius.z = 1; }

  bool isMax = true;
  float localMax = (float) READ_IMAGE(src, sampler, POS_src_INSTANCE(x,y,z,0)).x;
  const int4 pos = (int4){x,y,z,0};

  for (int dz = -radius.z; dz <= radius.z; ++dz) {
    for (int dy = -radius.y; dy <= radius.y; ++dy) {
      for (int dx = -radius.x; dx <= radius.x; ++dx) {

        int4 localPos = pos + (int4){dx,dy,dz,0};

        // we don't want to compare the pixel with itself
        if( localPos.x == pos.x && localPos.y == pos.y && localPos.z == pos.z) {
          continue;
        }

        // if pixel is outside the image, we don't want to compare it
        if( (localPos.x < 0 || localPos.y < 0 || localPos.z < 0) && (localPos.x > GET_IMAGE_WIDTH(src) || localPos.y > GET_IMAGE_HEIGHT(src) || localPos.z > GET_IMAGE_DEPTH(src)) ) {
          continue;
        }

        // we get the value of the pixel and test if it is greater than the current maximum
        const float value = (float) READ_IMAGE(src, sampler, POS_src_INSTANCE(localPos.x,localPos.y,localPos.z,0)).x;
        if (value >= localMax) {
            isMax = false;
        }
        
      }
    }
  }

  IMAGE_dst_PIXEL_TYPE result = (isMax) ? 1 : 0;
  WRITE_IMAGE(dst, POS_dst_INSTANCE(x,y,z,0), result);
}