// __constant sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP_TO_EDGE | CLK_FILTER_NEAREST;

// __kernel void detect_maxima(
//     IMAGE_src_TYPE  src,
//     IMAGE_dst_TYPE  dst
// )
// { 
//   const int x = get_global_id(0);
//   const int y = get_global_id(1);
//   const int z = get_global_id(2);

//   int4 r = (int4){0,0,0,0};
//   if (GET_IMAGE_WIDTH(src)  > 1) { radius.x = 1; }
//   if (GET_IMAGE_HEIGHT(src) > 1) { radius.y = 1; }
//   if (GET_IMAGE_DEPTH(src)  > 1) { radius.z = 1; }

//   float localMax = (float) READ_IMAGE(src, sampler, POS_src_INSTANCE(x,y,z,0)).x -1;
//   int4 localMaxPos = (int4){x,y,z,0};
//   const int4 pos = (int4){x,y,z,0};
//   for (int rx = -radius.x; rx <= radius.x; ++rx) {
//       for (int ry = -radius.y; ry <= radius.y; ++ry) {
//           for (int rz = -radius.z; rz <= radius.z; ++rz) {
//               int4 localPos = localMaxPos + (int4){rx,ry,rz,0};
//               if( all(localPos >= 0) && any(localPos != pos) ) {
//                 const float value = (float) READ_IMAGE(src, sampler, POS_src_INSTANCE(localPos.x,localPos.y,localPos.z,0)).x;
//                 if (value > localMax) {
//                     localMax = value;
//                     localMaxPos = localPos;
//                 }
//               }
//           }
//       }
//   }

//   IMAGE_dst_PIXEL_TYPE result = 0;
//   if (localMaxPos.x == x && localMaxPos.y == y && localMaxPos.z == z) {
//       result = 1;
//   }
//   WRITE_IMAGE(dst, POS_dst_INSTANCE(x,y,z,0), result);
// }


__constant sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP_TO_EDGE | CLK_FILTER_NEAREST;

__kernel void detect_maxima(
    IMAGE_src_TYPE  src,
    IMAGE_dst_TYPE  dst
)
{ 
  const int x = get_global_id(0);
  const int y = get_global_id(1);
  const int z = get_global_id(2);

  int4 r = (int4){0,0,0,0};
  if (GET_IMAGE_WIDTH(src)  > 1) { radius.x = 1; }
  if (GET_IMAGE_HEIGHT(src) > 1) { radius.y = 1; }
  if (GET_IMAGE_DEPTH(src)  > 1) { radius.z = 1; }

  bool isMax = true;
  float localMax = (float) READ_IMAGE(src, sampler, POS_src_INSTANCE(x,y,z,0)).x;
  const int4 pos = (int4){x,y,z,0};

  for (int rx = -radius.x; rx <= radius.x; ++rx) {
      for (int ry = -radius.y; ry <= radius.y; ++ry) {
          for (int rz = -radius.z; rz <= radius.z; ++rz) {
              int4 localPos = pos + (int4){rx,ry,rz,0};
              if( localPos.x == pos.x && localPos.y == pos.y && localPos.z == pos.z) {
                continue;
              }
              if( localPos.x >= 0 && localPos.y >= 0 && localPos.z >= 0 ) {
                const float value = (float) READ_IMAGE(src, sampler, POS_src_INSTANCE(localPos.x,localPos.y,localPos.z,0)).x;
                if (value >= localMax) {
                    isMax = false;
                }
              }
          }
      }
  }

  IMAGE_dst_PIXEL_TYPE result = 0;
  if (isMax) {
      result = 1;
  }
  WRITE_IMAGE(dst, POS_dst_INSTANCE(x,y,z,0), result);
}