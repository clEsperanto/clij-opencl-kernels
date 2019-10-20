__constant sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP_TO_EDGE | CLK_FILTER_NEAREST;

__kernel void count_nonzero_slicewise_image3d
(
  DTYPE_IMAGE_OUT_3D dst, DTYPE_IMAGE_IN_3D src,
  const int Nx, const int Ny
)
{
  const int i = get_global_id(0), j = get_global_id(1), k = get_global_id(2);
  const int4 coord = (int4){i,j,k,0};

    const int4   e = (int4)  {(Nx-1)/2, (Ny-1)/2, 0, 0 };

    float aSquared = e.x * e.x;
    float bSquared = e.y * e.y;

    DTYPE_OUT sum = 0;
    int count = 0;

    for (int x = -e.x; x <= e.x; x++) {
        float xSquared = x * x;
        for (int y = -e.y; y <= e.y; y++) {
            float ySquared = y * y;
            if (xSquared / aSquared + ySquared / bSquared <= 1.0) {
                DTYPE_OUT value = (DTYPE_OUT)READ_IMAGE_3D(src,sampler,coord+((int4){x,y,k,0})).x;
                if (value != 0) {
                    count++;
                }
            }
        }
    }

  DTYPE_OUT res = CONVERT_DTYPE_OUT(count);
  WRITE_IMAGE_3D(dst, coord, res);
}


__kernel void count_nonzero_image2d
(
  DTYPE_IMAGE_OUT_2D dst, DTYPE_IMAGE_IN_2D src,
  const int Nx, const int Ny
)
{
  const int i = get_global_id(0), j = get_global_id(1);
  const int2 coord = (int2){i,j};

    const int4   e = (int4)  {(Nx-1)/2, (Ny-1)/2, 0, 0 };
    int count = 0;
    float sum = 0;

    float aSquared = e.x * e.x;
    float bSquared = e.y * e.y;

  for (int x = -e.x; x <= e.x; x++) {
      float xSquared = x * x;
      for (int y = -e.y; y <= e.y; y++) {
          float ySquared = y * y;
          if (xSquared / aSquared + ySquared / bSquared <= 1.0) {
              DTYPE_OUT value = (DTYPE_OUT)READ_IMAGE_2D(src,sampler,coord+((int2){x,y})).x;
              if (value != 0) {
                  count++;
              }
          }
      }
  }

  DTYPE_OUT res = CONVERT_DTYPE_OUT(count);
  WRITE_IMAGE_2D(dst, coord, res);
}

__kernel void count_nonzero_image3d
(
  DTYPE_IMAGE_OUT_3D dst, DTYPE_IMAGE_IN_3D src,
  const int Nx, const int Ny, const int Nz
)
{
  const int i = get_global_id(0), j = get_global_id(1), k = get_global_id(2);
  const int4 coord = (int4){i,j,k,0};

    const int4   e = (int4)  {(Nx-1)/2, (Ny-1)/2, (Nz-1)/2, 0 };
    int count = 0;
    float sum = 0;

    float aSquared = e.x * e.x;
    float bSquared = e.y * e.y;
    float cSquared = e.z * e.z;

    for (int x = -e.x; x <= e.x; x++) {
        float xSquared = x * x;
        for (int y = -e.y; y <= e.y; y++) {
            float ySquared = y * y;
            for (int z = -e.z; z <= e.z; z++) {
                float zSquared = z * z;
                if (xSquared / aSquared + ySquared / bSquared + zSquared / cSquared <= 1.0) {

                    int x1 = coord.x + x;
                    int x2 = coord.y + y;
                    int x3 = coord.z + z;
                    const int4 pos = (int4){x1,x2,x3,0};
                    float value_res = (float)READ_IMAGE_3D(src,sampler,pos).x;
                    if (value_res != 0) {
                    count++;
                    }
                }
            }
        }
    }


  DTYPE_OUT res = CONVERT_DTYPE_OUT(count);
  WRITE_IMAGE_3D(dst, coord, res);
}











