// adapted from: https://github.com/maweigert/gputools/blob/master/gputools/transforms/kernels/transformations.cl
//
// Copyright (c) 2016, Martin Weigert
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
// * Redistributions of source code must retain the above copyright notice, this
//   list of conditions and the following disclaimer.
//
// * Redistributions in binary form must reproduce the above copyright notice,
//   this list of conditions and the following disclaimer in the documentation
//   and/or other materials provided with the distribution.
//
// * Neither the name of gputools nor the names of its
//   contributors may be used to endorse or promote products derived from
//   this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS \"AS IS\"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
// DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
// FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
// DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
// SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
// CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
// OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

#ifndef SAMPLER_FILTER
#define SAMPLER_FILTER CLK_FILTER_LINEAR
#endif

#ifndef SAMPLER_ADDRESS
#define SAMPLER_ADDRESS CLK_ADDRESS_CLAMP
#endif

__constant sampler_t sampler = CLK_NORMALIZED_COORDS_TRUE| SAMPLER_ADDRESS | SAMPLER_FILTER;

__kernel void affine_transform_interpolate(
    IMAGE_src_TYPE  src,
    IMAGE_dst_TYPE  dst,
    IMAGE_mat_TYPE  mat
)
{
  const uint i = get_global_id(0);
  const uint j = get_global_id(1);
  const uint k = get_global_id(2);

  const uint Nx = GET_IMAGE_WIDTH(src);
  const uint Ny = GET_IMAGE_HEIGHT(src);
  const uint Nz = GET_IMAGE_DEPTH(src);

  const float x = i + 0.5f;
  const float y = j + 0.5f;
  const float z = k + 0.5f;

  const float x2 = mat[0] * x + mat[1] * y + mat[2]  * z + mat[3] ;
  const float y2 = mat[4] * x + mat[5] * y + mat[6]  * z + mat[7] ;
  const float z2 = mat[8] * x + mat[9] * y + mat[10] * z + mat[11];

  if (Nz > 1)
  {
  const float4 read_coord = (float4) {x2/Nx, y2/Ny, z2/Nz, 0.f};
  const int4 write_coord = (int4) {i, j, k, 0};

  const float pix = (float) READ_IMAGE(src, sampler, read_coord).x;
  WRITE_IMAGE(dst, write_coord, CONVERT_dst_PIXEL_TYPE(pix));
  }
  else
  {
  const float2 read_coord = (float2) {x2/Nx, y2/Ny};
  const int2 write_coord = (int2) {i, j};

  const float pix = (float) READ_IMAGE(src, sampler, read_coord).x;
  WRITE_IMAGE(dst, write_coord, CONVERT_dst_PIXEL_TYPE(pix));
  }
}