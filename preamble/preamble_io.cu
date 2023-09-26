#ifdef BUFFER_READ_WRITE
#define BUFFER_READ_WRITE
#define MINMAX_TYPE int

__device__ inline {pixel_type}2 read_buffer3d{short_pixel_type}(int read_buffer_width, int read_buffer_height, int read_buffer_depth, {pixel_type} * buffer_var, int sampler, int4 position )
{
    int4 pos = make_int4(position.x, position.y, position.z, 0);
    
    pos.x = max((MINMAX_TYPE)pos.x, (MINMAX_TYPE)0);
    pos.y = max((MINMAX_TYPE)pos.y, (MINMAX_TYPE)0);
    pos.z = max((MINMAX_TYPE)pos.z, (MINMAX_TYPE)0);
    pos.x = min((MINMAX_TYPE)pos.x, (MINMAX_TYPE)read_buffer_width - 1);
    pos.y = min((MINMAX_TYPE)pos.y, (MINMAX_TYPE)read_buffer_height - 1);
    pos.z = min((MINMAX_TYPE)pos.z, (MINMAX_TYPE)read_buffer_depth - 1);

    int pos_in_buffer = pos.x + pos.y * read_buffer_width + pos.z * read_buffer_width * read_buffer_height;
    if (pos.x < 0 || pos.x >= read_buffer_width || pos.y < 0 || pos.y >= read_buffer_height || pos.z < 0 || pos.z >= read_buffer_depth) {
        return make_{pixel_type}2(0, 0);
    }
    return make_{pixel_type}2(buffer_var[pos_in_buffer],0);
}

__device__ inline void write_buffer3d{short_pixel_type}(int write_buffer_width, int write_buffer_height, int write_buffer_depth, {pixel_type} * buffer_var, int4 pos, float value )
{
    int pos_in_buffer = pos.x + pos.y * write_buffer_width + pos.z * write_buffer_width * write_buffer_height;
    if (pos.x < 0 || pos.x >= write_buffer_width || pos.y < 0 || pos.y >= write_buffer_height || pos.z < 0 || pos.z >= write_buffer_depth) {
        return;
    }
    buffer_var[pos_in_buffer] = value;
}

__device__ inline {pixel_type}2 read_buffer2d{short_pixel_type}(int read_buffer_width, int read_buffer_height, int read_buffer_depth, {pixel_type} * buffer_var, int sampler, int2 position )
{
    int4 pos = make_int4(position.x, position.y, 0, 0);
    
    pos.x = max((MINMAX_TYPE)pos.x, (MINMAX_TYPE)0);
    pos.y = max((MINMAX_TYPE)pos.y, (MINMAX_TYPE)0);
    pos.z = max((MINMAX_TYPE)pos.z, (MINMAX_TYPE)0);
    pos.x = min((MINMAX_TYPE)pos.x, (MINMAX_TYPE)read_buffer_width - 1);
    pos.y = min((MINMAX_TYPE)pos.y, (MINMAX_TYPE)read_buffer_height - 1);
    pos.z = min((MINMAX_TYPE)pos.z, (MINMAX_TYPE)read_buffer_depth - 1);

    int pos_in_buffer = pos.x + pos.y * read_buffer_width;
    if (pos.x < 0 || pos.x >= read_buffer_width || pos.y < 0 || pos.y >= read_buffer_height) {
        return make_{pixel_type}2(0, 0);
    }
    return make_{pixel_type}2(buffer_var[pos_in_buffer],0);
}

__device__ inline void write_buffer2d{short_pixel_type}(int write_buffer_width, int write_buffer_height, int write_buffer_depth, {pixel_type} * buffer_var, int2 pos, float value )
{
    int pos_in_buffer = pos.x + pos.y * write_buffer_width;
    if (pos.x < 0 || pos.x >= write_buffer_width || pos.y < 0 || pos.y >= write_buffer_height) {
        return;
    }
    buffer_var[pos_in_buffer] = value;
}

__device__ inline {pixel_type}2 read_buffer1d{short_pixel_type}(int read_buffer_width, int read_buffer_height, int read_buffer_depth, {pixel_type} * buffer_var, int sampler, int position )
{
    int pos = position;

    pos = max((MINMAX_TYPE)pos, (MINMAX_TYPE)0);
    pos = min((MINMAX_TYPE)pos, (MINMAX_TYPE)read_buffer_width - 1);

    int pos_in_buffer = pos;
    if (pos < 0 || pos >= read_buffer_width) {
        return make_{pixel_type}2(0, 0);
    }
    return make_{pixel_type}2(buffer_var[pos_in_buffer],0);
}

__device__ inline void write_buffer1d{short_pixel_type}(int write_buffer_width, int write_buffer_height, int write_buffer_depth, {pixel_type} * buffer_var, int pos, ulong value )
{
    int pos_in_buffer = pos;
    if (pos < 0 || pos >= write_buffer_width) {
        return;
    }
    buffer_var[pos_in_buffer] = value;
}

#endif // BUFFER_READ_WRITE