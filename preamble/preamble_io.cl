#ifdef BUFFER_READ_WRITE
#define BUFFER_READ_WRITE
#define MINMAX_TYPE int

inline {pixel_type}2 read_buffer3d{short_pixel_type}(int read_buffer_width, int read_buffer_height, int read_buffer_depth, __global {pixel_type} * buffer_var, sampler_t sampler, int4 position )
{
    int4 pos = (int4){position.x, position.y, position.z, 0};
    if (true) { // if (CLK_ADDRESS_CLAMP_TO_EDGE & sampler) {
        pos.x = max((MINMAX_TYPE)pos.x, (MINMAX_TYPE)0);
        pos.y = max((MINMAX_TYPE)pos.y, (MINMAX_TYPE)0);
        pos.z = max((MINMAX_TYPE)pos.z, (MINMAX_TYPE)0);
        pos.x = min((MINMAX_TYPE)pos.x, (MINMAX_TYPE)read_buffer_width - 1);
        pos.y = min((MINMAX_TYPE)pos.y, (MINMAX_TYPE)read_buffer_height - 1);
        pos.z = min((MINMAX_TYPE)pos.z, (MINMAX_TYPE)read_buffer_depth - 1);
    }
    int pos_in_buffer = pos.x + pos.y * read_buffer_width + pos.z * read_buffer_width * read_buffer_height;
    if (pos.x < 0 || pos.x >= read_buffer_width || pos.y < 0 || pos.y >= read_buffer_height || pos.z < 0 || pos.z >= read_buffer_depth) {
        return ({pixel_type}2){0, 0};
    }
    return ({pixel_type}2){buffer_var[pos_in_buffer],0};
}

inline void write_buffer3d{short_pixel_type}(int write_buffer_width, int write_buffer_height, int write_buffer_depth, __global {pixel_type} * buffer_var, int4 pos, short value )
{
    int pos_in_buffer = pos.x + pos.y * write_buffer_width + pos.z * write_buffer_width * write_buffer_height;
    if (pos.x < 0 || pos.x >= write_buffer_width || pos.y < 0 || pos.y >= write_buffer_height || pos.z < 0 || pos.z >= write_buffer_depth) {
        return;
    }
    buffer_var[pos_in_buffer] = value;
}

inline {pixel_type}2 read_buffer2d{short_pixel_type}(int read_buffer_width, int read_buffer_height, int read_buffer_depth, __global {pixel_type} * buffer_var, sampler_t sampler, int2 position )
{
    int2 pos = (int2){position.x, position.y};
    if (true) { // if (CLK_ADDRESS_CLAMP_TO_EDGE & sampler) {
        pos.x = max((MINMAX_TYPE)pos.x, (MINMAX_TYPE)0);
        pos.y = max((MINMAX_TYPE)pos.y, (MINMAX_TYPE)0);
        pos.x = min((MINMAX_TYPE)pos.x, (MINMAX_TYPE)read_buffer_width - 1);
        pos.y = min((MINMAX_TYPE)pos.y, (MINMAX_TYPE)read_buffer_height - 1);
    }
    int pos_in_buffer = pos.x + pos.y * read_buffer_width;
    if (pos.x < 0 || pos.x >= read_buffer_width || pos.y < 0 || pos.y >= read_buffer_height) {
        return ({pixel_type}2){0, 0};
    }
    return ({pixel_type}2){buffer_var[pos_in_buffer],0};
}

inline void write_buffer2d{short_pixel_type}(int write_buffer_width, int write_buffer_height, int write_buffer_depth, __global {pixel_type} * buffer_var, int2 pos, ushort value )
{
    int pos_in_buffer = pos.x + pos.y * write_buffer_width;
    if (pos.x < 0 || pos.x >= write_buffer_width || pos.y < 0 || pos.y >= write_buffer_height) {
        return;
    }
    buffer_var[pos_in_buffer] = value;
}

inline {pixel_type}2 read_buffer1d{short_pixel_type}(int read_buffer_width, int read_buffer_height, int read_buffer_depth, __global {pixel_type} * buffer_var, sampler_t sampler, int position )
{
    int pos = (int){position};
    if (true) { // if (CLK_ADDRESS_CLAMP_TO_EDGE & sampler) {
        pos = max((MINMAX_TYPE)pos, (MINMAX_TYPE)0);
        pos = min((MINMAX_TYPE)pos, (MINMAX_TYPE)read_buffer_width - 1);
    }
    int pos_in_buffer = pos;
    if (pos < 0 || pos >= read_buffer_width) {
        return ({pixel_type}2){0,0};
    }
    return ({pixel_type}2){buffer_var[pos_in_buffer],0};
}

inline void write_buffer1d{short_pixel_type}(int write_buffer_width, int write_buffer_height, int write_buffer_depth, __global {pixel_type} * buffer_var, int pos, short value )
{
    int pos_in_buffer = pos;
    if (pos < 0 || pos >= write_buffer_width) {
        return;
    }
    buffer_var[pos_in_buffer] = value;
}

#endif // BUFFER_READ_WRITE
