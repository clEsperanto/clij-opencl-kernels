# clij-opencl-kernels

This repository contains a collection of [OpenCL](https://www.khronos.org/opencl/) [kernels for generic image 
processing](https://github.com/clij/clij-opencl-kernels/tree/master/src/main/java/net/haesleinhuepf/clij/kernels). 
The [CLIJ](https://clij.github.io) is build on top of it allowing
[ImageJ](https://imagej.nih.gov/ij/) / [Fiji](https://fiji.sc) users in doing
GPU-accelerated image processing without the need for learning OpenCL.

**If you use it, please cite it:**

Robert Haase, Loic Alain Royer, Peter Steinbach, Deborah Schmidt, 
Alexandr Dibrov, Uwe Schmidt, Martin Weigert, Nicola Maghelli, Pavel Tomancak, 
Florian Jug, Eugene W Myers. 
*CLIJ: GPU-accelerated image processing for everyone*. BioRxiv preprint. [https://doi.org/10.1101/660704](https://doi.org/10.1101/660704)

## Why a custom OpenCL-dialect?

OpenCL offers several pixel types, such as `uint8`, `unit16` and `float`. 
Theoretically, one has to write OpenCL-kernels specifically for given input- and output-images, such as a kernel for 
adding images of type `float` resulting in a `float` image and a kernel for adding image of type `uint8` resulting in 
an image of type `float`. Furthermore, OpenCL defines images and buffers. However, as both are arrays of pixel 
intensities in memory, we wanted to access them in a unified way. As this would result in a ridiculous large number of individual kernel implementations, we used
a dialect where placeholders such as `DTYPE_OUT` represent the pixel type of the output image.

## List of placeholders
The following list of placeholders are used at the moment:
<table border="1">

<tr>
<td><b>Place holder</b></td>
<td><b>Replacement during runtime</b></td>
</tr>

<tr>
<td><pre>CONVERT_DTYPE_IN</pre></td>
<td rowspan="2"><pre>
<a href="https://github.com/clij/clij-clearcl/blob/master/src/main/java/net/haesleinhuepf/clij/clearcl/ocllib/preamble/preamble.cl#L303">clij_convert_char_sat</a>
<a href="https://github.com/clij/clij-clearcl/blob/master/src/main/java/net/haesleinhuepf/clij/clearcl/ocllib/preamble/preamble.cl#L292">clij_convert_uchar_sat</a>
<a href="https://github.com/clij/clij-clearcl/blob/master/src/main/java/net/haesleinhuepf/clij/clearcl/ocllib/preamble/preamble.cl#L325">clij_convert_short_sat</a>
<a href="https://github.com/clij/clij-clearcl/blob/master/src/main/java/net/haesleinhuepf/clij/clearcl/ocllib/preamble/preamble.cl#L314">clij_convert_ushort_sat</a>
<a href="https://github.com/clij/clij-clearcl/blob/master/src/main/java/net/haesleinhuepf/clij/clearcl/ocllib/preamble/preamble.cl#L345">clij_convert_int_sat</a>
<a href="https://github.com/clij/clij-clearcl/blob/master/src/main/java/net/haesleinhuepf/clij/clearcl/ocllib/preamble/preamble.cl#L335">clij_convert_uint_sat</a>
<a href="https://github.com/clij/clij-clearcl/blob/master/src/main/java/net/haesleinhuepf/clij/clearcl/ocllib/preamble/preamble.cl#L355">clij_convert_float_sat</a>
</pre>
</td>
</tr>
<tr>
<td><pre>CONVERT_DTYPE_OUT</pre></td>
</tr>
<tr>
<td><pre>DTYPE_IMAGE_IN_2D</pre></td>
<td><pre>
__read_only image2d_t
__global char*
__global uchar*
__global short*
__global ushort*
__global float*
</pre></td>
</tr>
<tr>
<td><pre>DTYPE_IMAGE_IN_3D</pre></td>
<td><pre>
__read_only image3d_t
__global char*
__global uchar*
__global short*
__global ushort*
__global float*
</pre></td>
</tr>
<tr>
<td><pre>DTYPE_IMAGE_OUT_2D</pre></td>
<td><pre>
__write_only image2d_t
__global char*
__global uchar*
__global short*
__global ushort*
__global float*
</pre></td>
</tr>
<tr>
<td><pre>DTYPE_IMAGE_OUT_3D</pre></td>
<td><pre>
__write_only image3d_t
__global char*
__global uchar*
__global short*
__global ushort*
__global float*
</pre></td>
</tr>
<tr>
<td><pre>GET_IMAGE_DEPTH</pre></td>
<td>constant number</td>
</tr>
<tr>
<td><pre>GET_IMAGE_HEIGHT</pre></td>
<td>constant number</td>
</tr>
<tr>
<td><pre>GET_IMAGE_WIDTH</pre></td>
<td>constant number</td>
</tr>
<tr>
<td><pre>READ_IMAGE_2D</pre></td>
<td><pre>
<a href="https://www.khronos.org/registry/OpenCL/sdk/1.2/docs/man/xhtml/read_imagei2d.html">read_imageui</a>
<a href="https://www.khronos.org/registry/OpenCL/sdk/1.2/docs/man/xhtml/read_imagef2d.html">read_imagef</a>
<a href="https://github.com/clij/clij-clearcl/blob/master/src/main/java/net/haesleinhuepf/clij/clearcl/ocllib/preamble/preamble.cl#L167">read_buffer2dc</a>
<a href="https://github.com/clij/clij-clearcl/blob/master/src/main/java/net/haesleinhuepf/clij/clearcl/ocllib/preamble/preamble.cl#L383">read_buffer2duc</a>
<a href="https://github.com/clij/clij-clearcl/blob/master/src/main/java/net/haesleinhuepf/clij/clearcl/ocllib/preamble/preamble.cl#L199">read_buffer2di</a>
<a href="https://github.com/clij/clij-clearcl/blob/master/src/main/java/net/haesleinhuepf/clij/clearcl/ocllib/preamble/preamble.cl#L215">read_buffer2dui</a>
<a href="https://github.com/clij/clij-clearcl/blob/master/src/main/java/net/haesleinhuepf/clij/clearcl/ocllib/preamble/preamble.cl#L231">read_buffer2df</a>
</pre></td>
</tr>
<tr>
<td><pre>READ_IMAGE_3D</pre></td>
<td><pre>
<a href="https://www.khronos.org/registry/OpenCL/sdk/1.2/docs/man/xhtml/read_imagei3d.html">read_imageui</a>
<a href="https://www.khronos.org/registry/OpenCL/sdk/1.2/docs/man/xhtml/read_imagef3d.html">read_imagef</a>
<a href="https://github.com/clij/clij-clearcl/blob/master/src/main/java/net/haesleinhuepf/clij/clearcl/ocllib/preamble/preamble.cl#L32">read_buffer3dc</a>
<a href="https://github.com/clij/clij-clearcl/blob/master/src/main/java/net/haesleinhuepf/clij/clearcl/ocllib/preamble/preamble.cl#L50">read_buffer3duc</a>
<a href="https://github.com/clij/clij-clearcl/blob/master/src/main/java/net/haesleinhuepf/clij/clearcl/ocllib/preamble/preamble.cl#L68">read_buffer3di</a>
<a href="https://github.com/clij/clij-clearcl/blob/master/src/main/java/net/haesleinhuepf/clij/clearcl/ocllib/preamble/preamble.cl#L86">read_buffer3dui</a>
<a href="https://github.com/clij/clij-clearcl/blob/master/src/main/java/net/haesleinhuepf/clij/clearcl/ocllib/preamble/preamble.cl#L104">read_buffer3df</a>
</pre></td>
</tr>
<tr>
<td><pre>WRITE_IMAGE_2D</pre></td>
<td><pre>
<a href="https://www.khronos.org/registry/OpenCL/sdk/1.2/docs/man/xhtml/write_image2d.html">write_imageui</a>
<a href="https://www.khronos.org/registry/OpenCL/sdk/1.2/docs/man/xhtml/write_image2d.html">write_imagef</a>
<a href="https://github.com/clij/clij-clearcl/blob/master/src/main/java/net/haesleinhuepf/clij/clearcl/ocllib/preamble/preamble.cl#L247">write_buffer2dc</a>
<a href="https://github.com/clij/clij-clearcl/blob/master/src/main/java/net/haesleinhuepf/clij/clearcl/ocllib/preamble/preamble.cl#L256">write_buffer2duc</a>
<a href="https://github.com/clij/clij-clearcl/blob/master/src/main/java/net/haesleinhuepf/clij/clearcl/ocllib/preamble/preamble.cl#L265">write_buffer2di</a>
<a href="https://github.com/clij/clij-clearcl/blob/master/src/main/java/net/haesleinhuepf/clij/clearcl/ocllib/preamble/preamble.cl#L274">write_buffer2dui</a>
<a href="https://github.com/clij/clij-clearcl/blob/master/src/main/java/net/haesleinhuepf/clij/clearcl/ocllib/preamble/preamble.cl#L283">write_buffer2df</a>
</pre></td>
</tr>
<tr>
<td><pre>WRITE_IMAGE_3D</pre></td>
<td><pre>
<a href="https://www.khronos.org/registry/OpenCL/sdk/1.2/docs/man/xhtml/write_image3d.html">write_imageui</a>
<a href="https://www.khronos.org/registry/OpenCL/sdk/1.2/docs/man/xhtml/write_image3d.html">write_imagef</a>
<a href="https://github.com/clij/clij-clearcl/blob/master/src/main/java/net/haesleinhuepf/clij/clearcl/ocllib/preamble/preamble.cl#L122">write_buffer3dc</a>
<a href="https://github.com/clij/clij-clearcl/blob/master/src/main/java/net/haesleinhuepf/clij/clearcl/ocllib/preamble/preamble.cl#L131">write_buffer3duc</a>
<a href="https://github.com/clij/clij-clearcl/blob/master/src/main/java/net/haesleinhuepf/clij/clearcl/ocllib/preamble/preamble.cl#L140">write_buffer3di</a>
<a href="https://github.com/clij/clij-clearcl/blob/master/src/main/java/net/haesleinhuepf/clij/clearcl/ocllib/preamble/preamble.cl#L149">write_buffer3dui</a>
<a href="https://github.com/clij/clij-clearcl/blob/master/src/main/java/net/haesleinhuepf/clij/clearcl/ocllib/preamble/preamble.cl#L158">write_buffer3df</a>
</pre></td>
</tr>

</table>

## Known issues
* The described OpenCL dialect has a drawback: Input images all need to be of the same type and output images have to be 
of the same type. As this assumption might be violated in practice, we are working on an new version of the dialect to
be release in summer 2020.
* Some of the placeholders represent image size in x/y/z. This results in performance drop in case many images of 
different size are processed as individual kernels need to be implemented. We have this in mind for the summer 2020 
update of the OpenCL-dialect.

## Deprecation note
The opencl-kernels presented here will be replaced by Summer 2020. Background
is an upcoming update of the used OpenCL dialect. If you plan to build your 
software directly on our OpenCL kernels, you should consider waiting for this 
update. You can follow development live on the 
[development branch](https://github.com/clij/clij-opencl-kernels/tree/development/src/main/java/net/haesleinhuepf/clij/kernels).