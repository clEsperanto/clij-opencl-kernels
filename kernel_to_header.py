#!/usr/bin/python

# This script is used to convert OpenCL and CUDA kernel files to C++ header files.
# The header files are used to embed the kernel source code into the library.
# The script is called and managed at build time by CMake.

import sys
import os
import glob

def stringify(input_file: str, output_path: str, prefix: str):
    """
    Converts an OpenCL or CUDA file to const char * in a C++ header file, with respect to the indentation.

    Args:
        input_file (str): Path to the input kernel file.
        output_path (str): Directory where the output header file will be saved.
        prefix (str): Prefix to be used in the header file names and guards.
    """
    kernel_template = """// This file is auto generated at build time. Do not edit manually.

#ifndef {prefix}_{kernel_upcase}_H
#define {prefix}_{kernel_upcase}_H

namespace kernel {{

constexpr const char* {kernel_locase} =
{kernel_source}
}} // end of namespace kernel

#endif // {prefix}_{kernel_upcase}_H
"""

    try:
        with open(input_file, 'r') as file:
            kernel_source = file.read()
    except IOError as e:
        print(f"Error reading file {input_file}: {e}")
        return

    # Extract and format kernel names
    kernel_name = os.path.basename(input_file).split(".")[0]
    kernel_upcase = kernel_name.upper()
    kernel_locase = kernel_name.lower()

    if "preamble" in kernel_locase:
        kernel_locase = "preamble_cl" if input_file.endswith(".cl") else "preamble_cu"

    # Compose output file name and path
    output_file = os.path.join(output_path, f"{prefix.lower()}_{kernel_locase}.h")

    # Format kernel source lines
    formatted_lines = [f'\t"{line}\\n"' for line in kernel_source.split("\n")]
    formatted_lines[-1] = formatted_lines[-1][:-2] + '";\n'  # Remove the last newline escape sequence
    formatted_kernel_source = "\n".join(formatted_lines)

    # Write the formatted kernel source to the output file
    try:
        with open(output_file, 'w') as file:
            file.write(kernel_template.format(
                prefix=prefix.upper(),
                kernel_upcase=kernel_upcase,
                kernel_locase=kernel_locase,
                kernel_source=formatted_kernel_source
            ))
    except IOError as e:
        print(f"Error writing file {output_file}: {e}")



def main():
    """
    Main function to convert OpenCL and CUDA kernel files to C++ header files.
    """
    if len(sys.argv) < 3:
        print("Usage: python kernel_to_header.py <input_folder> <output_folder>")
        sys.exit(1)
        
    input_folder = sys.argv[1]
    output_folder = sys.argv[2]

    if not os.path.isdir(input_folder):
        print(f"Error: {input_folder} is not a valid directory.")
        sys.exit(1)

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Collect all .cl and .cu files from the input folder
    file_list = glob.glob(os.path.join(input_folder, '**/*.cl'), recursive=True)
    file_list += glob.glob(os.path.join(input_folder, '**/*.cu'), recursive=True)

    # Convert each kernel file to a C++ header file
    for file in file_list:
        stringify(file, output_folder, "cle")

if __name__ == "__main__":
    main()
