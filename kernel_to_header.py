#!/usr/bin/python

# This script is used to convert OpenCL and CUDA kernel files to C++ header files.
# The header files are used to embed the kernel source code into the library.
# The script is called and managed at build time by CMake.

import sys
import os
import glob

def stringify(input_file: str, output_path: str, prefix: str):
    kernel_template = """// This file is auto generated at build time. Do not edit manually.

#ifndef {prefix}_{kernel_upcase}_H
#define {prefix}_{kernel_upcase}_H

namespace kernel {{

constexpr const char* {kernel_locase} =
{kernel_source}
}} // end of namespace kernel

#endif // {prefix}_{kernel_upcase}_H
"""

    with open(input_file, 'r') as f:
        kernel_source = f.read()

    # manage name and file name
    upcase_name = input_file.split(os.sep)[-1].split(".")[0].upper()
    lowcase_name = upcase_name.lower()
    if lowcase_name.find("preamble") != -1:
        lowcase_name = "preamble_cl" if input_file.find(".cl") != -1 else "preamble_cu"

    # compose output file name and path using os.path.join
    output_file = os.path.join(output_path, prefix.lower() + "_" + lowcase_name + ".h")
    split_kernel_source = kernel_source.split("\n")

    lines = []
    for i, line in enumerate(split_kernel_source):
            if i == len(split_kernel_source) - 1:
                lines.append(f"\t\"{line}\";\n")
            else:
                lines.append(f"\t\"{line}\\n\"\n")
    kernel_source = "".join(lines)

    with open(output_file, 'w') as f:
        f.write(kernel_template.format(prefix=prefix.upper(), kernel_upcase=upcase_name, kernel_locase=lowcase_name, kernel_source=kernel_source))

def main():
    if len(sys.argv) < 3:
        print("Usage: python kernel_to_header.py <input_folder> <input_folder>")
        sys.exit(1)
    input_folder = sys.argv[1] if os.path.isdir(sys.argv[1]) else sys.exit(1)
    output_folder = sys.argv[2]
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    file_list = glob.glob( os.path.join(input_folder,'**/*.cl'), recursive=True)
    file_list += glob.glob( os.path.join(input_folder,'**/*.cu'), recursive=True)
    for file in file_list:
        stringify(file, output_folder, "cle")


if __name__ == "__main__":
    main()
