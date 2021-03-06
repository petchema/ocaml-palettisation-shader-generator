# ocaml-palettisation-shader-generator

Generates nearest-color palettization shader code using classic palettes from Daggerfall.
This has been used in a retro mode shader of Daggerfall Unity.

The algorithm is a k-d tree based nearest-neighbour search (Wikipedia: https://en.wikipedia.org/wiki/K-d_tree#Nearest_neighbour_search), with a cutoff when reaching a partition smaller than a given number of colors.

## Compilation

Using OCaml optimizing compiler: 

    ocamlopt -o palette palette.ml
    
(tested with OCaml 4.06.0)

## Execution

    ./palette [palette file name] [max size of color clusters]
    
Example

    ./palette ART_PAL.txt 42 > palette.txt

Input palette file should be a text file, containing R G B values in 0-255 range, one color per line.

Generated code will be sent to the standard output, while debugging messages may be sent to the standard error stream.
palette.txt is an example of code generated.

The main function provided by this code is simply

    fixed4 nearestColor(fixed4 color)
