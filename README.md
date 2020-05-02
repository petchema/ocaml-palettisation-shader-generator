# ocaml-palettisation-shader-generator

Generates nearest-color palettization shader code using ART_PAL classic palette from Daggerfall.
This has been used in a retro mode shader of Daggerfall Unity.

The algorithm is a k-d tree based nearest-neighbour search, with a cutoff when reaching a partition smaller than a given number of colors.

## Compilation

Using OCaml optimizing compiler: 

    ocamlopt -o palette palette.ml
    
(tested with OCaml 4.06.0)

## Execution

    ./palette [max size of color clusters]
    
Example

    ./palette 42

Generated code will be output to the standard output, while debugging message may be sent to the standard error stream.
palette.txt is an example of code generated.
