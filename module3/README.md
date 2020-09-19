Assignment: The following will build just the assignment and
output 'assignment.exe'

'make'

Example runs:
./assignment.exe
./assignment.exe 5000000 256


# Optional parameters (use --help for documentation)
./assignment.exe 5000000 256 --verify
./assignment.exe 24 16 --debug


Run
'make block'

in order to output 6 different executables
that will run the 'blocks.cu' file with varying array size
and block sizes. I made changes to accept compiler
arguments to define the block size and array size.


Run
'make grid'

to compile grid.cu into a single executable. I made more
iterations in the for-loop to display several interesting
new dimensions as output.
