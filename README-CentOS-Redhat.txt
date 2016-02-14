SDO_FE Code
Updated: March 25, 2012

Required libs: 

- Cfitsio (latest version)
- Armadillo 2.4.3 (with LAPACK, BLAS and ATLAS installed) 
- GSL 1.13
- Opencv 2.0 (code has been tested with 2.3 and it compiles with a minor correction in the make file)

Code compiles with "make".

Runs with 4 args: input file, output folder, segments per axis, thumbnail size (of one axis)

Example Run: ./sdoFE example.fits ouput/ 32 256

- example.fits is the file to extract parameters from
- output folder will hold the .head and lossless .png files with same file names as input files
- segments per axis is the number of splits per axis to make on the image, so for example, 32 would indicate and 32x32 grid.
- thumbnail size is for example 256x256 in the example listed


Instalation: (tested on CentOS 6 and Red Hat Enterprise Linux Server release 6.2)

Installation for CentOS 6 (as root)

After a fresh install of the OS: 

yum groupinstall "Development Tools"

yum install cmake

yum install boost-devel

yum install blas-devel lapack-devel atlas-devel

Versions installed:
altas-3.8.4
altas-devel- 3.8.4
blas-3.2.1
blas-devel-3.2.1
lapack-3.2.1
lapack-devel-3.2.1

yum install gsl-devel

Version installed:
gsl-1.13
gsl-devel-1.13

yum install opencv-devel

Version installed
opencv 2.0.0

- CFITSIO

1) Untar cfitsio tar file
2) ./configure --prefix=/usr/local
3) make         
4) make install

-Armadillo

1) untar armadillo tar file  (be sure LAPACK and BLAS are setup BEFORE this one)
2) cmake .     <-------- has to be exactly like this!!!
3) make 
4) sudo make install

ldconfig

----- END ----





