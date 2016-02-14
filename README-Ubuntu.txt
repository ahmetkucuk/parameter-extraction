SDO_FE Code
Updated: 07-03-2012

Required libs: 

- Cfitsio (latest version)
- Armadillo 2.4.3 (with LAPACK and BLAS installed) 
- GSL 1.15
- Opencv 2.1 (code has been tested with 2.3 and it compiles with a minor correction in the make file)

Code compiles with "make".

Runs with 4 args: input file, output folder, segments per axis, thumbnail size (of one axis)

Example Run: ./sdoFE example.fits ouput/ 32 256

- example.fits is the file to extract parameters from
- output folder will hold the .head and lossless .png files with same file names as input files
- segments per axis is the number of splits per axis to make on the image, so for example, 32 would indicate and 32x32 grid.
- thumbnail size is for example 256x256 in the example listed


Instalation:

We have extensively tested in Ubuntu 11.04 and this configuration worked fine for us:

sudo apt-get install build-essential
sudo apt-get install liblapack-dev
sudo apt-get install libblas-dev

- CFITSIO

1) Untar cfitsio tar file
2) ./configure --prefix=/usr/local
3) make         
4) sudo make install

-ARMADILLO 2.4.3
1) untar armadillo tar file  (be sure LAPACK and BLAS are setup BEFORE this one)
2) cmake .     <-------- has to be exactly like this!!!
3) make 
4) sudo make install

- GSL 1.15
1) untar gsl tar file
2) ./configure
3) make
4) sudo make install

-OpenCV 2.1 ( this one can be very tricky)
Modified from: http://www.samontab.com/web/2010/04/installing-opencv-2-1-in-ubuntu/

1) sudo apt-get install libgtk2.0-dev libavcodec-dev libavformat-dev libjpeg62-dev libtiff4-dev cmake libswscale-dev libjasper-dev
2) Untar the opencv tar file
2) cmake .     <-------- has to be exactly like this!!!
3) make 
4) sudo make install
5) sudo gedit /etc/ld.so.conf.d/opencv.conf

Add the following line at the end of the file(it may be an empty file, that is ok) and then save it:
	
/usr/local/lib

6) sudo ldconfig
7) sudo gedit /etc/bash.bashrc

Add these two lines at the end of the file and save it:
	PKG_CONFIG_PATH=$PKG_CONFIG_PATH:/usr/local/lib/pkgconfig
	export PKG_CONFIG_PATH

For simplicity you can install OpenCV 2.1 from ubuntu:

OpenCV 2.1 Instalation (from Ubuntu)
sudo apt-get install libcv-dev
sudo apt-get install libhighgui-dev

But somethimes this does not work properly, so resot to this as the last option.





