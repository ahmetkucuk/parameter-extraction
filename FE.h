/* 
 *  SDO Feature Extraction Code: C++ Version
 *
 *  Parameters:  <file in> <directory out> <cells> <thumbnnail size>
 *  NOTE: Cells is indicated by side, ie. 8 will indicate an 8 x 8 grid
 *  Example: ./sdoFE AIA20110212_000008_4500.fits output/ 8 256
 *
 *  Montana State University
 *  Data Mining Lab - http://dmlab.cs.montana.edu
 *  Juan M. Banda http://www.jmbanda.com
 *  Michael A. Schuh 
 * 
 */ 

#ifndef _FE_H_
#define _FE_H_

#include <string.h>
#include <stdio.h>
#include <math.h>
#include <algorithm>
#include "fitsio.h"
#include <iostream>
#include <fstream>
#include <vector>
#include <gsl/gsl_histogram.h>
#include <gsl/gsl_matrix.h>
#include <gsl/gsl_vector.h>
#include <gsl/gsl_linalg.h>
#include <gsl/gsl_math.h>
#include "armadillo"
#include <opencv/cv.h>
#include <opencv/highgui.h>

using namespace arma;
using namespace std;

using std::vector;
using std::string;

// General Use Functions

int dumpHead(fitsfile *fptr, string fileOut);
void getHeaderKeywordValues(fitsfile *fptr, vector<string> &keywords, vector<string> &values);
void getPixels(fitsfile *fptr, long *data, int totalPixels, int nullVal);
int runFE(string fileIn, string dirOut, int segSplits, int thSize);
mat normalizeFull(long* pixels, int wave, int intDMin, int intDMax, int totalPixels, int cols, int rows, double expTime);
uchar& pixel(IplImage* canvas, int row, int col, int channel);
int writeRSimage(mat pixels, int orHeight, int orWidth, int rsHeight, int rsWidth, string fileOut);
mat bytscl(mat toScaling);

// Parameters - and helper functions of them
double kurtosisOFV(vec chunkData); 
double skewnessOFV(vec chunkData);
double fractalDimension(mat tamA, int chunkX, int chunkY);
double tamuraContrastV (vec chunkData);
double tamuraDirectionality (mat A, int chunkY, int chunkX);
mat genHistGSL (mat A, int nBins, int out);

// OBSOLETE
//mat genHistogram (mat A, int nBins); //Obsolete but still left in the code
//mat scalingBR(mat toScale, double rangeMax, double rangeMin); //Obsolete but still left in the code
#endif
