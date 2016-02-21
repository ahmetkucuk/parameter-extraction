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

#include "FE.h"
#include "helper.h"


#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <iostream>
#include <fstream>

using namespace cv;

#define DEBUG 1

// 
// name: getPixels - Extract pixel values from FITS file, no preprocessing
// @param
// @return
// 

void getPixels(fitsfile *fptr, long *data, int totalPixels, int nullVal)
{
    int status = 0;
    long fpixel[2] = {1,1};
    int anyNull = 0; //indicates if null pixels were found
    fits_read_pix(fptr, TLONG, fpixel, totalPixels, 
                &nullVal, data, &anyNull, &status);
}


void rotate_90n(cv::Mat &src, cv::Mat &dst, int angle)
{
    dst.create(src.rows, src.cols, src.type());
    if(angle == 270 || angle == -90){
        // Rotate clockwise 270 degrees
        cv::flip(src.t(), dst, 0);
    }else if(angle == 180 || angle == -180){
        // Rotate clockwise 180 degrees
        cv::flip(src, dst, -1);
    }else if(angle == 90 || angle == -270){
        // Rotate clockwise 90 degrees
        cv::flip(src.t(), dst, 1);
    }else if(angle == 360 || angle == 0){
        if(src.data != dst.data){
            src.copyTo(dst);
        }
    }
}

void cvMatToArmaMat(cv::Mat &imaget, arma::mat &data) {
   	
   	cv::Mat image;
    rotate_90n(imaget, image, 90);

	int rows = image.rows;
	int cols = image.cols;

	unsigned char *input = (unsigned char*)(image.data);
	
	int r,g,b;
	for(int i = 0;i < cols;i++){
		for(int j = 0;j < rows;j++){
			b = input[cols * j + i ];
			//g = input[cols * j + i + 1];
			//r = input[cols * j + i + 2];
			double gray = b;//(0.2989 * r + 0.5870 * g + 0.1140 * b);

			int index = i*cols + j;
	        int tmp = (index / cols);
	        tmp = ((rows - tmp - 1)*cols) + (index % cols);
			data(tmp) = gray;
		}
	}
	image.release();
	imaget.release();
	//delete [] input;

}

/*
*
*
*/
cv::Mat readJP2Image(string pathToImage) {
	
	
	cv::Mat image;
    image = imread(pathToImage, 0);   // Read the file

    if(! image.data )                           // Check for invalid input
    {
        cout <<  "Could not open or find the image" << std::endl ;
    }
    return image;
}

void writePixelToFile(arma::mat &data, int totalPixels) {
	ofstream myfile;
	myfile.open ("data2.txt");
	int max = -10;
	for(int i = 0; i < totalPixels; i++) {
	  myfile << data(i) << "\n";
	  if(data(i) > max) {
	  	max = data(i);
	  }
	}
	cout << max << endl;
	myfile.close();
}

int runFE(string fileIn, string dirOut, int segSplits, int thSize)
{
	string fileOut = dirOut + trimToName(fileIn);
	

	cv::Mat image = readJP2Image(fileIn);
	//cv::Mat image;
	//resize(imaget, image, Size(), 0.25, 0.25);
	int rows = image.rows;
	int cols = image.cols;

	arma:mat data2(rows*cols, 1);
    cvMatToArmaMat(image, data2);
    image.release();


	//writeRSimage(data2,rows,cols,thSize,thSize,fileOut+"_th.png");

	int chunkY = rows/segSplits;
	int chunkX = cols/segSplits;
	int shiftP = rows - chunkX; //How Many to move forward
	int nxtRow =0;
	int countY=1;
	int totalP=0;
	int imgC=0;
	int newChunk;
	char numstr[21];
	//Prepare textfile for saving of parameters
	FILE * pFile;
	string textout;	
	textout = fileOut + ".txt";
	char *fileName = (char*)textout.c_str();
	//cout << " File name: " << fileName << endl;
	pFile = fopen(fileName,"w");
	
	//Chunks start here
	for (int cnkY=1; cnkY <= segSplits; cnkY++) {
		totalP=nxtRow;
		int countX=1;
	for (int cnkX=1; cnkX <= segSplits; cnkX++) {
		//NEW long array with the 'chunk' pixels
		long* dataChunk = (long *) malloc((chunkY*chunkX) * sizeof(long));
		double* dataChunkDBL = (double *) malloc((chunkY*chunkX) * sizeof(double));

		//cout << newChunk << endl;
		mat tamA(chunkY,chunkX);
		vec vecA(chunkY*chunkX,1);
		newChunk=0;
		for (int cJ=0; cJ<chunkY; cJ++){
			for (int rJ=0; rJ<chunkX; rJ++){
				// Represent in different variables - need to clean this up
				tamA(cJ,rJ) = data2(totalP);
				dataChunk[newChunk]=data2(totalP);
				vecA(newChunk)=data2(totalP);
				totalP++;
				newChunk++; //DO NOT REMOVE
			}
			totalP=totalP+shiftP; //One was added after exiting the loop
		}
		imgC++;
		sprintf(numstr, "%d", imgC);
		//
		//FEATURE EXTRACTION SECTION
		//
		//Make a histogram of the chunk of size chunkX*chunkY
		vec histogrmV(256,1);
		histogrmV.zeros();
		for (int hC=0; hC<newChunk; hC++){
			histogrmV(dataChunk[hC])++;			
		}
		histogrmV = histogrmV / newChunk; //Normalize to sum to one and we are good for the last parameters except tamuras

		// Parameter 1 - Entropy - from Vector calculations
		// Formula: -sum(p.*log2(p))
		double eps = 0.00000000000000000000000001; // Avoid Log of empty bins
		vec tmp= log2(histogrmV+eps);   // We need to add a very very small quantity here otw we get log2(0) for empty bins
		vec tmp2 = histogrmV % tmp;	// Element-wise multiplication (again put separatedly since this is faster
		double entropy = (-1) * sum( tmp2 );

		// Parameter 2 - Mean - From Vector Calculation
		double meanB = mean(vecA);

		// Parameter 3 - Standard Deviation - From Vector Calculation
		double stdev = stddev(vecA);

		// Parameter 4 - Skewness
		double skewness	= skewnessOFV(histogrmV);


		// Parameter 5 - Kurtosis		
		double kurtosis = kurtosisOFV(histogrmV);


	    // Parameter 6 - Fractal Dimension
		double fractal_dim = fabs(fractalDimension(tamA, chunkX, chunkY));


				
		// Parameter 7 - Uniformity - from Vector calculations
		// Taking Normalized to 1 histogram as in Matlab code
		vec tmphistogrmV =pow(histogrmV,2) ;  //Faster doing this temp vector than putting inside the sum and changing to two sums
		double uniformity = sum(tmphistogrmV);


		// Parameter 8 - Relative Smoothness - from Vector Calculations
		// Formula:  1-(1/(1+(std(p)^2))
		double stdevNV = stddev(histogrmV); //Calculate the STD of the normalized to 1 histogram
		double rs = 1-(1 / (1 + pow(stdevNV,2)));


		//Parameter 9 - Tamura Directionality
		double tDIRECTIONALITY = tamuraDirectionality(tamA, chunkY, chunkX);
		if (isinf(tDIRECTIONALITY)) {
			tDIRECTIONALITY = 36.0437; //Upper bound otw function returns infinites
		}
		//Parameter 10 - Tamura Contrast 
		double tContrast = tamuraContrastV(vecA);


		int outS = 0; //Secreen output -debug mode
		
		if (outS !=0) {
			cout << "Segment (row, col): " << countY << "," << countX << endl;
			cout << "Image parameter - 1 - Entropy: " << entropy << endl;
			cout << "Image parameter - 2 - Mean: " << meanB << endl;
			cout << "Image parameter - 3 - StandardDeviation: " << stdev << endl;
			cout << "Image parameter - 4 - Fractal Dimension: " << fractal_dim << endl;
			cout << "Image parameter - 5 - Skewness: " << skewness << endl;
			cout << "Image parameter - 6 - Kurtosis: " << kurtosis << endl;
			cout << "Image parameter - 7 - uniformity: " << uniformity << endl;
			cout << "Image parameter - 8 - Relative Smoothness: " << rs << endl;
			cout << "Image parameter - 9 - Tamura Directionality: " << tDIRECTIONALITY << endl;
			cout << "Image parameter - 10 - Tamura Contrast: " << tContrast << endl;
		}
		//Write to text file
		fprintf(pFile, "%d\t%d\t%.4f\t%.4f\t%.4f\t%.4f\t%.4f\t%.4f\t%.4f\t%.4f\t%.4f\t%.4f\n",countY,countX,entropy,meanB,stdev,fractal_dim,skewness,kurtosis,uniformity,rs,tDIRECTIONALITY,tContrast);
		//fprintf(pFile, "%d\t%d\t%.4f\t%.4f\t%.4f\t%.4f\t%.4f\t%.4f\t%.4f\t%.4f\t%.4f\t%.4f\n",countY,countX,entropy,meanB,stdev,skewness,kurtosis,uniformity,rs,tDIRECTIONALITY,tContrast);
		
		// Free malloc memory
		free (dataChunk);
		free (dataChunkDBL);		
		//Move to next segment
		totalP=nxtRow + (cnkX*chunkX);  // Move to next row stuff ---- Move before the end of the loop
		countX++;
    	}
	nxtRow = cnkY * chunkX * segSplits * chunkY; //Move to next row of segments
	countY++;
	}

	fclose(pFile);

    return 0;

}

// 
// name: kurtosisOFV - Calculate kurtosis of desired chunk of data
// @param chunkData - vector of elements to operate on
// @return: double value containing the kurtosis 
// 
double kurtosisOFV(vec chunkData) {
	double gamma=0;
	double sigma=0;
	double chunkMean= mean(chunkData);
	vec tmp =pow( (chunkData - chunkMean),4);
	vec tmp2 = pow( (chunkData - chunkMean),2);
	gamma = sum(tmp) / chunkData.n_elem ;
	sigma = sum( tmp2 ) /chunkData.n_elem;
	return gamma/pow(sigma,2);
}    
// 
// name: skewnessOFV - Calculate skewness of desired chunk of data
// @param chunkData - vector of elements to operate on
// @return: double value containing the skewness 
// 
double skewnessOFV(vec chunkData) {
	double gamma=0;
	double sigma=0;
	double chunkMean= mean(chunkData);
	vec tmp =pow( (chunkData - chunkMean),3);
	vec tmp2 = pow( (chunkData - chunkMean),2);
	gamma = sum(tmp) / chunkData.n_elem ;
	sigma = sqrt( sum( tmp2 ) /chunkData.n_elem );
	return gamma/pow(sigma,3);
}   
// 
// name: tamuraContrastV - Calculate the Tamura contrast of desired chunk of data
// @param chunkData - vector of elements to operate on
// @return: double value containing the tamura contrast
// 
double tamuraContrastV (vec chunkData) {
    	double tCont=0;
	double range = 255; //Max pixel value here for normalization by range
	chunkData = chunkData / range;
	double stdevTC=stddev(chunkData);
	if (fabs(stdevTC)<0.00000000001) {
		tCont=0;
	} else {
		double ktTC = kurtosisOFV(chunkData);
		double alfa = ktTC / (pow(stdevTC,4));
		tCont = stdevTC / pow(alfa,0.25);
	}
return tCont;
}
// 
// name: pixel - direct pixel access to openCV's iplImage object
// @param 
// @return
// 

uchar& pixel(IplImage* canvas, int row, int col, int channel) {
	return ((uchar*)(canvas->imageData + canvas->widthStep*row))[col*canvas->nChannels+channel];
}
// 
// name: writeRSimage - Write the resized thumbnail of the input FITS file after normalization
// @param mat pixels - all values, orginal height, original width, target height, target width, filename
// @return
// 

int writeRSimage(mat pixels, int orHeight, int orWidth, int rsHeight, int rsWidth, string fileOut) {
	//Create the original size image first
	CvSize size;
	size.height = orHeight;
	size.width = orWidth;
	IplImage* ipl_image_p = cvCreateImage(size,IPL_DEPTH_8U,1);
	int totalP=0;
	for (int cJ=0; cJ<orHeight; cJ++){
		for (int rJ=0; rJ<orWidth; rJ++){
			pixel(ipl_image_p,cJ,rJ,0) = pixels(totalP);
			totalP++;
		}
	}
	// Now we do another image with the resizing of the original one
	CvSize size2;
	size2.height = rsHeight;
	size2.width = rsWidth;
	IplImage* ipl_image_p2 = cvCreateImage(size2,IPL_DEPTH_8U,1);
	cvResize(ipl_image_p,ipl_image_p2,CV_INTER_LINEAR);
	char *fileNameSN;
	fileNameSN = (char*)fileOut.c_str();
	if (!cvSaveImage(fileNameSN,ipl_image_p2)) {
		return 1;
	} else {
		//Clean up memory by releasing the images
		cvReleaseImage(&ipl_image_p2);
		cvReleaseImage(&ipl_image_p);
		return 0;
	}
		
}



// 
// name: genHistGSL - Generate histogram using the GSL GNU library (quite fast)
// @param mat A - chunk to generate histogram for, int nBins - Number of bins, int out - output enabled
// @return
// 

mat genHistGSL (mat A, int nBins, int out) {
	gsl_histogram * h = gsl_histogram_alloc(nBins);
	gsl_histogram_set_ranges_uniform (h, min(min(A)), max(max(A))+0.00001);   // We need the 0.00001 to match correctly the last bin
	for (int kf=0; kf<A.n_elem; kf++) {
		gsl_histogram_increment(h, A(kf));
	}
	// Basic Error Checking
	if (out ==1) {
		gsl_histogram_fprintf (stdout, h, "%g" , "%g");
	}
	colvec xC(nBins);
	xC.zeros();
	for (int loopH=0; loopH<nBins; loopH++) {
		xC(loopH) = gsl_histogram_get(h, loopH); 
	}
	gsl_histogram_free (h);
	return xC;
}

// 
// name: fractalDimension - C++ interpretation of the Fractal dimension found here:
// @param
// @return
// 

double fractalDimension (mat tamA, int chunkX, int chunkY) {
	double fractal_dim=0;
	int steps=3;
	//cout << chunkX << "," << chunkY << endl;
	//Construct matrix
	mat xM(chunkX,1); 
	xM.zeros();
	xM = tamA.col(0);
	mat yM(chunkY,1);
	yM.zeros(); 
	yM = tamA.col(1);
	//cout << tamA <<endl;
	double glob_llx = min(min(xM,0));
	double glob_lly = min(min(yM,0));
	double glob_urx = max(max(xM,0));
	double glob_ury = max(max(yM,0));
	//cout << glob_llx << "," << glob_lly << ":" << glob_urx <<","<< glob_ury << endl;
	double glob_width = glob_urx - glob_llx;
	double glob_height = glob_ury - glob_lly;
	// find min and max of x and y of the segments
	mat xA(steps+1,1);
	mat yA(steps+1,1);
	xA.zeros();	
	yA.zeros();
	for (int stepF=0; stepF<=steps; stepF++) {
		int n_boxes = 0;
		double n_sds=pow(2,stepF);
		//cout << glob_width << endl;
		int loc_width = floor((glob_width / n_sds) + 0.5);     // Isues happen here since matlab does something weird 
		//cout << "loc_width:" << loc_width << endl;
		//cout << glob_height << endl;
		int loc_height = floor((glob_height / n_sds) + 0.5);   // Isues happen here since matlab does something weird
		//cout << "loc_height:" << loc_width << endl;
		for (int sd_x = 1; sd_x<=n_sds; sd_x++){
			double loc_llx = glob_llx + ((sd_x - 1) * loc_width);
			double loc_urx = glob_llx + (sd_x * loc_width);
			int fy_c=0;
				uvec fxdx = find(xM >= loc_llx);
				if (fxdx.n_elem != 0 ) {  //Empty vectors cause this to crash, so we need some ifs
				  //cout << "a: " << fxdx.n_elem << endl;
				  mat found0 = xM.elem((find(xM >= loc_llx)));
  				  uvec fxd1x = find( found0 < loc_urx);
					if (fxd1x.n_elem !=0 ) { //Empty vectors cause this to crash, so we need some ifs
						//cout << "b: "<< fxd1x.n_elem << endl;
						fy_c=fxd1x.n_rows; //Need to know how many we have
					}
				}
			
			int fy_c2=0;
			mat found_y(fy_c,1);
			found_y.zeros();
			for (int fnX=0; fnX<chunkX; fnX++){
				if ((xM(fnX) >= loc_llx) && (xM(fnX)< loc_urx)) {
					found_y(fy_c2)=yM(fnX);
					fy_c2++; //Determine how many elements for found_y
				}
			}
			for (int sd_y =1; sd_y<=n_sds; sd_y++){
				double loc_lly = glob_lly + ((sd_y - 1) * loc_height);
				double loc_ury = glob_lly + (sd_y*loc_height);
				uvec fxd = find(found_y >= loc_lly);
				if (fxd.n_elem != 0 ) {  //Empty vectors cause this to crash, so we need some ifs
				  mat found0 = found_y.elem(find(found_y >= loc_lly));
				  if (found0.n_elem != 0) { //Empty vectors cause this to crash, so we need some ifs
					uvec fxd1 = find( found0 < loc_ury);
					if (fxd1.n_elem !=0 ) { //Empty vectors cause this to crash, so we need some ifs
						n_boxes++;  // Found one element inside, we augment the box count
					}
				  }
				}
			}
		}
			//cout << "boxes: " << n_boxes <<endl;
			xA(stepF,0)=stepF*log(2);
			if (n_boxes==0) {
				yA(stepF,0)=0;
			} else {
				yA(stepF,0)=log(n_boxes);
			}
	} //END LOOP of steps
	mat AF(steps+1,2);

	//Original code using Armadillo/LAPACK - Working in Ubuntu
	//qr(fdQ,fdR,AF);//QR decomposition of AF
	//mat fdc = trans(fdQ)*yA;
	//vec fdparam = solve(fdR,fdc);
	//fractal_dim = fdparam(0);
	//Modified code to only use GSL linear algebra solvers

	gsl_matrix *sampleAF, *samplefdQ, *samplefdR;
	gsl_vector *sample_tau,*sample_tau2;
	sampleAF = gsl_matrix_alloc(steps+1,2); //4x2
	samplefdQ = gsl_matrix_alloc(steps+1,steps+1); //4x4
	samplefdR = gsl_matrix_alloc(steps+1,2); //4x2
	for (int gA=0; gA<(steps+1); gA++) {
		AF(gA,0)=xA(gA,0);
		AF(gA,1)=1;
	}

	for (int gslx1=0; gslx1<(steps+1); gslx1++) {
		for (int gslx2=0; gslx2<2; gslx2++) {
			gsl_matrix_set(sampleAF, gslx1, gslx2, AF(gslx1, gslx2));
		}
	}
	sample_tau= gsl_vector_alloc(2); //Smallest dimension of the 4x2 matrix
	gsl_linalg_QR_decomp(sampleAF, sample_tau);
	gsl_linalg_QR_unpack(sampleAF, sample_tau, samplefdQ, samplefdR);
	//cout << xA << endl;
	//cout << AF << endl;
	//cout << yA << endl;
	mat fdQ(steps+1,steps+1);
	mat fdR(steps+1,2);
	for (int gxh1=0; gxh1<(steps+1); gxh1++) {
		for (int gxh2=0; gxh2<2; gxh2++) {
			fdR(gxh1,gxh2)=gsl_matrix_get(samplefdR,gxh1,gxh2);
		}
	}
	//cout << fdQ << endl;
	//cout << fdR << endl;
	int errC=0;
	gsl_matrix_transpose(samplefdQ);
	for (int gsh1=0; gsh1<(steps+1); gsh1++) {
		for (int gsh2=0; gsh2<(steps+1); gsh2++) {
			fdQ(gsh1,gsh2)=gsl_matrix_get(samplefdQ,gsh1,gsh2);
		}
	}
	mat fdc = fdQ*yA;
	//cout << fdc << endl;
	//Need to get QR decom of fdR
	sample_tau2= gsl_vector_alloc(2); //Smallest dimension of the 4x2 matrix
	gsl_linalg_QR_decomp(samplefdR, sample_tau2);
	gsl_vector *samplefdc, *samplefdparam, *sampleRem;
	samplefdc= gsl_vector_alloc(4);
	samplefdparam =gsl_vector_alloc(2);
	sampleRem =gsl_vector_alloc(4);
	//Put values onto fdc
	for (int gshv=0; gshv < 4; gshv++) {
		gsl_vector_set(samplefdc,gshv,fdc(gshv));
	}
	//Least Squares Solutions
	gsl_linalg_QR_lssolve(samplefdR, sample_tau2, samplefdc, samplefdparam, sampleRem);
	fractal_dim=gsl_vector_get(samplefdparam,0);

	if (isnan(fractal_dim)){
		fractal_dim=1.646;
	} else {
		if (isinf(fractal_dim)) {
			if (fractal_dim <0) { 
				fractal_dim = 0;
			} else {
				fractal_dim = 1.646;
			}
		} else {
			if (fractal_dim < 0) { 
				fractal_dim = 0;
			} else {
				fractal_dim = fractal_dim;
			}
		}
	}
	return fractal_dim;
}
// 
// name: tamuraDirectionality - Calculate the Tamura Directionality. C++ interpretation of the Matlab function found here:
// @param
// @return
// 

double tamuraDirectionality (mat A, int chunkY, int chunkX) {
	//Normalize by range (same as in matlab)
	A = A/ 255;
	int ndim = 2;  // Will always be 2 dimensional (x and y)
	colvec perm(ndim);
	perm(0) = 2;  //Cyclic permutations for 2 dimensions
	perm(1) = 1;
	mat gy;
	mat gx;
	mat f = A;
	int n = f.n_rows;
	int p = f.n_cols;
	for (int k = 1; k<= ndim; k++) {
		rowvec h(n);
		for (int tc = 0; tc<n; tc++) {  //Fill a vector with 1 to n 
			h(tc) = tc+1;			
		}
		mat g(n,p);
		g.zeros(); //Initialize all to zeros
		if (n > 1) {
			//Note: ALL elements are shifted one since C++ vector/elements start at 0 not 1 like in matlab
			g.row(0)=f.row(1)-f.row(0) / ( h(1) - h(0) );
			g.row(n-1)=f.row(n-1)- f.row(n-2) / (h(n-1) - h(n-2));
		}
		if (n > 2) {
			//Note: ALL elements are shifted one since C++ vector/elements start at 0 not 1 like in matlab
			rowvec h2 = h.subvec(2,n-1) - h.subvec(0,n-3);
			mat tmprm(h2.n_elem,n);   //Equals to th h(:,ones(p,1))  part of the matlab code
			mat tmpdiv = f.submat(span(2,n-1),span::all) - f.submat(span(0,n-3), span::all );
			tmprm.fill(perm(1-1));
			g.submat(span(1,n-2),span::all)= ( tmpdiv / tmprm );
		}
		// ALL OTHER PERMUTATION operations do not do anything for 2D
		if (k==1) { //We got out gy
			gy = g;
		} else { // k = 2 so we get gx now
			gx = trans(g);
		}
		// Need to permute f for next loop and getting gx
		f= trans(f);
	}
	gx.reshape(gx.n_elem, 1);
	gy.reshape(gy.n_elem, 1);
	colvec tempR(gx.n_elem);
	colvec temT(gx.n_elem);
	//Calculate the atan2 and hypot of all elements to represent in Polar coordinates
	for (int ko = 0; ko < gx.n_elem; ko++) {
		temT(ko) = atan2(gy(ko),gx(ko));
		tempR(ko) = hypot(gx(ko),gy(ko));
	}

	temT.reshape(n,p);
	int nbins=125;
	double tmpRv = (15 * max(tempR)) * 0.01;   //The .15*max(r(:) means exactly this
	double tmpFto =  0.0001;
	colvec t0 = temT;
	for (int yu=0 ; yu< tempR.n_elem; yu++){
		if (tempR(yu) < tmpRv) {
			tempR(yu)=0;
		}
		if (fabs(tempR(yu)) < tmpFto) {    //Populate tO
			t0(yu)=0;
		}
	}
	tempR.reshape(n,p);	
	t0.reshape(n,p);	
	tempR = trans(tempR);
	t0 = trans(t0);
	t0.reshape(A.n_elem,1,0);
	tempR.reshape(A.n_elem,1,0);
	mat Hd = genHistGSL (t0, nbins,0);
	mat nrth = pow(tempR,2) + pow(t0,2);
	mat nrm = genHistGSL (nrth, nbins,0);
	int fmx=0;
	//Can optimize this one
	for (int nyh=0; nyh < Hd.n_elem; nyh++) {
		if (Hd(nyh) == max(max(Hd))) {
			fmx = nyh;
			nyh= Hd.n_elem;
		}
	}
	//uvec fmx = find(Hd==max(max(Hd)));  -- ISUES to get back from uvec to actual integer value
	colvec ff(Hd.n_elem);
	// Can optimize this one
	for (int jm=0; jm<Hd.n_elem; jm++) {  //Another vector that has from 1 to Hd.n_elem
		ff(jm)=jm+1;
	}
	mat ff2 = pow((ff- (fmx+1)),2);
	mat FdirTop = sum(Hd % ff2);
	vec FdirBot = sum(nrm);
	vec Fdir1 = FdirTop/ FdirBot ;
	vec Fdir = log(Fdir1);  //Where INF might occur
	double FFdir =Fdir(0);
	FFdir = fabs(FFdir);
	return FFdir;
}
