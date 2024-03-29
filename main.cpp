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
#include <stdio.h>
#include "stdlib.h"
#include "string.h"
#include <iostream>
#include <fstream>
#include <sstream>
#include <ctime>
#include <omp.h>
#include <stdlib.h>


#include "math.h"
#include "helper.h"
#include "FE.h"
#include "FileReader.cpp"


using namespace std;
//////////////////////////////////////////////////////////////////////
// Command-line arguments need to match otw, exit

int main(int argc, char *argv[])
{

    time_t begin = time(0);

    if(argc < 4)
    {
        cout << "Invalid number of arguments" << endl;
        return 1;
    }
   
    string filein = argv[1];
    string dirOut = argv[2];
    int segSplits = atoi(argv[3]);
    int thSiz =atoi(argv[4]);
    int threadCount =atoi(argv[5]);
    int offset = atoi(argv[6]);
    int limit = atoi(argv[7]);
    int LOCAL = atoi(argv[8]);
    //int statFE = runFE(filein, dirOut, segSplits, thSiz);
    //if (statFE ==1) {
	//	cout << "Feature Extraction Failed" << endl;
	//	return 1;
	//} else {
	//	return 0;
	//}
//

    int counter = 0;
    FileReader reader(filein);
    string line = reader.next();
    while(counter < offset) {
        counter++;
        line = reader.next();
    }

    #pragma omp parallel
    {
        #pragma omp single nowait
        {
            while(!line.empty() && counter < offset + limit) {
                
                #pragma omp task firstprivate(line)
                {

                    string finalOutputDir = dirOut;
                    string finalFileName = line;
                    
                    if(LOCAL == 0) {
                        std::size_t found = line.find_last_of("/\\");
                        finalOutputDir = dirOut + line.substr(0,found) + "/";
                        system( ("mkdir -p " + finalOutputDir).c_str() );
                        finalFileName = "/data4/STORE/" + line;
                    }

                    int statFE = runFE(finalFileName, finalOutputDir, segSplits, thSiz);
                    if (statFE == 1) {
                        cout << "Feature Extraction Failed " + finalFileName << endl;
                    }
                }
                
                counter++;
                line = reader.next();
                #pragma omp flush(counter)
                {  
			if(counter % 1000 == 0) {
                    		cout << counter << endl;   
			}
                }
            }
        }

        #pragma omp barrier
        {
            #pragma omp single
            {
                time_t end = time(0);
                double elapsed_secs = difftime(end, begin);
                cout << "Total Time Elapsed: " << elapsed_secs << endl;
            }
        }
        
    }
    return 0;
}
