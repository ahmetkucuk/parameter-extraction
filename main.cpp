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
#include <thread>
#include <ctime>
#include <stdlib.h>


#include <ctime>
#include <vector>
#include "math.h"

#include "helper.h"
#include "FE.h"
#include "FileReader.cpp"

#define LOCAL 1

using namespace std;
//////////////////////////////////////////////////////////////////////
// Command-line arguments need to match otw, exit


std::vector<std::thread> Pool;

void waitUntilThreadsFinsihed() {

    for (std::vector<thread>::iterator it = Pool.begin() ; it != Pool.end(); ++it) {
        (*it).join();
    }
    Pool.clear();
    
//
    //for(int i = 0; i < Pool.size(); i++) {
    //    Pool.at(i).join();
    //    Pool.at(i).
    //    Pool.erase(Pool.begin() + i);
    //}
}

int main(int argc, char *argv[])
{

    clock_t begin = clock();

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
    //int statFE = runFE(filein, dirOut, segSplits, thSiz);
    //if (statFE ==1) {
	//	cout << "Feature Extraction Failed" << endl;
	//	return 1;
	//} else {
	//	return 0;
	//}
//
    FileReader reader(filein);
    std::string line;
    int counter = 0;
    while(true) {

        line = reader.next();
        if(line.empty()) {
            break;
        }

        if(counter < offset) {
            continue;
        }

        if(counter >= offset + limit) {
            break;
        }

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
            cout << "Feature Extraction Failed" << endl;
        }
        
        /*
        Pool.push_back(std::thread(runFE, line, dirOut, segSplits, thSiz));
        if(Pool.size() >= threadCount) {
            waitUntilThreadsFinsihed();
        }
        cout << line << endl;
        */
        counter++;
    }
    //waitUntilThreadsFinsihed();

    clock_t end = clock();
    double elapsed_secs = double(end - begin) / CLOCKS_PER_SEC;
    cout << "Total Time Elapsed: " << elapsed_secs << endl;
    
}
