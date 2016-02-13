//comment line

#ifndef _HELPER_H_
#define _HELPER_H_


#include <iostream>
#include <fstream>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <vector>
#include <time.h>
#include <algorithm>
#include <sstream>
#include <sys/time.h>


#include <sys/types.h>
#include <dirent.h>

using std::string;
using std::vector;

inline string trimStr(string& str);
const char* stringToUpper(const char* cStr);
string trimToName(string str);

string convertInt(int number);

void getFiles(vector<string>& files, string dir);
void sortFiles(vector<string> &files);

void test();

#endif
