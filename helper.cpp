//comment


#include "helper.h"

#define DEBUG 1

using namespace std;

/********************************************************
 * Trims leading and trailing whitespace characters,
 *   including space, tab, and new lines
 * 
 * Input: string
 * Output: trimmed string
 * 
 */
inline std::string trimStr(std::string& str)
{
  str.erase(0, str.find_first_not_of(" \t\r\n"));  
  str.erase(str.find_last_not_of(" \t\r\n")+1);         
  
  return str;
}

const char* stringToUpper(const char* cStr)
{
  string str = cStr;
  int i=0;
  char c;
  while (cStr[i])
  {
    c=cStr[i];
    str[i] = (toupper(c));
    i++;
  }
  return str.c_str();
}

std::string trimToName(std::string str)
{
  str.erase(0, str.find_last_of("/\\")+1);  
  str.erase(str.find_last_of("."));         
  
  return str;
}

string convertInt(int number) {
	stringstream ss;
	ss << number;
	return ss.str();
}


void getFiles(vector<string> &files, string dir)
{

    struct dirent *de=NULL;
    DIR *d=NULL;
    char file[256];

    string ftsDir = dir;
    string fname;

    d=opendir(ftsDir.c_str());
    if(d == NULL)
    {
        perror("Couldn't open directory");
        return;
    }

    // Loop while not NULL
    while(de = readdir(d))
    {

//        printf("%s\n",de->d_name);

        if(
           (strcmp(de->d_name, ".") == 0) ||
           (strcmp(de->d_name, "..") == 0) )
        {

            if(DEBUG)
            {
                cout << "skipping: " << de->d_name << endl;
            }
            continue;
        }

     //   cout << de->d_name << endl;
        files.push_back(de->d_name);

    }


    if(DEBUG)
    {
        cout << "Total files: " << files.size() << endl;
    }

}



void sortFiles(vector<string> &files)
{
  
  sort(files.begin(), files.end());
  
}

void test()
{
    cout << "Helper DEBUG:" << DEBUG << endl;
}
