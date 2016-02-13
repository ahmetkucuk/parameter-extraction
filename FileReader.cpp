#include <string>
#include <iostream>
#include <fstream>

using namespace std;

class FileReader;

class FileReader {
	std::ifstream readStream;
	public:
		inline std::string next();
		inline FileReader(std::string filePath);
};

FileReader::FileReader (std::string filePath) {

	readStream.open(filePath.c_str());
}

std::string FileReader::next () {
	std::string line;
	std::getline(readStream, line);
  	return line;
}