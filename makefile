
sdoFE : main.o helper.o FE.o
	g++ -ggdb -fopenmp `pkg-config --cflags opencv` -debug -O1 -Wall main.o helper.o FE.o FileReader.o -o sdoFE -L/usr/lib64/ -I/usr/local/include -lm -lcfitsio -larmadillo -lgsl `pkg-config --libs opencv`
main.o: main.cpp helper.h FE.h
	g++ -c -fopenmp main.cpp FileReader.cpp

helper.o: helper.cpp helper.h
	g++ -c helper.cpp 

FE.o: FE.cpp
	g++ -c FE.cpp

clean:
	\rm *.o *~ sdoFE

tar:
	tar cfv sdoFE.tar main.cpp helper.cpp helper.h FE.cpp FE.h makefile README.txt



