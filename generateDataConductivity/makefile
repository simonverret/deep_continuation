# Makefile

# *****************************************************
# Variables to control Makefile operation

CXX = g++
CXXFLAGS = -Wall -g -O3 -std=c++11

# ****************************************************
# Targets needed to bring the executable up to date

main: main.o OpticalConductivity.o  
	$(CXX) $(CXXFLAGS) -o generateDataCond main.o OpticalConductivity.o

# The main.o target can be written more simply

main.o: main.cpp 
	$(CXX) $(CXXFLAGS) -c main.cpp

OpticalConductivity.o: OpticalConductivity.hpp


