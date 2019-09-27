# Makefile

# *****************************************************
# Variables to control Makefile operation

CXX = g++
CXXFLAGS = -Wall -g -O3

# ****************************************************
# Targets needed to bring the executable up to date

main: main.o opticalConductivity.o GaussLegendreIntegration.o opticalPolarization.o  
	$(CXX) $(CXXFLAGS) -o generateCondData main.o opticalConductivity.o

# The main.o target can be written more simply

main.o: main.cpp 
	$(CXX) $(CXXFLAGS) -c main.cpp

opticalConductivity.o: opticalConductivity.hpp
GaussLegendreIntegration.o: GaussLegendreIntegration.hpp
opticalPolarization.o: opticalPolarization.hpp


