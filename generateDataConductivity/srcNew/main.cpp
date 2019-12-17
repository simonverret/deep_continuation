//
//  main.cpp
//  generateCondData
//
//  Created by Reza Nourafkan on 2019-09-19.
//  Copyright © 2019 Reza Nourafkan. All rights reserved.
//

#include <iostream>
#include <fstream>
#include <string>
#include <map>
#include <random>
#include <chrono>
#include "opticalConductivity.hpp"
#include "opticalPolarization.hpp"
#include "GaussLegendreIntegration.hpp"

int main() {
    //
    auto start = std::chrono::high_resolution_clock::now();
    // read input
    std::ifstream inFile;
    inFile.open("Input.txt", std::ios::in);
    if (inFile.fail())
    {
        std::cout << "Unable to open Input.txt file!" << std::endl;
        return 1;
    }
    std::string key;
    double  value;
    std::map<std::string, double> input;
    while (inFile >> key && inFile >> value){
        std::cout << key << ": " << value << std::endl;
        input[key] = value;
    }
    inFile.close();
    // Initiate Freq. meshes on real and imaginary axis
    opticalConductivity::setFrequencyMesh((int)input["numIntervalOmega"], input["bandWidth"]);
    opticalPolarization::setBosonicMatsubaraFrequencyMesh(input["beta"], (int) input["maxBosonicMatsubaraIndex"]);
    //
    std::vector<opticalConductivity> sigma;
    std::vector<opticalPolarization> Pi;
    //
    std::random_device rd;  //Will be used to obtain a seed for the random number engine
    std::mt19937 gen(rd()); //Standard mersenne_twister_engine seeded with rd()
    std::uniform_int_distribution<> dis(0, (int)input["maxNumLowFreqPeaks"]);
    std::random_device rd1;
    std::mt19937 gen1(rd1());
    std::uniform_int_distribution<> dis1(3, (int)input["maxNumMidInfraredPeaks"]);
    std::random_device rd2;
    std::mt19937 gen2(rd2());
    std::uniform_real_distribution<double> dis2(0., 1.0);
    for (unsigned int i=0; i<(int)input["numData"]; ++i)
    {
        unsigned int numLowFreqPeaks = dis(gen);
        std::vector<double> plasmaFreqs;
        std::vector<double> scatteringRates;
        for (unsigned int j=0; j<numLowFreqPeaks; ++j)
        {
            double wp = dis2(gen2)*5.+1.;
            double tau = dis2(gen2)*0.02*pow((double)input["beta"], 2.)+1.0; // in FL, 1/tau scales as T^2
            //
            plasmaFreqs.push_back(wp);
            scatteringRates.push_back(tau);
        }
        //
        unsigned int numMidInfraredPeaks = dis1(gen1);
        std::vector<double> positions;
        std::vector<double> widths;
        std::vector<double> strengths;
        for (unsigned int j=0; j<numMidInfraredPeaks; ++j)
        {
            double w = dis2(gen2)*8.+2.3;
            double gamma = dis2(gen2)*1.2+0.8;
            double omega = dis2(gen2)*6.+0.6;
            //
            positions.push_back(w);
            widths.push_back(gamma);
            strengths.push_back(omega);
        }
        //
        opticalConductivity sigmaTemp(numLowFreqPeaks, plasmaFreqs, scatteringRates, numMidInfraredPeaks, positions, widths, strengths);
        sigma.push_back(sigmaTemp);
        //
        plasmaFreqs.clear();
        scatteringRates.clear();
        positions.clear();
        widths.clear();
        strengths.clear();
    }
    //
    for (unsigned int i=0; i<(int)input["numData"]; ++i)
    {
        opticalPolarization PiTemp(sigma[i]);
        Pi.push_back(PiTemp);
    }
    // set the normalization equal to pi
    for (unsigned int i=0; i<(int)input["numData"]; ++i)
    {
        double wp = M_PI;
        sigma[i].setNormalization(wp);
    }
    // calculate optical conductivity
    for (unsigned int i=0; i<(int)input["numData"]; ++i)
    {
        sigma[i].calculateSigma();
    }
    // calculate optical polarization
    for (unsigned int i=0; i<(int)input["numData"]; ++i)
    {
        Pi[i].setFreqCutOff(2.0*input["bandWidth"]);
        GaussLegendreIntegration gl((int)input["GLorder"]);
        Pi[i].calculatePi(gl, (int)input["GLorder"]);
    }
    // adding Gaussian noise to Pi
    for (unsigned int i=0; i<(int)input["numData"]; ++i)
    {
        Pi[i].addNoise(input["noiseLevel"]);
    }
    // write into file
    csvWrite(sigma);
    csvWrite(Pi);
    // write samples
    sigma[0].csvWrite();
    Pi[0].csvWrite();
    //
    auto stop = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::seconds>(stop - start);
    std::cout << "Time taken by the code: " << duration.count() << " seconds" <<  std::endl;
    //
    return 0;
}
