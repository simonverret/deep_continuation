//
//  main.cpp
//  generateDataConductivity
//
//  Created by Reza Nourafkan on 2019-08-23.
//  Copyright © 2019 Reza Nourafkan. All rights reserved.
//

#include <iostream>
#include <random>
#include <chrono>
#include <string>

#include "OpticalConductivity.hpp"


int main() {
    std::string basisFunction = "Gaussian";
    //
    double beta = 20;
    unsigned int numData = 50000;
    unsigned int numIntervalOmega = 512;
    double bandWidth = 20.0;
    //
    unsigned int maxBosonicMatsubaraIndex = 128;
    bool normalization= true;
    //
    unsigned int maxNumDrude = 3;
    unsigned int maxNumDrudeLorentz = 16;
    //
    auto start = std::chrono::high_resolution_clock::now();
    // Initiate Freq. mesh
    OpticalConductivity::initFrequencyMesh(numIntervalOmega, bandWidth);
    OpticalConductivity::initBosonicMatsubaraFrequencyMesh(beta, maxBosonicMatsubaraIndex);
    //
    std::vector<OpticalConductivity> sigma;
    //
    std::random_device rd;  //Will be used to obtain a seed for the random number engine
    std::mt19937 gen(rd()); //Standard mersenne_twister_engine seeded with rd()
    std::uniform_int_distribution<> dis(0, maxNumDrude);
    std::random_device rd1;
    std::mt19937 gen1(rd1());
    std::uniform_int_distribution<> dis1(3, maxNumDrudeLorentz);
    std::random_device rd2;
    std::mt19937 gen2(rd2());
    std::uniform_real_distribution<double> dis2(0., 1.0);
    for (unsigned int i=0; i<numData; ++i)
    {
        unsigned int numDrude = dis(gen);
        std::vector<double> plasmaFreqs;
        std::vector<double> scatteringRates;
        for (unsigned int j=0; j<numDrude; ++j)
        {
            double wp = dis2(gen2)*5.+1.;
            double tau = dis2(gen2)*0.02*pow(beta, 2.)+1.0; // in FL, 1/tau scales as T^2
//
            plasmaFreqs.push_back(wp);
            scatteringRates.push_back(tau);
        }
        //
        unsigned int numDrudeLorentz = dis1(gen1);
        std::vector<double> positions;
        std::vector<double> widths;
        std::vector<double> strengths;
        for (unsigned int j=0; j<numDrudeLorentz; ++j)
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
        OpticalConductivity sigmaTemp(numDrude, plasmaFreqs, scatteringRates, numDrudeLorentz, positions, widths, strengths);
        //
        sigma.push_back(sigmaTemp);
        //
        plasmaFreqs.clear();
        scatteringRates.clear();
        positions.clear();
        widths.clear();
        strengths.clear();
    }
    //
    for (unsigned int i=0; i<numData; ++i)
    {
        if (normalization)
        {
            // the total spectral weight is distributed randomly in [1., 4.]
            double wp = 1.0/M_PI;///(1.0 + dis2(gen2)*3.);
            sigma[i].setNormalization(wp);
        }
    }
    //
    if (basisFunction == "Lorentzian")
    {
        for (unsigned int i=0; i<numData; ++i)
        {
            sigma[i].calculateOpticalConductivityLorentzian();
            sigma[i].calculateOpticalPolarizationLorentzian(2.*bandWidth);
        }
    } else if (basisFunction == "Gaussian")
    {
        for (unsigned int i=0; i<numData; ++i)
        {
            sigma[i].calculateOpticalConductivityGaussian();
            sigma[i].calculateOpticalPolarizationGaussian(2.*bandWidth);
        }
    }
    // sample conductivity
    sigma[numData-1].csvWrite();
    //
    csvWrite(sigma, maxNumDrude, maxNumDrudeLorentz);
    //
    auto stop = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::seconds>(stop - start);
    std::cout << "Time taken by the code: " << duration.count() << " seconds" <<  std::endl;
    //
    return 0;
}
