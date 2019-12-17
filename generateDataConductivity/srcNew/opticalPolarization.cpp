//
//  opticalPolarization.cpp
//  generateCondData
//
//  Created by Reza Nourafkan on 2019-09-24.
//  Copyright © 2019 Reza Nourafkan. All rights reserved.
//

#include "opticalPolarization.hpp"
#include "opticalConductivity.hpp"
#include "GaussLegendreIntegration.hpp"
#include <cmath>
#include <iostream>
#include <fstream>
#include <iomanip>
#include <random>

namespace{
    const double DEFAULT_FREQCUTOFF = 50.0;
}

opticalPolarization::opticalPolarization(opticalConductivity &s)
:m_freqCutOff(DEFAULT_FREQCUTOFF),
 m_sigma(s)
{
    
}

opticalPolarization::~opticalPolarization()
{
    
}

opticalPolarization::opticalPolarization(const opticalPolarization &p)
:m_freqCutOff(p.m_freqCutOff),
 m_sigma(p.m_sigma)
{
    
}

opticalPolarization &opticalPolarization::operator=(const opticalPolarization &p)
{
    if (this != &p)
    {
        this->m_freqCutOff = p.m_freqCutOff;
        this->m_sigma = p.m_sigma;
    }
    return *this;
}

void opticalPolarization::setFreqCutOff(const double &cutOff)
{
    m_freqCutOff = cutOff;
}

double opticalPolarization::kernel(const double &omega, const double &nu)
{
    double tiny = 1.e-12;
    //
    double s = 2.* m_sigma.calculateSigma(omega) / M_PI;
    s *= (omega*omega+tiny)/(omega*omega+nu*nu+tiny);
    return s;
}

void opticalPolarization::calculatePi(const GaussLegendreIntegration &gl, const int &n)
{
    //
    const std::vector<double> & weight = gl.getWeight();
    const std::vector<double> & abscissa = gl.getAbscissa();
    //
    double precision= 1.e-9;
    unsigned int maxRecursion= 4;
    //
    for (auto&& nui : bosonicMatsubaraFrequencyMesh)
    {
        double previousS= 0.;
        double S= 0.0;
        unsigned int numIntervals= 256;
        //
        for (unsigned int k=0; k<maxRecursion; k++, numIntervals *=2)
        {
            S= 0.0;
            double lowerBound = 0.0;
            double upperBound = 0.0;
            for (unsigned int i=0; i<numIntervals; ++i)
            {
                double intSize = pow((double)(i+1)/(double)numIntervals, 2.);
                intSize -= pow((double)i/(double)numIntervals, 2.);
                intSize *= m_freqCutOff;
                upperBound = lowerBound + intSize;
                const double width = 0.5* (upperBound- lowerBound);
                const double mean  = 0.5* (lowerBound+ upperBound);
                for(unsigned int j= 1; j <= n; j++)
                {
                    S += width * weight[j]*kernel(width * abscissa[j] + mean, nui);
                }
                lowerBound += intSize;
            }
            if (fabs(S-previousS) <= precision*fabs(previousS))
            {
                break;
            }
            previousS = S;
        }
        m_Pi.push_back(S); // best approximation
    }
}

void opticalPolarization::addNoise(const double &noiseLevel)
{
    // Define random generator with Gaussian distribution
    const double mean = 0.0;
    const double stddev = noiseLevel;
    std::random_device rd{};
    std::mt19937 gen{rd()};
    std::normal_distribution<double> dist(mean, stddev);
    for (unsigned int j=1; j<bosonicMatsubaraFrequencyMesh.size(); ++j)
    {
       m_Pi[j] += dist(gen);
    }
}

std::vector<double> opticalPolarization::bosonicMatsubaraFrequencyMesh;
void opticalPolarization::setBosonicMatsubaraFrequencyMesh(const double &beta, const unsigned int &nMax)
{
    for (unsigned int i=0; i<nMax; ++i)
    {
        double val = 2.*i*M_PI / beta;
        bosonicMatsubaraFrequencyMesh.push_back(val);
    }
}

void opticalPolarization::csvWrite()
{
    std::fstream inFilePi;
    inFilePi.open("Pi_sample.csv", std::ios::out);
    if (inFilePi.fail())
    {
        std::cout << "Unable to open Pi_sample.csv file!" << std::endl;
        return;
    }
    for (unsigned int j=0; j<bosonicMatsubaraFrequencyMesh.size(); ++j)
    {
        inFilePi << std::fixed << std::setprecision(10) << bosonicMatsubaraFrequencyMesh[j] << ", " << m_Pi[j] << "\n";
    }
    inFilePi.close();
}

void csvWrite(const std::vector<opticalPolarization> &p)
{
    std::fstream inFilePi;
    inFilePi.open("Pi.csv", std::ios::out);
    if (inFilePi.fail())
    {
        std::cout << "Unable to open Pi.csv file!" << std::endl;
        return;
    }
    for (unsigned int i=0; i<p.size(); ++i)
    {
        for (unsigned int j=0; j<p[0].bosonicMatsubaraFrequencyMesh.size()-1; ++j)
        {
            inFilePi << std::fixed << std::setprecision(10) << p[i].m_Pi[j] << ",";
        }
        inFilePi << std::fixed << std::setprecision(10) << p[i].m_Pi.back() << "\n";
    }
    inFilePi.close();
}

