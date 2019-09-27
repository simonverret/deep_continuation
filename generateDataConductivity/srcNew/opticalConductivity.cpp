//
//  opticalConductivity.cpp
//  generateCondData
//
//  Created by Reza Nourafkan on 2019-09-20.
//  Copyright © 2019 Reza Nourafkan. All rights reserved.
//

#include "opticalConductivity.hpp"
#include <vector>
#include <cmath>
#include <iostream>
#include <fstream>
#include <iomanip>

namespace{
    const double DEFAULT_NORMALIZATION = 1.0;
}

opticalConductivity::opticalConductivity(int numLowFreqPeaks, std::vector<double> plasmaFrequencies, std::vector<double> scatteringRates, int numMidInfraredPeaks, std::vector<double> positions, std::vector<double> widths, std::vector<double> strengths)
:m_numLowFreqPeaks(numLowFreqPeaks),
m_plasmaFrequencies(plasmaFrequencies),
m_scatteringRates(scatteringRates),
m_numMidInfraredPeaks(numMidInfraredPeaks),
m_positions(positions),
m_widths(widths),
m_strengths(strengths),
m_normalization(DEFAULT_NORMALIZATION)
{
    
}

opticalConductivity::~opticalConductivity()
{
    
}

opticalConductivity::opticalConductivity(const opticalConductivity &s)
:m_numLowFreqPeaks(s.m_numLowFreqPeaks),
m_plasmaFrequencies(s.m_plasmaFrequencies),
m_scatteringRates(s.m_scatteringRates),
m_numMidInfraredPeaks(s.m_numMidInfraredPeaks),
m_positions(s.m_positions),
m_widths(s.m_widths),
m_strengths(s.m_strengths),
m_normalization(s.m_normalization)
{
    
}

opticalConductivity &opticalConductivity::operator=(const opticalConductivity &s)
{
    if (this != &s)
    {
        this->m_numLowFreqPeaks=s.m_numLowFreqPeaks;
        this->m_plasmaFrequencies=s.m_plasmaFrequencies;
        this->m_scatteringRates=s.m_scatteringRates;
        this->m_numMidInfraredPeaks=s.m_numMidInfraredPeaks;
        this->m_positions=s.m_positions;
        this->m_widths=s.m_widths;
        this->m_strengths=s.m_strengths;
        this->m_normalization=s.m_normalization;
    }
    return *this;
}

void opticalConductivity::setNormalization(const double &wp)
{
    double norm = 0.;
    for (unsigned int i=0; i<m_numLowFreqPeaks; ++i)
    {
        norm += pow(m_plasmaFrequencies[i], 2.);
    }
    for (unsigned int j=0; j<m_numMidInfraredPeaks; ++j)
    {
        norm += 2.0*pow(m_strengths[j], 2.);
    }
    m_normalization = norm/wp;
}

double opticalConductivity::calculateSigma(const double &omega)
{
    double s = 0.;
    for (unsigned int i=0; i<m_numLowFreqPeaks; ++i)
    {
        double exponent = omega;
        exponent *= -exponent;
        exponent *= pow(m_scatteringRates[i],2.)/2.;
        double result = m_scatteringRates[i]*exp(exponent)/sqrt(2. * M_PI);
        s += pow(m_plasmaFrequencies[i], 2.)* result;
    }
    for (unsigned int j=0; j<m_numMidInfraredPeaks; ++j)
    {
        double exponent = omega - m_positions[j];
        exponent *= -exponent;
        exponent /= 2. * pow(m_widths[j],2.);
        double result = exp(exponent);
        result /= m_widths[j] * sqrt(2. * M_PI);
        s += pow(m_strengths[j], 2.)* result;
        //
        exponent = omega + m_positions[j];
        exponent *= -exponent;
        exponent /= 2. * pow(m_widths[j],2.);
        result = exp(exponent);
        result /= m_widths[j] * sqrt(2. * M_PI);
        s += pow(m_strengths[j], 2.)* result;
    }
    return s/m_normalization;
}

void opticalConductivity::calculateSigma()
{
    if (frequencyMesh.empty())
    {
        return;
    }
    //
    for (auto&& omegai : frequencyMesh)
    {
        double s = 0.;
        for (unsigned int i=0; i<m_numLowFreqPeaks; ++i)
        {
            double exponent = omegai;
            exponent *= -exponent;
            exponent *= pow(m_scatteringRates[i],2.)/2.;
            double result = m_scatteringRates[i]*exp(exponent)/sqrt(2. * M_PI);
            s += pow(m_plasmaFrequencies[i], 2.)* result;
        }
        for (unsigned int j=0; j<m_numMidInfraredPeaks; ++j)
        {
            double exponent = omegai - m_positions[j];
            exponent *= -exponent;
            exponent /= 2. * pow(m_widths[j],2.);
            double result = exp(exponent);
            result /= m_widths[j] * sqrt(2. * M_PI);
            s += pow(m_strengths[j], 2.)* result;
            //
            exponent = omegai + m_positions[j];
            exponent *= -exponent;
            exponent /= 2. * pow(m_widths[j],2.);
            result = exp(exponent);
            result /= m_widths[j] * sqrt(2. * M_PI);
            s += pow(m_strengths[j], 2.)* result;
        }
        m_sigma.push_back(s/m_normalization);
    }
}

void opticalConductivity::csvWrite()
{
    std::fstream inFileSigma;
    inFileSigma.open("SigmaRe_sample.csv", std::ios::out);
    if (inFileSigma.fail())
    {
        std::cout << "Unable to open SigmaRe_sample.csv file!" << std::endl;
        return;
    }
    for (unsigned int j=0; j<frequencyMesh.size(); ++j)
    {
        inFileSigma << std::fixed << std::setprecision(10) << frequencyMesh[j] << ", " << m_sigma[j] << "\n";
    }
    inFileSigma.close();
}

//non-uniform freq. mesh with finer mesh at low frequency (behave like w^2)
std::vector<double> opticalConductivity::frequencyMesh;
void opticalConductivity::setFrequencyMesh(const unsigned int &numInterval, const double &freqCutOff)
{
    for (unsigned int i=0; i<numInterval; ++i)
    {
        double val = pow((double)i / (double)numInterval, 2.) * freqCutOff;
        frequencyMesh.push_back(val);
    }
}

void csvWrite(const std::vector<opticalConductivity> &op)
{
    std::fstream inFileSigma;
    inFileSigma.open("SigmaRe.csv", std::ios::out);
    if (inFileSigma.fail())
    {
        std::cout << "Unable to open SigmaRe.csv file!" << std::endl;
        return;
    }
    for (unsigned int i=0; i<op.size(); ++i)
    {
        for (unsigned int j=0; j<op[0].frequencyMesh.size()-1; ++j)
        {
            inFileSigma << std::fixed << std::setprecision(10) << op[i].m_sigma[j] << ",";
        }
        inFileSigma << std::fixed << std::setprecision(10) << op[i].m_sigma.back() << "\n";
    }
    inFileSigma.close();
}
