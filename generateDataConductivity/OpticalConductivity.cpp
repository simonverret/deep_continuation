//
//  OpticalConductivity.cpp
//  generateDataConductivity
//
//  Created by Reza Nourafkan on 2019-08-23.
//  Copyright © 2019 Reza Nourafkan. All rights reserved.
//

#include "OpticalConductivity.hpp"
#include <iostream>
#include <fstream>
#include <cmath>
#include <iomanip>

namespace{
    const double DEFAULT_NORMALIZATION = 1.0;
}

OpticalConductivity::OpticalConductivity(int numDrude, std::vector<double> plasmaFrequencies, std::vector<double> scatteringRates, int numDrudeLorentz, std::vector<double> positions, std::vector<double> widths, std::vector<double> strengths)
:m_numDrude(numDrude),
 m_plasmaFrequencies(plasmaFrequencies),
 m_scatteringRates(scatteringRates),
 m_numDrudeLorentz(numDrudeLorentz),
 m_positions(positions),
 m_widths(widths),
 m_strengths(strengths),
 m_normalization(DEFAULT_NORMALIZATION)
{
    
}

OpticalConductivity::~OpticalConductivity()
{
    
}

OpticalConductivity::OpticalConductivity(const OpticalConductivity &s)
:m_numDrude(s.m_numDrude),
m_plasmaFrequencies(s.m_plasmaFrequencies),
m_scatteringRates(s.m_scatteringRates),
m_numDrudeLorentz(s.m_numDrudeLorentz),
m_positions(s.m_positions),
m_widths(s.m_widths),
m_strengths(s.m_strengths),
m_normalization(s.m_normalization)
{
    
}

OpticalConductivity &OpticalConductivity::operator=(const OpticalConductivity &s)
{
    if (this != &s)
    {
        this->m_numDrude=s.m_numDrude;
        this->m_plasmaFrequencies=s.m_plasmaFrequencies;
        this->m_scatteringRates=s.m_scatteringRates;
        this->m_numDrudeLorentz=s.m_numDrudeLorentz;
        this->m_positions=s.m_positions;
        this->m_widths=s.m_widths;
        this->m_strengths=s.m_strengths;
        this->m_normalization=s.m_normalization;
    }
    return *this;
}

void OpticalConductivity::setNormalization(const double &wp)
{
    double norm = 0.;
    for (unsigned int i=0; i<m_numDrude; ++i)
    {
        norm += pow(m_plasmaFrequencies[i], 2.);
    }
    for (unsigned int j=0; j<m_numDrudeLorentz; ++j)
    {
        norm += pow(m_strengths[j], 2.);
    }
    norm *= M_PI;
    m_normalization = wp*norm;
}


std::vector<double> OpticalConductivity::frequencyMesh;
void OpticalConductivity::initFrequencyMesh(const unsigned int numInterval, const double bandWidth)
{
    //non-uniform freq. mesh with finer mesh at low frequency
    for (unsigned int i=0; i<numInterval; ++i)
    {
        double val = pow((double)i / (double)numInterval, 2.) * bandWidth;
        frequencyMesh.push_back(val);
    }
}

std::vector<double> OpticalConductivity::BosonicMatsubaraFrequencyMesh;
void OpticalConductivity::initBosonicMatsubaraFrequencyMesh(const double beta, const unsigned int nMax)
{
    for (unsigned int i=0; i<nMax; ++i)
    {
        double val = 2.*i*M_PI / beta;
        BosonicMatsubaraFrequencyMesh.push_back(val);
    }
}


double OpticalConductivity::calculateOpticalConductivity(const double &omega)
{
    double s = 0.;
    for (unsigned int i=0; i<m_numDrude; ++i)
    {
        double n = pow(m_plasmaFrequencies[i], 2.)* m_scatteringRates[i];
        double d = 1.+ pow(omega*m_scatteringRates[i], 2.);
        s += n/d;
    }
    for (unsigned int j=0; j<m_numDrudeLorentz; ++j)
    {
        double n = pow(omega*m_strengths[j], 2.)* m_widths[j];
        double d = pow(m_positions[j], 2.) - pow(omega, 2.);
        d *= d;
        d += pow(omega*m_widths[j], 2.);
        s += n/d;
    }
    return s/m_normalization;
}

double OpticalConductivity::kernel(const double &omega, const double &nu)
{
    double tiny = 1.e-9;
//
    double s = 2.* calculateOpticalConductivity(omega) / M_PI;
    s *= (omega*omega+tiny)/(omega*omega+nu*nu+tiny);
    return s;
}

void OpticalConductivity::calculateOpticalConductivity()
{
    if (frequencyMesh.empty())
    {
        return;
    }
    //
    for (auto&& omegai : frequencyMesh)
    {
        double s = 0.;
        for (unsigned int i=0; i<m_numDrude; ++i)
        {
            double n = pow(m_plasmaFrequencies[i], 2.)* m_scatteringRates[i];
            double d = 1. + pow(omegai*m_scatteringRates[i], 2.);
            s += n/d;
        }
        for (unsigned int j=0; j<m_numDrudeLorentz; ++j)
        {
            double n = pow(omegai*m_strengths[j], 2.)* m_widths[j];
            double d = pow(m_positions[j], 2.) - pow(omegai, 2.);
            d *= d;
            d += pow(omegai*m_widths[j], 2.);
            s += n/d;
        }
        m_sigma.push_back(s/m_normalization);
    }
}

void OpticalConductivity::calculateOpticalPolarization( const double &bandWidth)
{
    if (BosonicMatsubaraFrequencyMesh.empty())
    {
        return;
    }
    //
    unsigned int numIntervals = 1024;
    for (auto&& nui : BosonicMatsubaraFrequencyMesh)
    {
        //simple Simpson integration. This should be improved!
        double a = 0.0;
        double b = bandWidth;
        double S = 0.;
        double w = a;
        for (unsigned int i=0; i<numIntervals; ++i)
        {
            double intSize = pow((double)(i+1)/(double)numIntervals, 2.);
            intSize -= pow((double)i/(double)numIntervals, 2.);
            intSize *= (b - a);
            S += (intSize / 6.) * ( kernel(w, nui) + kernel(w+intSize, nui) + 4.* kernel((w+w+intSize)/2., nui) );
            w += intSize;
        }
        // add the integration of the tail
        double tail = 0.0;
        double multi = 0.0;
        if (nui == 0.)
        {
            multi = 1./bandWidth/m_normalization;
        }else
        {
            multi = (M_PI/2.-atan(bandWidth/nui))/nui/m_normalization;
        }
        for (unsigned int i=0; i<m_numDrude; ++i)
        {
            tail += (2.0/M_PI)*(pow(m_plasmaFrequencies[i], 2.)/m_scatteringRates[i])*multi;
        }
        for (unsigned int j=0; j<m_numDrudeLorentz; ++j)
        {
            tail += (2.0/M_PI)*(pow(m_strengths[j], 2.)*m_widths[j])*multi;
        }
//        std::cout << tail << std::endl;
        m_Pi.push_back(S+tail);
    }
}

void OpticalConductivity::csvWrite()
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
//
    std::fstream inFilePi;
    inFilePi.open("Pi_sample.csv", std::ios::out);
    if (inFilePi.fail())
    {
        std::cout << "Unable to open Pi_sample.csv file!" << std::endl;
        return;
    }
    for (unsigned int j=0; j<BosonicMatsubaraFrequencyMesh.size(); ++j)
    {
        inFilePi << std::fixed << std::setprecision(10) << BosonicMatsubaraFrequencyMesh[j] << ", " << m_Pi[j] << "\n";
    }
    inFilePi.close();
}

void csvWrite(const std::vector<OpticalConductivity> &op, const unsigned int& maxNumDrude, const unsigned int& maxNumDrudeLorentz)
{
    /*std::fstream inFileParam;
    inFileParam.open("Params.csv", std::ios::out);
    if (inFileParam.fail())
    {
        std::cout << "Unable to open Params.csv file!" << std::endl;
        return;
    }
    for (unsigned int i=0; i<op.size(); ++i)
    {
        const double zero= 0.0;
        for (unsigned int j=0; j<op[i].m_numDrude; ++j)
        {
            inFileParam << std::fixed << std::setprecision(10) << op[i].m_plasmaFrequencies[j] / op[i].m_normalization << "," << op[i].m_scatteringRates[j] << ",";
        }
        for (unsigned int j=op[i].m_numDrude; j<maxNumDrude; ++j)
        {
            inFileParam << std::fixed << std::setprecision(10) << zero << "," << zero << ",";
        }
        if (op[i].m_numDrudeLorentz == maxNumDrudeLorentz)
        {
            for (unsigned int j=0; j<op[i].m_numDrudeLorentz-1; ++j)
            {
                inFileParam << std::fixed << std::setprecision(10) << op[i].m_strengths[j] / op[i].m_normalization << "," << op[i].m_widths[j] << "," << op[i].m_positions[j] << ",";
            }
            unsigned int j = op[i].m_numDrudeLorentz-1;
            inFileParam << std::fixed << std::setprecision(10) << op[i].m_strengths[j] / op[i].m_normalization << "," << op[i].m_widths[j] << "," << op[i].m_positions[j] << "\n";
        }else
        {
            for (unsigned int j=0; j<op[i].m_numDrudeLorentz; ++j)
            {
                inFileParam << std::fixed << std::setprecision(10) << op[i].m_strengths[j] / op[i].m_normalization << "," << op[i].m_widths[j] << "," << op[i].m_positions[j] << ",";
            }
            for (unsigned int j=op[i].m_numDrudeLorentz; j< maxNumDrudeLorentz-1; ++j)
            {
                inFileParam << std::fixed << std::setprecision(10) << zero << "," << zero << "," << zero << ",";
            }
            inFileParam << std::fixed << std::setprecision(10) << zero << "," << zero << "," << zero << "\n";
        }
    }
    inFileParam.close();*/
    //
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
//
    std::fstream inFilePi;
    inFilePi.open("Pi.csv", std::ios::out);
    if (inFilePi.fail())
    {
        std::cout << "Unable to open Pi.csv file!" << std::endl;
        return;
    }
    for (unsigned int i=0; i<op.size(); ++i)
    {
        for (unsigned int j=0; j<op[0].BosonicMatsubaraFrequencyMesh.size()-1; ++j)
        {
            inFilePi << std::fixed << std::setprecision(10) << op[i].m_Pi[j] << ",";
        }
        inFilePi << std::fixed << std::setprecision(10) << op[i].m_Pi.back() << "\n";
    }
    inFilePi.close();
}
