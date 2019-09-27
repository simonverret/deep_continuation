//
//  opticalConductivity.hpp
//  generateCondData
//
//  Created by Reza Nourafkan on 2019-09-20.
//  Copyright © 2019 Reza Nourafkan. All rights reserved.
//

#ifndef opticalConductivity_hpp
#define opticalConductivity_hpp
#include <vector>

class opticalConductivity{
    friend void csvWrite(const std::vector<opticalConductivity> &op);
public:
    opticalConductivity(int numLowFreqPeaks, std::vector<double> plasmaFrequencies, std::vector<double> scatteringRates, int numMidInfraredPeaks, std::vector<double> positions, std::vector<double> widths, std::vector<double> strengths);
    opticalConductivity(const opticalConductivity &s);
    opticalConductivity &operator=(const opticalConductivity &s);
    ~opticalConductivity();
    //
    void setNormalization(const double &wp);
    double calculateSigma(const double &omega);
    void calculateSigma();
    void csvWrite();
    //
    static void setFrequencyMesh(const unsigned int &numInterval, const double &freqCutOff);
    
private:
    unsigned int m_numLowFreqPeaks;
    std::vector<double> m_plasmaFrequencies;
    std::vector<double> m_scatteringRates;
    //
    unsigned int m_numMidInfraredPeaks;
    std::vector<double> m_positions;
    std::vector<double> m_widths;
    std::vector<double> m_strengths;
    //
    double m_normalization;
    static std::vector<double> frequencyMesh;
    std::vector<double> m_sigma;
    
};

void csvWrite(const std::vector<opticalConductivity> &op);

#endif /* opticalConductivity_hpp */
