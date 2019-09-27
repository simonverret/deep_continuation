//
//  opticalPolarization.hpp
//  generateCondData
//
//  Created by Reza Nourafkan on 2019-09-24.
//  Copyright © 2019 Reza Nourafkan. All rights reserved.
//

#ifndef opticalPolarization_hpp
#define opticalPolarization_hpp
#include <vector>

class GaussLegendreIntegration;
class opticalConductivity;

class opticalPolarization{
    friend void csvWrite(const std::vector<opticalPolarization> &op);
public:
    opticalPolarization(opticalConductivity &s);
    opticalPolarization (const opticalPolarization &p);
    opticalPolarization &operator=(const opticalPolarization &p);
    ~opticalPolarization();
    //
    void setFreqCutOff(const double &cutOff);
    double kernel(const double &omega, const double &nu);
    void calculatePi(const GaussLegendreIntegration &gl, const int &n);
    void addNoise(const double &noiseLevel);
    void csvWrite();
    //
    static void setBosonicMatsubaraFrequencyMesh(const double &beta, const unsigned int &nMax);
private:
    double m_freqCutOff;
    opticalConductivity &m_sigma;
    std::vector<double> m_Pi;
    //
    static std::vector<double> bosonicMatsubaraFrequencyMesh;
};

void csvWrite(const std::vector<opticalPolarization> &op);

#endif /* opticalPolarization_hpp */
