//
//  OpticalConductivity.hpp
//  generateDataConductivity
//
//  Created by Reza Nourafkan on 2019-08-23.
//  Copyright © 2019 Reza Nourafkan. All rights reserved.
//

#ifndef OpticalConductivity_hpp
#define OpticalConductivity_hpp

#include <vector>

class OpticalConductivity{
    friend void csvWrite(const std::vector<OpticalConductivity> &op, const unsigned int& maxNumDrude, const unsigned int& maxNumDrudeLorentz);
public:
    OpticalConductivity(int numDrude, std::vector<double> plasmaFrequencies, std::vector<double> scatteringRates, int numDrudeLorentz, std::vector<double> positions, std::vector<double> widths, std::vector<double> strengths);
    OpticalConductivity(const OpticalConductivity &s);
    OpticalConductivity &operator=(const OpticalConductivity &s);
    ~OpticalConductivity();
    //
    static void initFrequencyMesh(const unsigned int numInterval, const double bandWidth);
    static void initBosonicMatsubaraFrequencyMesh(const double beta, const unsigned int nMax);
    double calculateOpticalConductivityLorentzian(const double &omega);
    double calculateOpticalConductivityGaussian(const double &omega);
    double kernelLorentzian(const double &omega, const double &nu);
    double kernelGaussian(const double &omega, const double &nu);
    void setNormalization(const double &wp);
    void calculateOpticalConductivityLorentzian();
    void calculateOpticalConductivityGaussian();
    void calculateOpticalPolarizationLorentzian(const double& bandWidth);
    void calculateOpticalPolarizationGaussian(const double& bandWidth);
    void csvWrite();
    //
private:
    unsigned int m_numDrude;
    std::vector<double> m_plasmaFrequencies;
    std::vector<double> m_scatteringRates;
    //
    unsigned int m_numDrudeLorentz;
    std::vector<double> m_positions;
    std::vector<double> m_widths;
    std::vector<double> m_strengths;
    //
    static std::vector<double> frequencyMesh;
    static std::vector<double> BosonicMatsubaraFrequencyMesh;
    //
    double m_normalization;
    std::vector<double> m_sigma;
    std::vector<double> m_Pi;
    //
};

void csvWrite(const std::vector<OpticalConductivity> &op);

#endif /* OpticalConductivity_hpp */
