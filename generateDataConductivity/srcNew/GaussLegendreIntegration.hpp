//
//  GaussLegendreIntegration.hpp
//  Integration
//
//  Created by Reza Nourafkan on 2019-09-15.
//  Copyright © 2019 Reza Nourafkan. All rights reserved.
//

#ifndef GaussLegendreIntegration_hpp
#define GaussLegendreIntegration_hpp

#include <vector>

class GaussLegendreIntegration{
public:
    GaussLegendreIntegration(const int &numberOfIterations);
    GaussLegendreIntegration(const GaussLegendreIntegration &lp);
    GaussLegendreIntegration &operator=(const GaussLegendreIntegration &lp);
    ~GaussLegendreIntegration();
    //
    const std::vector<double> &getWeight() const;
    const std::vector<double> &getAbscissa() const;
    
private:
    double m_EPSILON;
    int m_numberOfIterations;
    std::vector<double> m_weight;
    std::vector<double> m_abscissa;
    //
    struct Result {
        double value;
        double derivative;
        
        Result() : value(0), derivative(0) {}
        Result(double val, double deriv) : value(val), derivative(deriv) {}
    };
    void calculateWeightAndRoot();
    Result calculatePolynomialValueAndDerivative(const double& x);
};

#endif /* GaussLegendreIntegration_hpp */
