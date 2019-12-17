//
//  GaussLegendreIntegration.cpp
//  Integration
//
//  Created by Reza Nourafkan on 2019-09-15.
//  Copyright © 2019 Reza Nourafkan. All rights reserved.
//

#include "GaussLegendreIntegration.hpp"
#include <vector>
#include <cmath>

namespace {
    const double DEFAULT_EPSILON = 1.e-15;
}

GaussLegendreIntegration::GaussLegendreIntegration (const int &numberOfIterations)
:m_numberOfIterations(numberOfIterations),
 m_EPSILON(DEFAULT_EPSILON)
{
   calculateWeightAndRoot();
}

GaussLegendreIntegration::~GaussLegendreIntegration()
{
    
}

GaussLegendreIntegration::GaussLegendreIntegration(const GaussLegendreIntegration &lp)
:m_numberOfIterations(lp.m_numberOfIterations),
 m_weight(lp.m_weight),
 m_abscissa(lp.m_abscissa),
 m_EPSILON(lp.m_EPSILON)
{
    
}

GaussLegendreIntegration &GaussLegendreIntegration::operator=(const GaussLegendreIntegration &lp)
{
    if (this != &lp)
    {
        this->m_numberOfIterations= lp.m_numberOfIterations;
        this->m_weight= lp.m_weight;
        this->m_abscissa= lp.m_abscissa;
        this->m_EPSILON= lp.m_EPSILON;
    }
    return *this;
}


const std::vector<double> & GaussLegendreIntegration::getWeight() const {
    return m_weight;
}

const std::vector<double> & GaussLegendreIntegration::getAbscissa() const {
    return m_abscissa;
}

// abscissas (x_i's) are given by roots of Legendre polynominal P_n(x)
// weights are 2/[(1-x_i^2)*[P'_n(x_i)]^2]
void GaussLegendreIntegration::calculateWeightAndRoot() {
    for(unsigned int step = 0; step <= m_numberOfIterations; step++) {
        double root = cos(M_PI * (step-0.25)/(m_numberOfIterations+0.5));
        Result result = calculatePolynomialValueAndDerivative(root);
        
        double newtonRaphsonRatio;
        do {
            newtonRaphsonRatio = result.value/result.derivative;
            root -= newtonRaphsonRatio;
            result = calculatePolynomialValueAndDerivative(root);
        } while (fabs(newtonRaphsonRatio) > m_EPSILON);
        
        m_abscissa.push_back(root);
        m_weight.push_back(2.0/((1.0-root*root)*result.derivative*result.derivative));
    }
}

// using the recursion formula
// P_m(x) = (1/m)[(2m-1)P_{m-1}(x)-(m-1)P_{m-1}(x)]
GaussLegendreIntegration::Result GaussLegendreIntegration::calculatePolynomialValueAndDerivative(const double &x) {
    Result result(x, 0);
    
    double valueMinusOne = 1;
    const double f = 1/(x*x-1);
    for(int step = 2; step <= m_numberOfIterations; step++) {
        const double value = ((2*step-1)*x*result.value-(step-1)*valueMinusOne)/step;
        result.derivative = step*f*(x*value - result.value);
        
        valueMinusOne = result.value;
        result.value = value;
    }
    
    return result;
}
