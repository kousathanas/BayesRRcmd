#ifndef SPARSEMARKER_H
#define SPARSEMARKER_H

#include "marker.h"

struct SparseMarker : public Marker
{
    double mean = 0;
    double sd = 0;
    double sqrdZ= 0;
    double Zsum = 0;

    double epsilonSum = 0;

    double computeNum(VectorXd &epsilon, const double beta_old) override;

    void updateEpsilon(VectorXd &epsilon,
                       const double beta_old,
                       const double beta) override;

    virtual void updateStatistics(unsigned int allele1, unsigned int allele2);

    size_t size() const override;
    void write(std::ostream *outStream) const override;

protected:
    virtual double computeNum(VectorXd &epsilon,
                              const double beta_old,
                              const double epsilonSum);

    virtual double dot(const VectorXd &epsilon) const = 0;

    virtual double computeEpsilonSumUpdate(const double beta_old,
                                           const double beta) const;
};

#endif // SPARSEMARKER_H
