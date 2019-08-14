#ifndef SRC_SPARSEBAYESRRG_H_
#define SRC_SPARSEBAYESRRG_H_

#include "BayesRBase.hpp"
#include <Eigen/Eigen>
class Data;

class SparseBayesRRG : public BayesRBase
{
    friend class SparseParallelGraph;

public:
    explicit SparseBayesRRG(const Data *data, const Options *opt);
    ~SparseBayesRRG() override;

    std::unique_ptr<Kernel> kernelForMarker(const Marker *marker) const override;
    MarkerBuilder *markerBuilder() const override;

    void updateGlobal(Kernel *kernel, const double beta_old, const double beta, const VectorXd &deltaEps ) override;
   void updateMu(double old_mu,double N);
protected:
    double m_asyncEpsilonSum = 0.0;

    VectorXd m_ones;

    void init(int K, unsigned int markerCount, unsigned int individualCount) override;
    void prepareForAnylsis() override;

    void prepare(BayesRKernel *kernel) override;
    void readWithSharedLock(BayesRKernel *kernel) override;
    void writeWithUniqueLock(BayesRKernel *kernel) override;
   
};

#endif /* SRC_SPARSEBAYESRRG_H_ */
