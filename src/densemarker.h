#ifndef DENSEMARKER_H
#define DENSEMARKER_H

#include "marker.h"
#include "markerbuilder.h"

struct DenseMarker : public Marker
{
    double component = 0;

    std::shared_ptr<unsigned char[]> buffer = nullptr;
    std::shared_ptr<Map<VectorXd>> Cx = nullptr;

    double computeNum(VectorXd &epsilon, const double beta_old) override;
    void updateEpsilon(VectorXd &epsilon,
                       const double beta_old,
                       const double beta) override;

    CompressedMarker compress() const override;
    void write(std::ostream *outStream) const override;
};

#endif // DENSEMARKER_H
