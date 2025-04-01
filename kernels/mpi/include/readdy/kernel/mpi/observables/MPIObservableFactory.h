/********************************************************************
 * Copyright © 2019 Computational Molecular Biology Group,          *
 *                  Freie Universität Berlin (GER)                  *
 *                                                                  *
 * Redistribution and use in source and binary forms, with or       *
 * without modification, are permitted provided that the            *
 * following conditions are met:                                    *
 *  1. Redistributions of source code must retain the above         *
 *     copyright notice, this list of conditions and the            *
 *     following disclaimer.                                        *
 *  2. Redistributions in binary form must reproduce the above      *
 *     copyright notice, this list of conditions and the following  *
 *     disclaimer in the documentation and/or other materials       *
 *     provided with the distribution.                              *
 *  3. Neither the name of the copyright holder nor the names of    *
 *     its contributors may be used to endorse or promote products  *
 *     derived from this software without specific                  *
 *     prior written permission.                                    *
 *                                                                  *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND           *
 * CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES,      *
 * INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF         *
 * MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE         *
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR            *
 * CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,     *
 * SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,         *
 * BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; *
 * LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER *
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT,      *
 * STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)    *
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF      *
 * ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.                       *
 ********************************************************************/

/**
 * @file MPIObservableFactory.h
 * @brief Declaration of observable factory for the MPI kernel
 * @author chrisfroe
 * @date 03.06.19
 */

#pragma once

#include <readdy/model/observables/ObservableFactory.h>

namespace readdy::kernel::mpi {
class MPIKernel;
namespace observables {

class MPIObservableFactory : public readdy::model::observables::ObservableFactory {

public:
    explicit MPIObservableFactory(MPIKernel* kernel);

    [[nodiscard]] std::unique_ptr<readdy::model::observables::Energy>
    energy(Stride stride, ObsCallback<readdy::model::observables::Energy> callback) const override;

    [[nodiscard]] std::unique_ptr<readdy::model::observables::Virial>
    virial(Stride stride, ObsCallback<readdy::model::observables::Virial> callback) const override;

    [[nodiscard]] std::unique_ptr<readdy::model::observables::HistogramAlongAxis>
    histogramAlongAxis(Stride stride, std::vector<scalar> binBorders, std::vector<std::string> typesToCount,
                       unsigned int axis, ObsCallback<readdy::model::observables::HistogramAlongAxis> callback) const override;

    [[nodiscard]] std::unique_ptr<readdy::model::observables::NParticles>
    nParticles(Stride stride, std::vector<std::string> typesToCount, ObsCallback<readdy::model::observables::NParticles> callback) const override;

    [[nodiscard]] std::unique_ptr<readdy::model::observables::Forces>
    forces(Stride stride, std::vector<std::string> typesToCount, ObsCallback<readdy::model::observables::Forces> callback) const override;

    [[nodiscard]] std::unique_ptr<readdy::model::observables::Positions>
    positions(Stride stride, std::vector<std::string> typesToCount, ObsCallback<readdy::model::observables::Positions> callback) const override;

    [[nodiscard]] std::unique_ptr<readdy::model::observables::RadialDistribution>
    radialDistribution(Stride stride, std::vector<scalar> binBorders, std::vector<std::string> typeCountFrom,
                       std::vector<std::string> typeCountTo, scalar particleDensity,
                       ObsCallback <readdy::model::observables::RadialDistribution> callback) const override;

    [[nodiscard]] std::unique_ptr<readdy::model::observables::Particles>
    particles(Stride stride, ObsCallback<readdy::model::observables::Particles> callback) const override;

    [[nodiscard]] std::unique_ptr<readdy::model::observables::Reactions>
    reactions(Stride stride, ObsCallback<readdy::model::observables::Reactions> callback) const override;

    [[nodiscard]] std::unique_ptr<readdy::model::observables::ReactionCounts>
    reactionCounts(Stride stride, ObsCallback<readdy::model::observables::ReactionCounts> callback) const override;

private:
    MPIKernel *const kernel;
};

}
}
