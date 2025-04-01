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
 * « detailed description »
 *
 * @file MPIActionsFactory.h
 * @brief « brief description »
 * @author chrisfroe
 * @date 03.06.19
 */

#pragma once

#include <readdy/model/actions/ActionFactory.h>

namespace readdy::kernel::mpi {

class MPIKernel;

namespace actions {

class MPIActionFactory : public readdy::model::actions::ActionFactory {
    MPIKernel *const kernel;
public:
    explicit MPIActionFactory(MPIKernel *kernel);

    [[nodiscard]] std::unique_ptr<readdy::model::actions::AddParticles>
    addParticles(const std::vector<readdy::model::Particle> &particles) const override;

    [[nodiscard]] std::unique_ptr<readdy::model::actions::EulerBDIntegrator> eulerBDIntegrator(scalar timeStep) const override;

    [[nodiscard]] std::unique_ptr<readdy::model::actions::MdgfrdIntegrator> mdgfrdIntegrator(scalar timeStep) const override;

    [[nodiscard]] std::unique_ptr<readdy::model::actions::CalculateForces> calculateForces() const override;

    [[nodiscard]] std::unique_ptr<readdy::model::actions::CreateNeighborList> createNeighborList(scalar interactionDistance) const override;

    [[nodiscard]] std::unique_ptr<readdy::model::actions::UpdateNeighborList> updateNeighborList() const override;

    [[nodiscard]] std::unique_ptr<readdy::model::actions::ClearNeighborList> clearNeighborList() const override;

    [[nodiscard]] std::unique_ptr<readdy::model::actions::EvaluateCompartments> evaluateCompartments() const override;

    [[nodiscard]] std::unique_ptr<readdy::model::actions::reactions::UncontrolledApproximation>
    uncontrolledApproximation(scalar timeStep) const override;

    [[nodiscard]] std::unique_ptr<readdy::model::actions::reactions::Gillespie>
    gillespie(scalar timeStep) const override;

    [[nodiscard]] std::unique_ptr<readdy::model::actions::reactions::DetailedBalance>
    detailedBalance(scalar timeStep) const override;

    [[nodiscard]] std::unique_ptr<readdy::model::actions::top::EvaluateTopologyReactions>
    evaluateTopologyReactions(scalar timeStep) const override;

    [[nodiscard]] std::unique_ptr<readdy::model::actions::top::BreakBonds>
    breakBonds(scalar timeStep, readdy::model::actions::top::BreakConfig config) const override;

    [[nodiscard]] std::unique_ptr<readdy::model::actions::EvaluateObservables> evaluateObservables() const override;

    [[nodiscard]] std::unique_ptr<readdy::model::actions::MakeCheckpoint>
    makeCheckpoint(std::string base, std::size_t maxNSaves, std::string checkpointFormat) const override;

    [[nodiscard]] std::unique_ptr<readdy::model::actions::InitializeKernel> initializeKernel() const override;
};

}
}