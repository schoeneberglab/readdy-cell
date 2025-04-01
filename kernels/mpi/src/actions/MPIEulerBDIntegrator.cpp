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
 * @file MPIEulerBDIntegrator.cpp
 * @brief Propagate positions of particles according to Brownian Dynamics
 * @author chrisfroe
 * @date 03.06.19
 */

#include <readdy/kernel/mpi/actions/MPIActions.h>
#include <readdy/common/boundary_condition_operations.h>

namespace readdy::kernel::mpi::actions {

void MPIEulerBDIntegrator::perform() {
    if (kernel->domain().isWorkerRank()) {
        const auto &context = kernel->context();
        const auto &pbc = context.periodicBoundaryConditions().data();
        const auto &kbt = context.kBT();
        const auto &box = context.boxSize().data();
        auto& stateModel = kernel->getMPIKernelStateModel();
        auto pd = stateModel.getParticleData();
        for(auto& entry : *pd) {
            if(!entry.is_deactivated() and entry.responsible) {
                const scalar D = context.particleTypes().diffusionConstantOf(entry.type);
                const auto randomDisplacement = std::sqrt(2. * D * _timeStep) *
                                                (readdy::model::rnd::normal3<readdy::scalar>());
                entry.pos += randomDisplacement;
                const auto deterministicDisplacement = entry.force * _timeStep * D / kbt;
                entry.pos += deterministicDisplacement;
                bcs::fixPosition(entry.pos, box, pbc);
            }
        }
    } else {
        readdy::log::trace("MPIEulerBDIntegrator::perform is noop for non workers");
    }
}

}

