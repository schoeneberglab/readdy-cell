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
 * @file Utils.h
 * @brief « brief description »
 * @author chrisfroe
 * @date 15.10.19
 */

#pragma once

#include <readdy/model/topologies/reactions/StructuralTopologyReaction.h>
#include <readdy/common/index_persistent_vector.h>

namespace readdy::model::actions::top {

template<typename Kernel, typename Topology, typename TopologyRef, typename ParticleData>
void executeStructuralReaction(readdy::util::index_persistent_vector<TopologyRef> &topologies,
                               std::vector<Topology> &newTopologies,
                               TopologyRef &topology,
                               const readdy::model::top::reactions::StructuralTopologyReaction &reaction,
                               std::size_t topologyIdx,
                               ParticleData &particleData,
                               Kernel *kernel) {
    auto result = reaction.execute(*topology, kernel);
    if (!result.empty()) {
        // we had a topology fission, so we need to actually remove the current topology from the
        // data structure
        topologies.erase(topologies.begin() + topologyIdx);
        assert(topology->isDeactivated());
        for (auto &it : result) {
            if (!it.isNormalParticle(*kernel)) {
                newTopologies.push_back(std::move(it));
            }
        }
    } else {
        if (topology->isNormalParticle(*kernel)) {
            auto it = topology->graph().vertices().begin();
            if(it == topology->graph().vertices().end()) {
                throw std::runtime_error("Topology had size 1 but no active vertices!");
            }
            particleData.entry_at((*it)->particleIndex).topology_index = -1;
            topologies.erase(topologies.begin() + topologyIdx);
            assert(topology->isDeactivated());
        }
    }
}

}
