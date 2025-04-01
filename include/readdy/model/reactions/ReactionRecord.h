/********************************************************************
 * Copyright © 2018 Computational Molecular Biology Group,          *
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
 * << detailed description >>
 *
 * @file ReactionRecord.h
 * @brief << brief description >>
 * @author clonker
 * @date 07.03.17
 * @copyright BSD-3
 */

#pragma once
#include <spdlog/fmt/ostr.h>
#include <readdy/common/common.h>
#include <readdy/model/Particle.h>
#include <readdy/model/reactions/Reaction.h>

namespace readdy::model::reactions {

struct ReactionRecord {
    int type{0}; // int corresponding to the enum readdy::model::reactions::ReactionType
    std::array<ParticleId, 2> educts{{0, 0}};
    std::array<ParticleId, 2> products{{0, 0}};
    std::array<ParticleTypeId, 2> types_from{{0, 0}};
    Vec3 where{0, 0, 0};
    /**
     * unique reaction id
     */
    ReactionId id{0};

    friend std::ostream &operator<<(std::ostream &os, const ReactionRecord &record) {
        auto reactionType = ReactionType(record.type);
        os << "ReactionRecord[type: " << reactionType;
        switch (reactionType) {
            case ReactionType::Decay: {
                os << ", educt: " << record.educts[0];
                break;
            }
            case ReactionType::Conversion: {
                os << ", educt: " << record.educts[0] << ", product: " << record.products[0];
                break;
            }
            case ReactionType::Fusion: {
                os << ", educts: [" << record.educts[0] << "," << record.educts[1] << "], product: "
                   << record.products[0];
                break;
            }
            case ReactionType::Fission: {
                os << ", educt: " << record.educts[0] << ", products: [" << record.products[0] << ","
                   << record.products[1] << "]";
                break;
            }
            case ReactionType::Enzymatic: {
                os << ", educts: [" << record.educts[0] << "," << record.educts[1] << "]";
                os << ", products: [" << record.products[0] << "," << record.products[1] << "]";
                break;
            }
        }
        os << ", location: " << record.where << "]";
        return os;
    };
};

using ReactionCounts = std::unordered_map<ReactionId, std::size_t>;
using SpatialTopologyReactionCounts = std::unordered_map<ReactionId, std::size_t>;
using StructuralTopologyReactionCounts = std::unordered_map<ReactionId, std::size_t>;

}
