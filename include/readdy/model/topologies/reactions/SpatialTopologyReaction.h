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
 * @file TopologyFusionReaction.h
 * @brief << brief description >>
 * @author clonker
 * @date 23.06.17
 * @copyright BSD-3
 */

#pragma once

#include <readdy/common/common.h>
#include <readdy/common/ParticleTypeTuple.h>
#include <readdy/model/topologies/TopologyParticleTypeMap.h>
#include <readdy/common/string.h>

namespace readdy::model {
class Particle;
namespace top {
class TopologyRegistry;
class GraphTopology;
namespace reactions {

/**
 * TT - Topology-Topology
 * TP - Topology-Particle
 */
enum class STRMode {
   TT_ENZYMATIC = 0, TT_FUSION, TT_FUSION_ALLOW_SELF, TP_ENZYMATIC,
   TP_FUSION
};

class STRParser;

class SpatialTopologyReaction {
public:
    /**
     * rate function type, yielding a rate
     */

    using rate_function = std::function<scalar(const GraphTopology &,
                                               const GraphTopology &)>;


    SpatialTopologyReaction(std::string name, util::particle_type_pair types, topology_type_pair top_types,
                            util::particle_type_pair types_to, topology_type_pair top_types_to, scalar rate,
                            scalar radius, STRMode mode)
            : _name(std::move(name)), _types(std::move(types)), _types_to(std::move(types_to)), _rate(rate),
              _radius(radius), _mode(mode), _top_types(std::move(top_types)), _top_types_to(std::move(top_types_to)),
              _rate_is_const(true)
    {
        if (_rate <=0) {
            throw std::invalid_argument(fmt::format("The rate of an structural topology reaction ({}) "
                                                    "should always be positive", _name));
        }
    };

    SpatialTopologyReaction(std::string name, util::particle_type_pair types, topology_type_pair top_types,
                            util::particle_type_pair types_to, topology_type_pair top_types_to,
                            rate_function rate_func,
                            scalar radius, STRMode mode)
            : _name(std::move(name)), _types(std::move(types)), _types_to(std::move(types_to)),
              _rate_function(std::move(rate_func)),
              _radius(radius), _mode(mode), _top_types(std::move(top_types)), _top_types_to(std::move(top_types_to)),
              _rate_is_const(false)
    {
        if (std::get<1>(_top_types) == EmptyTopologyId) {
            throw std::invalid_argument("Dynamic rate functions are only implemented for topology-topology "
                                        "reactions. For topology-particle reactions use a constant rate.");
        }
    };

    ~SpatialTopologyReaction() = default;

    SpatialTopologyReaction(const SpatialTopologyReaction &) = default;

    SpatialTopologyReaction &operator=(const SpatialTopologyReaction &) = default;

    SpatialTopologyReaction(SpatialTopologyReaction &&) = default;

    SpatialTopologyReaction &operator=(SpatialTopologyReaction &&) = default;

    std::string_view name() const {
        return _name;
    }

    const ParticleTypeId type1() const {
        return std::get<0>(_types);
    }

    const ParticleTypeId type2() const {
        return std::get<1>(_types);
    }

    const util::particle_type_pair &types() const {
        return _types;
    }

    const ParticleTypeId type_to1() const {
        return std::get<0>(_types_to);
    }

    const ParticleTypeId type_to2() const {
        return std::get<1>(_types_to);
    }

    const util::particle_type_pair &types_to() const {
        return _types_to;
    }

    const TopologyTypeId top_type1() const {
        return std::get<0>(_top_types);
    }

    const TopologyTypeId top_type2() const {
        return std::get<1>(_top_types);
    }

    const TopologyTypeId top_type_to1() const {
        return std::get<0>(_top_types_to);
    }

    const TopologyTypeId top_type_to2() const {
        return std::get<1>(_top_types_to);
    }

    bool is_topology_particle_reaction() const {
        return top_type2() == EmptyTopologyId;
    }

    bool is_topology_topology_reaction() const {
        return !is_topology_particle_reaction();
    }

    const bool is_enzymatic() const {
        return _mode == STRMode::TT_ENZYMATIC || _mode == STRMode::TP_ENZYMATIC;
    }

    const bool is_fusion() const {
      return _mode == STRMode::TT_FUSION || _mode == STRMode::TT_FUSION_ALLOW_SELF || _mode == STRMode::TP_FUSION;
    }

    [[nodiscard]] scalar rate(const GraphTopology &topology1,
                              const GraphTopology &topology2) const {
        if (_rate_is_const) return _rate;
        scalar current_rate = _rate_function(topology1, topology2);
        if (current_rate < 0.0) {
            throw std::runtime_error(
                    fmt::format("Your rate function for reaction {} produced a rate "
                                "smaller than 0, make sure rate functions only return values >= 0.",
                                _name)
            );
        }
        return current_rate;
    }

    [[nodiscard]] scalar rate() const {
        if (!_rate_is_const) {
            throw std::runtime_error("rate is not static, use method rate with two topologies as arguments");
        }
        return _rate;
    }

    const scalar radius() const {
        return _radius;
    }

    const bool allow_self_connection() const {
      return _mode == STRMode::TT_FUSION_ALLOW_SELF;
    }

    const STRMode &mode() const {
        return _mode;
    }

    ReactionId id() const {
        return _id;
    }

  const std::size_t min_graph_distance() const {
    return _min_graph_distance;
  }
  
private:

    static ReactionId counter;
    ReactionId _id {counter++};

    friend class STRParser;

    SpatialTopologyReaction() = default;

    std::string _name;
    util::particle_type_pair _types;
    util::particle_type_pair _types_to;
    topology_type_pair _top_types;
    topology_type_pair _top_types_to;
    scalar _rate{0};
    scalar _radius{0};
    STRMode _mode{STRMode::TP_ENZYMATIC};
    unsigned _min_graph_distance{0};
    bool _rate_is_const{true};
    rate_function _rate_function;
};

class STRParser {
public:

    explicit STRParser(const TopologyRegistry &registry) : _topology_registry(registry) {};

    /**
     *  Pass descriptor of form for Topology<->Topology
     * - fusion type:
     *     name: T1 (p1) + T2 (p2) -> T3 (p3--p4) [self=true|false]
     * - enzymatic type:
     *     name: T1 (p1) + T2 (p2) -> T3 (p3) + T4 (p4)
     *
     * and for Topology<->Particle
     * - fusion type:
     *     name: T1 (p1) + (p2) -> T2 (p3--p4)
     * - enzymatic type:
     *     name: T1 (p1) + (p2) -> T2 (p3) + (p4)
     *
     * @param descriptor the descriptor
     * @param rate the rate
     * @param radius the radius
     * @return the parsed reaction object
     */
    SpatialTopologyReaction parse(const std::string &descriptor, scalar rate, scalar radius) const;

    SpatialTopologyReaction parse(const std::string &descriptor,
                                  std::function<scalar(const GraphTopology &,
                                                       const GraphTopology &)> rate_func,
                                  scalar radius) const;

    void parse_descriptor(const std::string &descriptor, SpatialTopologyReaction &reaction) const;


    /**
     * Take complete option string and split at commata into single option strings.
     */
    std::vector<std::string> parse_options(const std::string &option_str) const;
  
    /**
     * Search options vector for self=true option
     * @return true if option is present, false otherwise.
     */
    bool has_option_allow_self(const std::vector<std::string> &options) const;
  
    /**
     * Search options vector for distance option.
     * @return -1 if no distnace option was found, or parsed distance value otherwise.
     */
    int get_distance(const std::vector<std::string> &options) const;
  
  
private:
    std::reference_wrapper<const TopologyRegistry> _topology_registry;
};
  
}
}
}
