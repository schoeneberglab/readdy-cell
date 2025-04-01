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
 * @file TopologyFusionReaction.cpp
 * @brief << brief description >>
 * @author clonker
 * @date 23.06.17
 * @copyright BSD-3
 */

#include <readdy/model/topologies/reactions/SpatialTopologyReaction.h>

#include <regex>
#include <readdy/model/topologies/TopologyRegistry.h>
#include <readdy/model/Utils.h>

namespace readdy::model::top::reactions {

  ReactionId SpatialTopologyReaction::counter = 0;

  SpatialTopologyReaction STRParser::parse(const std::string &descriptor, scalar rate, scalar radius) const {
      SpatialTopologyReaction reaction;
      reaction._rate = rate;
      reaction._radius = radius;

      parse_descriptor(descriptor, reaction);
      return reaction;
  }

  SpatialTopologyReaction STRParser::parse(const std::string &descriptor,
                                           std::function<scalar(const GraphTopology &, const GraphTopology&)> rate_func,
                                           scalar radius) const {
      SpatialTopologyReaction reaction;
      reaction._rate_is_const = false;
      reaction._rate_function = std::move(rate_func);
      reaction._radius = radius;

      parse_descriptor(descriptor, reaction);
      if (reaction.is_topology_particle_reaction()) {
          throw std::invalid_argument("Dynamic rate functions are only implemented for topology-topology "
                                      "reactions. For topology-particle reactions use a constant rate.");
      }
      return reaction;
  }

  void STRParser::parse_descriptor(const std::string &descriptor, SpatialTopologyReaction &reaction) const {
      namespace mutil = readdy::model::util;
      namespace rutil = readdy::util;

      log::trace("begin parsing \"{}\"", descriptor);
      auto arrowPos = descriptor.find(mutil::arrow);
      if (arrowPos == std::string::npos) {
          throw std::invalid_argument(fmt::format(
                  "the descriptor must contain an arrow (\"{}\") to indicate lhs and rhs.", mutil::arrow
          ));
      }
      if (descriptor.find(mutil::arrow, arrowPos + 1) != std::string::npos) {
          throw std::invalid_argument(fmt::format(
                  "the descriptor must not contain more than one arrow (\"{}\").", mutil::arrow
          ));
      }
      auto lhs = descriptor.substr(0, arrowPos);
      auto rhs = descriptor.substr(arrowPos + std::strlen(mutil::arrow), std::string::npos);

      rutil::str::trim(lhs);
      rutil::str::trim(rhs);

      std::string name;
      {
          auto colonPos = lhs.find(':');
          if (colonPos == std::string::npos) {
              throw std::invalid_argument("The descriptor did not contain a colon ':' to specify the end of the name.");
          }
          name = rutil::str::trim_copy(lhs.substr(0, colonPos));
          lhs = rutil::str::trim_copy(lhs.substr(colonPos + 1, std::string::npos));
      }
      reaction._name = name;

      static std::regex particleTypeRegex(R"(\(([^\)]*)\))");
      static std::regex topologyTypeRegex(R"([^(]*)");

      static auto getTop = [](const std::string &s) {
          std::smatch topMatch;
          if (std::regex_search(s, topMatch, topologyTypeRegex)) {
              return rutil::str::trim_copy(topMatch.str());
          }
          throw std::invalid_argument(fmt::format("The term \"{}\" did not contain a topology type.", s));
      };
      static auto getParticleType = [](const std::string &s) {
          std::smatch ptMatch;
          if (std::regex_search(s, ptMatch, particleTypeRegex)) {
              auto pt = rutil::str::trim_copy(ptMatch.str());
              return rutil::str::trim_copy(pt.substr(1, pt.size() - 2));
          }
          throw std::invalid_argument(fmt::format("The term \"{}\" did not contain a particle type.", s));
      };

      static auto treatTerm = [](const std::string &s) {
          return std::make_tuple(getParticleType(s), getTop(s));
      };

      static auto treatSide = [](const std::string &s) {
          auto plusPos = s.find('+');
          if (plusPos == std::string::npos) {
              throw std::invalid_argument("The left hand side of the topology reaction did not contain a '+'.");
          }
          auto educt1 = rutil::str::trim_copy(s.substr(0, plusPos));
          auto educt2 = rutil::str::trim_copy(s.substr(plusPos + 1, std::string::npos));

          std::string t1, t2, p1, p2;
          std::tie(p1, t1) = treatTerm(educt1);
          std::tie(p2, t2) = treatTerm(educt2);

          return std::make_tuple(p1, t1, p2, t2);
      };

      std::string lhs_p1, lhs_t1, lhs_p2, lhs_t2;
      {
          std::tie(lhs_p1, lhs_t1, lhs_p2, lhs_t2) = treatSide(lhs);
      }

      std::string rhs_p1, rhs_t1, rhs_p2, rhs_t2;
      bool rhs_fusion {false};
      {
          auto plusPos = rhs.find('+');
          rhs_fusion = plusPos == std::string::npos;
          if (plusPos == std::string::npos) {
              // fusion type
              std::string fuse;
              std::tie(fuse, rhs_t1) = treatTerm(rhs);
              rhs_t2 = "";

              auto separatorPos = fuse.find(mutil::bond);
              if (separatorPos == std::string::npos) {
                  throw std::invalid_argument(fmt::format(
                          "The right-hand side was of fusion type but there was no bond \"{}\" defined.", mutil::bond
                  ));
              }

              rhs_p1 = rutil::str::trim_copy(fuse.substr(0, separatorPos));
              rhs_p2 = rutil::str::trim_copy(fuse.substr(separatorPos + std::strlen(mutil::bond), std::string::npos));
          } else {
              // enzymatic type
              std::tie(rhs_p1, rhs_t1, rhs_p2, rhs_t2) = treatSide(rhs);
          }

          log::trace(R"(got lhs with toplogies "{}" and "{}", particle types "{}" and "{}")", lhs_t1, lhs_t2, lhs_p1,
                     lhs_p2);
          log::trace(R"(got rhs with topologies "{}" and "{}", particles "{}" and "{}")", rhs_t1, rhs_t2, rhs_p1, rhs_p2);
      }

      const auto &particle_types = _topology_registry.get().particleTypeRegistry();

      if(lhs_t2.empty()) {
          // we are in the topology-particle case
          reaction._top_types = std::make_tuple(_topology_registry.get().idOf(lhs_t1), EmptyTopologyId);
          reaction._types = std::make_tuple(particle_types.idOf(lhs_p1), particle_types.idOf(lhs_p2));
          reaction._types_to = std::make_tuple(particle_types.idOf(rhs_p1), particle_types.idOf(rhs_p2));
          reaction._top_types_to = std::make_tuple(_topology_registry.get().idOf(rhs_t1), EmptyTopologyId);
          if(rhs_fusion) {
              // we are in the fusion case
              reaction._mode = STRMode::TP_FUSION;
          } else {
              // we are in the enzymatic case
              reaction._mode = STRMode::TP_ENZYMATIC;
          }
      } else {
          // we are in the topology-topology case
          reaction._top_types = std::make_tuple(_topology_registry.get().idOf(lhs_t1),
                                                _topology_registry.get().idOf(lhs_t2));
          reaction._types = std::make_tuple(particle_types.idOf(lhs_p1), particle_types.idOf(lhs_p2));
          reaction._types_to = std::make_tuple(particle_types.idOf(rhs_p1), particle_types.idOf(rhs_p2));
          if(rhs_fusion) {
              // we are in the fusion case
              std::smatch option_match;
              std::regex option_rx(R"(\[(.*?)\])");
              // check for options
              if(std::regex_search(rhs, option_match, option_rx)) {
                  auto option_str = option_match.str();
                  std::vector<std::string> options = parse_options(
                          option_str.substr(1, option_str.length()-2)
                  );
                  if (!has_option_allow_self(options)) {
                      throw std::runtime_error("Option \"self=true\" is missing. This option is currently required for all valid options. If you don't want to allow self fusion, ommit the option block (in square brackets).");
                  }
                  reaction._mode = STRMode::TT_FUSION_ALLOW_SELF;
                  int d = get_distance(options);
                  if (d == -1) reaction._min_graph_distance = 0;
                  else reaction._min_graph_distance = (unsigned) d;
              } else {
                  reaction._mode = STRMode::TT_FUSION;
              }
              reaction._top_types_to = std::make_tuple(_topology_registry.get().idOf(rhs_t1), EmptyTopologyId);
          } else {
              // we are in the enzymatic case
              reaction._mode = STRMode::TT_ENZYMATIC;
              reaction._top_types_to = std::make_tuple(_topology_registry.get().idOf(rhs_t1),
                                                       _topology_registry.get().idOf(rhs_t2));
          }
      }
  }

  std::vector<std::string> STRParser::parse_options(const std::string &option_str) const {
    std::vector<std::string> options;
    size_t pos = 0;
    std::string s = option_str.substr();
    std::string o;
    while ((pos = s.find(",")) != std::string::npos) {
      o = s.substr(0, pos);
      readdy::util::str::trim(o);
      options.push_back(o);
      s.erase(0, pos + 1);
    }
    readdy::util::str::trim(s);
    options.push_back(s);
    return options;
  }

  bool STRParser::has_option_allow_self(const std::vector<std::string> &options) const {
    auto it = std::find(options.begin(),
			options.end(),
			"self=true");
    if (it != options.end()) return true;
    return false;
  }

  int STRParser::get_distance(const std::vector<std::string> &options) const {    
    for (auto &o: options) {
      if(o.find("distance>") == std::string::npos) continue;
      auto dist = std::stoi(o.substr(9));
      return dist;
    }
    return -1;
  }
}
