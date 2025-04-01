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
 * @file SCPUEvaluateTopologyReactions.cpp
 * @brief << brief description >>
 * @author clonker
 * @date 12.06.17
 * @copyright BSD-3
 */

#include <readdy/kernel/singlecpu/actions/SCPUEvaluateTopologyReactions.h>
#include <readdy/common/algorithm.h>
#include <readdy/common/boundary_condition_operations.h>
#include <readdy/model/actions/Utils.h>

namespace readdy::kernel::scpu::actions::top {

/**
 * Struct holding information about a topology reaction event.
 * If:
 *   - reaction_idx == -1: external topology reaction
 *   - reaction_idx >= 0: internal (fission/conversion-type) topology reaction
 */
struct SCPUEvaluateTopologyReactions::TREvent {
    using index_type = std::size_t;

    rate_t cumulativeRate{0};
    rate_t rate{0};
    std::size_t topology_idx{0};
    std::ptrdiff_t topology_idx2{-1};
    std::size_t reaction_idx{0};
    ParticleTypeId t1{0}, t2{0};
    // idx1 is always the particle that belongs to a topology
    index_type idx1{0}, idx2{0};
    bool spatial{false};
    readdy::ReactionId reactionId{0};

};

SCPUEvaluateTopologyReactions::SCPUEvaluateTopologyReactions(SCPUKernel *const kernel, scalar timeStep)
        : EvaluateTopologyReactions(timeStep), kernel(kernel) {}

template<bool approximated>
bool performReactionEvent(const scalar rate, const scalar timeStep) {
    if (approximated) {
        return readdy::model::rnd::uniform_real() < rate * timeStep;
    } else {
        return readdy::model::rnd::uniform_real() < 1 - std::exp(-rate * timeStep);
    }
}

void SCPUEvaluateTopologyReactions::perform() {
    auto &model = kernel->getSCPUKernelStateModel();
    const auto &context = kernel->context();
    auto &topologies = model.topologies();

    if (context.recordReactionCounts()) {
        model.resetTopologyReactionCounts();
    }

    if (!topologies.empty()) {

        auto events = gatherEvents();

        if (!events.empty()) {
            std::vector<readdy::model::top::GraphTopology> new_topologies;

            auto shouldEval = [this](const TREvent &event) {
                return performReactionEvent<false>(event.rate, _timeStep);
            };
            auto depending = [this](const TREvent &e1, const TREvent &e2) {
                return eventsDependent(e1, e2);
            };

            auto eval = [&](const TREvent &event) {
                auto &topology = topologies.at(event.topology_idx);
                if (topology->isDeactivated()) {
                    throw std::logic_error(
                            fmt::format("deactivated topology with idx {} for {} event", event.topology_idx,
                                        event.spatial ? "spatial" : "structural"));
                }
                assert(!topology->isDeactivated());
                if (!event.spatial) {
                    handleStructuralReactionEvent(topologies, new_topologies, event, topology);
                    if (context.recordReactionCounts()) {
                        kernel->getSCPUKernelStateModel().structuralReactionCounts()[event.reactionId] += 1;
                    }
                } else {
                    if (event.topology_idx2 >= 0) {
                        auto &top2 = topologies.at(static_cast<std::size_t>(event.topology_idx2));
                        if (event.topology_idx2 >= 0 && topologyDeactivated(static_cast<size_t>(event.topology_idx2))) {
                            throw std::logic_error(
                                    fmt::format("encountered deactivated topology here, oh no (ix {}).",
                                                event.topology_idx2));
                        }
                        handleTopologyTopologyReaction(topology, top2, event);
                    } else {
                        handleTopologyParticleReaction(topology, event);
                    }
                    if (context.recordReactionCounts()) {
                        kernel->getSCPUKernelStateModel().spatialReactionCounts()[event.reactionId] += 1;
                    }
                }
            };

            algo::performEvents(events, shouldEval, depending, eval);

            if (!new_topologies.empty()) {
                for (auto &&top : new_topologies) {
                    if (!top.isNormalParticle(*kernel)) {
                        // we have a new topology here, update data accordingly.
                        top.updateReactionRates(context.topologyRegistry().structuralReactionsOf(top.type()));
                        top.configure();
                        model.insert_topology(std::move(top));
                    } else {
                        // if we have a single particle that is not of flavor topology, remove from topology structure!
                        auto it = top.graph().vertices().begin();
                        while (it != top.graph().vertices().end() && !it->deactivated()) {
                            ++it;
                        }
                        if (it == top.graph().vertices().end()) {
                            throw std::logic_error("Graph had size 1 but no active vertex!");
                        }
                        model.getParticleData()->entry_at((*it)->particleIndex).topology_index = -1;
                    }
                }
            }
        }
    }

}

SCPUEvaluateTopologyReactions::topology_reaction_events SCPUEvaluateTopologyReactions::gatherEvents() {
    const auto &context = kernel->context();
    const auto &topology_registry = context.topologyRegistry();
    topology_reaction_events events;
    {
        rate_t current_cumulative_rate = 0;
        std::size_t topology_idx = 0;
        for (auto &top : kernel->getSCPUKernelStateModel().topologies()) {
            if (!top->isDeactivated()) {
                std::size_t reaction_idx = 0;
                for (const auto &reaction : topology_registry.structuralReactionsOf(top->type())) {
                    TREvent event{};
                    event.rate = top->rates().at(reaction_idx);
                    event.cumulativeRate = event.rate + current_cumulative_rate;
                    current_cumulative_rate = event.cumulativeRate;
                    event.topology_idx = topology_idx;
                    event.reaction_idx = reaction_idx;
                    event.reactionId = reaction.id();

                    events.push_back(event);
                    ++reaction_idx;
                }
            }
            ++topology_idx;
        }


        if (!topology_registry.spatialReactionRegistry().empty()) {
            const auto &box = context.boxSize().data();
            const auto &pbc = context.periodicBoundaryConditions().data();
            const auto &stateModel = kernel->getSCPUKernelStateModel();
            const auto &data = *stateModel.getParticleData();
            const auto &nl = *stateModel.getNeighborList();
            auto &topologies = stateModel.topologies();

            for (auto cell = 0_z; cell < nl.nCells(); ++cell) {
                for (auto it = nl.particlesBegin(cell); it != nl.particlesEnd(cell); ++it) {
                    auto pidx = *it;
                    auto &entry = data.entry_at(pidx);
                    const auto tidx1 = entry.topology_index;
                    if (entry.deactivated || (tidx1 >= 0 && topologyDeactivated(static_cast<std::size_t>(tidx1)))) {
                        continue;
                    }
                    TopologyTypeId tt1 =
                            tidx1 >= 0 ? stateModel.topologies().at(static_cast<std::size_t>(tidx1))->type()
                                       : static_cast<TopologyTypeId>(-1);

                    nl.forEachNeighbor(it, cell, [&](std::size_t neighborIdx) {
                        auto &neighbor = data.entry_at(neighborIdx);
                        if (!neighbor.deactivated) {
                            const auto tidx2 = neighbor.topology_index;

                            if (tidx2 >= 0 && topologyDeactivated(static_cast<std::size_t>(tidx2))) {
                                return;
                            }

                            TopologyTypeId tt2 =
                                    tidx2 >= 0 ? stateModel.topologies().at(static_cast<std::size_t>(tidx2))->type()
                                               : static_cast<TopologyTypeId>(-1);
                            if (tt1 == -1 && tt2 == -1) return;
                            const auto &reactions = topology_registry.spatialReactionsByType(entry.type, tt1,
                                                                                             neighbor.type, tt2);
                            const auto distSquared = bcs::distSquared(entry.pos, neighbor.pos, box, pbc);
                            std::size_t reaction_index = 0;
                            for (const auto &reaction : reactions) {
                                if (distSquared < reaction.radius() * reaction.radius()) {
                                    TREvent event{};
                                    event.reactionId = reaction.id();
                                    current_cumulative_rate = event.cumulativeRate;
                                    switch (reaction.mode()) {
                                        case readdy::model::top::reactions::STRMode::TT_FUSION:
                                            if (tidx1 >= 0 && tidx2 >= 0 && tidx1 != tidx2) {
                                                event.topology_idx = static_cast<std::size_t>(tidx1);
                                                event.topology_idx2 = tidx2;
                                                event.t1 = entry.type;
                                                event.t2 = neighbor.type;
                                                event.idx1 = pidx;
                                                event.idx2 = neighborIdx;
                                                event.reaction_idx = reaction_index;
                                                event.spatial = true;
                                                event.rate = reaction.rate(
                                                        *topologies.at(event.topology_idx),
                                                        *topologies.at(event.topology_idx2)
                                                );
                                                event.cumulativeRate = event.rate + current_cumulative_rate;


                                                events.push_back(event);
                                            }
                                            break;
                                        case readdy::model::top::reactions::STRMode::TT_ENZYMATIC:
                                            if (tidx1 >= 0 && tidx2 >= 0) {
                                                event.topology_idx = static_cast<std::size_t>(tidx1);
                                                event.topology_idx2 = tidx2;
                                                event.t1 = entry.type;
                                                event.t2 = neighbor.type;
                                                event.idx1 = pidx;
                                                event.idx2 = neighborIdx;
                                                event.reaction_idx = reaction_index;
                                                event.spatial = true;
                                                event.rate = reaction.rate(
                                                        *topologies.at(event.topology_idx),
                                                        *topologies.at(event.topology_idx2)
                                                );
                                                event.cumulativeRate = event.rate + current_cumulative_rate;

                                                events.push_back(event);
                                            }
                                            break;
                                        case readdy::model::top::reactions::STRMode::TT_FUSION_ALLOW_SELF:
                                            if (tidx1 >= 0 && tidx2 >= 0) {
                                                if (tidx1 == tidx2) {
                                                    const auto &t1 = topologies.at(static_cast<std::size_t>(tidx1));
                                                    const auto &gr = t1->graph();
                                                    auto v1 = t1->vertexIteratorForParticle(pidx);
                                                    auto v2 = t1->vertexIteratorForParticle(neighborIdx);
                                                    auto d = gr.graphDistance(v1, v2);
                                                    if ((d != -1 && d <= reaction.min_graph_distance())
                                                         || (!context.legacyTopologySelfFusion() && d == 1)) {
                                                        break;
                                                    }
                                                }
                                                event.topology_idx = static_cast<std::size_t>(tidx1);
                                                event.topology_idx2 = tidx2;
                                                event.t1 = entry.type;
                                                event.t2 = neighbor.type;
                                                event.idx1 = pidx;
                                                event.idx2 = neighborIdx;
                                                event.reaction_idx = reaction_index;
                                                event.spatial = true;
                                                event.rate = reaction.rate(
                                                        *topologies.at(event.topology_idx),
                                                        *topologies.at(event.topology_idx2)
                                                );
                                                event.cumulativeRate = event.rate + current_cumulative_rate;
                                                events.push_back(event);
                                            }
                                            break;
                                        case readdy::model::top::reactions::STRMode::TP_ENZYMATIC: // fall through
                                        case readdy::model::top::reactions::STRMode::TP_FUSION:
                                            event.rate = reaction.rate();
                                            event.cumulativeRate = event.rate + current_cumulative_rate;
                                            if (tidx1 >= 0 && tidx2 < 0) {
                                                event.topology_idx = static_cast<std::size_t>(tidx1);
                                                event.t1 = entry.type;
                                                event.t2 = neighbor.type;
                                                event.idx1 = pidx;
                                                event.idx2 = neighborIdx;
                                                event.reaction_idx = reaction_index;
                                                event.spatial = true;

                                                events.push_back(event);
                                            } else if (tidx1 < 0 && tidx2 >= 0) {
                                                event.topology_idx = static_cast<std::size_t>(tidx2);
                                                event.t1 = neighbor.type;
                                                event.t2 = entry.type;
                                                event.idx1 = neighborIdx;
                                                event.idx2 = pidx;
                                                event.reaction_idx = reaction_index;
                                                event.spatial = true;

                                                events.push_back(event);
                                            }
                                            break;
                                    }
                                }
                                ++reaction_index;
                            }
                        }
                    });
                }
            }
        }
    }
    return events;
}

bool SCPUEvaluateTopologyReactions::eventsDependent(const SCPUEvaluateTopologyReactions::TREvent &evt1,
                                                    const SCPUEvaluateTopologyReactions::TREvent &evt2) const {
    if (evt1.topology_idx == evt2.topology_idx || evt1.topology_idx == evt2.topology_idx2) {
        return true;
    }
    auto tpWithSameP = evt1.topology_idx2 < 0 && evt2.topology_idx2 < 0 && evt1.idx2 == evt2.idx2;
    if (tpWithSameP) {
        if (!evt1.spatial && !evt2.spatial) {
            return false;
        }
        return true;
    }
    return evt1.topology_idx2 >= 0 &&
           (evt1.topology_idx2 == evt2.topology_idx || evt1.topology_idx2 == evt2.topology_idx2);
}

void SCPUEvaluateTopologyReactions::handleTopologyTopologyReaction(SCPUStateModel::topology_ref &t1,
                                                                   SCPUStateModel::topology_ref &t2,
                                                                   const SCPUEvaluateTopologyReactions::TREvent &event) {
    const auto &context = kernel->context();
    const auto &top_registry = context.topologyRegistry();
    const auto &reaction = top_registry.spatialReactionsByType(event.t1, t1->type(), event.t2, t2->type()).at(
            event.reaction_idx);

    auto &model = kernel->getSCPUKernelStateModel();
    auto &data = *model.getParticleData();

    auto &entry1 = data.entry_at(event.idx1);
    auto &entry2 = data.entry_at(event.idx2);
    auto &entry1Type = entry1.type;
    auto &entry2Type = entry2.type;
    auto top_type_to1 = reaction.top_type_to1();
    auto top_type_to2 = reaction.top_type_to2();

    if (reaction.is_fusion() && event.topology_idx == event.topology_idx2) {
        // self-fusion

        // introduce edge if not already present
        auto ix1 = t1->vertexIndexForParticle(event.idx1);
        auto ix2 = t2->vertexIndexForParticle(event.idx2);

        if(!t1->containsEdge(ix1, ix2)) {
            t1->addEdge(ix1, ix2);
        }
        t1->type() = reaction.top_type_to1();

        if (entry1Type == reaction.type1()) {
            entry1Type = reaction.type_to1();
            entry2Type = reaction.type_to2();
        } else {
            entry1Type = reaction.type_to2();
            entry2Type = reaction.type_to1();
        }
    } else {
        if (entry1Type == reaction.type1() && t1->type() == reaction.top_type1()) {
            entry1Type = reaction.type_to1();
            entry2Type = reaction.type_to2();
        } else {
            std::swap(top_type_to1, top_type_to2);
            entry1Type = reaction.type_to2();
            entry2Type = reaction.type_to1();
        }

        // topology - topology fusion
        if (reaction.is_fusion()) {
            // merge topologies
            for (auto pidx : t2->particleIndices()) {
                data.entry_at(pidx).topology_index = event.topology_idx;
            }
            auto &topologies = model.topologies();
            t1->appendTopology(*t2, event.idx2, event.idx1, reaction.top_type_to1());
            topologies.erase(topologies.begin() + event.topology_idx2);
        } else {
            t1->type() = top_type_to1;
            t2->type() = top_type_to2;

            t2->updateReactionRates(context.topologyRegistry().structuralReactionsOf(t2->type()));
            t2->updateSpatialReactionRates(
                    context.topologyRegistry().spatialReactionsByType(event.t2, t2->type(),
                                                                      event.t1, t1->type()),
                    t1
            );
            t2->configure();
        }
    }
    t1->updateReactionRates(context.topologyRegistry().structuralReactionsOf(t1->type()));
    t1->updateSpatialReactionRates(
            context.topologyRegistry().spatialReactionsByType(event.t1, t1->type(),
                                                              event.t2, t2->type()),
            t2
    );
    t1->configure();
}

void SCPUEvaluateTopologyReactions::handleTopologyParticleReaction(SCPUStateModel::topology_ref &topology,
                                                                   const SCPUEvaluateTopologyReactions::TREvent &event) {
    const auto &context = kernel->context();
    const auto &top_registry = context.topologyRegistry();
    const auto &reaction = top_registry.spatialReactionsByType(event.t1, topology->type(), event.t2,
                                                               EmptyTopologyId).at(event.reaction_idx);

    auto &model = kernel->getSCPUKernelStateModel();
    auto &data = *model.getParticleData();

    auto &entry1 = data.entry_at(event.idx1);
    auto &entry2 = data.entry_at(event.idx2);
    auto &entry1Type = entry1.type;
    auto &entry2Type = entry2.type;
    if (entry1Type == reaction.type1()) {
        entry1Type = reaction.type_to1();
        entry2Type = reaction.type_to2();
    } else {
        entry1Type = reaction.type_to2();
        entry2Type = reaction.type_to1();
    }
    if (event.topology_idx2 < 0) {
        entry1.topology_index = event.topology_idx;
        entry2.topology_index = event.topology_idx;

        if (reaction.is_fusion()) {
            topology->appendParticle(event.idx2, event.idx1);
        }
    } else {
        throw std::logic_error("this branch should never be reached as topology-topology reactions are "
                               "handeled in a different method");
    }
    if (topology->type() == reaction.top_type1()) {
        topology->type() = reaction.top_type_to1();
    } else {
        topology->type() = reaction.top_type_to2();
    }
    topology->updateReactionRates(context.topologyRegistry().structuralReactionsOf(topology->type()));
    topology->configure();
}

void SCPUEvaluateTopologyReactions::handleStructuralReactionEvent(SCPUStateModel::topologies_vec &topologies,
                                                                  std::vector<SCPUStateModel::topology> &new_topologies,
                                                                  const SCPUEvaluateTopologyReactions::TREvent &event,
                                                                  SCPUStateModel::topology_ref &topology) const {
    const auto &context = kernel->context();
    const auto &reactions = context.topologyRegistry().structuralReactionsOf(topology->type());
    const auto &reaction = reactions.at(static_cast<std::size_t>(event.reaction_idx));
    readdy::model::actions::top::executeStructuralReaction(topologies, new_topologies, topology, reaction,
                                                           event.topology_idx,
                                                           *(kernel->getSCPUKernelStateModel().getParticleData()),
                                                           kernel);
}

bool SCPUEvaluateTopologyReactions::topologyDeactivated(std::size_t index) const {
    return kernel->getSCPUKernelStateModel().topologies().at(index)->isDeactivated();
}

}
