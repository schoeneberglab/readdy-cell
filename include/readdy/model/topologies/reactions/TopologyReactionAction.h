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
 * @file TopologyReactionOperation.h
 * @brief << brief description >>
 * @author clonker
 * @date 03.04.17
 * @copyright BSD-3
 */

#pragma once

#include <memory>

#include "../common.h"

namespace readdy::model::top {
class GraphTopology;
namespace reactions::actions {

class TopologyReactionAction {
public:
    /**
     * creates a new topology reaction action w.r.t. the given topology
     * @param topology the topology
     */
    explicit TopologyReactionAction(GraphTopology *topology);

    /**
     * default delete
     */
    virtual ~TopologyReactionAction() = default;

    /**
     * default copy
     */
    TopologyReactionAction(const TopologyReactionAction &) = default;

    /**
     * no copy assign
     */
    TopologyReactionAction &operator=(const TopologyReactionAction &) = delete;

    /**
     * default move
     */
    TopologyReactionAction(TopologyReactionAction &&) = default;

    /**
     * no move assign
     */
    TopologyReactionAction &operator=(TopologyReactionAction &&) = delete;

    /**
     * base method for executing the action
     */
    virtual void execute() = 0;

protected:
    /**
     * a pointer to the topology on which this action should be executed
     */
    GraphTopology *const topology;
};

class ChangeParticleType : public TopologyReactionAction {
public:

    /**
     * Creates an action that changes the particle type of one of the particles in the topology. Since the data
     * structures may vary with the selected kernel, the implementation of do/undo is kernel specific.
     *
     * @param topology the respective topology
     * @param v the vertex pointing to the particle whose type should be changed
     * @param type_to the target type
     */
    ChangeParticleType(GraphTopology *topology, Graph::PersistentVertexIndex v, ParticleTypeId type_to);

protected:
    /**
     * a reference to the vertex
     */
    Graph::PersistentVertexIndex _vertex;
    /**
     * the target type
     */
    ParticleTypeId type_to;
    /**
     * the previous particle type, stored for undo
     */
    ParticleTypeId previous_type;
};

class ChangeParticlePosition : public TopologyReactionAction {
public:
    ChangeParticlePosition(GraphTopology *topology, Graph::PersistentVertexIndex v, Vec3 posTo);

protected:
    Graph::PersistentVertexIndex _vertex;
    Vec3 _posTo;
};

class AppendParticle : public TopologyReactionAction {
public:
    AppendParticle(GraphTopology *topology, std::vector<Graph::PersistentVertexIndex> neighbors,
                   ParticleTypeId type, Vec3 pos);

protected:
    std::vector<Graph::PersistentVertexIndex> neighbors;
    ParticleTypeId type;
    Vec3 pos;
};

class AddEdge : public TopologyReactionAction {
public:
    /**
     * Creates an action that introduces an edge in the graph.
     * @param topology the topology
     * @param edge the edge to introduce
     */
    AddEdge(GraphTopology *topology, Graph::Edge edge);

    void execute() override;

private:
    /**
     * the edge to introduce
     */
    top::Graph::Edge label_edge_;
};

class RemoveEdge : public TopologyReactionAction {
public:
    /**
     * Creates an action that removes an edge in the topology.
     * @param topology the topology
     * @param edge the edge to remove
     */
    RemoveEdge(GraphTopology *topology, Graph::Edge edge);

    /**
     * execute me
     */
    void execute() override;

private:
    /**
     * the edge to remove
     */
    top::Graph::Edge label_edge_;
};

class ChangeTopologyType : public TopologyReactionAction {
public:
    /**
     * Creates an action that changes the topology type of this topology
     * @param topology this topology
     * @param newType the target type
     */
    ChangeTopologyType(GraphTopology *topology, TopologyTypeId newType);

    /**
     * execute me
     */
    void execute() override;

private:
    TopologyTypeId _newType;
};

}
}
