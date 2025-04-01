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
 * This header files contains a base class for all operations on graph topologies, as well as definitions for:
 *   * ChangeParticleType
 *   * ChangeTopologyType
 *   * AddEdge
 *   * RemoveEdge
 *
 * @file Operations.h
 * @brief Definitions for various operations that can be performed on graph topologies.
 * @author clonker
 * @date 13.04.17
 * @copyright BSD-3
 */

#pragma once

#include <memory>
#include "TopologyReactionAction.h"
#include "TopologyReactionActionFactory.h"

namespace readdy::model::top {
class GraphTopology;
namespace reactions::op {

class Operation {
public:
    /**
     * Reference to an operation
     */
    using OperationPtr = std::shared_ptr<Operation>;
    /**
     * Reference to the respective topology reaction action factory
     */
    using FactoryPtr = const actions::TopologyReactionActionFactory *const;
    /**
     * Reference to the respective graph topology
     */
    using TopologyPtr = GraphTopology *const;
    /**
     * pointer type to a topology reaction action
     */
    using ActionPtr = actions::TopologyReactionActionFactory::ActionPtr;

    /**
     * Interface of the create_action method which will create the corresponding action on the selected kernel.
     * @param topology the topology this action should act upon
     * @param factory the factory
     * @return a unique pointer to the action
     */
    virtual ActionPtr create_action(TopologyPtr topology, FactoryPtr factory) const = 0;
};

class ChangeParticleType : public Operation {
public:
    /**
     * Creates an action that changes the particle type of the particle pointed to by vertex.
     * @param vertex the vertex
     * @param type_to the target type
     */
    ChangeParticleType(const Graph::PersistentVertexIndex &vertex, ParticleTypeId type_to) : _vertex(vertex), _type_to(type_to) {}

    /**
     * Create the corresponding action.
     * @param topology the topology
     * @param factory the action factory
     * @return a pointer to the change particle type action
     */
    ActionPtr create_action(TopologyPtr topology, FactoryPtr factory) const override {
        return factory->createChangeParticleType(topology, _vertex, _type_to);
    }

private:
    Graph::PersistentVertexIndex _vertex;
    ParticleTypeId _type_to;
};

class AppendParticle : public Operation {
public:

    AppendParticle(std::vector<Graph::PersistentVertexIndex> neighbors, ParticleTypeId type, const Vec3 &pos)
            : neighbors(std::move(neighbors)), type(type), pos(pos) {};

    ActionPtr create_action(TopologyPtr topology, FactoryPtr factory) const override {
        return factory->createAppendParticle(topology, neighbors, type, pos);
    }

private:
    std::vector<Graph::PersistentVertexIndex> neighbors;
    ParticleTypeId type;
    Vec3 pos;
};

class ChangeParticlePosition : public Operation {
public:
    ChangeParticlePosition(const Graph::PersistentVertexIndex &vertex, Vec3 position) : _vertex(vertex), _pos(position) {};

    ActionPtr create_action(TopologyPtr topology, FactoryPtr factory) const override {
        return factory->createChangeParticlePosition(topology, _vertex, _pos);
    }

private:
    Graph::PersistentVertexIndex _vertex;
    Vec3 _pos;
};

class ChangeTopologyType : public Operation {
public:
    /**
     * Creates an action that changes the topology type of the belonging topology.
     * @param type_to the target type
     */
    explicit ChangeTopologyType(std::string type_to) : _type_to(std::move(type_to)) {};

    /**
     * Create the corresponding action.
     * @param topology the topology
     * @param factory the action factory
     * @return a pointer to the action
     */
    ActionPtr create_action(TopologyPtr topology, FactoryPtr factory) const override {
        return factory->createChangeTopologyType(topology, _type_to);
    }

private:
    std::string _type_to;
};

class AddEdge : public Operation {
public:
    /**
     * Adds the specified edge on the graph.
     * @param edge the edge
     */
    explicit AddEdge(Graph::Edge edge) : _edge(std::move(edge)) {};

    /**
     * Create the corresponding action
     * @param topology the topology
     * @param factory the action factory
     * @return a pointer to the respective action
     */
    ActionPtr create_action(TopologyPtr topology, FactoryPtr factory) const override {
        return factory->createAddEdge(topology, _edge);
    }

private:
    Graph::Edge _edge;
};

class RemoveEdge : public Operation {
public:
    /**
     * Operation for removing the specified edge on the graph.
     * @param edge the edge
     */
    explicit RemoveEdge(Graph::Edge edge) : _edge(std::move(edge)) {};

    /**
     * Create the corresponding action
     * @param topology the topology
     * @param factory the action factory
     * @return a pointer to the respective action
     */
    ActionPtr create_action(TopologyPtr topology, FactoryPtr factory) const override {
        return factory->createRemoveEdge(topology, _edge);
    }

private:
    Graph::Edge _edge;
};

}
}
