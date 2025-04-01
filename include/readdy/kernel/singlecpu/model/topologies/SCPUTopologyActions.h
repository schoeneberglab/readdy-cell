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
 * @file SCPUTopologyActions.h
 * @brief << brief description >>
 * @author clonker
 * @date 30.01.17
 * @copyright BSD-3
 */

#pragma once

#include <utility>

#include <readdy/model/topologies/potentials/TopologyPotentialActions.h>
#include <readdy/kernel/singlecpu/SCPUStateModel.h>
#include <readdy/model/topologies/Topology.h>
#include <readdy/common/boundary_condition_operations.h>

namespace readdy::kernel::scpu::model::top {

class SCPUCalculateHarmonicBondPotential : public readdy::model::top::pot::CalculateHarmonicBondPotential {

    const harmonic_bond *const potential;
    SCPUParticleData<model::Entry> *const data;
    model::ObservableData *const observableData;

public:
    SCPUCalculateHarmonicBondPotential(const readdy::model::Context *const context,
                                       SCPUParticleData<model::Entry> *const data, model::ObservableData * observableData,
                                       const harmonic_bond *const potential)
            : CalculateHarmonicBondPotential(context), potential(potential), data(data),
              observableData(observableData) {}

    scalar perform(const readdy::model::top::GraphTopology *const topology) override {
        scalar energy = 0;
        for (const auto &bond : potential->getBonds()) {
            if (bond.forceConstant == 0) continue;

            Vec3 forceUpdate{0, 0, 0};
            auto &e1 = data->entry_at(bond.idx1);
            auto &e2 = data->entry_at(bond.idx2);
            const auto x_ij = bcs::shortestDifference(e1.position(), e2.position(), context->boxSize().data(),
                                                      context->periodicBoundaryConditions().data());
            potential->calculateForce(forceUpdate, x_ij, bond);
            e1.force += forceUpdate;
            e2.force -= forceUpdate;
            energy += potential->calculateEnergy(x_ij, bond);
        }
        return energy;
    }

};

class SCPUCalculateHarmonicAnglePotential : public readdy::model::top::pot::CalculateHarmonicAnglePotential {
    const harmonic_angle *const potential;
    SCPUParticleData<model::Entry> *const data;
    model::ObservableData *const observableData;
public:
    SCPUCalculateHarmonicAnglePotential(const readdy::model::Context *const context, SCPUParticleData<model::Entry> *const data,
                                        model::ObservableData * observableData, const harmonic_angle *const potential)
            : CalculateHarmonicAnglePotential(context), potential(potential), data(data),
              observableData(observableData) {}

    scalar perform(const readdy::model::top::GraphTopology *const topology) override {
        scalar energy = 0;

        for (const auto &angle : potential->getAngles()) {
            auto &e1 = data->entry_at(angle.idx1);
            auto &e2 = data->entry_at(angle.idx2);
            auto &e3 = data->entry_at(angle.idx3);
            const auto x_ji = bcs::shortestDifference(e2.pos, e1.pos, context->boxSize().data(),
                                                      context->periodicBoundaryConditions().data());
            const auto x_jk = bcs::shortestDifference(e2.pos, e3.pos, context->boxSize().data(),
                                                      context->periodicBoundaryConditions().data());
            energy += potential->calculateEnergy(x_ji, x_jk, angle);
            potential->calculateForce(e1.force, e2.force, e3.force, x_ji, x_jk, angle);
        }
        return energy;
    }

};

class SCPUCalculateCosineDihedralPotential : public readdy::model::top::pot::CalculateCosineDihedralPotential {
    const cos_dihedral *const potential;
    SCPUParticleData<model::Entry> *const data;
public:
    SCPUCalculateCosineDihedralPotential(const readdy::model::Context *const context, SCPUParticleData<model::Entry> *const data,
                                         const cos_dihedral *const pot)
            : CalculateCosineDihedralPotential(context), potential(pot), data(data) {
    }

    scalar perform(const readdy::model::top::GraphTopology *const topology) override {
        scalar energy = 0;
        for (const auto &dih : potential->getDihedrals()) {
            auto &e_i = data->entry_at(dih.idx1);
            auto &e_j = data->entry_at(dih.idx2);
            auto &e_k = data->entry_at(dih.idx3);
            auto &e_l = data->entry_at(dih.idx4);
            const auto x_ji = bcs::shortestDifference(e_j.pos, e_i.pos, context->boxSize().data(),
                                                      context->periodicBoundaryConditions().data());
            const auto x_kj = bcs::shortestDifference(e_k.pos, e_j.pos, context->boxSize().data(),
                                                      context->periodicBoundaryConditions().data());
            const auto x_kl = bcs::shortestDifference(e_k.pos, e_l.pos, context->boxSize().data(),
                                                      context->periodicBoundaryConditions().data());
            energy += potential->calculateEnergy(x_ji, x_kj, x_kl, dih);
            potential->calculateForce(e_i.force, e_j.force, e_k.force, e_l.force, x_ji, x_kj, x_kl, dih);
        }
        return energy;
    }
};

namespace reactions::op {

class SCPUChangeParticleType : public readdy::model::top::reactions::actions::ChangeParticleType {
    SCPUParticleData<model::Entry> *const data;
public:
    SCPUChangeParticleType(SCPUParticleData<model::Entry> *const data, top::GraphTopology *const topology,
                           const top::Graph::PersistentVertexIndex &v, const ParticleTypeId &type_to)
                           : ChangeParticleType(topology, v, type_to), data(data) {}

    void execute() override {
        const auto idx = topology->graph().vertices().at(_vertex)->particleIndex;
        std::swap(data->entry_at(idx).type, previous_type);
    }

};

class SCPUChangeParticlePosition : public readdy::model::top::reactions::actions::ChangeParticlePosition {
    SCPUParticleData<model::Entry> *const data;
public:
    SCPUChangeParticlePosition(SCPUParticleData<model::Entry> *const data, top::GraphTopology *const topology,
                               const top::Graph::PersistentVertexIndex &v, Vec3 posTo)
                               : ChangeParticlePosition(topology, v, posTo), data(data) {}

    void execute() override {
        const auto idx = topology->graph().vertices().at(_vertex)->particleIndex;
        std::swap(data->entry_at(idx).pos, _posTo);
    }

};

class SCPUAppendParticle : public readdy::model::top::reactions::actions::AppendParticle {
    SCPUParticleData<model::Entry> *const data;
    readdy::model::Particle particle;
    SCPUParticleData<model::Entry>::EntryIndex insertIndex {};
public:
    SCPUAppendParticle(SCPUParticleData<model::Entry> *const data, top::GraphTopology *topology,
                       std::vector<top::Graph::PersistentVertexIndex> neighbors, ParticleTypeId type, Vec3 pos)
            : AppendParticle(topology, std::move(neighbors), type, pos), data(data), particle(pos, type) {};

    void execute() override {
        auto entry = Entry(particle);
        insertIndex = data->addEntry(entry);
        auto firstNeighbor = neighbors[0];
        // append particle forming edge to the first neighbor
        auto ix = topology->appendParticle(insertIndex, firstNeighbor);
        // add remaining edges
        for(std::size_t i = 1; i < neighbors.size(); ++i) {
            topology->addEdge(ix, neighbors[i]);
        }
    }
};

}

}
