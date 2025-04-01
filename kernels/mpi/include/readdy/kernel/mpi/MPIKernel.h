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
 * @file MPIKernel.h
 * @brief Header file for readdy kernel that parallelizes using the message passing interface (MPI)
 * @author chrisfroe
 * @date 28.05.2019
 */

#pragma once


#include <readdy/model/Kernel.h>
#include <readdy/kernel/mpi/MPIStateModel.h>
#include <readdy/kernel/mpi/actions/MPIActionFactory.h>
#include <readdy/kernel/mpi/observables/MPIObservableFactory.h>
#include <readdy/kernel/mpi/model/MPIDomain.h>
#include <readdy/common/Timer.h>

#include <utility>

namespace readdy::kernel::mpi {

// fixme MPIKernel cannot be constructed as dynamically loaded plugin in a useful way currently
// because only the ctor with context provides a ready-to-use kernel,
// the default ctor provides a kernel that is not (/cannot be) initialized
class MPIKernel : public readdy::model::Kernel {
public:
    static const std::string name;

    MPIKernel();

    ~MPIKernel() override = default;

    explicit MPIKernel(const readdy::model::Context &ctx);

    MPIKernel(const MPIKernel &) = delete;

    MPIKernel &operator=(const MPIKernel &) = delete;

    MPIKernel(MPIKernel &&) = delete;

    MPIKernel &operator=(MPIKernel &&) = delete;

    // factory method
    static readdy::model::Kernel *create();

    // factory method with context
    static readdy::model::Kernel *create(const readdy::model::Context &);

    const MPIStateModel &getMPIKernelStateModel() const {
        return _stateModel;
    }

    MPIStateModel &getMPIKernelStateModel() {
        return _stateModel;
    }

    const readdy::model::StateModel &stateModel() const override {
        return _stateModel;
    }

    readdy::model::StateModel &stateModel() override {
        return _stateModel;
    }

    const readdy::model::actions::ActionFactory &actions() const override {
        return _actions;
    }

    readdy::model::actions::ActionFactory &actions() override {
        return _actions;
    }

    const readdy::model::observables::ObservableFactory &observe() const override {
        return _observables;
    }

    readdy::model::observables::ObservableFactory &observe() override {
        return _observables;
    }

    const readdy::model::top::TopologyActionFactory *const getTopologyActionFactory() const override {
        return nullptr;
    }

    readdy::model::top::TopologyActionFactory *const getTopologyActionFactory() override {
        return nullptr;
    }

    bool supportsGillespie() const override {
        return false;
    }

    const model::MPIDomain &domain() const {
        return _domain;
    }

    const MPI_Comm &commUsedRanks() const {
        return _commUsedRanks;
    }

    virtual void evaluateObservables(TimeStep t) override {
        if (not _domain.isIdleRank()) {
            _signal(t);
        }
    }

protected:
    // order here is order of initialization
    // https://en.cppreference.com/w/cpp/language/initializer_list#Initialization_order
    // domain needs context
    // data needs domain
    // state model needs data and domain
    model::MPIDomain _domain;
    MPIStateModel::Data _data;
    MPIStateModel _stateModel;
    actions::MPIActionFactory _actions;
    observables::MPIObservableFactory _observables;


    // The communicator for the subgroup of actually used workers
    MPI_Comm _commUsedRanks = MPI_COMM_WORLD;
};

}

extern "C" const char *name();

extern "C" readdy::model::Kernel *createKernel();

