/**
 * « detailed description »
 *
 * @file CPUActionReaction.h
 * @brief « brief description »
 * @author Eric Arkfeld
* @date 11.24.24
 */

#pragma once

#include <readdy/model/actions/Actions.h>
#include <readdy/kernel/cpu/CPUKernel.h>

namespace readdy::kernel::cpu::actions::top {

class CPUActionReaction : public readdy::model::actions::top::ActionReaction {
public:
    explicit CPUActionReaction(CPUKernel *kernel, readdy::model::actions::top::ReactionConfig config)
            : ActionReaction(std::move(config)), kernel(kernel) {}

    void perform() override {
        auto &topologies = kernel->getCPUKernelStateModel().topologies();
        auto &model = kernel->getCPUKernelStateModel();
        auto &particleData = *(kernel->getCPUKernelStateModel().getParticleData());
        genericPerform(topologies, model, kernel, particleData);
    }

private:
    CPUKernel *kernel;
};

}
