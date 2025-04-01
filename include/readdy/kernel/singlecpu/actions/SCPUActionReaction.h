/**
 * @file SCPUActionReaction.h
 * @brief Single CPU kernel implementation of the action ActionReaction
 * @author EricArkfeld
 * @date 11.24.24
 */

#pragma once

#include <readdy/model/actions/Actions.h>
#include <readdy/kernel/singlecpu/SCPUKernel.h>
#include <readdy/model/actions/Utils.h>

namespace readdy::kernel::scpu::actions::top {

class SCPUActionReaction : public readdy::model::actions::top::ActionReaction {
public:
    explicit SCPUActionReaction(SCPUKernel *kernel, readdy::model::actions::top::ReactionConfig config)
            : ActionReaction(std::move(config)), kernel(kernel) {}

    void perform() override {
        auto &topologies = kernel->getSCPUKernelStateModel().topologies();
        auto &model = kernel->getSCPUKernelStateModel();
        auto &particleData = *(kernel->getSCPUKernelStateModel().getParticleData());
        genericPerform(topologies, model, kernel, particleData);
    }

private:
    SCPUKernel *kernel;
};

}  // namespace readdy::kernel::scpu::actions::top

