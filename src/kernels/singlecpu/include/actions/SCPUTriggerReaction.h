/**
 * @file SCPUTriggerReaction.h
 * @brief Single CPU kernel implementation of the action TriggerReaction
 * @author EricArkfeld
 * @date 11.24.24
 */

#pragma once

#include <model/actions/Actions.h>
#include <SCPUKernel.h>
#include <model/actions/Utils.h>

namespace readdy::kernel::scpu::actions::top {

class SCPUTriggerReaction : public readdy::model::actions::top::TriggerReaction {
public:
    explicit SCPUTriggerReaction(SCPUKernel *kernel, readdy::model::actions::top::ReactionConfig config)
            : TriggerReaction(std::move(config)), kernel(kernel) {}

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

