/**
 * « detailed description »
 *
 * @file CPUTriggerReaction.h
 * @brief « brief description »
 * @author Eric Arkfeld
* @date 11.24.24
 */

#pragma once

#include <model/actions/Actions.h>
#include <CPUKernel.h>

namespace readdy::kernel::cpu::actions::top {

class CPUTriggerReaction : public readdy::model::actions::top::TriggerReaction {
public:
    explicit CPUTriggerReaction(CPUKernel *kernel, readdy::model::actions::top::ReactionConfig config)
            : TriggerReaction(std::move(config)), kernel(kernel) {}

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
