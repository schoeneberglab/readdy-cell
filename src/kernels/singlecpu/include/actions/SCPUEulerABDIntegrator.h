#pragma once

#include <model/actions/Actions.h>
#include <SCPUKernel.h>

namespace readdy::kernel::scpu::actions {

	struct ForceInputData {
    	std::vector<std::size_t> ids;
    	std::vector<Vec3> forces;
	};

    class EulerABDIntegrator : public readdy::model::actions::EulerABDIntegrator {
    public:
        explicit EulerABDIntegrator(SCPUKernel *kernel, scalar timeStep);
        void perform(const void *data) override;
        void perform() override;

    private:
        SCPUKernel *const kernel;
    };

}