#pragma once
#include <CPUKernel.h>
#include <model/actions/Actions.h>

namespace readdy::kernel::cpu::actions {

	struct ForceInputData {
    	std::vector<std::size_t> ids;
    	std::vector<Vec3> forces;
	};

    class EulerABDIntegrator : public readdy::model::actions::EulerABDIntegrator {
    public:
        explicit EulerABDIntegrator(CPUKernel *kernel, scalar timeStep);

        void perform(const void *data) override;

        void perform() override;

    private:
        CPUKernel *const kernel;
    };

}