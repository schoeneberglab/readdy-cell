#include <actions/SCPUEulerABDIntegrator.h>
#include <SCPUKernel.h>
#include <SCPUStateModel.h>
#include <common/boundary_condition_operations.h>
#include <spdlog/spdlog.h>

namespace readdy::kernel::scpu::actions {

    EulerABDIntegrator::EulerABDIntegrator(SCPUKernel *kernel, scalar timeStep)
        : readdy::model::actions::EulerABDIntegrator(timeStep), kernel(kernel) {}

    void EulerABDIntegrator::perform(const void *data) {
        const auto &context = kernel->context();
        const auto &pbc = context.periodicBoundaryConditions().data();
        const auto &kbt = context.kBT();
        const auto &box = context.boxSize().data();
        //const scalar _timeStep = 0.01; // Temp for testing
        auto& stateModel = kernel->getSCPUKernelStateModel();

        auto pdata = stateModel.getParticleData();

        // Check for valid input data
        const auto *input = static_cast<const ForceInputData *>(data);
        if (input->ids.size() != input->forces.size()) {
            spdlog::error("EulerABDIntegrator: Mismatch between ids and forces vector sizes ({} vs {}).",
                          input->ids.size(), input->forces.size());
            return;
        }

        // extract the ids and forces from the input data
        auto ids = input->ids;
        auto forces = input->forces;

        for (auto& entry : *pdata) {
            if (!entry.is_deactivated()) {
                const auto &D = context.particleTypes().diffusionConstantOf(entry.type);
                auto randomDisplacement = readdy::model::rnd::normal3<scalar>() * sqrt(2. * D * _timeStep);
                auto deterministicDisplacement = entry.force * D * _timeStep / kbt;
                auto artificialDisplacement = Vec3{0, 0, 0};
                // randomDisplacement *= sqrt(2. * D * _timeStep);

                if (std::find(ids.begin(), ids.end(), entry.id) != ids.end()) {
                    // std::cout << "[EulerABDIntegrator] Found id: " << entry.id << std::endl;
                    // Apply the force located at the id index

                    artificialDisplacement += forces[std::distance(ids.begin(), std::find(ids.begin(), ids.end(), entry.id))] * _timeStep / kbt;
                    // artificialDisplacement += forces[std::distance(ids.begin(), std::find(ids.begin(), ids.end(), entry.id))];
                }
                entry.pos += randomDisplacement + deterministicDisplacement + artificialDisplacement;
                bcs::fixPosition(entry.pos, box, pbc);
            }
        }
    }

    void EulerABDIntegrator::perform() {
        spdlog::warn("EulerABDIntegrator::perform() was called without input data. This is a no-op.");
    }

}