#include <actions/CPUEulerABDIntegrator.h>
#include <CPUKernel.h>
#include <CPUStateModel.h>
#include <spdlog/spdlog.h>
#include <unordered_map>

namespace readdy::kernel::cpu::actions
{

    EulerABDIntegrator::EulerABDIntegrator(CPUKernel *kernel, scalar timeStep)
        : readdy::model::actions::EulerABDIntegrator(timeStep), kernel(kernel) {}

    void EulerABDIntegrator::perform(const void *data)
    {
        auto pdata = kernel->getCPUKernelStateModel().getParticleData();
        const auto pdata_size = pdata->size();

        const auto &context = kernel->context();
        const auto dt = timeStep();

        const auto *input = static_cast<const ForceInputData *>(data);
        if (!input || input->ids.size() != input->forces.size())
        {
            spdlog::error("EulerABDIntegrator: Mismatch between ids and forces vector sizes ({} vs {}).",
                          input ? input->ids.size() : 0, input ? input->forces.size() : 0);
            return;
        }

        const auto &ids = input->ids;
        const auto &forces = input->forces;

        std::unordered_map<std::size_t, Vec3> id_to_force;
        for (std::size_t i = 0; i < ids.size(); ++i)
        {
            id_to_force[ids[i]] = forces[i];
        }

        auto &pool = kernel->pool();
        std::vector<util::thread::joining_future<void>> waitingFutures;
        waitingFutures.reserve(kernel->getNThreads());

        const std::size_t nThreads = kernel->getNThreads();
        const std::size_t grainSize = pdata_size / nThreads;

        auto it = pdata->begin();
        std::size_t idx = 0;

        for (std::size_t i = 0; i < nThreads - 1; ++i)
        {
            auto itNext = it + grainSize;

            waitingFutures.emplace_back(pool.push([&, begin = it, end = itNext, startIdx = idx](int)
                                                  {
            const auto kbt = context.kBT();
            std::size_t localIdx = startIdx;

            for (auto p = begin; p != end; ++p, ++localIdx) {
                if (!p->deactivated) {
                    const auto D = context.particleTypes().diffusionConstantOf(p->type);
                    auto randomDisplacement = readdy::model::rnd::normal3<scalar>() * std::sqrt(2. * D * dt);
                    auto deterministicDisplacement = p->force * D * dt / kbt;
                    auto artificialDisplacement = Vec3{0, 0, 0};

                    auto id_it = id_to_force.find(p->id);
                    if (id_it != id_to_force.end()) {
                        // artificialDisplacement += id_it->second * D * dt / kbt;
                        artificialDisplacement += id_it->second * dt / kbt;
                    }

                    p->pos += randomDisplacement + deterministicDisplacement + artificialDisplacement;
                }
            } }));

            it = itNext;
            idx += grainSize;
        }

        // Remainder
        if (it != pdata->end())
        {
            waitingFutures.emplace_back(pool.push([&, begin = it, end = pdata->end(), startIdx = idx](int)
                                                  {
            const auto kbt = context.kBT();
            std::size_t localIdx = startIdx;

            for (auto p = begin; p != end; ++p, ++localIdx) {
                if (!p->deactivated) {
                    const auto D = context.particleTypes().diffusionConstantOf(p->type);
                    auto randomDisplacement = readdy::model::rnd::normal3<scalar>() * std::sqrt(2. * D * dt);
                    auto deterministicDisplacement = p->force * D * dt / kbt;
                    auto artificialDisplacement = Vec3{0, 0, 0};

                    auto id_it = id_to_force.find(p->id);
                    if (id_it != id_to_force.end()) {
                        artificialDisplacement += id_it->second * dt / kbt;
                    }

                    p->pos += randomDisplacement + deterministicDisplacement + artificialDisplacement;
                }
            } }));
        }
    }

    void EulerABDIntegrator::perform()
    {
        spdlog::warn("EulerABDIntegrator::perform() was called without input data. This is a no-op.");
    }
}