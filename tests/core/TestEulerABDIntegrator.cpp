//
// Created by Eric Arkfeld on 4/8/25.
//
#include <catch2/catch_template_test_macros.hpp>
#include <catch2/catch_approx.hpp>

#include <KernelTest.h>
#include <Utils.h>
#include <model/actions/Actions.h>   // for ForceInputData

namespace m = readdy::model;
using namespace readdytesting::kernel;

TEMPLATE_TEST_CASE("EulerABDIntegrator action behavior", "[applyforcefunctional]", SingleCPU, CPU) {
    auto kernel = create<TestType>();
    auto &ctx = kernel->context();
    auto &stateModel = kernel->stateModel();

    ctx.boxSize() = {10.0, 10.0, 10.0};
    ctx.periodicBoundaryConditions() = {false, false, false};

    // Register particle type and topology
    ctx.particleTypes().add("A", 0.0, m::particleflavor::TOPOLOGY);
    auto &topReg = ctx.topologyRegistry();
    topReg.addType("T");

    const auto typeId = ctx.particleTypes().idOf("A");

    SECTION("Apply a force vector to a single particle") {
        const readdy::scalar timeStep = 1.0;
        const readdy::Vec3 initialPosition{0.0, 0.0, 0.0};
        const readdy::Vec3 expectedForce{1.0, 0.0, 0.0};

        // Add particle and topology
        std::vector<m::Particle> particles = {m::Particle(initialPosition, typeId)};
        auto top = stateModel.addTopology(topReg.idOf("T"), particles);
        auto particle_id = particles[0].id();

        // Create observable to check positions
        auto posObs = kernel->observe().positions(1);
        auto conn = kernel->connectObservable(posObs.get());

        posObs->evaluate();
        const auto &positionsBefore = posObs->getResult();
        REQUIRE(positionsBefore.size() == 1);
        const auto x0 = positionsBefore[0][0];

        // Apply the force
        readdy::model::actions::ForceInputData input{
            .ids = {particle_id},
            .forces = {expectedForce}
        };
        auto integrator = kernel->actions().eulerABDIntegrator(timeStep);
        integrator->perform(&input);

        posObs->evaluate();
        const auto &positionsAfter = posObs->getResult();
        REQUIRE(positionsAfter.size() == 1);
        const auto dx = positionsAfter[0][0] - x0;

        std::cout << "[Displacement] Δx: " << dx << std::endl;
        REQUIRE(dx > 0.0);
    }

    SECTION("Apply zero force to multiple particles") {
        const readdy::scalar timeStep = 0.1;

        ctx.particleTypes().add("B", 0.0, m::particleflavor::TOPOLOGY);
        const auto typeIdB = ctx.particleTypes().idOf("B");

        std::vector<m::Particle> particles = {
            m::Particle({1.0, 0.0, 0.0}, typeId),
            m::Particle({0.0, 1.0, 0.0}, typeIdB)
        };
        stateModel.addParticles(particles);

        auto posObs = kernel->observe().positions(1);
        auto conn = kernel->connectObservable(posObs.get());

        posObs->evaluate();
        const auto &positionsBefore = posObs->getResult();
        REQUIRE(positionsBefore.size() == 2);
        const auto x0 = positionsBefore[0][0];
        const auto y1 = positionsBefore[1][1];

        readdy::model::actions::ForceInputData input{
            .ids = {particles[0].id(), particles[1].id()},
            .forces = {
                {0.0, 0.0, 0.0},
                {0.0, 0.0, 0.0}
            }
        };
        auto integrator = kernel->actions().eulerABDIntegrator(timeStep);
        integrator->perform(&input);

        posObs->evaluate();
        const auto &positionsAfter = posObs->getResult();
        REQUIRE(positionsAfter.size() == 2);

        const auto dx = positionsAfter[0][0] - x0;
        const auto dy = positionsAfter[1][1] - y1;

        std::cout << "[Zero Force Displacement] Particle 0 Δx: " << dx << ", Particle 1 Δy: " << dy << std::endl;

        REQUIRE(dx == Catch::Approx(0.0));
        REQUIRE(dy == Catch::Approx(0.0));
    }

    SECTION("Simulate particle with nonzero diffusion and check for biased displacement in x-direction") {
        const std::string particleType = "C";
        const readdy::Vec3 initialPosition{0.0, 0.0, 0.0};
        const readdy::Vec3 appliedForce{1.0, 0.0, 0.0};

        ctx.particleTypes().add(particleType, 0.0, m::particleflavor::NORMAL); // non-zero diffusion
        const auto typeId = ctx.particleTypes().idOf(particleType);

        // Add single particle
        auto particle = m::Particle(initialPosition, typeId);
        auto id = particle.id();
        stateModel.addParticle(particle);

        // Configure integrator and timestep
        const readdy::scalar timeStep = 0.1;
        auto integrator = kernel->actions().eulerABDIntegrator(timeStep);
        auto calcForces = kernel->actions().calculateForces();

        // Setup force data
        readdy::model::actions::ForceInputData input{
            .ids = {id},
            .forces = {appliedForce}
        };

        // Observables
        auto posObs = kernel->observe().positions(1);
        auto connPos = kernel->connectObservable(posObs.get());

        posObs->evaluate();
        const auto &positionsBefore = posObs->getResult();
        REQUIRE(positionsBefore.size() == 1);
        const auto x0 = positionsBefore[0][0];
        const auto y0 = positionsBefore[0][1];
        const auto z0 = positionsBefore[0][2];

        // Simulation loop
        const unsigned int nSteps = 1000;
        for (unsigned int step = 0; step < nSteps; ++step) {
            calcForces->perform();         // compute total forces from potentials
            integrator->perform(&input);   // call integrator & apply external (input) force
        }

        // Check biased displacement
        posObs->evaluate();
        const auto &positionsAfter = posObs->getResult();
        REQUIRE(positionsAfter.size() == 1);
        const auto dx = positionsAfter[0][0] - x0;
        const auto dy = positionsAfter[0][1] - y0;
        const auto dz = positionsAfter[0][2] - z0;

        std::cout << "[Displacement] Δx: " << dx << ", Δy: " << dy << ", Δz: " << dz << std::endl;

        REQUIRE(dx > 0.0);                            // Displacement should be forward in x
        REQUIRE(std::abs(dx) > std::abs(dy));         // x-bias dominates over y
        REQUIRE(std::abs(dx) > std::abs(dz));         // x-bias dominates over z
    }
}