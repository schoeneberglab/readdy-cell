//
// Created by Eric Arkfeld on 4/8/25.
//
#include <catch2/catch_template_test_macros.hpp>

#include <KernelTest.h>
#include <Utils.h>

using namespace readdytesting::kernel;

TEMPLATE_TEST_CASE("Test deterministic triggering of structural topology reactions.", "[triggerreaction]", SingleCPU, CPU) {
    auto kernel = readdytesting::kernel::create<TestType>();
    auto &ctx = kernel->context();
    ctx.boxSize() = {10., 10., 10.};
    auto &types = ctx.particleTypes();
    auto &topReg = ctx.topologyRegistry();
    auto &stateModel = kernel->stateModel();

    types.add("X", 1.0, readdy::model::particleflavor::TOPOLOGY);
    topReg.addType("T");

    GIVEN("A dimer X-X with a registered structural reaction") {
        std::vector<readdy::model::Particle> particles{
            {0., 0., 0., types.idOf("X")},
            {0., 0., 1., types.idOf("X")}
        };

        auto topology = stateModel.addTopology(topReg.idOf("T"), particles);
        topology->addEdge({0}, {1});

        // Register a simple structural reaction that removes the first edge
        auto removeEdgeReaction = readdy::model::top::reactions::StructuralTopologyReaction(
            "remove_edge", [&](readdy::model::top::GraphTopology &top) {
                readdy::model::top::reactions::Recipe recipe(top);
                for (const auto &edge : top.graph().edges()) {
                    recipe.removeEdge(edge);
                    break;
                }
                return recipe;
            }, 0.0 // rate doesn't matter for this test
        );

        topReg.addStructuralReaction(topReg.idOf("T"), removeEdgeReaction);

        WHEN("a simulation loop manually triggers the reaction") {
            readdy::model::actions::top::ReactionConfig config;
            config.registerReaction("remove_edge");
            auto &&trigger = kernel->actions().triggerReaction(config);

            trigger->perform();

            THEN("the dimer has split into two topologies") {
                auto tops = stateModel.getTopologies();
                REQUIRE(tops.size() == 2);
                for (const auto &top : tops) {
                    REQUIRE(top->nParticles() == 1);
                }
            }
        }

        WHEN("the reaction is not registered in config") {
            readdy::model::actions::top::ReactionConfig config;
            config.registerReaction("non_existent_reaction");
            auto &&trigger = kernel->actions().triggerReaction(config);

            trigger->perform();

            THEN("the topology remains unchanged") {
                auto tops = stateModel.getTopologies();
                REQUIRE(tops.size() == 1);
                REQUIRE(tops.at(0)->nParticles() == 2);
            }
        }
    }
}