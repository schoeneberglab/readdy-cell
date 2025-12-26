# coding=utf-8

# Copyright © 2020 Computational Molecular Biology Group,
#                  Freie Universität Berlin (GER)
#
# Redistribution and use in source and binary forms, with or
# without modification, are permitted provided that the
# following conditions are met:
#  1. Redistributions of source code must retain the above
#     copyright notice, this list of conditions and the
#     following disclaimer.
#  2. Redistributions in binary form must reproduce the above
#     copyright notice, this list of conditions and the following
#     disclaimer in the documentation and/or other materials
#     provided with the distribution.
#  3. Neither the name of the copyright holder nor the names of
#     its contributors may be used to endorse or promote products
#     derived from this software without specific
#     prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND
# CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES,
# INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF
# MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR
# CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
# SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
# BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
# LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT,
# STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF
# ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

"""
Created on 17.03.20

@author: chrisfroe
"""

import os
import readdy
from readdy.api.experimental.action_factory import BreakConfig, ReactionConfig, ActionFactory
import unittest
import numpy as np
import tempfile
import shutil
from tqdm import tqdm
from readdy.api.utils import vec3_of


class TestCustomLoop(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.dir = tempfile.mkdtemp("test-custom-loop")

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.dir, ignore_errors=True)

    def test_displace_particle(self):
        rds = readdy.ReactionDiffusionSystem(box_size=[10., 10., 10.], unit_system=None)
        rds.add_species("A", 1.0)
        simulation = rds.simulation("SingleCPU")
        simulation.add_particle("A", [0., 0., 0.])
        dt = 1.0
        # action factory is an experimental feature and thus _hidden
        integrator = simulation._actions.integrator_euler_brownian_dynamics(dt)
        p = simulation.current_particles[0]
        self.assertTrue(np.all(p.pos == np.array([0., 0., 0.])))
        integrator()
        p = simulation.current_particles[0]
        self.assertFalse(np.all(p.pos == np.array([0., 0., 0.])))

    def test_build_and_run_custom_loop(self):
        rds = readdy.ReactionDiffusionSystem(box_size=[7., 7., 7.], unit_system=None)
        rds.add_species("A", 1.0)
        rds.add_species("B", 1.0)
        rds.add_species("C", 1.0)
        rds.reactions.add("fusion: A +(2) B -> C", 10.)
        simulation = rds.simulation("SingleCPU")
        simulation.add_particle("A", [0., 0., 0.])
        simulation.add_particle("B", [3., 0., 0.])
        simulation.output_file = os.path.join(self.dir, "customlooptest1.h5")
        simulation.observe.number_of_particles(1, ["A", "B", "C"])

        def loop():
            nonlocal simulation
            dt = 1.0
            n_steps = 10000
            base_path = os.path.join(self.dir, "ckpts")
            os.makedirs(base_path, exist_ok=True)
            max_n_saves = 2

            init = simulation._actions.initialize_kernel()
            diff = simulation._actions.integrator_euler_brownian_dynamics(dt)
            calc_forces = simulation._actions.calculate_forces()
            create_nl = simulation._actions.create_neighbor_list(rds.calculate_max_cutoff())
            update_nl = simulation._actions.update_neighbor_list()
            reac = simulation._actions.reaction_handler_uncontrolled_approximation(dt)
            obs = simulation._actions.evaluate_observables()
            check = simulation._actions.make_checkpoint(base_path, max_n_saves)

            init()
            create_nl()
            calc_forces()
            update_nl()
            obs(0)
            for t in tqdm(range(1, n_steps + 1)):
                diff()
                update_nl()
                reac()
                update_nl()
                calc_forces()
                obs(t)  # striding of observables is done internally
                if t % 100 == 0:
                    check(t)

        simulation._run_custom_loop(loop)

        traj = readdy.Trajectory(simulation.output_file)
        ts, ns = traj.read_observable_number_of_particles()

        self.assertEqual(ns[0, 0], 1)
        self.assertEqual(ns[0, 1], 1)
        self.assertEqual(ns[0, 2], 0)
        self.assertEqual(ns[-1, 0], 0)
        self.assertEqual(ns[-1, 1], 0)
        self.assertEqual(ns[-1, 2], 1)

    def test_break_bonds_integration(self):
        # basically the "Break bonds due to pulling" test from IntegrationTests.cpp translated to python
        self._break_bonds_integration(True)  # test with pulling, topology is split in twain
        self._break_bonds_integration(False)  # test without pulling, topology stays intact

    def _break_bonds_integration(self, pull):
        system = readdy.ReactionDiffusionSystem(
            box_size=[20., 10., 10], periodic_boundary_conditions=[True, True, True], unit_system=None)
        system.kbt = 0.01
        system.add_topology_species("head", 0.1)
        system.add_topology_species("A", 0.1)
        system.add_topology_species("tail", 0.1)
        system.potentials.add_cylinder("A", 100., [0., 0., 0.], [1., 0., 0.], 0.01, True)
        system.potentials.add_cylinder("head", 100., [0., 0., 0.], [1., 0., 0.], 0.01, True)
        system.potentials.add_cylinder("tail", 100., [0., 0., 0.], [1., 0., 0.], 0.01, True)
        system.potentials.add_box("head", 10., [-4., -12.5, -12.5], [0.000001, 25., 25.])
        if pull:
            system.potentials.add_box("tail", 100., [+8, -12.5, -12.5], [0.000001, 25., 25.])
        system.topologies.add_type("T1")
        system.topologies.add_type("T2")
        system.topologies.configure_harmonic_bond("head", "A", 10., 2.)
        system.topologies.configure_harmonic_bond("A", "A", 10., 4.)
        system.topologies.configure_harmonic_bond("A", "tail", 10., 2.)

        simulation = system.simulation("CPU")

        pos = np.array([
            [-4., 0., 0.],
            [-2., 0., 0.],
            [2., 0., 0.],
            [4., 0., 0.],
        ])
        types = ["head", "A", "A", "tail"]

        topology = simulation.add_topology("T1", types, pos)
        topology.graph.add_edge(0, 1)
        topology.graph.add_edge(1, 2)
        topology.graph.add_edge(2, 3)

        conf = BreakConfig()
        id_a = system._context.particle_types.id_of("A")
        conf.add_breakable_pair(id_a, id_a, 1.0, 1.0)

        def loop():
            nonlocal simulation
            nonlocal conf
            dt = 0.0005
            n_steps = 100000
            # init is needed here to configure topologies
            init = simulation._actions.initialize_kernel()
            diff = simulation._actions.integrator_euler_brownian_dynamics(dt)
            forces = simulation._actions.calculate_forces()
            break_bonds = simulation._actions.break_bonds(dt, conf)

            init()
            forces()
            for t in tqdm(range(1, n_steps + 1)):
                diff()
                break_bonds()
                forces()

        simulation._run_custom_loop(loop)

        tops_after = simulation.current_topologies
        if pull:
            self.assertTrue(len(tops_after) == 2)
        else:
            self.assertTrue(len(tops_after) == 1)

        for top in tops_after:
            for vertex in top.graph.vertices:
                self.assertTrue(len(vertex.neighbors()) <= 2)

    def test_trigger_reaction_integration(self):
        self._trigger_reaction_integration()

    def _trigger_reaction_integration(self):
        system = readdy.ReactionDiffusionSystem(
            box_size=[5., 5., 5], periodic_boundary_conditions=[True, True, True], unit_system=None)

        system.topologies.add_type("T")
        system.add_topology_species("A", 0.)
        system.add_topology_species("B", 0.)

        def A2B_rfn(topology):
            # Reaction which changes a particle from type A to type B
            recipe = readdy.StructuralReactionRecipe(topology)
            graph = topology.graph
            v = graph.vertices[0]
            recipe.change_particle_type(v, "B")
            return recipe

        system.topologies.add_structural_reaction("A2B", "T", A2B_rfn, lambda x: 1.e-32)

        simulation = system.simulation("CPU")

        rconf = ReactionConfig()
        rconf.register_reaction("A2B")

        simulation.add_topology("T", ["A"], np.array([[0., 0., 0.]]))

        def loop():
            nonlocal simulation
            nonlocal rconf
            dt = 0.0005
            n_steps = 5

            # init is needed here to configure topologies
            init = simulation._actions.initialize_kernel()
            diff = simulation._actions.integrator_euler_brownian_dynamics(dt)
            forces = simulation._actions.calculate_forces()
            trigger_reaction = simulation._actions.trigger_reaction(rconf)

            init()
            forces()
            for _ in tqdm(range(1, n_steps + 1)):
                diff()
                trigger_reaction()
                forces()

        simulation._run_custom_loop(loop)

        tops_after = simulation.current_topologies

        self.assertTrue(len(tops_after) == 1)

        # check that the type of the vertex has changed
        top = tops_after[0]
        vertex = top.graph.vertices[0]
        self.assertTrue(top.particle_type_of_vertex(vertex) == "B")

    def test_apply_force_functional(self):
        rds = readdy.ReactionDiffusionSystem(box_size=[10., 10., 10.],
                                             unit_system=None,
                                             periodic_boundary_conditions=[False, False, False])
        
        rds.kbt = 1.
        rds.add_species("A", diffusion_constant=0.0)  # No diffusion to isolate force effect
        simulation = rds.simulation("CPU")

        initial_position = [0., 0., 0.]
        simulation.add_particle("A", initial_position)

        dt = 1.0
        force_vector = np.array([1.0, 0.0, 0.0])
        particle_ids = [0]
        forces = [vec3_of(force_vector)]

        init = simulation._actions.initialize_kernel()
        integrator = simulation._actions.integrator_euler_active_brownian_dynamics(dt)

        def loop():
            # nonlocal simulation
            # nonlocal integrator
            n_steps = 3
            init()
            for _ in range(n_steps):
                integrator.perform(particle_ids, forces)

        simulation._run_custom_loop(loop)

        final_pos = simulation.current_particles[0].pos
        self.assertTrue((final_pos == np.array([3., 0., 0.])).all(),
                        f"Final position should be [3., 0., 0.], but got {final_pos}")

if __name__ == '__main__':
    unittest.main()
