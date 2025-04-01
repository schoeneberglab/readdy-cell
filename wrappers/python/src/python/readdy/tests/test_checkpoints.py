import os
import shutil
import tempfile
import unittest

import numpy as np

import readdy
from readdy.util.testing_utils import ReaDDyTestCase


class TestCheckpoints(ReaDDyTestCase):

    def setUp(self):
        self.dir = tempfile.mkdtemp("test-checkpoints")

    def tearDown(self):
        shutil.rmtree(self.dir, ignore_errors=True)

    def _set_up_system(self):
        system = readdy.ReactionDiffusionSystem(box_size=[10, 10, 10])
        system.add_species("A", 1.)
        system.add_species("B", 1.)
        system.reactions.add("conv1: A -> B", rate=1.)
        system.reactions.add("conv2: B -> A", rate=1.)
        system.reactions.add("fus: A +(1.) B -> A", rate=1.)
        system.reactions.add("fiss: A -> A +(1.) B", rate=1.)
        system.topologies.add_type("TT1")
        system.topologies.add_type("TT2")
        system.topologies.add_type("Dummy")
        system.add_topology_species("Dummy", 10.)
        system.add_topology_species("T1", 1.)
        system.add_topology_species("T2", 1.)
        system.topologies.configure_harmonic_bond("T1", "T1")
        system.topologies.configure_harmonic_bond("T1", "T2")
        system.topologies.configure_harmonic_bond("T2", "T2")
        system.topologies.configure_harmonic_bond("Dummy", "Dummy")
        return system

    def _run_test(self, with_topologies, with_particles, fname):
        system = self._set_up_system()
        sim = system.simulation()

        if with_topologies:
            t1_initial_pos = np.random.normal(0, 1, size=(4, 3))
            t1 = sim.add_topology("TT1", ["T1", "T2", "T1", "T2"], t1_initial_pos)
            t1.graph.add_edge(0, 1)
            t1.graph.add_edge(1, 2)
            t1.graph.add_edge(2, 3)
            t1.graph.add_edge(3, 0)
            t2_initial_pos = np.random.normal(0, 1, size=(4, 3))
            t2 = sim.add_topology("TT2", ["T2", "T1", "T2", "T1"], t2_initial_pos)
            t2.graph.add_edge(0, 1)
            t2.graph.add_edge(1, 2)
            t2.graph.add_edge(2, 3)

        if with_particles:
            a_particles_initial_pos = np.random.normal(0, 1, size=(20, 3))
            sim.add_particles("A", a_particles_initial_pos)
            b_particles_initial_pos = np.random.normal(0, 1, size=(50, 3))
            sim.add_particles("B", b_particles_initial_pos)

        def topologies_callback(_):
            if with_topologies:
                if len(sim.current_topologies) % 2 == 0:
                    sim.add_topology("Dummy", "Dummy", np.random.random(size=(1, 3)))
                else:
                    t = sim.add_topology("Dummy", "Dummy", np.random.random(size=(5, 3)))
                    t.graph.add_edge(0, 1)
                    t.graph.add_edge(1, 2)
                    t.graph.add_edge(2, 3)
                    t.graph.add_edge(3, 4)
                    t.configure()

        sim.make_checkpoints(7, output_directory=self.dir, max_n_saves=7)
        sim.record_trajectory()
        sim.observe.topologies(1, callback=topologies_callback)
        sim.output_file = os.path.join(self.dir, fname)
        sim.show_progress = False
        sim.run(120, 1e-2, show_summary=False)

        traj = readdy.Trajectory(sim.output_file)
        entries = traj.read()
        _, traj_tops = traj.read_observable_topologies()

        system = self._set_up_system()
        sim = system.simulation()

        ckpt_files = sim.list_checkpoint_files(self.dir)
        ckpt_file = sim.get_latest_checkpoint_file(self.dir)
        checkpoints = sim.list_checkpoints(ckpt_file)
        checkpoint = checkpoints[-1]

        latest_checkpoint_step = 120 // 7 * 7
        assert checkpoint['step'] == latest_checkpoint_step, "expected {} but got {} (file {} of files {})"\
            .format(latest_checkpoint_step, checkpoint['step'], ckpt_file, ckpt_files)

        sim.load_particles_from_checkpoint(ckpt_file)

        current_entries = entries[latest_checkpoint_step]
        current_particles = sim.current_particles

        if with_topologies:
            tops = sim.current_topologies
            assert len(tops) == len(traj_tops[latest_checkpoint_step]), \
                f"expected {len(traj_tops[latest_checkpoint_step])} topologies, " \
                f"got {len(tops)} (file {ckpt_file})"
            assert tops[0].type == "TT1"
            assert tops[0].graph.has_edge(0, 1)
            assert tops[0].graph.has_edge(1, 2)
            assert tops[0].graph.has_edge(2, 3)
            assert tops[0].graph.has_edge(3, 0)
            assert not tops[0].graph.has_edge(0, 2)
            assert not tops[0].graph.has_edge(1, 3)
            assert tops[1].type == "TT2"
            assert tops[1].graph.has_edge(0, 1)
            assert tops[1].graph.has_edge(1, 2)
            assert tops[1].graph.has_edge(2, 3)

            topologies = traj_tops[checkpoint['step']]

            # check whether restored topologies are o.k.
            assert len(topologies) == len(tops)
            for ix, topology_record in enumerate(topologies):
                restored_topology = tops[ix]
                for edge in topology_record.edges:
                    assert tops[ix].graph.has_edge(*edge)
                for pix, particle_ix in enumerate(topology_record.particles):
                    particle = current_entries[particle_ix]
                    restored_particle = restored_topology.particles[pix]
                    assert np.array_equal(restored_particle.pos, np.array(particle.position))
                    assert restored_particle.type == particle.type

        # check whether restored particles are o.k.
        for entry in current_entries:
            # see if entry is available in current particles
            ix = 0
            for ix, particle in enumerate(current_particles):
                if particle.type == entry.type and np.array_equal(particle.pos, entry.position):
                    break
            assert ix < len(current_particles), f"entry {entry} was not found in particles!"
            current_particles.pop(ix)
        assert len(current_particles) == 0

        sim.show_progress = False
        sim.run(500, 1e-3, show_summary=False)

    def test_continue_simulation_full(self):
        self._run_test(with_topologies=True, with_particles=True, fname='full.h5')

    def test_continue_simulation_no_topologies(self):
        self._run_test(with_topologies=False, with_particles=True, fname='no_topologies.h5')

    def test_continue_simulation_no_free_particles(self):
        self._run_test(with_topologies=True, with_particles=False, fname='no_free_particles')


if __name__ == '__main__':
    unittest.main()
