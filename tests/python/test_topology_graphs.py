# coding=utf-8

# Copyright © 2018 Computational Molecular Biology Group,
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
Created on 24.03.17

@author: clonker
"""

from __future__ import print_function

import unittest

import numpy as np
import readdy._internal.readdybinding.common as common
from readdy._internal.readdybinding.api import BondedPotentialConfiguration
from readdy._internal.readdybinding.api import ParticleTypeFlavor
from readdy._internal.readdybinding.api import Simulation
from readdy._internal.readdybinding.api import Context

from testing_utils import ReaDDyTestCase


class TestTopologyGraphs(ReaDDyTestCase):
    def test_sanity(self):
        context = Context()
        context.box_size = [10., 10., 10.]
        context.topologies.add_type("TA")
        context.particle_types.add("T", 1.0, flavor=ParticleTypeFlavor.TOPOLOGY)
        context.topologies.configure_bond_potential("T", "T", BondedPotentialConfiguration(10., 11., "harmonic"))
        sim = Simulation("SingleCPU", context)
        np.testing.assert_equal(sim.kernel_supports_topologies(), True)
        particles = [sim.create_topology_particle("T", common.Vec(x, 0, 0)) for x in range(4)]
        top = sim.add_topology("TA", particles)
        graph = top.graph
        graph.add_edge(0, 1)
        graph.add_edge(1, 2)
        graph.add_edge(2, 3)
        np.testing.assert_equal(len(graph.get_vertices()), 4)

        for v in graph.vertices:

            if v.particle_index == 0:
                np.testing.assert_equal(top.position_of_vertex(v), common.Vec(0, 0, 0))
                np.testing.assert_equal(len(v.neighbors()), 1)
                neigh = v.neighbors()[0]
                np.testing.assert_(neigh in graph.vertices)
                np.testing.assert_equal(1 in [vv.particle_index for vv in v], True)
            if v.particle_index == 1:
                np.testing.assert_equal(top.position_of_vertex(v), common.Vec(1, 0, 0))
                np.testing.assert_equal(len(v.neighbors()), 2)
                np.testing.assert_equal(0 in [vv.get().particle_index for vv in v], True)
                np.testing.assert_equal(2 in [vv.get().particle_index for vv in v], True)
            if v.particle_index == 2:
                np.testing.assert_equal(top.position_of_vertex(v), common.Vec(2, 0, 0))
                np.testing.assert_equal(len(v.neighbors()), 2)
                np.testing.assert_equal(1 in [vv.get().particle_index for vv in v], True)
                np.testing.assert_equal(3 in [vv.get().particle_index for vv in v], True)
            if v.particle_index == 3:
                np.testing.assert_equal(top.position_of_vertex(v), common.Vec(3, 0, 0))
                np.testing.assert_equal(len(v.neighbors()), 1)
                np.testing.assert_equal(2 in [vv.get().particle_index for vv in v], True)
        top.configure()
        sim.run(0, 1)

    def test_unconnected_graph(self):
        context = Context()
        context.topologies.add_type("TA")
        context.box_size = [10., 10., 10.]
        context.particle_types.add("T", 1.0, flavor=ParticleTypeFlavor.TOPOLOGY)
        context.topologies.configure_bond_potential("T", "T", BondedPotentialConfiguration(10, 11, "harmonic"))
        sim = Simulation("SingleCPU", context)
        np.testing.assert_equal(sim.kernel_supports_topologies(), True)
        particles = [sim.create_topology_particle("T", common.Vec(0, 0, 0)) for _ in range(4)]
        top = sim.add_topology("TA", particles)
        graph = top.get_graph()
        graph.add_edge(0, 1)
        graph.add_edge(1, 2)
        with (np.testing.assert_raises(ValueError)):
            top.configure()

    def test_unbonded_edge(self):
        context = Context()
        context.box_size = [10., 10., 10.]
        context.topologies.add_type("TA")
        context.particle_types.add("T", 1.0, flavor=ParticleTypeFlavor.TOPOLOGY)
        context.particle_types.add("D", 1.0, flavor=ParticleTypeFlavor.TOPOLOGY)
        context.topologies.configure_bond_potential("T", "T", BondedPotentialConfiguration(10., 11., "harmonic"))
        sim = Simulation("SingleCPU", context)
        np.testing.assert_equal(sim.kernel_supports_topologies(), True)
        particles = [sim.create_topology_particle("T", common.Vec(0, 0, 0)) for _ in range(3)]
        particles.append(sim.create_topology_particle("D", common.Vec(0, 0, 0)))
        top = sim.add_topology("TA", particles)
        graph = top.get_graph()
        graph.add_edge(0, 1)
        graph.add_edge(1, 2)
        graph.add_edge(2, 3)
        with (np.testing.assert_raises(ValueError)):
            top.configure()

    def test_get_vertices_with_filters(self):
        """Test the extended get_vertices method with filtering by type, degree, and particle ID."""
        context = Context()
        context.box_size = [10., 10., 10.]
        context.topologies.add_type("TA")
        context.particle_types.add("TypeA", 1.0, flavor=ParticleTypeFlavor.TOPOLOGY)
        context.particle_types.add("TypeB", 1.0, flavor=ParticleTypeFlavor.TOPOLOGY)
        context.particle_types.add("TypeAB", 1.0, flavor=ParticleTypeFlavor.TOPOLOGY)
        context.topologies.configure_bond_potential("TypeA", "TypeA", BondedPotentialConfiguration(10., 11., "harmonic"))
        context.topologies.configure_bond_potential("TypeA", "TypeB", BondedPotentialConfiguration(10., 11., "harmonic"))
        context.topologies.configure_bond_potential("TypeB", "TypeB", BondedPotentialConfiguration(10., 11., "harmonic"))
        context.topologies.configure_bond_potential("TypeAB", "TypeA", BondedPotentialConfiguration(10., 11., "harmonic"))
        context.topologies.configure_bond_potential("TypeAB", "TypeB", BondedPotentialConfiguration(10., 11., "harmonic"))

        sim = Simulation("SingleCPU", context)

        # Create a topology with mixed particle types
        # Structure: TypeA(0) - TypeB(1) - TypeA(2) - TypeAB(3)
        #                        |
        #                     TypeB(4)
        particles = [
            sim.create_topology_particle("TypeA", common.Vec(0, 0, 0)),   # vertex 0, degree 1
            sim.create_topology_particle("TypeB", common.Vec(1, 0, 0)),   # vertex 1, degree 3
            sim.create_topology_particle("TypeA", common.Vec(2, 0, 0)),   # vertex 2, degree 2
            sim.create_topology_particle("TypeAB", common.Vec(3, 0, 0)),  # vertex 3, degree 1
            sim.create_topology_particle("TypeB", common.Vec(1, 1, 0))    # vertex 4, degree 1
        ]

        top = sim.add_topology("TA", particles)
        graph = top.graph

        # Create edges
        graph.add_edge(0, 1)  # TypeA - TypeB
        graph.add_edge(1, 2)  # TypeB - TypeA
        graph.add_edge(2, 3)  # TypeA - TypeAB
        graph.add_edge(1, 4)  # TypeB - TypeB

        # Test 1: Get all vertices (no filters)
        all_vertices = graph.get_vertices()
        np.testing.assert_equal(len(all_vertices), 5)

        # Test 2: Filter by exact particle type
        type_a_vertices = graph.get_vertices(vertex_type="TypeA", exact_match=True)
        np.testing.assert_equal(len(type_a_vertices), 2)
        type_a_indices = [v.particle_index for v in type_a_vertices]
        np.testing.assert_equal(sorted(type_a_indices), [0, 2])

        type_b_vertices = graph.get_vertices(vertex_type="TypeB", exact_match=True)
        np.testing.assert_equal(len(type_b_vertices), 2)
        type_b_indices = [v.particle_index for v in type_b_vertices]
        np.testing.assert_equal(sorted(type_b_indices), [1, 4])

        type_ab_vertices = graph.get_vertices(vertex_type="TypeAB", exact_match=True)
        np.testing.assert_equal(len(type_ab_vertices), 1)
        np.testing.assert_equal(type_ab_vertices[0].particle_index, 3)

        # Test 3: Filter by substring match (exact_match=False)
        type_containing_a = graph.get_vertices(vertex_type="A", exact_match=False)
        np.testing.assert_equal(len(type_containing_a), 3)  # TypeA (2) + TypeAB (1)
        type_a_indices = [v.particle_index for v in type_containing_a]
        np.testing.assert_equal(sorted(type_a_indices), [0, 2, 3])

        type_containing_b = graph.get_vertices(vertex_type="B", exact_match=False)
        np.testing.assert_equal(len(type_containing_b), 3)  # TypeB (2) + TypeAB (1)
        type_b_indices = [v.particle_index for v in type_containing_b]
        np.testing.assert_equal(sorted(type_b_indices), [1, 3, 4])

        # Test 4: Filter by degree
        degree_1_vertices = graph.get_vertices(vertex_degree=1)
        np.testing.assert_equal(len(degree_1_vertices), 3)
        degree_1_indices = [v.particle_index for v in degree_1_vertices]
        np.testing.assert_equal(sorted(degree_1_indices), [0, 3, 4])

        degree_2_vertices = graph.get_vertices(vertex_degree=2)
        np.testing.assert_equal(len(degree_2_vertices), 1)
        np.testing.assert_equal(degree_2_vertices[0].particle_index, 2)

        degree_3_vertices = graph.get_vertices(vertex_degree=3)
        np.testing.assert_equal(len(degree_3_vertices), 1)
        np.testing.assert_equal(degree_3_vertices[0].particle_index, 1)

        # Test 5: Combined filters - type and degree
        type_a_degree_1 = graph.get_vertices(vertex_type="TypeA", vertex_degree=1, exact_match=True)
        np.testing.assert_equal(len(type_a_degree_1), 1)
        np.testing.assert_equal(type_a_degree_1[0].particle_index, 0)

        type_b_degree_1 = graph.get_vertices(vertex_type="TypeB", vertex_degree=1, exact_match=True)
        np.testing.assert_equal(len(type_b_degree_1), 1)
        np.testing.assert_equal(type_b_degree_1[0].particle_index, 4)

        # Test 6: Filter by particle ID
        target_vertex = all_vertices[2]  # TypeA vertex at index 2
        target_id = top.particle_id_of_vertex(target_vertex)

        id_filtered_vertices = graph.get_vertices(vertex_id=target_id)
        np.testing.assert_equal(len(id_filtered_vertices), 1)
        np.testing.assert_equal(id_filtered_vertices[0].particle_index, 2)

        # Test 7: Filter by particle ID with additional type filter (should match)
        id_type_filtered = graph.get_vertices(vertex_id=target_id, vertex_type="TypeA", exact_match=True)
        np.testing.assert_equal(len(id_type_filtered), 1)
        np.testing.assert_equal(id_type_filtered[0].particle_index, 2)

        # Test 8: Filter by particle ID with incompatible type filter (should not match)
        id_type_no_match = graph.get_vertices(vertex_id=target_id, vertex_type="TypeB", exact_match=True)
        np.testing.assert_equal(len(id_type_no_match), 0)

        # Test 9: Filter by particle ID with additional degree filter (should match)
        id_degree_filtered = graph.get_vertices(vertex_id=target_id, vertex_degree=2)
        np.testing.assert_equal(len(id_degree_filtered), 1)
        np.testing.assert_equal(id_degree_filtered[0].particle_index, 2)

        # Test 10: Filter by particle ID with incompatible degree filter (should not match)
        id_degree_no_match = graph.get_vertices(vertex_id=target_id, vertex_degree=1)
        np.testing.assert_equal(len(id_degree_no_match), 0)

        # Test 11: Filter by non-existent particle ID
        non_existent_id = 99999
        no_match_vertices = graph.get_vertices(vertex_id=non_existent_id)
        np.testing.assert_equal(len(no_match_vertices), 0)

        # Test 12: Filter with impossible combinations
        impossible_combo = graph.get_vertices(vertex_type="TypeA", vertex_degree=3, exact_match=True)
        np.testing.assert_equal(len(impossible_combo), 0)

    def test_vertex_neighbors_with_filters(self):
        """Test the extended neighbors method with filtering by type, degree, and particle ID."""
        context = Context()
        context.box_size = [10., 10., 10.]
        context.topologies.add_type("TA")
        context.particle_types.add("TypeA", 1.0, flavor=ParticleTypeFlavor.TOPOLOGY)
        context.particle_types.add("TypeB", 1.0, flavor=ParticleTypeFlavor.TOPOLOGY)
        context.particle_types.add("TypeAB", 1.0, flavor=ParticleTypeFlavor.TOPOLOGY)
        context.topologies.configure_bond_potential("TypeA", "TypeA", BondedPotentialConfiguration(10., 11., "harmonic"))
        context.topologies.configure_bond_potential("TypeA", "TypeB", BondedPotentialConfiguration(10., 11., "harmonic"))
        context.topologies.configure_bond_potential("TypeB", "TypeB", BondedPotentialConfiguration(10., 11., "harmonic"))
        context.topologies.configure_bond_potential("TypeAB", "TypeA", BondedPotentialConfiguration(10., 11., "harmonic"))
        context.topologies.configure_bond_potential("TypeAB", "TypeB", BondedPotentialConfiguration(10., 11., "harmonic"))

        sim = Simulation("SingleCPU", context)

        # Create a more complex topology for neighbor testing
        # Structure:    TypeA(0) - TypeB(1) - TypeA(2) - TypeAB(3)
        #                            |            |
        #                         TypeB(4)   TypeB(5)
        #                                        |
        #                                    TypeA(6)
        particles = [
            sim.create_topology_particle("TypeA", common.Vec(0, 0, 0)),   # vertex 0, degree 1
            sim.create_topology_particle("TypeB", common.Vec(1, 0, 0)),   # vertex 1, degree 3
            sim.create_topology_particle("TypeA", common.Vec(2, 0, 0)),   # vertex 2, degree 3
            sim.create_topology_particle("TypeAB", common.Vec(3, 0, 0)),  # vertex 3, degree 1
            sim.create_topology_particle("TypeB", common.Vec(1, 1, 0)),   # vertex 4, degree 1
            sim.create_topology_particle("TypeB", common.Vec(2, 1, 0)),   # vertex 5, degree 2
            sim.create_topology_particle("TypeA", common.Vec(2, 2, 0))    # vertex 6, degree 1
        ]

        top = sim.add_topology("TA", particles)
        graph = top.graph

        # Create edges to form the structure described above
        graph.add_edge(0, 1)  # TypeA - TypeB
        graph.add_edge(1, 2)  # TypeB - TypeA
        graph.add_edge(2, 3)  # TypeA - TypeAB
        graph.add_edge(1, 4)  # TypeB - TypeB
        graph.add_edge(2, 5)  # TypeA - TypeB
        graph.add_edge(5, 6)  # TypeB - TypeA

        vertices = graph.get_vertices()

        # Get specific vertices for testing
        vertex_1 = next(v for v in vertices if v.particle_index == 1)  # TypeB with 3 neighbors
        vertex_2 = next(v for v in vertices if v.particle_index == 2)  # TypeA with 3 neighbors

        # Test 1: Get all neighbors (no filters) - should match original behavior
        all_neighbors_v1 = vertex_1.neighbors()
        np.testing.assert_equal(len(all_neighbors_v1), 3)
        neighbor_indices_v1 = sorted([n.particle_index for n in all_neighbors_v1])
        np.testing.assert_equal(neighbor_indices_v1, [0, 2, 4])

        all_neighbors_v2 = vertex_2.neighbors()
        np.testing.assert_equal(len(all_neighbors_v2), 3)
        neighbor_indices_v2 = sorted([n.particle_index for n in all_neighbors_v2])
        np.testing.assert_equal(neighbor_indices_v2, [1, 3, 5])

        # Test 2: Filter neighbors by exact particle type
        # vertex_1 (TypeB) has neighbors: vertex_0 (TypeA), vertex_2 (TypeA), vertex_4 (TypeB)
        type_a_neighbors_v1 = vertex_1.neighbors(vertex_type="TypeA", exact_match=True)
        np.testing.assert_equal(len(type_a_neighbors_v1), 2)  # vertices 0 and 2 are both TypeA
        type_a_indices = sorted([n.particle_index for n in type_a_neighbors_v1])
        np.testing.assert_equal(type_a_indices, [0, 2])

        type_b_neighbors_v1 = vertex_1.neighbors(vertex_type="TypeB", exact_match=True)
        np.testing.assert_equal(len(type_b_neighbors_v1), 1)
        np.testing.assert_equal(type_b_neighbors_v1[0].particle_index, 4)

        type_ab_neighbors_v2 = vertex_2.neighbors(vertex_type="TypeAB", exact_match=True)
        np.testing.assert_equal(len(type_ab_neighbors_v2), 1)
        np.testing.assert_equal(type_ab_neighbors_v2[0].particle_index, 3)

        # Test 3: Filter neighbors by substring match (exact_match=False)
        # vertex_2 neighbors: vertex_1 (TypeB), vertex_3 (TypeAB), vertex_5 (TypeB)
        type_containing_a_v2 = vertex_2.neighbors(vertex_type="A", exact_match=False)
        np.testing.assert_equal(len(type_containing_a_v2), 1)  # TypeAB only
        np.testing.assert_equal(type_containing_a_v2[0].particle_index, 3)

        type_containing_b_v2 = vertex_2.neighbors(vertex_type="B", exact_match=False)
        np.testing.assert_equal(len(type_containing_b_v2), 3)  # TypeB (vertices 1,5) + TypeAB (vertex 3)
        type_b_indices = sorted([n.particle_index for n in type_containing_b_v2])
        np.testing.assert_equal(type_b_indices, [1, 3, 5])

        # Test 4: Filter neighbors by degree
        # vertex_1 neighbors: vertex_0 (degree 1), vertex_2 (degree 3), vertex_4 (degree 1)
        degree_1_neighbors_v1 = vertex_1.neighbors(vertex_degree=1)
        np.testing.assert_equal(len(degree_1_neighbors_v1), 2)  # vertices 0 and 4
        degree_1_indices = sorted([n.particle_index for n in degree_1_neighbors_v1])
        np.testing.assert_equal(degree_1_indices, [0, 4])

        degree_3_neighbors_v1 = vertex_1.neighbors(vertex_degree=3)
        np.testing.assert_equal(len(degree_3_neighbors_v1), 1)
        np.testing.assert_equal(degree_3_neighbors_v1[0].particle_index, 2)

        # vertex_2 neighbors: vertex_1 (degree 3), vertex_3 (degree 1), vertex_5 (degree 2)
        degree_1_neighbors_v2 = vertex_2.neighbors(vertex_degree=1)
        np.testing.assert_equal(len(degree_1_neighbors_v2), 1)
        np.testing.assert_equal(degree_1_neighbors_v2[0].particle_index, 3)

        # Test 5: Combined filters - type and degree
        # vertex_1 has TypeA neighbors at indices 0 (degree 1) and 2 (degree 3)
        type_a_degree_1_v1 = vertex_1.neighbors(vertex_type="TypeA", vertex_degree=1, exact_match=True)
        np.testing.assert_equal(len(type_a_degree_1_v1), 1)
        np.testing.assert_equal(type_a_degree_1_v1[0].particle_index, 0)  # vertex 0 has degree 1
        
        type_a_degree_3_v1 = vertex_1.neighbors(vertex_type="TypeA", vertex_degree=3, exact_match=True)
        np.testing.assert_equal(len(type_a_degree_3_v1), 1)
        np.testing.assert_equal(type_a_degree_3_v1[0].particle_index, 2)  # vertex 2 has degree 3

        type_b_degree_1_v1 = vertex_1.neighbors(vertex_type="TypeB", vertex_degree=1, exact_match=True)
        np.testing.assert_equal(len(type_b_degree_1_v1), 1)
        np.testing.assert_equal(type_b_degree_1_v1[0].particle_index, 4)

        # Test 6: Filter neighbors by particle ID
        target_neighbor = vertex_1.neighbors()[0]  # Get first neighbor
        target_id = top.particle_id_of_vertex(target_neighbor)

        id_filtered_neighbors = vertex_1.neighbors(vertex_id=target_id)
        np.testing.assert_equal(len(id_filtered_neighbors), 1)
        np.testing.assert_equal(id_filtered_neighbors[0].particle_index, target_neighbor.particle_index)

        # Test 7: Filter by particle ID with additional type filter (should match)
        target_type = target_neighbor.particle_type()
        id_type_filtered = vertex_1.neighbors(vertex_id=target_id, vertex_type=target_type, exact_match=True)
        np.testing.assert_equal(len(id_type_filtered), 1)
        np.testing.assert_equal(id_type_filtered[0].particle_index, target_neighbor.particle_index)

        # Test 8: Filter by particle ID with incompatible type filter (should not match)
        incompatible_type = "TypeAB" if target_type != "TypeAB" else "TypeA"
        id_type_no_match = vertex_1.neighbors(vertex_id=target_id, vertex_type=incompatible_type, exact_match=True)
        np.testing.assert_equal(len(id_type_no_match), 0)

        # Test 9: Filter by particle ID with additional degree filter (should match)
        target_degree = len(target_neighbor.neighbors())
        id_degree_filtered = vertex_1.neighbors(vertex_id=target_id, vertex_degree=target_degree)
        np.testing.assert_equal(len(id_degree_filtered), 1)
        np.testing.assert_equal(id_degree_filtered[0].particle_index, target_neighbor.particle_index)

        # Test 10: Filter by particle ID with incompatible degree filter (should not match)
        incompatible_degree = target_degree + 1
        id_degree_no_match = vertex_1.neighbors(vertex_id=target_id, vertex_degree=incompatible_degree)
        np.testing.assert_equal(len(id_degree_no_match), 0)

        # Test 11: Filter by non-existent particle ID
        non_existent_id = 99999
        no_match_neighbors = vertex_1.neighbors(vertex_id=non_existent_id)
        np.testing.assert_equal(len(no_match_neighbors), 0)

        # Test 12: Filter with impossible combinations
        # vertex_1 has no TypeA neighbors with degree 2 (vertex 0 has degree 1, vertex 2 has degree 3)
        impossible_combo = vertex_1.neighbors(vertex_type="TypeA", vertex_degree=2, exact_match=True)
        np.testing.assert_equal(len(impossible_combo), 0)

        # Test 13: Verify vertex with no neighbors handles filters correctly
        vertex_0 = next(v for v in vertices if v.particle_index == 0)  # Has only 1 neighbor
        vertex_6 = next(v for v in vertices if v.particle_index == 6)  # Has only 1 neighbor
        
        # These should return empty lists for filters that don't match their single neighbor
        no_type_match = vertex_0.neighbors(vertex_type="TypeAB", exact_match=True)
        np.testing.assert_equal(len(no_type_match), 0)
        
        no_degree_match = vertex_6.neighbors(vertex_degree=1)  # vertex 6's neighbor (vertex 5) has degree 2
        np.testing.assert_equal(len(no_degree_match), 0)

        # Test 14: Test filtering with multiple compatible neighbors
        # vertex_2 has neighbors: TypeB(1), TypeAB(3), TypeB(5)
        # Filter for all TypeB neighbors
        all_type_b_neighbors = vertex_2.neighbors(vertex_type="TypeB", exact_match=True)
        np.testing.assert_equal(len(all_type_b_neighbors), 2)
        type_b_indices = sorted([n.particle_index for n in all_type_b_neighbors])
        np.testing.assert_equal(type_b_indices, [1, 5])

if __name__ == '__main__':
    unittest.main()
