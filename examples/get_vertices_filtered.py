#!/usr/bin/env python3
"""
Demonstration script for the extended get_vertices method functionality.

This script shows how to use the new filtering capabilities of the 
topology.graph.get_vertices() method.
"""

import readdy._internal.readdybinding.common as common
from readdy._internal.readdybinding.api import BondedPotentialConfiguration
from readdy._internal.readdybinding.api import ParticleTypeFlavor
from readdy._internal.readdybinding.api import Simulation
from readdy._internal.readdybinding.api import Context


def main():
    print("=== Extended get_vertices() Method Demonstration ===\n")
    
    # Setup context and particle types
    context = Context()
    context.box_size = [10., 10., 10.]
    context.topologies.add_type("TA")
    context.particle_types.add("TypeA", 1.0, flavor=ParticleTypeFlavor.TOPOLOGY)
    context.particle_types.add("TypeB", 1.0, flavor=ParticleTypeFlavor.TOPOLOGY)
    context.particle_types.add("TypeAB", 1.0, flavor=ParticleTypeFlavor.TOPOLOGY)
    
    # Configure bond potentials
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
    
    print("Created topology with the following structure:")
    print("TypeA(0) - TypeB(1) - TypeA(2) - TypeAB(3)")
    print("             |")
    print("          TypeB(4)")
    print()
    
    # Demonstrate different filtering options
    print("1. Get all vertices (no filters):")
    all_vertices = graph.get_vertices(vertex_type="Type", exact_match=False)
    print(f"   Found {len(all_vertices)} vertices")
    for v in all_vertices:
        vertex_type = top.particle_type_of_vertex(v)
        print(f"   - Vertex {v.particle_index}: {vertex_type} (degree: {len(v)})")
    print()

    print("2. Get all vertices with 'Type' in the vertex type:")
    all_vertices = graph.get_vertices(vertex_type="Type", exact_match=False)
    print(f"   Found {len(all_vertices)} vertices")
    for v in all_vertices:
        vertex_type = top.particle_type_of_vertex(v)
        print(f"   - Vertex {v.particle_index}: {vertex_type} (degree: {len(v)})")
    print()
    
    print("3. Filter by exact particle type:")
    type_a_vertices = graph.get_vertices(vertex_type="TypeA", exact_match=True)
    print(f"   TypeA vertices: {[v.particle_index for v in type_a_vertices]}")
    
    type_b_vertices = graph.get_vertices(vertex_type="TypeB", exact_match=True)
    print(f"   TypeB vertices: {[v.particle_index for v in type_b_vertices]}")
    print()
    
    print("4. Filter by substring match (exact_match=False):")
    containing_a = graph.get_vertices(vertex_type="A", exact_match=False)
    print(f"   Vertices containing 'A': {[v.particle_index for v in containing_a]}")
    
    containing_b = graph.get_vertices(vertex_type="B", exact_match=False)
    print(f"   Vertices containing 'B': {[v.particle_index for v in containing_b]}")
    print()
    
    print("5. Filter by degree:")
    degree_1 = graph.get_vertices(vertex_degree=1)
    print(f"   Degree 1 vertices: {[v.particle_index for v in degree_1]}")
    
    degree_2 = graph.get_vertices(vertex_degree=2)
    print(f"   Degree 2 vertices: {[v.particle_index for v in degree_2]}")
    
    degree_3 = graph.get_vertices(vertex_degree=3)
    print(f"   Degree 3 vertices: {[v.particle_index for v in degree_3]}")
    print()
    
    print("6. Combined filters (type + degree):")
    type_a_degree_1 = graph.get_vertices(vertex_type="TypeA", vertex_degree=1, exact_match=True)
    print(f"   TypeA vertices with degree 1: {[v.particle_index for v in type_a_degree_1]}")
    
    type_b_degree_1 = graph.get_vertices(vertex_type="TypeB", vertex_degree=1, exact_match=True)
    print(f"   TypeB vertices with degree 1: {[v.particle_index for v in type_b_degree_1]}")
    print()
    
    print("7. Filter by particle ID:")
    target_vertex = all_vertices[2]  # TypeA vertex at index 2
    target_id = top.particle_id_of_vertex(target_vertex)
    print(f"   Looking for vertex with particle ID {target_id}")
    
    id_filtered = graph.get_vertices(vertex_id=target_id)
    if id_filtered:
        v = id_filtered[0]
        vertex_type = top.particle_type_of_vertex(v)
        print(f"   Found: Vertex {v.particle_index} ({vertex_type}, degree: {len(v)})")
    print()
    
    print("8. Filter by particle ID with additional constraints:")
    # This should match (TypeA vertex with degree 2)
    id_type_match = graph.get_vertices(vertex_id=target_id, vertex_type="TypeA", exact_match=True)
    print(f"   ID {target_id} + TypeA: {len(id_type_match) > 0}")
    
    # This should not match (TypeA vertex has degree 2, not 1)
    id_degree_no_match = graph.get_vertices(vertex_id=target_id, vertex_degree=1)
    print(f"   ID {target_id} + degree 1: {len(id_degree_no_match) > 0}")
    
    print("\n=== Demonstration complete ===")


if __name__ == "__main__":
    main()
