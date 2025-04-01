# Optimized version of the TrajectoryData class for larger simulations

import copy
import logging
from typing import Any, Dict, List

import numpy as np

from src.visualization.agent_data_custom import AgentData
from simulariumio import UnitData, MetaData, DisplayData, DimensionData

###############################################################################

log = logging.getLogger(__name__)

###############################################################################

BUFFER_SIZE_INC: DimensionData = DimensionData(
    total_steps=1,
    max_agents=1,
    max_subpoints=1,
)


class TrajectoryData:
    meta_data: MetaData
    agent_data: AgentData
    time_units: UnitData
    spatial_units: UnitData
    plots: List[Dict[str, Any]]

    def __init__(
        self,
        meta_data: MetaData,
        agent_data: AgentData,
        time_units: UnitData = None,
        spatial_units: UnitData = None,
        plots: List[Dict[str, Any]] = None,
    ):
        """
        This object holds simulation trajectory outputs
        and plot data

        Parameters
        ----------
        meta_data : MetaData
            An object containing metadata for the trajectory
            including box size, scale factor, and camera defaults
        agent_data : AgentData
            An object containing data for each agent
            at each timestep
        time_units: UnitData (optional)
            multiplier and unit name for time values
            Default: 1.0 second
        spatial_units: UnitData (optional)
            multiplier and unit name for spatial values
            Default: 1.0 meter
        plots : List[Dict[str, Any]] (optional)
            An object containing plot data already
            in Simularium format
        """
        self.meta_data = meta_data
        self.agent_data = agent_data
        self.time_units = time_units if time_units is not None else UnitData("s")
        self.spatial_units = (
            spatial_units if spatial_units is not None else UnitData("m")
        )
        self.plots = plots if plots is not None else []

    @classmethod
    def from_buffer_data(
        cls, buffer_data: Dict[str, Any], display_data: Dict[int, DisplayData] = None
    ):
        """
        Create TrajectoryData from a simularium JSON dict containing buffers
        """
        if display_data is None:
            display_data = {}
        return cls(
            meta_data=MetaData.from_dict(buffer_data["trajectoryInfo"]),
            agent_data=AgentData.from_buffer_data(buffer_data, display_data),
            time_units=UnitData.from_dict(
                buffer_data["trajectoryInfo"]["timeUnits"], default_mag=1.0
            ),
            spatial_units=UnitData.from_dict(
                buffer_data["trajectoryInfo"]["spatialUnits"], default_mag=1.0
            ),
            plots=buffer_data["plotData"]["data"],
        )

    # def append_agents(self, new_agents: AgentData):
    #     """
    #     Concatenate the new AgentData with the current data,
    #     generate new unique IDs and type IDs as needed
    #     """
    #     # create appropriate length buffer with current agents
    #     current_dimensions = self.agent_data.get_dimensions()
    #     added_dimensions = new_agents.get_dimensions()
    #     new_dimensions = current_dimensions.add(added_dimensions, axis=1)
    #     result = self.agent_data.check_increase_buffer_size(
    #         new_dimensions.max_agents - 1, axis=1, buffer_size_inc=BUFFER_SIZE_INC
    #     )
    #     # add new agents
    #     result.n_agents = np.add(result.n_agents, new_agents.n_agents)
    #     start_i = current_dimensions.max_agents
    #     end_i = start_i + added_dimensions.max_agents
    #     result.viz_types[:, start_i:end_i] = new_agents.viz_types[:]
    #     result.positions[:, start_i:end_i] = new_agents.positions[:]
    #     result.radii[:, start_i:end_i] = new_agents.radii[:]
    #     result.rotations[:, start_i:end_i] = new_agents.rotations[:]
    #     result.n_subpoints[:, start_i:end_i] = new_agents.n_subpoints[:]
    #     if len(new_agents.subpoints.shape) > 2:
    #         result.subpoints[:, start_i:end_i] = new_agents.subpoints[:]
    #     # generate new unique IDs and type IDs so they don't overlap
    #     used_uids = list(np.unique(self.agent_data.unique_ids).astype(int))
    #     new_uids = {}
    #     for time_index in range(new_dimensions.total_steps):
    #         new_agent_index = int(self.agent_data.n_agents[time_index])
    #         n_a = int(new_agents.n_agents[time_index])
    #         for agent_index in range(n_a):
    #             raw_uid = int(new_agents.unique_ids[time_index][agent_index])
    #             if raw_uid not in new_uids:
    #                 uid = raw_uid
    #                 while uid in used_uids:
    #                     uid += 1
    #                 new_uids[raw_uid] = uid
    #                 used_uids.append(uid)
    #             result.unique_ids[time_index][new_agent_index] = new_uids[raw_uid]
    #             result.types[time_index].append(
    #                 new_agents.types[time_index][agent_index]
    #             )
    #             new_agent_index += 1
    #     result.display_data.update(new_agents.display_data)
    #     self.agent_data = result

    import numpy as np

    def append_agents(self, new_agents: AgentData):
        """
        Concatenate the new AgentData with the current data,
        generate new unique IDs and type IDs as needed
        """
        # create appropriate length buffer with current agents
        current_dimensions = self.agent_data.get_dimensions()
        added_dimensions = new_agents.get_dimensions()
        new_dimensions = current_dimensions.add(added_dimensions, axis=1)
        result = self.agent_data.check_increase_buffer_size(
            new_dimensions.max_agents - 1, axis=1, buffer_size_inc=BUFFER_SIZE_INC
        )

        # add new agents
        result.n_agents = np.add(result.n_agents, new_agents.n_agents)
        start_i = current_dimensions.max_agents
        end_i = start_i + added_dimensions.max_agents
        result.viz_types[:, start_i:end_i] = new_agents.viz_types
        result.positions[:, start_i:end_i] = new_agents.positions
        result.radii[:, start_i:end_i] = new_agents.radii
        result.rotations[:, start_i:end_i] = new_agents.rotations
        result.n_subpoints[:, start_i:end_i] = new_agents.n_subpoints

        if new_agents.subpoints.ndim > 2:
            result.subpoints[:, start_i:end_i] = new_agents.subpoints

        # generate new unique IDs and type IDs so they don't overlap
        used_uids = np.unique(self.agent_data.unique_ids).tolist()
        uid_map = {}

        # Vectorized unique id generation
        new_uids = np.vectorize(lambda raw_uid: raw_uid if raw_uid not in uid_map else uid_map[raw_uid])(
            new_agents.unique_ids)
        raw_uids = np.unique(new_agents.unique_ids)

        for raw_uid in raw_uids:
            if raw_uid not in uid_map:
                uid = raw_uid
                while uid in used_uids:
                    uid += 1
                uid_map[raw_uid] = uid
                used_uids.append(uid)

        for time_index in range(new_dimensions.total_steps):
            new_agent_index = int(self.agent_data.n_agents[time_index])
            n_a = int(new_agents.n_agents[time_index])
            raw_uids = new_agents.unique_ids[time_index, :n_a]
            uids = np.vectorize(uid_map.get)(raw_uids)

            result.unique_ids[time_index, new_agent_index:new_agent_index + n_a] = uids
            result.types[time_index].extend(new_agents.types[time_index, :n_a])

        result.display_data.update(new_agents.display_data)
        self.agent_data = result

    def append_agents_with_subpoints(self, new_agents: AgentData):
      """
          Concatenate the new AgentData with the current data,
          generate new unique IDs and type IDs as needed
          """
      # create appropriate length buffer with current agents
      current_dimensions = self.agent_data.get_dimensions()
      added_dimensions = new_agents.get_dimensions()
      new_dimensions = current_dimensions.add(added_dimensions, axis=1)
      result = self.agent_data.check_increase_buffer_size(
        new_dimensions.max_agents - 1, axis=1, buffer_size_inc=BUFFER_SIZE_INC
      )
      # add new agents
      result.n_agents = np.add(result.n_agents, new_agents.n_agents)
      start_i = current_dimensions.max_agents
      end_i = start_i + added_dimensions.max_agents
      result.viz_types[:, start_i:end_i] = new_agents.viz_types[:]
      result.positions[:, start_i:end_i] = new_agents.positions[:]
      result.radii[:, start_i:end_i] = new_agents.radii[:]
      result.rotations[:, start_i:end_i] = new_agents.rotations[:]
      result.n_subpoints[:, start_i:end_i] = new_agents.n_subpoints[:]

      # Check if new_agents.subpoints has a valid shape before assigning
      if new_agents.subpoints.size > 0:  # Ensures there are subpoints to add
        result.subpoints[:, start_i:end_i] = new_agents.subpoints[:]
      else:
        # Handle the case where there are no subpoints to add
        logging.warning(f"new_agents.subpoints is empty for agents in range {start_i} to {end_i}")

      # generate new unique IDs and type IDs so they don't overlap
      used_uids = list(np.unique(self.agent_data.unique_ids).astype(int))
      new_uids = {}
      for time_index in range(new_dimensions.total_steps):
        new_agent_index = int(self.agent_data.n_agents[time_index])
        n_a = int(new_agents.n_agents[time_index])
        for agent_index in range(n_a):
          raw_uid = int(new_agents.unique_ids[time_index][agent_index])
          if raw_uid not in new_uids:
            uid = raw_uid
            while uid in used_uids:
              uid += 1
            new_uids[raw_uid] = uid
            used_uids.append(uid)
          result.unique_ids[time_index][new_agent_index] = new_uids[raw_uid]
          result.types[time_index].append(
            new_agents.types[time_index][agent_index]
          )
          new_agent_index += 1
      result.display_data.update(new_agents.display_data)
      self.agent_data = result

    def append_agents_with_subpoints_optimized(self, new_agents: AgentData):
      """
      Concatenate the new AgentData with the current data,
      generate new unique IDs and type IDs as needed
      """
      # create appropriate length buffer with current agents
      current_dimensions = self.agent_data.get_dimensions()
      added_dimensions = new_agents.get_dimensions()
      new_dimensions = current_dimensions.add(added_dimensions, axis=1)
      result = self.agent_data.check_increase_buffer_size(
        new_dimensions.max_agents - 1, axis=1, buffer_size_inc=BUFFER_SIZE_INC
      )
      # add new agents
      result.n_agents = np.add(result.n_agents, new_agents.n_agents)
      start_i = current_dimensions.max_agents
      end_i = start_i + added_dimensions.max_agents
      result.viz_types[:, start_i:end_i] = new_agents.viz_types[:]
      result.positions[:, start_i:end_i] = new_agents.positions[:]
      result.radii[:, start_i:end_i] = new_agents.radii[:]
      result.rotations[:, start_i:end_i] = new_agents.rotations[:]
      result.n_subpoints[:, start_i:end_i] = new_agents.n_subpoints[:]

      if new_agents.subpoints.size > 0:  # Ensures there are subpoints to add
        result.subpoints[:, start_i:end_i] = new_agents.subpoints[:]
      else:
        logging.warning(f"new_agents.subpoints is empty for agents in range {start_i} to {end_i}")

      # Efficient unique ID generation using vectorized operations
      used_uids = np.unique(self.agent_data.unique_ids)
      new_uids = np.zeros_like(new_agents.unique_ids)

      # Vectorized approach to generate new unique IDs
      for time_index in range(new_dimensions.total_steps):
        new_uids_at_time = new_agents.unique_ids[time_index]
        offset = used_uids.max() + 1  # Start new IDs from max existing UID + 1

        # Create a mapping for new unique IDs
        unique_mapping = {old_uid: new_uid for old_uid, new_uid in zip(
          np.unique(new_uids_at_time),
          np.arange(offset, offset + len(np.unique(new_uids_at_time)))
        )}

        # Apply the mapping
        new_uids[time_index] = np.vectorize(unique_mapping.get)(new_uids_at_time)
        used_uids = np.concatenate((used_uids, list(unique_mapping.values())))

      # Update unique IDs and types in a vectorized manner
      result.unique_ids = np.hstack((self.agent_data.unique_ids, new_uids))

      # Use NumPy concatenate to efficiently append new types
      for time_index in range(new_dimensions.total_steps):
        result.types[time_index] = np.concatenate(
          (result.types[time_index], new_agents.types[time_index])
        )

      result.display_data.update(new_agents.display_data)
      self.agent_data = result

    def __deepcopy__(self, memo):
        result = type(self)(
            meta_data=copy.deepcopy(self.meta_data, memo),
            agent_data=copy.deepcopy(self.agent_data, memo),
            time_units=copy.copy(self.time_units),
            spatial_units=copy.copy(self.spatial_units),
            plots=copy.deepcopy(self.plots, memo),
        )
        return result

    def __eq__(self, other):
        if isinstance(other, TrajectoryData):
            return (
                self.meta_data == other.meta_data
                and self.agent_data == other.agent_data
                and self.time_units == other.time_units
                and self.spatial_units == other.spatial_units
                and self.plots == other.plots
            )
        return False
