#!/usr/bin/env python
# -*- coding: utf-8 -*-

import logging
from typing import Any, Tuple, Callable

import numpy as np
import readdy

from simulariumio.trajectory_converter import TrajectoryConverter
from simulariumio.data_objects import TrajectoryData, AgentData, DimensionData, DisplayData
from simulariumio.constants import DISPLAY_TYPE, VIZ_TYPE
from simulariumio.readdy.readdy_data import ReaddyData
from simulariumio.exceptions import InputDataError

###############################################################################

log = logging.getLogger(__name__)

###############################################################################


class ReaddyConverter(TrajectoryConverter):
    def __init__(
        self,
        input_data: ReaddyData,
        progress_callback: Callable[[float], None] = None,
        callback_interval: float = 10,
    ):
        """
        This object reads simulation trajectory outputs
        from ReaDDy (https://readdy.github.io/)
        and plot data and writes them in the JSON format used
        by the Simularium viewer

        Parameters
        ----------
        input_data : ReaddyData
            An object containing info for reading
            ReaDDy simulation trajectory outputs and plot data
        progress_callback : Callable[[float], None] (optional)
            Callback function that accepts 1 float argument and returns None
            which will be called at a given progress interval, determined by
            callback_interval requested, providing the current percent progress
            Default: None
        callback_interval : float (optional)
            If a progress_callback was provided, the period between updates
            to be sent to the callback, in seconds
            Default: 10
        """
        super().__init__(input_data, progress_callback, callback_interval)
        self._data = self._read(input_data)

    @staticmethod
    def _get_raw_trajectory_data(
        input_data: ReaddyData,
    ) -> Tuple[AgentData, Any, np.ndarray]:
        """
        Return agent data populated from a ReaDDy .h5 trajectory file
        """
        # load the trajectory
        traj = readdy.Trajectory(input_data.path_to_readdy_h5)
        n_agents, positions, type_ids, ids = traj.to_numpy(start=0, stop=None)
        return (traj, n_agents, positions, type_ids, ids)

    def _get_agent_data(self, input_data: ReaddyData) -> AgentData:
        """
        Pack raw ReaDDy trajectory data into AgentData,
        ignoring particles with type names in ignore_types.
        """
        traj, n_agents, positions, type_ids, ids = ReaddyConverter._get_raw_trajectory_data(input_data)
        data_dimensions = DimensionData(
            total_steps=n_agents.shape[0],
            max_agents=int(np.amax(n_agents)),
        )
        result = AgentData.from_dimensions(data_dimensions)
        result.times = input_data.timestep * np.arange(data_dimensions.total_steps)
        result.viz_types = np.full(
            shape=(data_dimensions.total_steps, data_dimensions.max_agents),
            fill_value=VIZ_TYPE.DEFAULT
        )

        ignore_types = set(input_data.ignore_types)  # Convert to set for O(1) membership testing

        for time_index in range(data_dimensions.total_steps):
            traj_time = traj
            agent_count = int(n_agents[time_index])
            tids = type_ids[time_index]
            pos = positions[time_index]
            ids_time = ids[time_index]

            new_agent_index = 0
            for agent_index in range(agent_count):
                tid = tids[agent_index]
                type_name = traj_time.species_name(tid)

                if type_name in ignore_types:
                    continue

                input_display_data = TrajectoryConverter._get_display_data_for_agent(
                    type_name, input_data.display_data
                )
                display_data = input_display_data if input_display_data is not None else DisplayData(
                    name=type_name, display_type=DISPLAY_TYPE.SPHERE
                )

                result.unique_ids[time_index][new_agent_index] = ids_time[agent_index]
                result.types[time_index].append(display_data.name)
                result.display_data[display_data.name] = display_data
                result.positions[time_index][new_agent_index] = pos[agent_index]
                result.radii[time_index][new_agent_index] = display_data.radius or 1.0

                new_agent_index += 1

            result.n_agents[time_index] = new_agent_index
            self.check_report_progress(time_index / data_dimensions.total_steps)

        return TrajectoryConverter.scale_agent_data(result, input_data.meta_data.scale_factor)

    def _read(
        self,
        input_data: ReaddyData,
    ) -> TrajectoryData:
        """
        Return an object containing the data shaped for Simularium format
        """
        print("Reading ReaDDy Data -------------")
        try:
            agent_data, scale_factor = self._get_agent_data(input_data)
        except Exception as e:
            raise InputDataError(f"Error reading input Readdy data: {e}")

        # get display data (geometry and color)
        for tid in input_data.display_data:
            display_data = input_data.display_data[tid]
            agent_data.display_data[display_data.name] = display_data
        input_data.spatial_units.multiply(1.0 / scale_factor)
        input_data.meta_data.scale_factor = scale_factor
        input_data.meta_data._set_box_size(np.array([0., 0., 0.]))
        return TrajectoryData(
            meta_data=input_data.meta_data,
            agent_data=agent_data,
            time_units=input_data.time_units,
            spatial_units=input_data.spatial_units,
            plots=input_data.plots,
        )
