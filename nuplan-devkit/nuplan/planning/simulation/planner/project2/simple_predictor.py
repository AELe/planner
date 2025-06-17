import numpy as np
from typing import List, Type, Optional, Tuple
from nuplan.planning.simulation.observation.observation_type import DetectionsTracks, Observation
from nuplan.common.actor_state.ego_state import DynamicCarState, EgoState
from nuplan.planning.simulation.planner.project2.abstract_predictor import AbstractPredictor
from nuplan.common.actor_state.tracked_objects_types import TrackedObjectType
from nuplan.common.actor_state.agent import Agent
from nuplan.planning.simulation.trajectory.predicted_trajectory import PredictedTrajectory
from nuplan.common.actor_state.waypoint import Waypoint
from nuplan.common.actor_state.state_representation import StateSE2, StateVector2D, TimePoint
from nuplan.common.actor_state.oriented_box import OrientedBox


class SimplePredictor(AbstractPredictor):
    def __init__(self, ego_state: EgoState, observations: Observation, duration: float, sample_time: float) -> None:
        self._ego_state = ego_state
        self._observations = observations
        self._duration = duration
        self._sample_time = sample_time
        self._occupancy_map_radius = 40

 

    def predict(self):
        """使用定速度模型预测物体轨迹"""
        if isinstance(self._observations, DetectionsTracks):
            prediction_time_step = 0.1  # 采样时间间隔（秒）
            prediction_horizon = 3.0    # 预测总时长（秒）
            
            objects_init = self._observations.tracked_objects.tracked_objects
            objects = [
                obj for obj in objects_init
                if np.linalg.norm(self._ego_state.center.array - obj.center.array) < self._occupancy_map_radius
            ]

            for obj in objects:
                current_pos = obj.center.array[:2]
                current_vel = obj.velocity.array[:2] if hasattr(obj, 'velocity') else np.zeros(2)
                current_heading = obj.center.heading if hasattr(obj.center, 'heading') else 0.0
                
                waypoints = []
                for t in np.arange(0, prediction_horizon + prediction_time_step, prediction_time_step):
                    predicted_pos = current_pos + current_vel * t
                     
                    center = StateSE2(
                        x=float(predicted_pos[0]),
                        y=float(predicted_pos[1]),
                        heading=current_heading
                    )
                     
                    oriented_box = OrientedBox(
                        center=center,
                        length=obj.box.length if hasattr(obj, 'box') else 4.0,
                        width=obj.box.width if hasattr(obj, 'box') else 2.0,
                        height=obj.box.height if hasattr(obj, 'box') else 1.5
                    )
                    
                    waypoint = Waypoint(
                        time_point=TimePoint(int(t * 1e6)),
                        oriented_box=oriented_box,
                        velocity=StateVector2D(x=float(current_vel[0]), y=float(current_vel[1]))
                    )
                    waypoints.append(waypoint)

                obj.predictions = [PredictedTrajectory(
                    probability=1.0,
                    waypoints=waypoints
                )]

            return objects
        else:
            raise ValueError(f"Unsupported input type: {type(self._observations)}")

