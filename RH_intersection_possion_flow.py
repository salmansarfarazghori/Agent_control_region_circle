# -*- coding: utf-8 -*-
# =============================================================================
# Title: MILP-based Cross-Flow Intersection Simulator (X–Y flows, circular control zone)
# Description:
#   - Discrete-time MILP/MPC controller that schedules autonomous agents through
#     a single intersection within a circular control region of radius `self.radius`.
#   - Supports fairness (order-reversal) constraints, safe-separation, and
#     arrival processes for two orthogonal flows (x and y).
#   - Logs solver MIP gaps and writes per-iteration positions for visualization.
#
# How to run:
#   - Requires: Python 3.x, gurobipy (Gurobi), numpy, matplotlib.
#   - Execute this file directly; outputs JSON logs, GIF (if enabled),
#     and Gurobi logs into a dated results directory.
#
# Citation:
#   If you use or build upon this code in academic work, please cite the
#   associated research paper and repository.
#
# License:

# =============================================================================

import sys
import gurobipy as gp
import numpy as np
from gurobipy import GRB
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.patches as patches
import time
from datetime import datetime
import matplotlib.collections
import json
import random
import itertools
import uuid
import math
import os
# import psutil  # Optional: for memory profiling

class traj():
    """
    Trajectory/Simulation container for MILP-based cross-flow intersection control.

    This class manages:
      - Agent arrivals on x and y axes (stochastic inter-arrival times).
      - Building a MILP at each control step with state/control variables, safety,
        intersection, and optional fairness constraints.
      - Solving with Gurobi, logging MIP gaps, and storing per-step agent positions
        for visualization and post-processing.

    Notes on important attributes:
      N:              MPC horizon (timesteps).
      delta_t:        Discretization step [s].
      radius:         Control-zone radius [m]; vehicles are controlled inside the circle.
      fair:           If True, activates fairness (reversal) constraints.
      safe_dis:       Same-lane safe separation distance [m].
      buff_ins:       Center buffer (sum of |x|+|y| distance) to avoid conflicts.
      Big_M, Big_W:   Big-M constants for indicator/logic constraints.
      V_max, A_max:   Speed/accel bounds used in constraints/objective shaping.
      lambda_x/y:     Poisson/exponential arrival rates (vehicles per second).
      min_steps:      Minimum discrete steps between arrivals to avoid overlap.
      directory_name: Output folder name for logs/plots of current run.

    The flow:
      simulation_loop() -> update_vehicle_positions() -> default() which:
        -> setup_variables(), setup_constraints(), setup_objective(), optimize()
        -> update_vehicle_state(), remove_vehicle()
    """

    def __init__(self):

        # Make runs reproducible
        random.seed(20)
        np.random.seed(20)

        # MPC horizon and discretization
        self.N = 60
        self.delta_t = 0.1  # Time step (s)

        # Polygon approximation for norms (not used directly in current constraints)
        self.N_polygon = 8
        self.total_iterations = 36000  # Total sim steps

        # Vehicle counters for ID assignment
        self.vehicle_id_x = 0
        self.vehicle_id_y = 0
        self.vehicle_id_x_y = 0

        # Control-zone radius (meters)
        self.radius = 60

        # Stop adding vehicles in last stop_add iterations
        self.stop_add = 0

        # Fairness toggle and safety parameters
        self.fair = False
        self.safe_dis = 3         # Same-lane spacing [m]
        self.buff_ins = 4         # Center buffer for conflict avoidance [m]
        self.proximity = 4        # Fairness close-pair proximity threshold [m]
        self.Big_M = 1e3          # Big-M for logic constraints
        self.Big_W = 1e6          # Extra large M for strong relaxation

        # Problem sizes
        self.K = 0  # Number of controlled vehicles in the current MILP
        self.L = 0  # Number of stationary obstacles (unused placeholder)

        # Bounds and weights
        self.V_min = 0
        self.V_max = 22
        self.A_max = 3
        self.theta = [2 * np.pi * k / self.N_polygon for k in range(1, self.N_polygon+1)]

        # State bounds: [x, y, xdot, ydot]
        # These include a bit of slack beyond control zone to keep MILP well-posed
        self.smin = [-(self.radius + 2), -(self.radius + 29), -15, -15]
        self.smax = [ self.radius + 29,  self.radius + 2,   15,  15]

        # Control bounds: [ax, ay]
        self.umin = [-5, -5]
        self.umax = [ 5,  5]

        # Objective weights
        self.qo = [1] * 4         # Slack (state deviation) weights
        self.ro = [1] * 2         # (Unused here) control-effort auxiliary weights
        self.po = [100] * 4       # Terminal slack weights
        self.reversal_cost = 1    # Penalty for fairness relaxation (S_pq)
        self.velocity_weight = 1  # Weight for velocity term

        # (Legacy) Bernoulli-block arrival parameters (some kept for reproducibility)
        self.lambda_rate = 0.9
        self.block_options = [2, 3, 4]
        self.block_weights = [0.1, 0.1, 0.8]
        self.next_block_x = 0
        self.next_block_y = 0

        # Same-lane minimal distance (unused placeholders d_x, d_y kept)
        self.d_x = 1
        self.d_y = 1

        self.constraint_texts = []         # Stores subset of constraint names when parsing IIS
        self.vehicles_positions = []       # For plotting
        self.hlines = []                   # For plotting

        # Color utilities for plotting
        self.colors = [
            'b', 'g', 'r', 'c', 'm', 'y', 'k',
            '#FF5733', '#33FF57', '#3357FF', '#FF33FC', '#57FFC4',
            '#FFD133', '#8D33FF', '#FF5733', '#33D5FF'
        ]
        self.color_cycle = itertools.cycle(self.colors)
        self.vehicle_colors = {}  # vehicle_id -> color
        self.color_index = 0

        # Lane geometry (straight, orthogonal flows)
        self.lane_start_x, self.lane_start_y = -240, 240
        self.lane_end_x,   self.lane_end_y   =  240, -240

        self.vehicle_velocity = 15  # Nominal cruise speed magnitude

        # Runtime vehicle lists (position, axis flag, numeric IDs)
        self.vehicles_x = []
        self.vehicles_y = []
        self.vehicle_it1 = []
        self.vehicle_it2 = []

        # Per-iteration position storage (for animation/logging)
        self.vehicle_positions_over_time = []
        self.vehicle_positions_uniques_id = []

        # Trigger flags for first MILP build and subsequent solves
        self.tau = 0   # First vehicle flag: 0 -> no MILP yet, 1 -> MILP will be set
        self.mi  = 0   # After first solve, turn to 1 and keep solving

        # Cached sets for sorting by axis
        self.x_axis_vehicles = []
        self.y_axis_vehicles = []

        # Blocked/Bernoulli legacy arrival parameters
        self.lambda_per_block = 0.40
        self.block_duration = 4 * self.delta_t  # 0.4s blocks
        self.lambda_poisson = -np.log(1 - 0.45)

        self.x_center_dists = {}
        self.y_center_dists = {}

        self.lambda_str = str(self.lambda_per_block).replace('.', '_')  # For filenames

        # Whether fairness constraints were included (used in folder names)
        self.const_fair = False

        # Next arrival indices (exponential inter-arrivals)
        self.next_arrival_x = 0
        self.next_arrival_y = 0

        # Poisson/exponential arrival rates for x and y flows (veh/s)
        self.lambda_x = 0.9
        self.lambda_y = 0.9

        # Minimum discrete steps between arrivals to avoid instantaneous stacking
        self.min_steps = 2

        # Output directory; updated when saving
        self.directory_name = f"Week_{self.radius}_step_{self.lambda_x}_S_{self.safe_dis}_I_{self.buff_ins}_H_{self.N}_C_{self.const_fair}"

    def save_vehicles_positions_to_file(self):
        """
        Save (x,y,id) histories and (x,y,vx,vy,axis,uid) histories as JSON
        into a timestamped filename within the current results directory.
        """
        self.directory_name = f"Week_{self.radius}_step_{self.lambda_str}_S_{self.safe_dis}_I_{self.buff_ins}_H_{self.N}_C_{self.const_fair}"
        if not os.path.exists(self.directory_name):
            os.makedirs(self.directory_name)

        current_time = datetime.now().strftime('%Y-%m-%d--%H-%M-%S')

        filename  = f"vehicles_positions_R_{self.radius}_Rnd_{self.lambda_str}_SD_{self.safe_dis}_BUI_{self.buff_ins}_H_{self.N}_T_{current_time}_C_{self.const_fair}.json"
        filename1 = f"vehicles_positions_id_R_{self.radius}_Rnd_{self.lambda_str}_SD_{self.safe_dis}_BUI_{self.buff_ins}_H_{self.N}_T_{current_time}_C_{self.const_fair}.json"

        with open(os.path.join(self.directory_name, filename), 'w') as file:
            json.dump(self.vehicle_positions_over_time, file, indent=4)

        with open(os.path.join(self.directory_name, filename1), 'w') as file:
            json.dump(self.vehicle_positions_uniques_id, file, indent=4)

        print(f"Vehicles positions saved to {filename}.")
        print(f"Vehicles positions saved to {filename1}.")

    def save_to_file(self, base_filename):
        """(Legacy) Save current `self.vehicles_positions` as JSON with timestamp."""
        current_time = datetime.now().strftime('%Y-%m-%d--%H-%M-%S')
        filename = f"{base_filename}_{current_time}.json"
        with open(filename, 'w') as file:
            json.dump(self.vehicles_positions, file)

    def add_vehicle(self, new_position, final_position, velocity, final_velocity, axis, id, U_id):
        """
        Push a new controlled vehicle into the current MILP by appending to
        initial_conditions/final_conditions and bumping K.

        Parameters
        ----------
        new_position : float
            Initial scalar position on moving axis (x for 'x', y for 'y').
        final_position : float
            Target scalar position to exit control zone.
        velocity : float
            Initial signed velocity along active axis.
        final_velocity : float
            Desired terminal signed velocity along active axis.
        axis : {'x','y'}
            Flow axis for the agent.
        id : int
            Per-axis integer ID.
        U_id : int
            Global unique integer ID across both axes.
        """
        if axis == 'y':
            # y-flow vehicle
            self.initial_conditions.append({'x': 0, 'y': new_position, 'xdot': 0, 'ydot': velocity})
            self.final_conditions.append({'x': 0, 'y': final_position, 'xdot': 0, 'ydot': -final_velocity, 'id': id, 'uid_': U_id})
            self.K += 1

        if axis == 'x':
            # x-flow vehicle
            self.initial_conditions.append({'x': new_position, 'y': 0, 'xdot': velocity, 'ydot': 0})
            self.final_conditions.append({'x': final_position, 'y': 0, 'xdot': final_velocity, 'ydot': 0, 'id': id, 'uid_': U_id})
            self.K += 1

        # Assign a plotting color for visualization (by UUID key)
        vehicle_id = str(uuid.uuid4())
        color = self.colors[self.color_index % len(self.colors)]
        self.color_index += 1
        self.vehicle_colors[vehicle_id] = color

    def next_step(self, current_step, lam):
        """
        Draw an exponential inter-arrival time with rate `lam` (veh/s),
        convert to discrete steps, and enforce a minimum step gap.

        Returns
        -------
        int
            Next step index when a new arrival should be spawned.
        """
        dt_arrival = np.random.exponential(1.0 / lam)
        k = math.ceil(dt_arrival / self.delta_t)
        k = max(k, self.min_steps)
        return current_step + k

    def add_vehicles(self, i, add_vehicle_every_n_steps_x, add_vehicle_every_n_steps_y, velocity):
        """
        (Legacy helper) Add vehicles to x and y lists at fixed step intervals.
        Kept for reproducibility; main code uses exponential arrivals instead.
        """
        if i > 0 and i % add_vehicle_every_n_steps_x == 0:
            self.vehicle_id_x += 1
            self.vehicle_id_x_y += 1
            self.vehicles_x.append((self.lane_start_x, 0, velocity, 'x', self.vehicle_id_x, self.vehicle_id_x_y))

        if i > 0 and i % add_vehicle_every_n_steps_y == 0:
            self.vehicle_id_y += 1
            self.vehicle_id_x_y += 1
            self.vehicles_y.append((0, self.lane_start_y, velocity, 'y', self.vehicle_id_y, self.vehicle_id_x_y))

    def bernoulli_arrival(self, p):
        """Return True with probability p (legacy arrival process)."""
        return random.random() < p

    def poisson_arrival(self):
        """Return True if at least one arrival occurs in a block (legacy mode)."""
        return np.random.poisson(self.lambda_poisson) >= 1

    def update_vehicle_positions(self, iteration, vehicles, velocity, axis):
        """
        Move 'free-flight' vehicles (outside MILP control) one time step,
        detect crossing into control zone boundary, and when the first vehicle
        enters, build & solve the first MILP.

        Returns
        -------
        updated_vehicles : list
            Updated (x, y, vel, axis, id, uid) tuples after moving.
        iteration_positions : list
            Positions for logging/visualization.
        iteration_positions_uid : list
            Positions + velocities + axis + uid for logging.
        """
        updated_vehicles = []
        iteration_positions = []
        iteration_positions_uid = []

        if axis == 'x':
            # Move along +x
            for position, _, _, _, id, Uid in vehicles:
                new_position = position + velocity * self.delta_t

                # Before entering control zone
                if new_position <= -self.radius:
                    updated_vehicles.append((new_position, 0, velocity, 'x', id, Uid))
                    iteration_positions.append((new_position, 0, id))
                    iteration_positions_uid.append((new_position, 0, velocity, 0, 'x', Uid))

                # After exiting control zone (still within plot bounds)
                if new_position >= (self.radius) and new_position < (self.lane_end_x):
                    updated_vehicles.append((new_position, 0, velocity, 'x', id, Uid))
                    iteration_positions.append((new_position, 0, id))
                    iteration_positions_uid.append((new_position, 0, velocity, 0, 'x', Uid))

                # Check if we crossed the control-zone boundary this step
                if position <= -self.radius < new_position or new_position <= -self.radius < position:
                    if self.tau == 0:
                        # First controlled vehicle triggers MPC/MILP initialization
                        self.tau += 1
                        print("first vehicle from x-axis", self.tau)

                        self.vehicles_positions = []
                        self.initialize()
                        self.add_vehicle(new_position, self.radius + 28, velocity, 15, 'x', id, Uid)

                        self.vehicles = []

                        # Build first MILP
                        self.m = gp.Model("vehicle_motion_planning")
                        self.m.setParam('Threads', 32)
                        self.m.setParam(GRB.Param.Method, -1)
                        self.m.setParam(GRB.Param.Presolve, -1)

                        self.default(iteration)

                        # Extract one-step positions for logging
                        for t, vehicle in enumerate(self.vehicles):
                            x_positions = [vehicle['s'][i, 0].X for i in range(1)]
                            y_positions = [vehicle['s'][i, 1].X for i in range(1)]
                            x_velocitys = [vehicle['s'][i, 2].X for i in range(1)]
                            y_velocitys = [vehicle['s'][i, 3].X for i in range(1)]

                            iteration_positions.append((x_positions, y_positions, self.final_conditions[t]['id']))  # CHCK
                            iteration_positions_uid.append((x_positions, y_positions, x_velocitys, y_velocitys, 'x', self.final_conditions[t]['uid_']))

                        del self.m
                    else:
                        # After first MILP, just add more vehicles to the controlled set
                        self.add_vehicle(new_position, self.radius + 28, velocity, 15, 'x', id, Uid)

        else:
            # axis == 'y': moving toward -y (note sign of velocity when appended)
            for _, position, _, _, id, Uid in vehicles:
                new_position = position + velocity * self.delta_t
                if new_position >= self.radius:
                    updated_vehicles.append((0, new_position, velocity, 'y', id, Uid))
                    iteration_positions.append((0, new_position, id))
                    iteration_positions_uid.append((0, new_position, 0, velocity, 'y', Uid))

                if new_position <= -(self.radius) and new_position > (self.lane_end_y):
                    updated_vehicles.append((0, new_position, velocity, 'y', id, Uid))
                    iteration_positions.append((0, new_position, id))
                    iteration_positions_uid.append((0, new_position, 0, velocity, 'y', Uid))

                # Entering control zone boundary for y
                if position >= self.radius > new_position or new_position >= self.radius > position:
                    print((0, new_position, velocity, 'y'))

                    if self.tau == 0:
                        self.tau += 1
                        print("first vehicle from x-axis", self.tau)

                        self.vehicles_positions = []
                        self.initialize()
                        self.add_vehicle(new_position, -(self.radius + 28), velocity, 15, 'y', id, Uid)

                        self.vehicles = []
                        self.m = gp.Model("vehicle_motion_planning")
                        self.m.setParam('Threads', 32)
                        self.m.setParam(GRB.Param.Method, -1)
                        self.m.setParam(GRB.Param.Presolve, -1)

                        self.default(iteration)
                        for t, vehicle in enumerate(self.vehicles):
                            x_positions = [vehicle['s'][i, 0].X for i in range(1)]
                            y_positions = [vehicle['s'][i, 1].X for i in range(1)]
                            x_velocitys = [vehicle['s'][i, 2].X for i in range(1)]
                            y_velocitys = [vehicle['s'][i, 3].X for i in range(1)]
                            iteration_positions.append((x_positions, y_positions, self.final_conditions[t]['id']))  # chck
                            iteration_positions_uid.append((x_positions, y_positions, x_velocitys, y_velocitys, 'y', self.final_conditions[t]['uid_']))

                        del self.m
                    else:
                        self.add_vehicle(new_position, -(self.radius + 28), velocity, 15, 'y', id, Uid)

        return updated_vehicles, iteration_positions, iteration_positions_uid

    def simulation_loop(self):
        """
        Main simulation loop:
          - Spawns vehicles on each axis using exponential inter-arrival times.
          - Advances free vehicles one step; when inside control zone, builds/solves MILP.
          - Logs per-iteration positions (raw and with UID/velocities).
        """
        # Schedule first arrivals (discrete step indices)
        self.next_arrival_x = self.next_step(0, self.lambda_x)
        self.next_arrival_y = self.next_step(0, self.lambda_y)

        self.temp_iteration = []
        self.all_mip_gaps = {}

        for i in range(self.total_iterations):
            print("iteration no: ", i)

            # Spawn new x-axis arrival
            if i == self.next_arrival_x and i < (self.total_iterations - self.stop_add):
                self.vehicle_id_x += 1
                self.vehicle_id_x_y += 1
                self.vehicles_x.append((self.lane_start_x, 0, self.vehicle_velocity, 'x', self.vehicle_id_x, self.vehicle_id_x_y))
                self.next_arrival_x = self.next_step(i, self.lambda_x)

            # Spawn new y-axis arrival
            if i == self.next_arrival_y and i < (self.total_iterations - self.stop_add):
                self.vehicle_id_y += 1
                self.vehicle_id_x_y += 1
                self.vehicles_y.append((0, self.lane_start_y, -self.vehicle_velocity, 'y', self.vehicle_id_y, self.vehicle_id_x_y))
                self.next_arrival_y = self.next_step(i, self.lambda_y)

            # Move free-flight vehicles and capture their positions
            self.vehicles_x, self.vehicle_it1, self.vehicleidx = self.update_vehicle_positions(i, self.vehicles_x, self.vehicle_velocity, 'x')
            self.vehicles_y, self.vehicle_it2, self.vehicleidy = self.update_vehicle_positions(i, self.vehicles_y, -self.vehicle_velocity, 'y')

            if self.mi > 0:
                # After the first MILP has been built once, we keep solving each step
                self.vehicles = []
                self.m = gp.Model("vehicle_motion_planning")
                self.m.setParam('Threads', 32)
                self.m.setParam(GRB.Param.Method, -1)
                self.m.setParam(GRB.Param.Presolve, -1)

                self.default(i)

                iteration_pos = []
                iteration_pos_uid = []
                for t, vehicle in enumerate(self.vehicles):
                    x_positions = [vehicle['s'][i, 0].X for i in range(1)]
                    y_positions = [vehicle['s'][i, 1].X for i in range(1)]
                    x_velocitys = [vehicle['s'][i, 2].X for i in range(1)]
                    y_velocitys = [vehicle['s'][i, 3].X for i in range(1)]
                    iteration_pos.append((x_positions, y_positions, self.final_conditions[t]['id']))

                    # Detect axis based on velocity/state zeroing convention
                    if y_positions[0] == 0 and y_velocitys[0] == 0:
                        iteration_pos_uid.append((x_positions, y_positions, x_velocitys, y_velocitys, 'x', self.final_conditions[t]['uid_']))
                    elif x_positions[0] == 0 and x_velocitys[0] == 0:
                        iteration_pos_uid.append((x_positions, y_positions, x_velocitys, y_velocitys, 'y', self.final_conditions[t]['uid_']))
                    else:
                        print('x,y,x_v,y_v', x_positions, y_positions, x_velocitys, y_velocitys)
                        print("couldn't print it")

                self.vehicle_positions_over_time.append(self.vehicle_it1 + self.vehicle_it2 + iteration_pos)
                self.vehicle_positions_uniques_id.append(self.vehicleidx + self.vehicleidy + iteration_pos_uid)
                del self.m
            else:
                # Before first MILP solve, just store free-flight positions
                self.vehicle_positions_over_time.append(self.vehicle_it1 + self.vehicle_it2)
                self.vehicle_positions_uniques_id.append(self.vehicleidx + self.vehicleidy)

            # Flip flag to start solving after the very first MILP
            if self.tau == 1 and self.mi == 0:
                self.mi = 1

        # Optionally: self.create_animation()

    def remove_vehicle(self):
        """
        Remove vehicles that have exited the control zone and re-queue them onto
        their free-flight lists so they continue moving outside MPC control.
        """
        removed_vehicles = []
        initial_removed_count = 0

        for i in reversed(range(len(self.vehicles))):
            vehicle = self.vehicles[i]
            x_position = vehicle['s'][1, 0].X
            y_position = vehicle['s'][1, 1].X
            x_velocity = vehicle['s'][1, 2].X
            y_velocity = vehicle['s'][1, 3].X

            if (x_position >= (self.radius+3) or y_position <= -(self.radius+3)):
                vehicle_id = i
                removed_vehicles.append(vehicle_id)
                tid = self.final_conditions[i]['id']
                utid = self.final_conditions[i]['uid_']
                self.vehicles.pop(i)
                self.initial_conditions.pop(i)
                self.final_conditions.pop(i)
                initial_removed_count += 1

                # Re-inject into free-flight streams
                if x_position == 0 and y_position <= -(self.radius+3):
                    self.vehicles_y.append((0, y_position, y_velocity, 'y', tid, utid))
                if y_position == 0 and x_position >= (self.radius+3):
                    self.vehicles_x.append((x_position, 0, x_velocity, 'x', tid, utid))

        self.K -= initial_removed_count

        if removed_vehicles:
            print(f"Removed vehicles: {removed_vehicles}")
            print(f"Total removed this iteration: {initial_removed_count}")

    def default(self, iteration):
        """
        One MPC/MILP iteration:
          - Create variables and constraints,
          - Build objective and optimize,
          - Update vehicle states with the 1-step solution.
        """
        self.setup_variables(iteration)
        self.setup_constraints(iteration)
        self.setup_objective()
        self.optimize(iteration)
        self.update_vehicle_state(iteration)

    def update_vehicle_state(self, iteration):
        """
        After a successful MILP solve, write back the first-step state into
        initial_conditions so next solve is warm-started from that point.
        """
        if self.m.status == GRB.OPTIMAL:
            for i in reversed(range(len(self.vehicles))):
                vehicle = self.vehicles[i]

                x_position = vehicle['s'][1, 0].X
                y_position = vehicle['s'][1, 1].X
                x_velocity = vehicle['s'][1, 2].X
                y_velocity = vehicle['s'][1, 3].X

                self.initial_conditions[i] = {
                    'x': x_position, 'y': y_position, 'xdot': x_velocity, 'ydot': y_velocity
                }

            self.remove_vehicle()
        else:
            print(f"Skipping vehicle state update for iteration {iteration} due to optimization failure.")

    def update_visualization_positions(self):
        """Collect current MILP states for plotting (used by animation helpers)."""
        iteration_positions = []
        for vehicle in self.vehicles:
            x_positions = [vehicle['s'][i, 0].X for i in range(1)]
            y_positions = [vehicle['s'][i, 1].X for i in range(1)]
            iteration_positions.append((x_positions, y_positions))
        self.vehicles_positions.append(iteration_positions)

    def initialize(self, update='False'):
        """Initialize/reset controlled vehicle lists."""
        if update == 'False':
            self.initial_conditions = []
            self.final_conditions = []

    def sort_vehicles_by_axis(self):
        """
        Partition current controlled set into x-flow and y-flow indices,
        based on the axis with nonzero nominal velocity.
        """
        x_axis_vehicles = []
        y_axis_vehicles = []
        for i, vehicle in enumerate(self.initial_conditions):
            if abs(vehicle['xdot']) > abs(vehicle['ydot']) and vehicle['ydot'] == 0 and vehicle['y'] == 0:
                x_axis_vehicles.append(i)
            else:
                y_axis_vehicles.append(i)
        return x_axis_vehicles, y_axis_vehicles

    def setup_variables(self, iteration):
        """Add state s, control u, slack w, aux v for each controlled vehicle."""
        for p in range(self.K):
            vehicle = {
                's': self.m.addVars(self.N, 4, lb=[self.smin] * self.N, ub=[self.smax] * self.N, name=f"s_{p}"),
                'u': self.m.addVars(self.N, 2, lb=[self.umin] * self.N, ub=[self.umax] * self.N, name=f"u_{p}"),
                'w': self.m.addVars(self.N, 4, name=f"w_{p}"),
                'v': self.m.addVars(self.N, 2, name=f"v_{p}"),
            }
            self.vehicles.append(vehicle)
        self.m.update()

    def setup_constraints(self, iteration):
        """Assemble all deterministic constraints; optionally fairness constraints."""
        self.state_constraints(iteration)
        self.control_constraints()
        self.state_transition_constraints()
        self.initial_final_condition_constraints()
        self.vehicle_intersection_constraints()
        self.apply_safe_distance_constraints()
        if (self.fair == True):
            self.define_crossing_and_proximity_constraints()

    def define_crossing_and_proximity_constraints(self):
        """
        Fairness activation:
          - Defines binary crossing indicators w_p (x) and w_q (y) by Big-M.
          - Defines proximity (close pairs) and enforces order depending on UID-based priority pi_{p,q}.
          - S_pq allows a reversal with penalty in objective.
        """
        self.const_fair = True

        x_axis_vehicles, y_axis_vehicles = self.sort_vehicles_by_axis()

        # Crossing indicator variables
        self.w_p = {p: self.m.addVars(self.N, vtype=GRB.BINARY, name=f"w_p_{p}") for p in x_axis_vehicles}
        self.w_q = {q: self.m.addVars(self.N, vtype=GRB.BINARY, name=f"w_q_{q}") for q in y_axis_vehicles}

        x_c, y_c = 0, 0
        buffer = self.proximity
        M = self.Big_M

        # Big-M logic to mark crossing for x-flow vehicles
        for p in x_axis_vehicles:
            for i in range(self.N):
                self.m.addConstr(self.vehicles[p]['s'][i, 0] >= x_c - M * (1 - self.w_p[p][i]), f"Crossed_X_BigM_{p}_{i}")
                self.m.addConstr(self.vehicles[p]['s'][i, 0] <= x_c - 0.1 + M * self.w_p[p][i], f"Not_Crossed_X_BigM_{p}_{i}")

        # Big-M logic to mark crossing for y-flow vehicles
        for q in y_axis_vehicles:
            for i in range(self.N):
                self.m.addConstr(self.vehicles[q]['s'][i, 1] <= y_c + M * (1 - self.w_q[q][i]), f"Crossed_Y_BigM_{q}_{i}")
                self.m.addConstr(self.vehicles[q]['s'][i, 1] >= y_c + 0.1 - M * self.w_q[q][i], f"Not_Crossed_Y_BigM_{q}_{i}")

        # L1 distances to center for proximity test
        self.d_p = {p: self.m.addVars(self.N, lb=0, name=f"d_p_{p}") for p in x_axis_vehicles}
        self.d_q = {q: self.m.addVars(self.N, lb=0, name=f"d_q_{q}") for q in y_axis_vehicles}
        self.delta_pq = {(p, q): self.m.addVars(self.N, lb=0, name=f"Delta_{p}_{q}") for p in x_axis_vehicles for q in y_axis_vehicles}

        for p in x_axis_vehicles:
            for i in range(self.N):
                self.m.addConstr(self.d_p[p][i] >= self.vehicles[p]['s'][i, 0] - x_c, f"Dist_Pos_X_{p}_{i}")
                self.m.addConstr(self.d_p[p][i] >= -(self.vehicles[p]['s'][i, 0] - x_c), f"Dist_Neg_X_{p}_{i}")

        for q in y_axis_vehicles:
            for i in range(self.N):
                self.m.addConstr(self.d_q[q][i] >= self.vehicles[q]['s'][i, 1] - y_c, f"Dist_Pos_Y_{q}_{i}")
                self.m.addConstr(self.d_q[q][i] >= -(self.vehicles[q]['s'][i, 1] - y_c), f"Dist_Neg_Y_{q}_{i}")

        for p in x_axis_vehicles:
            for q in y_axis_vehicles:
                for i in range(self.N):
                    self.m.addConstr(self.delta_pq[(p, q)][i] >= self.d_p[p][i] - self.d_q[q][i], f"Delta_Pos_{p}_{q}_{i}")
                    self.m.addConstr(self.delta_pq[(p, q)][i] >= self.d_q[q][i] - self.d_p[p][i], f"Delta_Neg_{p}_{q}_{i}")

        # Identify candidate close pairs (heuristic filter to reduce constraints)
        close_pairs = []
        for p in x_axis_vehicles:
            for q in y_axis_vehicles:
                if self.initial_conditions[p]['x'] < 1 and self.initial_conditions[q]['y'] > -1:
                    close_pairs.append((p, q))
        print(close_pairs)

        # Priority pi_{p,q} based on UID ordering
        self.pi_pq = {}
        for (p, q) in close_pairs:
            uid_p = self.final_conditions[p]['uid_']
            uid_q = self.final_conditions[q]['uid_']
            self.pi_pq[(p, q)] = 0 if uid_p < uid_q else 1

        # Reversal slack S_pq and close-pair activation binaries
        self.S_pq = {}
        self.close_pq = {}
        ε = 0.1
        W = self.Big_W

        for (p, q) in close_pairs:
            self.S_pq[(p, q)] = self.m.addVar(vtype=GRB.BINARY, name=f"S_{p}_{q}")
            for i in range(self.N):
                self.close_pq[(p, q, i)] = self.m.addVar(vtype=GRB.BINARY, name=f"close_{p}_{q}_{i}")

                # Enforce close_pq==1 iff delta_pq <= buffer (within Big-M tolerance)
                self.m.addConstr(self.delta_pq[(p, q)][i] <= buffer + M * (1 - self.close_pq[(p, q, i)]), f"Close_Pair_Upper_{p}_{q}_{i}")
                self.m.addConstr(self.delta_pq[(p, q)][i] >= buffer + ε - M * self.close_pq[(p, q, i)], f"Close_Pair_Lower_{p}_{q}_{i}")

                # Priority-enforced ordering with relaxation via S_pq (and deactivation when not close)
                if self.pi_pq[(p, q)] == 0:
                    # p has priority -> constrain q to wait before y_c unless p already crossed
                    self.m.addConstr(
                        self.vehicles[q]['s'][i, 1] >= y_c + ε - M * self.w_p[p][i] + M * self.S_pq[(p, q)] + W * (1 - self.close_pq[(p, q, i)]),
                        f"Vehicle_q_Constraint_{p}_{q}_{i}"
                    )
                else:
                    # q has priority -> constrain p to wait before x_c unless q already crossed
                    self.m.addConstr(
                        self.vehicles[p]['s'][i, 0] <= x_c - ε + M * self.w_q[q][i] + M * self.S_pq[(p, q)] + M * (1 - self.close_pq[(p, q, i)]),
                        f"Vehicle_p_Constraint_{p}_{q}_{i}"
                    )

    def apply_safe_distance_constraints(self):
        """Same-lane spacing constraints for each axis and timestep."""
        safe_distance = self.safe_dis
        x_axis_vehicles, y_axis_vehicles = self.sort_vehicles_by_axis()

        # X-axis followers keep >= safe_distance
        for i in x_axis_vehicles:
            for j in x_axis_vehicles:
                if i != j:
                    if (self.initial_conditions[i]['xdot'] * self.initial_conditions[j]['xdot']) > 0:
                        for t in range(1, self.N):
                            if self.initial_conditions[i]['x'] > self.initial_conditions[j]['x']:
                                self.m.addConstr(self.vehicles[i]['s'][t, 0] - self.vehicles[j]['s'][t, 0] >= safe_distance,
                                                 f"safe_dist_x_{i}_{j}_{t}")
                            else:
                                self.m.addConstr(self.vehicles[j]['s'][t, 0] - self.vehicles[i]['s'][t, 0] >= safe_distance,
                                                 f"safe_dist_x_{j}_{i}_{t}")

        # Y-axis followers keep >= safe_distance
        for i in y_axis_vehicles:
            for j in y_axis_vehicles:
                if i != j and (self.initial_conditions[i]['ydot'] * self.initial_conditions[j]['ydot']) > 0:
                    for t in range(1, self.N):
                        if self.initial_conditions[i]['y'] > self.initial_conditions[j]['y']:
                            self.m.addConstr(self.vehicles[i]['s'][t, 1] - self.vehicles[j]['s'][t, 1] >= safe_distance,
                                             f"safe_dist_y_{i}_{j}_{t}")
                        else:
                            self.m.addConstr(self.vehicles[j]['s'][t, 1] - self.vehicles[i]['s'][t, 1] >= safe_distance,
                                             f"safe_dist_y_{j}_{i}_{t}")

    def state_constraints(self, iteration):
        """
        Add slack-based state deviation constraints relative to desired final conditions,
        and apply position bounds on [x,y] over the horizon.
        """
        for p, vehicle in enumerate(self.vehicles):
            s, u, w, v_vars = vehicle['s'], vehicle['u'], vehicle['w'], vehicle['v']

            # Soft tracking to final (x,y,xdot,ydot) via slacks w[i,j]
            for i in range(1, self.N):
                for j in range(4):
                    self.m.addConstr(
                        s[i, j] - self.final_conditions[p][['x', 'y',  'xdot', 'ydot', 'id', 'uid_'][j]] <= w[i, j],
                        f"State_Dev_{j}_{p}_{i}"
                    )
                    self.m.addConstr(
                        -s[i, j] + self.final_conditions[p][['x', 'y', 'xdot', 'ydot', 'id', 'uid_'][j]] <= w[i, j],
                        f"State_Dev_Neg_{j}_{p}_{i}"
                    )

            # Position bounds for [x,y] (vel bounds handled by dynamics/control)
            for i in range(1, self.N):
                self.m.addConstr(s[i, 0] >= self.smin[0], f"Pos_X_Lower_Bound_{p}_{i}")
                self.m.addConstr(s[i, 0] <= self.smax[0], f"Pos_X_Upper_Bound_{p}_{i}")
                self.m.addConstr(s[i, 1] >= self.smin[1], f"Pos_Y_Lower_Bound_{p}_{i}")
                self.m.addConstr(s[i, 1] <= self.smax[1], f"Pos_Y_Upper_Bound_{p}_{i}")

    def control_constraints(self):
        """
        Constrain accelerations to lie only along the motion axis:
          - x-flow: ay == 0,
          - y-flow: ax == 0,
        and enforce control bounds.
        """
        x_axis_vehicles, y_axis_vehicles = self.sort_vehicles_by_axis()

        for i in range(self.N):
            for p in x_axis_vehicles:
                self.m.addConstr(self.vehicles[p]['u'][i, 1] == 0, f"Zero_Accel_Y_{p}_{i}")
                self.m.addConstr(self.vehicles[p]['u'][i, 0] <= self.umax[0], f"Control_Effort_Max_X_{p}_{i}")
                self.m.addConstr(self.vehicles[p]['u'][i, 0] >= self.umin[0], f"Control_Effort_Min_X_{p}_{i}")

            for p in y_axis_vehicles:
                self.m.addConstr(self.vehicles[p]['u'][i, 0] == 0, f"Zero_Accel_X_{p}_{i}")
                self.m.addConstr(self.vehicles[p]['u'][i, 1] <= self.umax[1], f"Control_Effort_Max_Y_{p}_{i}")
                self.m.addConstr(self.vehicles[p]['u'][i, 1] >= self.umin[1], f"Control_Effort_Min_Y_{p}_{i}")

    def state_transition_constraints(self):
        """
        Trapezoidal (midpoint) integration scheme for the active axis and
        hold-constant constraints for the orthogonal axis.
        """
        x_axis_vehicles, y_axis_vehicles = self.sort_vehicles_by_axis()

        for i in range(self.N - 1):
            for p in x_axis_vehicles:
                # x dynamics, y held
                self.m.addConstr(self.vehicles[p]['s'][i + 1, 0] == self.vehicles[p]['s'][i, 0] + (self.delta_t/2) *
                                 (self.vehicles[p]['s'][i, 2]+self.vehicles[p]['s'][i+1, 2]), f"Dynamics_x_{p}_{i}")
                self.m.addConstr(self.vehicles[p]['s'][i + 1, 2] == self.vehicles[p]['s'][i, 2] + (self.delta_t/2) *
                                 (self.vehicles[p]['u'][i, 0]+self.vehicles[p]['u'][i+1, 0]), f"Dynamics_xdot_{p}_{i}")
                self.m.addConstr(self.vehicles[p]['s'][i + 1, 1] == self.vehicles[p]['s'][i, 1], f"Static_y_{p}_{i}")
                self.m.addConstr(self.vehicles[p]['s'][i + 1, 3] == 0, f"Static_ydot_{p}_{i}")

            for p in y_axis_vehicles:
                # y dynamics, x held
                self.m.addConstr(self.vehicles[p]['s'][i + 1, 1] == self.vehicles[p]['s'][i, 1] + (self.delta_t/2) *
                                 (self.vehicles[p]['s'][i, 3]+self.vehicles[p]['s'][i+1, 3]), f"Dynamics_y_{p}_{i}")
                self.m.addConstr(self.vehicles[p]['s'][i + 1, 3] == self.vehicles[p]['s'][i, 3] + (self.delta_t/2) *
                                 (self.vehicles[p]['u'][i, 1]+self.vehicles[p]['u'][i+1, 1]), f"Dynamics_ydot_{p}_{i}")
                self.m.addConstr(self.vehicles[p]['s'][i + 1, 0] == self.vehicles[p]['s'][i, 0], f"Static_x_{p}_{i}")
                self.m.addConstr(self.vehicles[p]['s'][i + 1, 2] == 0, f"Static_xdot_{p}_{i}")

    def initial_final_condition_constraints(self):
        """Pin s[0,:] to current initial_conditions (per vehicle)."""
        for p, vehicle in enumerate(self.vehicles):
            s, u, w, v_vars = vehicle['s'], vehicle['u'], vehicle['w'], vehicle['v']
            self.m.addConstrs((s[0, j] == self.initial_conditions[p][key]
                               for j, key in enumerate(['x', 'y', 'xdot', 'ydot'])), f"Initial_{p}")

    def vehicle_intersection_constraints(self):
        """
        Intersection conflict-avoidance via center-buffer:
          For each (x-vehicle, y-vehicle) pair, use |x| + |y| >= buff_ins near the center
          while both are still approaching (prevents simultaneous occupancy).
        """
        center_buffer = self.buff_ins

        self.x_axis_vehicles, self.y_axis_vehicles = self.sort_vehicles_by_axis()

        for i in range(1, self.N):
            self.x_center_dists = {}
            self.y_center_dists = {}

            # Only enforce before crossing threshold to reduce constraints
            for vehicle_index in self.x_axis_vehicles:
                if self.initial_conditions[vehicle_index]['x'] > 1:
                    continue
                self.x_center_dists[vehicle_index] = self.m.addVar(lb=0, name=f"x_center_dist_{vehicle_index}_{i}")
                self.m.addGenConstrAbs(self.x_center_dists[vehicle_index], self.vehicles[vehicle_index]['s'][i, 0],
                                       f"X_Distance_{vehicle_index}_{i}")

            for vehicle_index in self.y_axis_vehicles:
                if self.initial_conditions[vehicle_index]['y'] < -1:
                    continue
                self.y_center_dists[vehicle_index] = self.m.addVar(lb=0, name=f"y_center_dist_{vehicle_index}_{i}")
                self.m.addGenConstrAbs(self.y_center_dists[vehicle_index], self.vehicles[vehicle_index]['s'][i, 1],
                                       f"Y_Distance_{vehicle_index}_{i}")

            for x_index in self.x_axis_vehicles:
                if x_index in self.x_center_dists:
                    for y_index in self.y_axis_vehicles:
                        if y_index in self.y_center_dists:
                            self.m.addConstr((self.x_center_dists[x_index] + self.y_center_dists[y_index]) >= center_buffer,
                                             f"Center_Buffer_{x_index}_{y_index}_{i}")

    def setup_objective(self):
        """
        Build linear objectives:
          obj    : minimize slacks (state deviation) + terminal slacks,
          obj1   : (optional) fairness reversal penalty via S_pq,
          obj_vel: encourage higher x-vel and lower y-vel (as written).
                   NOTE: sign convention used via setObjectiveN below.
        """
        self.obj = gp.LinExpr()
        self.obj1 = gp.LinExpr()
        self.obj_vel = gp.LinExpr()

        for p, vehicle in enumerate(self.vehicles):
            s, u, w, v_vars = vehicle['s'], vehicle['u'], vehicle['w'], vehicle['v']

            # Running slack cost on [x,y] only (j in 0..1)
            for i in range(1, self.N-1):
                for j in range(2):
                    self.obj += self.qo[j] * w[i, j]

            # Velocity term (as difference), later maximized by minimizing its negative
            for i in range(0, self.N - 1):
                self.obj_vel += self.velocity_weight * (s[i, 2] - s[i, 3])

            # Terminal slacks on [x,y]
            for j in range(2):
                self.obj += self.po[j] * w[self.N-1, j]

        if (self.fair == True):
            # Fairness penalty: sum of S_pq
            for (p, q) in self.S_pq:
                self.obj1 += self.reversal_cost * self.S_pq[(p, q)]

    def optimize(self, it):
        """
        Solve the MILP with prioritized multiobjective:
          0: minimize slacks,
          1: maximize velocity term   (via minimize -obj_vel),
          2: (optional) minimize fairness reversals.
        Logs MIP gap data via a callback and writes Gurobi log to file.
        """
        self.directory_name = f"Week_{self.radius}_step_{self.lambda_str}_S_{self.safe_dis}_I_{self.buff_ins}_H_{self.N}_C_{self.const_fair}"
        if not os.path.exists(self.directory_name):
             os.makedirs(self.directory_name)

        current_time = datetime.now().strftime('%Y-%m-%d--%H-%M-%S')
        self.m.setParam("LogFile", os.path.join(self.directory_name, f"gurobi_{it}_R_{self.radius}_Rnd_{self.lambda_str}_SD_{self.safe_dis}_BUI_{self.buff_ins}_H_{self.N}_T_{current_time}_C_{self.const_fair}.log"))

        # Priority-1 objectives: slacks (index 0) and -velocity (index 1)
        self.m.setObjectiveN(self.obj, index=0, priority=1)
        self.m.setObjectiveN(-self.obj_vel, index=1, priority=1)

        if (self.fair == True):
            self.m.setObjectiveN(self.obj1, index=2, priority=1)

        self.m.setParam(GRB.Param.OutputFlag, 1)

        self.mip_gaps = []

        def mip_gap_callback(model, where):
            """
            Callback to record MIP gap, bounds, node counts, solution count,
            etc., across time—useful for solver performance diagnostics.
            """
            if where == GRB.Callback.MIP:
                current_runtime = model.cbGet(GRB.Callback.RUNTIME)
                objbst = model.cbGet(GRB.Callback.MIP_OBJBST)
                objbnd = model.cbGet(GRB.Callback.MIP_OBJBND)

                if math.isclose(objbnd, objbst, abs_tol=1.0e-9):
                    gap = 0.0
                elif objbst == 0.0:
                    gap = float('inf')
                else:
                    gap = abs((objbnd - objbst) / objbst)

                node_count = model.cbGet(GRB.Callback.MIP_NODCNT)
                node_left  = model.cbGet(GRB.Callback.MIP_NODLFT)
                iter_count = model.cbGet(GRB.Callback.MIP_ITRCNT)
                sol_count  = model.cbGet(GRB.Callback.MIP_SOLCNT)

                self.mip_gaps.append({
                    "time": current_runtime,
                    "gap": gap,
                    "OBJBST": objbst,
                    "OBJBND": objbnd,
                    "node_count": node_count,
                    "node_left": node_left,
                    "iter_count": iter_count,
                    "sol_count": sol_count
                })

        self.m.optimize(mip_gap_callback)
        self.all_mip_gaps[f"iteration_{it}"] = self.mip_gaps

        # If infeasible, write IIS and dump an .ilp snapshot for debugging
        if self.m.status == GRB.INF_OR_UNBD or self.m.status == GRB.INFEASIBLE:
            print(f"Optimization was infeasible for iteration {it}.")
            print('The model is infeasible; computing IIS')
            self.m.computeIIS()

            filename = f"R_{self.radius}_step_{self.lambda_per_block}_S_{self.safe_dis}_I_{self.buff_ins}_H_{self.N}_{datetime.now().strftime('%Y-%m-%d--%H-%M-%S')}.ilp"
            self.m.write(filename)
            print(f"IIS written to file {filename}")

            vehicle_constraints = self.parse_ilp_file(filename)
            self.interpret_constraints(vehicle_constraints)
            # self.highlight_infeasible_vehicles()
            self.create_animation()
            self.save_vehicles_positions_to_file()
            sys.exit(1)

    def parse_ilp_file(self, file_path):
        """
        Parse a Gurobi .ilp model snapshot to extract certain constraint names
        associated with potential infeasibility (for debugging).
        """
        vehicle_constraints = {}
        parsing = False

        self.constraint_texts = []

        with open(file_path, 'r') as file:
            for line in file:
                if "safe_dist_y_" in line or "safe_dist_x_" in line or "Center_Buffer_" in line:
                    self.constraint_texts.append(line.strip())

                line = line.strip()

                if line.startswith("Subject To"):
                    parsing = True
                    continue

                if line.startswith("Bounds"):
                    break

                if parsing and (':' in line):
                    parts = line.split(':')
                    constraint_name = parts[0].strip()

                    if any(prefix in constraint_name for prefix in ("Dynamics_y_", "Dynamics_ydot_", "Dynamics_x_", "Dynamics_xdot_")):
                        *_, vehicle_no, _ = constraint_name.split('_')
                        vehicle_constraints.setdefault(vehicle_no, []).append(constraint_name)

        return vehicle_constraints

    def interpret_constraints(self, vehicle_constraints):
        """Print a small report of constraints involving vehicles flagged in parse_ilp_file()."""
        self.infeasible_vehicles = []
        for vehicle, constraints in vehicle_constraints.items():
            print(f"Vehicle {vehicle} is involved in the following constraints indicating potential infeasibility issues:")
            for constraint in constraints:
                print(f" - {constraint}")
            self.infeasible_vehicles.append(vehicle)
            print("vehicle == ", self.infeasible_vehicles)

    def highlight_infeasible_vehicles(self):
        """
        (Optional) On-plot highlighting of vehicles involved in IIS;
        call after a plotting figure is active.
        """
        if not hasattr(self, 'infeasible_vehicles') or not self.infeasible_vehicles:
            return

        for vehicle_id in self.infeasible_vehicles:
            vehicle_index = int(vehicle_id)-1
            print("vehicle after == ", vehicle_index, len(self.lines))
            if vehicle_index < len(self.lines1):
                x, y = self.vehicles_positions[-1][vehicle_index]
                print(x, y, "coordinate")
                self.ax.plot(x, y, 'o', ms=20, mfc='none', mec='red', mew=2, alpha=0.5)
                self.ax1.plot(x, y, 'o', ms=20, mfc='none', mec='red', mew=2, alpha=0.5)

            self.fig.canvas.draw_idle()
            self.fig1.canvas.draw_idle()
            plt.pause(0.05)

    def plot(self):
        """Initialize a basic static plot for the intersection and lanes."""
        self.fig, self.ax = plt.subplots(figsize=(10, 8))
        self.ax.set_xlim([-100, 100])
        self.ax.set_ylim([-100, 100])
        self.ax.set_xlabel('X Axis')
        self.ax.set_ylabel('Y Axis')

        self.ax.axhline(y=2, color='black', linestyle='--')
        self.ax.axhline(y=-2, color='black', linestyle='--')
        self.ax.axvline(x=2, color='black', linestyle='--')
        self.ax.axvline(x=-2, color='black', linestyle='--')

        intersection = plt.Rectangle((-2, -2), 4, 4, color='grey', alpha=0.3)
        self.ax.add_patch(intersection)

        self.ax.legend(['Lanes', 'Intersection'], loc="upper right")
        self.lines = [self.ax.plot([], [], 'o-', linewidth=1, markersize=5)[0] for _ in range(self.K)]

    def plot1(self):
        """Initialize a large figure used by the animation/update() method."""
        self.fig1, self.ax1 = plt.subplots(figsize=(15, 15))

    def update(self, frame):
        """Animation callback: draw per-frame vehicle positions and control-zone circle."""
        self.ax1.cla()
        self.ax1.set_xlim([-200, 200])
        self.ax1.set_ylim([-200, 200])
        self.ax1.set_xlabel('X Axis')
        self.ax1.set_ylabel('Y Axis')

        self.ax1.axhline(y=2, color='black', linestyle='--')
        self.ax1.axhline(y=-2, color='black', linestyle='--')
        self.ax1.axvline(x=2, color='black', linestyle='--')
        self.ax1.axvline(x=-2, color='black', linestyle='--')

        intersection1 = plt.Rectangle((-2, -2), 4, 4, color='grey', alpha=0.6)
        self.ax1.add_patch(intersection1)

        circle = patches.Circle((0, 0), self.radius, fill=False, color='green', linestyle='--')
        self.ax1.add_patch(circle)

        for vehicle_id, vehicle_positions in enumerate(self.vehicle_positions_over_time[frame]):
            x_data, y_data, tid = vehicle_positions
            color = self.vehicle_colors.get(vehicle_id, 'blue')
            self.ax1.plot(x_data, y_data, 'o-', linewidth=1, markersize=3.5, color=color)

    def create_animation(self):
        """Render and save a GIF animation of stored positions (if any were recorded)."""
        self.directory_name = f"Week_{self.radius}_step_{self.lambda_str}_S_{self.safe_dis}_I_{self.buff_ins}_H_{self.N}_C_{self.const_fair}"
        if self.vehicle_positions_over_time:
            ani = animation.FuncAnimation(self.fig1, self.update, frames=len(self.vehicle_positions_over_time),
                                          blit=False, repeat=True)

            current_time = datetime.now().strftime('%Y-%m-%d--%H-%M-%S')
            filename = f"animation__ITE_{self.total_iterations}_R_{self.radius}_Rnd_{self.lambda_per_block}_SD_{self.safe_dis}_BUI_{self.buff_ins}_H_{self.N}_T_{current_time}.gif"

            if not os.path.exists(self.directory_name):
                os.makedirs(self.directory_name)

            ani.save(os.path.join(self.directory_name, filename), writer='imagemagick', fps=10, dpi=150)
            print(f"Animation saved as {filename}")
        else:
            print("No data available for animation.")

    def plot_current_positions(self, iteration_positions):
        """(Legacy) Update static figures with current positions."""
        for line in self.lines:
            line.set_data([], [])
        for line in self.lines1:
            line.set_data([], [])

        for i, vehicle_positions in enumerate(iteration_positions):
            color = self.colors[i % len(self.colors)]
            x_data, y_data = vehicle_positions
            if i < len(self.lines):
                self.lines[i].set_data(x_data, y_data)
                self.lines[i].set_color(color)
                self.lines1[i].set_data(x_data, y_data)
                self.lines1[i].set_color(color)
            else:
                self.lines.append(self.ax.plot(x_data, y_data, 'o-', linewidth=1, markersize=5, color=color)[0])
                self.lines1.append(self.ax1.plot(x_data, y_data, 'o-', linewidth=1, markersize=5, color=color)[0])

        self.fig.canvas.draw_idle()
        self.fig1.canvas.draw_idle()
        plt.pause(0.05)

    def update_plot_lines(self):
        """Keep plot line lists in sync with the number of vehicles (legacy plotting)."""
        while len(self.lines) > len(self.vehicles):
            line_to_remove = self.lines.pop()
            line_to_remove.remove()
            line_to_remove1 = self.lines1.pop()
            line_to_remove1.remove()

        while len(self.lines) < len(self.vehicles):
            color = next(self.color_cycle)
            new_line, = self.ax.plot([], [], 'o-', linewidth=1, markersize=5, color=color)
            self.lines.append(new_line)
            new_line1, = self.ax1.plot([], [], 'o-', linewidth=1, markersize=5, color=color)
            self.lines1.append(new_line1)


optimization = None  # Global handle so we can access results after main()

def main():
    """
    Entry point:
      - Build traj(), run the simulation loop, persist outputs (positions and MIP-gap logs).
    """
    global optimization
    optimization = traj()
    optimization.simulation_loop()
    optimization.save_vehicles_positions_to_file()

    current_time = datetime.now().strftime('%Y-%m-%d--%H-%M-%S')

    if not os.path.exists(optimization.directory_name):
        os.makedirs(optimization.directory_name)

    filename = f"gmip_file-{optimization.radius}-_iter-{optimization.total_iterations}-_T_{current_time}.png"

    filename1 = f"MIP_file-{optimization.radius}-_iter-{optimization.total_iterations}-_T_{current_time}.json"
    with open(os.path.join(optimization.directory_name, filename1), 'w') as file:
        json.dump(optimization.all_mip_gaps, file)


# Time the full run (optional; memory lines left commented)
start_time = time.perf_counter()
# start_memory = psutil.Process().memory_info().rss / 1e6  # MB

main()

# End tracking
end_time = time.perf_counter()
# end_memory = psutil.Process().memory_info().rss / 1e6

elapsed_time = end_time - start_time
# memory_used = end_memory - start_memory

current_time = datetime.now().strftime('%Y-%m-%d--%H-%M-%S')

# Summarize some run metrics
if optimization is not None:
    metrics = {
        "Code execution time (seconds)": round(elapsed_time, 6),
        # "Memory Used (MB)": round(memory_used, 2),
        "No. of agent along x": optimization.vehicle_id_x,
        "No. of agent along y": optimization.vehicle_id_y
    }

    with open(os.path.join(optimization.directory_name, f"runtime_duration_{optimization.radius}_Rnd_{optimization.lambda_str}_SD_{optimization.safe_dis}_BUI_{optimization.buff_ins}_H_{optimization.N}_T_{current_time}_C_{optimization.const_fair}"), 'w') as file:
        json.dump(metrics, file, indent=4)
else:
    print("Optimization is not initialized.")
