import gymnasium as gym
from gymnasium import spaces

import timeit

import numpy as np
import traci
import traci.constants as tc

cell_length = 7
detection_length = 350
n_channels = 2
width = 16
height = int(detection_length / cell_length)
car_occupancy = 1
bus_occupancy = 1
min_left_green_time = 5
min_through_green_time = 12

edges = {
    'east_edge': (0, '-E2'),
    'south_edge': (4, '-E3'),
    'west_edge': (8, 'E0'),
    'north_edge': (12, 'E1')
}
incoming_edges = ['-E2', '-E3', 'E0', 'E1']
action_state_map = {
    0: 'grrrgrrGGgrrrgrrGG',  # WL EL
    1: 'grrrgrrrrgrrrgGGGG',  # WL WT
    2: 'grrrgGGrrgrrrgGGrr',  # WT ET
    3: 'grrrgGGGGgrrrgrrrr',  # EL ET
    4: 'grrGgrrrrgrrGgrrrr',  # SL NL
    5: 'grrrgrrrrgGGGgrrrr',  # SL ST
    6: 'gGGrgrrrrgGGrgrrrr',  # ST NT
    7: 'gGGGgrrrrgrrrgrrrr'  # NL NT
}


class SumoEnv(gym.Env):
    """Custom Environment that follows gym interface
    :param sumo_cmd: The command for the sumo.
    :param obs_type: Sets the output type ('img': image, 'vec': vector, 'comb': combined)
    for observations in the environment.
    :param cv_only: Controls whether only the CV can be detected.
    """

    metadata = {'render.modes': ['human', 'rgb_array']}

    def __init__(self, sumo_cmd, obs_type='img', cv_only=False):
        super(SumoEnv, self).__init__()
        # Define action and observation space
        # They must be gym.spaces objects
        # Initiate action space:
        self.action_space = spaces.Discrete(8)

        # Initiate observation space:
        self.obs_type = obs_type

        if self.obs_type == 'comb':
            self.observation_space = spaces.Box(low=0, high=255, shape=(n_channels+1, height, width), dtype=np.uint8)
            # self.observation_space = spaces.Dict(
            #     {
            #         'img': spaces.Box(low=0, high=255, shape=(n_channels, height, width), dtype=np.uint8),
            #         'vec': spaces.Box(low=0, high=1, shape=(width,), dtype=np.float64),
            #     }
            # )

        elif self.obs_type == 'vec':
            self.observation_space = spaces.Box(low=0, high=1, shape=(width,), dtype=np.float64)

        else:
            self.observation_space = spaces.Box(low=0, high=255, shape=(n_channels, height, width), dtype=np.uint8)

        self.cv_det = cv_only
        self.episode = 0
        self.total_rewards = []
        self.sumo_cmd = sumo_cmd
        self.yellow_time = 3
        self.red_time = 2
        self.min_left_green_time = min_left_green_time
        self.min_through_green_time = min_through_green_time

    def reset(self, seed=None, options=None):
        self.episode += 1
        try:
            traci.close()
        except:
            pass

        self.done = False
        self.ep_reward = 0
        self.start_time = timeit.default_timer()
        self.sim_step = 0
        self.ep_step = 0

        # print(f'---Episode: {self.episode}--- Simulating...')
        traci.start(self.sumo_cmd)

        # Warm up 10 minutes
        while self.sim_step <= 600:
            if self.sim_step == 600:
                self.last_state, self.last_tot_person_delay = self.get_state()
                last_phase = traci.trafficlight.getRedYellowGreenState('J1')
                for k, v in action_state_map.items():
                    if v == last_phase:
                        self.last_action = k
                    else:
                        self.last_action = 0
                return self.last_state, {}
            traci.simulationStep()
            self.sim_step += 1
        # print(self.last_state)

    def step(self, action):
        # Take the action: Signal control
        current_action = action
        if current_action != self.last_action:
            self.set_yellow_red(current_action, self.last_action)
            if current_action == 2 or current_action == 6:
                self.set_green(current_action, self.min_through_green_time)
            else:
                self.set_green(current_action, self.min_left_green_time)
        else:
            self.simulate(1)

        # Get the info after taking the action
        self.current_state, current_tot_person_delay = self.get_state()
        self.reward = self.last_tot_person_delay - current_tot_person_delay
        # print(self.reward)

        # Update the last action and total person delay for the next step
        self.last_action = current_action
        self.last_tot_person_delay = current_tot_person_delay

        self.ep_reward += self.reward
        terminated = False
        if self.sim_step > 4400:
            self.done = True
            # print(f'Episode: {self.episode}---Total Steps: {self.ep_step}---Total Sim Steps: {self.sim_step}')
            # simulation_time = round(timeit.default_timer() - self.start_time, 1)
            # info.update({'Simulation_time': simulation_time})
            # print(f'Simulation time: {simulation_time} seconds -- '
            #       f'Total reward: {self.ep_reward} -- ')
            traci.close()
            self.save_episode_stats()
        self.ep_step += 1

        return self.current_state, self.reward, terminated, self.done, {'ep_step': self.ep_step}

    def render(self):
        pass

    def close(self):
        traci.close()

    # Get the state
    def get_state(self):
        img_state = np.zeros((n_channels, height, width))
        queue_state = np.zeros(width)
        tot_person_delay = 0
        tot_queue_veh = 0

        for veh_id in traci.vehicle.getIDList():
            traci.vehicle.subscribe(veh_id, (tc.VAR_NEXT_TLS, tc.VAR_LANE_ID, tc.VAR_SPEED, tc.VAR_TYPE,
                                             tc.VAR_TIMELOSS))
        p = traci.vehicle.getAllSubscriptionResults()
        for x in p:
            v_type = p[x][tc.VAR_TYPE]
            # Not the only cv detection mode
            if not self.cv_det:
                if p[x][tc.VAR_NEXT_TLS]:
                    ps_tls = p[x][tc.VAR_NEXT_TLS][0][2]  # get the distance to the traffic light
                else:
                    ps_tls = -1  # vehicle crossing the stop line is set to a negative value

                if p[x][tc.VAR_LANE_ID]:
                    ln_id, ln_idx = p[x][tc.VAR_LANE_ID].split('_')  # get the lane id and index

                spd = p[x][tc.VAR_SPEED]  # get the speed

                if ps_tls > 0:  # vehicle not crossing the stop line
                    delay = p[x][tc.VAR_TIMELOSS]
                else:  # vehicle already crossing the stop line
                    delay = 0

                # get the vehicle type and assign the occupancy
                if v_type == 'car':
                    v_occupancy = car_occupancy
                    person_delay = delay * car_occupancy
                else:
                    v_occupancy = bus_occupancy
                    person_delay = delay * bus_occupancy
                tot_person_delay += person_delay

                # get the position in state array
                if 0 < ps_tls < detection_length:
                    height_index = int(ps_tls / cell_length)
                    for edge in edges.values():
                        if edge[1] in ln_id:
                            width_index = int(ln_idx) + edge[0]
                            img_state[:, height_index, width_index] = (v_occupancy, spd)
            # The only cv detection mode on
            else:
                if v_type == 'cv' or v_type == 'bus':
                    if p[x][tc.VAR_NEXT_TLS]:
                        ps_tls = p[x][tc.VAR_NEXT_TLS][0][2]  # get the distance to the traffic light
                    else:
                        ps_tls = -1  # vehicle crossing the stop line is set to a negative value

                    if p[x][tc.VAR_LANE_ID]:
                        ln_id, ln_idx = p[x][tc.VAR_LANE_ID].split('_')  # get the lane id and index

                    spd = p[x][tc.VAR_SPEED]  # get the speed

                    if ps_tls > 0:  # vehicle not crossing the stop line
                        delay = p[x][tc.VAR_TIMELOSS]
                    else:  # vehicle already crossing the stop line
                        delay = 0

                    # get the vehicle type and assign the occupancy
                    if v_type == 'cv':
                        v_occupancy = car_occupancy
                        person_delay = delay * car_occupancy
                    else:
                        v_occupancy = bus_occupancy
                        person_delay = delay * bus_occupancy
                    tot_person_delay += person_delay

                    # get the position in state array
                    if 0 < ps_tls < detection_length:
                        height_index = int(ps_tls / cell_length)
                        for edge in edges.values():
                            if edge[1] in ln_id:
                                width_index = int(ln_idx) + edge[0]
                                img_state[:, height_index, width_index] = (v_occupancy, spd)

        for edge in edges.values():
            for i in range(4):
                width_index = i + edge[0]
                ln_id = f'{edge[1]}_{i}'
                queue_veh_lane = traci.lane.getLastStepHaltingNumber(ln_id)
                queue_state[width_index] = traci.lane.getLastStepHaltingNumber(ln_id)
                tot_queue_veh += queue_veh_lane

        if self.obs_type == 'img':
            state = img_state
            state = state.astype(np.uint8)
            # return state, tot_person_delay

        # Count the stopped vehicles on each lane, speed <= 0.1
        elif self.obs_type == 'comb':
            # Concatenate the queue array with the image state, after this, the dimension is (3, 50, 16)
            queue_array = np.zeros((1, height, width))
            queue_array[:, 0, :] += queue_state
            state = np.concatenate((img_state, queue_array), axis=0)
            state = state.astype(np.uint8)

            # state = {
            #     'img': img_state,
            #     'vec': queue_state
            # }
        else:
            state = queue_state

        # return state, tot_person_delay
        return state, tot_queue_veh

    # Execute the designated simulation step
    def simulate(self, steps_todo):
        while steps_todo > 0:
            traci.simulationStep()
            self.sim_step += 1
            steps_todo -= 1

    def set_green(self, action, min_green_time):
        """
        phase-movement mapping
        {0: (WL, EL), 1: (W, WL), 2: (W, E), 3: (E, EL), 4: (SL, NL), 5: (S, SL), 6: (S, N), 7: (N, NL)}
        """
        green_state = action_state_map[action]
        traci.trafficlight.setRedYellowGreenState('J1', green_state)
        self.simulate(min_green_time)
        # print('------Set green------')

    # Activate the corresponding yellow and red phase
    def set_yellow_red(self, action, last_action):
        action_state = action_state_map[action]
        old_action_state = action_state_map[last_action]
        yellow_state = []
        red_state = []
        for i in range(18):
            # print(action_state[i], old_action_state[i])
            if old_action_state[i] == 'G' and old_action_state[i] != action_state[i]:
                yellow_state.append('Y')
            else:
                yellow_state.append(old_action_state[i])
        yellow_state = ''.join(yellow_state)
        traci.trafficlight.setRedYellowGreenState('J1', yellow_state)
        self.simulate(self.yellow_time)

        for i in range(18):
            if yellow_state[i] == 'Y':
                red_state.append('r')
            else:
                red_state.append(yellow_state[i])
        red_state = ''.join(red_state)
        traci.trafficlight.setRedYellowGreenState('J1', red_state)
        self.simulate(self.red_time)

    def save_episode_stats(self):
        self.total_rewards.append(self.ep_reward)

    def get_stats(self):
        return {
            'Reward': self.total_rewards,
            # 'Mean Waiting Time (s)': np.divide(self.total_person_delays, self.step)
        }

    def save_stats(self, save_time):
        np.savetxt(f'result\\training_stats_{save_time}.csv', self.total_rewards, delimiter=',')
