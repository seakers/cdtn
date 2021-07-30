# ============================================================================================================
# === MISCELLANEOUS HELPER FUNCTIONS FOR THE REINFORCEMENT LEARNING
# ============================================================================================================
import pandas as pd


def change_route_to_cross_link(bundle):
    bundle_route = bundle.route
    index_mission_45 = bundle_route.index('Mission45')
    if index_mission_45 != (len(bundle_route) - 1):  # only change route if Mission45 is not the final destination
        if bundle_route[index_mission_45 + 1] == 'Mission0' or bundle_route[index_mission_45 + 1] == 'Mission34':
            bundle_route.insert(index_mission_45 + 1, 'Mission36')


def check_node_memory(node):
    incoming_traffic = 0
    outgoing_traffic = 0
    for bundle in node.in_queue:
        incoming_traffic += bundle[0].data_vol
    for bundle in node.limbo_queue:
        incoming_traffic += bundle[0].data_vol
    for nid, neighbor_manager in node.queues.items():
        # Skipping opportunistic queues
        if neighbor_manager is None:
            continue
        # Check number of bundles in neighbour queue and create record
        neighbor_queue = neighbor_manager.queue.queue
        for priority in neighbor_queue.priorities:
            # If this priority level is empty, continue
            if not any(neighbor_queue.items[priority]): continue
            for rtn_record in neighbor_queue.items[priority]:
                outgoing_traffic += rtn_record.bundle.data_vol

    return [incoming_traffic / node.maximum_capacity, outgoing_traffic / node.maximum_capacity]


def compute_performance_metric(results):
    arrived_bundles_table = results['arrived']
    bundles_dsn_0 = arrived_bundles_table[arrived_bundles_table['node'] == 'Mission0'].shape[0]
    bundles_dsn_34 = arrived_bundles_table[arrived_bundles_table['node'] == 'Mission34'].shape[0]
    performance_metric = bundles_dsn_0 + bundles_dsn_34

    return performance_metric


def vectorize_state(state):
    vector_state = []
    for dic_id in state:
        vector_state.extend(state[dic_id])

    return vector_state


def evaluate_lunar_scenario(env, model, output_pickle_path, n_episodes=100):
    history_rewards = []
    history_benefit = []
    history_benefit_mod = []
    history_cost = []
    history_memory = []
    history_rb_in = []
    history_rb_out = []
    history_memory_neighbors = []
    history_rewards_over_time = []
    history_benefit_over_time = []
    history_cost_over_time = []
    history_dropped_over_time = []
    history_dropped_low_over_time = []
    history_dropped_medium_over_time = []
    history_dropped_high_over_time = []
    history_actions_over_time = []

    for i in range(n_episodes):
        # Evaluate model
        cumulative_reward = 0
        cumulative_benefit = 0
        cumulative_benefit_mod = 0
        cumulative_cost = 0
        memory_over_time = []
        rb_in_over_time = []
        rb_out_over_time = []
        memory_neighbors_over_time = []
        reward_over_time = []
        benefit_over_time = []
        cost_over_time = []
        dropped_over_time = []
        dropped_low_over_time = []
        dropped_medium_over_time = []
        dropped_high_over_time = []
        action_over_time = []
        obs = env.reset()
        dones = False
        while not dones:
            print('State: {}'.format(obs))
            action, _states = model.predict(obs)
            print('Action: {}'.format(action))
            obs, rewards, dones, info = env.step(action)
            print('Reward: {}'.format(rewards))
            cumulative_reward = cumulative_reward + rewards
            cumulative_benefit = cumulative_benefit + info['benefit']
            cumulative_benefit_mod = cumulative_benefit_mod + info['benefit_mod']
            cumulative_cost = cumulative_cost + info['cost']

            ############################################
            # Uncomment the following block for scenario 'cdtn-ASCEND2020-v0'
            # vector_state = vectorize_state(env.observe_state())
            # memory = vector_state[0]
            # data_rate_in = vector_state[1]
            # data_rate_out = vector_state[2]
            ############################################

            ############################################
            # Uncomment the following block for scenarios 'cdtn-JAIS2021-v0', 'cdtn-prioritiesRL-v0'
            # and 'cdtn-prioritiesHybrid-v0'
            _, next_state_dict = env.observe_state()
            memory = next_state_dict['memory']
            data_rate_in = next_state_dict['in_data_rate']
            data_rate_out = next_state_dict['out_data_rate']
            ############################################

            memory_over_time.append(memory)
            rb_in_over_time.append(data_rate_in)
            rb_out_over_time.append(data_rate_out)
            memory_neighbors_over_time.append(env.observe_bits_in_memory_neighbors())
            reward_over_time.append(rewards)
            benefit_over_time.append(info['benefit'])
            cost_over_time.append(info['cost'])

            ############################################
            # Uncomment the following block for scenarios 'cdtn-prioritiesRL-v0' and 'cdtn-prioritiesHybrid-v0'
            # dropped_over_time.append(info['dropped'])
            # dropped_low_over_time.append(info['dropped_low'])
            # dropped_medium_over_time.append(info['dropped_medium'])
            # dropped_high_over_time.append(info['dropped_high'])
            # action_over_time.append(info['action'])
            #############################################

        print('Finished episode {} with cumulative reward {}'.format(i, cumulative_reward))
        print('\n')
        history_rewards.append(cumulative_reward)
        history_benefit.append(cumulative_benefit)
        history_benefit_mod.append(cumulative_benefit_mod)
        history_cost.append(cumulative_cost)

        history_memory.append(memory_over_time)
        history_rb_in.append(rb_in_over_time)
        history_rb_out.append(rb_out_over_time)
        history_memory_neighbors.append(memory_neighbors_over_time)
        history_rewards_over_time.append(reward_over_time)
        history_benefit_over_time.append(benefit_over_time)
        history_cost_over_time.append(cost_over_time)
        history_dropped_over_time.append(dropped_over_time)
        history_dropped_low_over_time.append(dropped_low_over_time)
        history_dropped_medium_over_time.append(dropped_medium_over_time)
        history_dropped_high_over_time.append(dropped_high_over_time)
        history_actions_over_time.append(action_over_time)
        print(history_rewards)
        print('Saving history of rewards, benefit and cost...')
        d = {'reward': history_rewards,  # array of episode cumulative rewards (index corresponds to run number)
             'benefit': history_benefit,  # array of episode cumulative benefits (index corresponds to run number)
             'benefit_mod': history_benefit_mod,
             'cost': history_cost,  # array of episode cumulative costs (index corresponds to run number)
             'memory': history_memory,  # array of arrays of gateway's memory (index1 corresponds to run number and index2 corresponds to time step number)
             'rb_in': history_rb_in,  # array of arrays of data rate in(index1 corresponds to run number and index2 corresponds to time step number)
             'rb_out': history_rb_out,  # array of arrays of data rate out (index1 corresponds to run number and index2 corresponds to time step number)
             'memory_neighbors': history_memory_neighbors,  # array of arrays of dicts containing the memory of the neighbors (index1 corresponds to run number and index2 corresponds to time step number. The keys are the ids of the neighbor nodes: 'Mission26',...)
             'rewards_over_time': history_rewards_over_time,  # array of arrays of step rewards(index1 corresponds to run number and index2 corresponds to time step number)
             'benefit_over_time': history_benefit_over_time,  # array of arrays of step benefits(index1 corresponds to run number and index2 corresponds to time step number)
             'cost_over_time': history_cost_over_time,  # array of arrays of step costs(index1 corresponds to run number and index2 corresponds to time step number)
             'dropped_over_time': history_dropped_over_time,  # array of arrays of total bits dropped (index1 corresponds to run number and index2 corresponds to time step number)
             'dropped_low_over_time': history_dropped_low_over_time,  # array of arrays of low priority bits dropped (index1 corresponds to run number and index2 corresponds to time step number)
             'dropped_medium_over_time': history_dropped_medium_over_time,  # array of arrays of medium priority bits dropped (index1 corresponds to run number and index2 corresponds to time step number)
             'dropped_high_over_time': history_dropped_high_over_time,  # array of arrays of high priority bits dropped (index1 corresponds to run number and index2 corresponds to time step number)
             'actions_over_time': history_actions_over_time  # array of arrays of actions taken (index1 corresponds to run number and index2 corresponds to time step number)
             }
        df = pd.DataFrame(data=d)
        df.to_pickle(output_pickle_path)


def evaluate_EO_scenario(env, model, output_pickle_path, n_episodes=100):
    history_rewards = []
    history_wrong_actions = []
    history_benefit = []
    history_cost = []
    history_memory = []
    history_rb = []
    history_rewards_over_time = []
    history_benefit_over_time = []
    history_cost_over_time = []
    history_dropped_over_time = []
    history_actions_over_time = []
    history_dropped_reasons = []

    for i in range(n_episodes):
        # Evaluate model
        cumulative_reward = 0
        cumulative_wrong_actions = 0
        cumulative_benefit = 0
        cumulative_cost = 0
        memory_over_time = []
        rb_over_time = []
        reward_over_time = []
        benefit_over_time = []
        cost_over_time = []
        dropped_over_time = []
        action_over_time = []
        drop_reasons = []
        obs = env.reset()
        dones = False
        while not dones:
            print('State: {}'.format(obs))
            action, _states = model.predict(obs)
            print('Action: {}'.format(action))
            wrong_action = env.wrong_action(action)
            # if wrong_action:
            #     print('Wrong action!')
            # else:
            #     print('Correct action!')
            obs, rewards, dones, info = env.step(action)
            print('Reward: {}'.format(rewards))
            cumulative_reward = cumulative_reward + rewards
            cumulative_wrong_actions = cumulative_wrong_actions + wrong_action
            cumulative_benefit = cumulative_benefit + info['benefit']
            cumulative_cost = cumulative_cost + info['cost']
            _, next_state_dict = env.observe_state()
            memory = next_state_dict['memory']
            data_rate = next_state_dict['data_rate']
            memory_over_time.append(memory)
            rb_over_time.append(data_rate)
            reward_over_time.append(rewards)
            benefit_over_time.append(info['benefit'])
            cost_over_time.append(info['cost'])
            dropped_over_time.append(info['dropped_total'])
            action_over_time.append(info['action'])
            drop_reasons2 = []
            for nid, node_i in env.env.nodes.items():
                node_dropped_queue = node_i.dropped
                for bundle in node_dropped_queue:
                    drop_reasons2.append(bundle.drop_reason)
            drop_reasons.append(drop_reasons2)

        print('Finished episode {} with cumulative reward {}'.format(i, cumulative_reward))
        print('\n')
        history_rewards.append(cumulative_reward)
        history_wrong_actions.append(cumulative_wrong_actions)
        history_benefit.append(cumulative_benefit)
        history_cost.append(cumulative_cost)

        history_memory.append(memory_over_time)
        history_rb.append(rb_over_time)
        history_rewards_over_time.append(reward_over_time)
        history_benefit_over_time.append(benefit_over_time)
        history_cost_over_time.append(cost_over_time)
        history_dropped_over_time.append(dropped_over_time)
        history_actions_over_time.append(action_over_time)
        history_dropped_reasons.append(drop_reasons)

        print(history_rewards)
        print('Saving history of rewards, benefit and cost...')
        d = {'reward': history_rewards,  # array of episode cumulative rewards (index corresponds to run number)
             'benefit': history_benefit,  # array of episode cumulative benefits (index corresponds to run number)
             'cost': history_cost,  # array of episode cumulative costs (index corresponds to run number)
             'memory': history_memory,  # array of arrays of arrays of all nodes memories (index1 corresponds to run number, index2 corresponds to time step number and index 3 corresponds to the node number)
             'data_rate': history_rb,  # array of arrays of arrays of all nodes transmitting data rates (index1 corresponds to run number, index2 corresponds to time step number and index 3 corresponds to the node number)
             'rewards_over_time': history_rewards_over_time,  # array of arrays of step rewards(index1 corresponds to run number and index2 corresponds to time step number)
             'benefit_over_time': history_benefit_over_time,  # array of arrays of step benefits(index1 corresponds to run number and index2 corresponds to time step number)
             'cost_over_time': history_cost_over_time,  # array of arrays of step costs(index1 corresponds to run number and index2 corresponds to time step number)
             'dropped_over_time': history_dropped_over_time,  # array of arrays of total bits dropped (index1 corresponds to run number and index2 corresponds to time step number)
             'actions_over_time': history_actions_over_time,  # array of arrays of actions taken (index1 corresponds to run number and index2 corresponds to time step number)
             'wrong_actions': history_wrong_actions,  # array of episode cumulative wrong actions (index corresponds to run number). This is a very subjective metric.
             'drop_reasons': history_dropped_reasons  # array of arrays of dropping packet reasons (index1 corresponds to run number and index2 corresponds to time step number)
             }
        df = pd.DataFrame(data=d)
        df.to_pickle(output_pickle_path)
