from copy import deepcopy
import itertools
import math
import numpy as np
from operator import attrgetter
from collections import defaultdict
from overcooked_ai_py.mdp.actions import Action
from overcooked_ai_py.planning.planners import MediumLevelActionManager, MotionPlanner, NO_COUNTERS_PARAMS, COUNTERS_MLG_PARAMS
from overcooked_ai_py.mdp.overcooked_mdp import Recipe

class Agent(object):

    def __init__(self):
        self.motion_goal = None
        self.chosen_goal = None
        self.hl_objective_switch = 0
        self.stuck_frames = 0
        self.reset()

    def action(self, state):
        """
        Should return an action, and an action info dictionary.
        If collecting trajectories of the agent with OvercookedEnv, the action
        info data will be included in the trajectory data under `ep_infos`.

        This allows agents to optionally store useful information about them
        in the trajectory for further analysis.
        """
        return NotImplementedError()

    def actions(self, states, agent_indices):
        """
        A multi-state version of the action method. This enables for parallized
        implementations that can potentially give speedups in action prediction. 

        Args:
            states (list): list of OvercookedStates for which we want actions for
            agent_indices (list): list to inform which agent we are requesting the action for in each state

        Returns:
            [(action, action_info), (action, action_info), ...]: the actions and action infos for each state-agent_index pair
        """
        return NotImplementedError()

    @staticmethod
    def a_probs_from_action(action):
        action_idx = Action.ACTION_TO_INDEX[action]
        return np.eye(Action.NUM_ACTIONS)[action_idx]

    @staticmethod
    def check_action_probs(action_probs, tolerance=1e-4):
        """Check that action probabilities sum to ≈ 1.0"""
        probs_sum = sum(action_probs)
        assert math.isclose(probs_sum, 1.0, rel_tol=tolerance), "Action probabilities {} should sum up to approximately 1 but sum up to {}".format(
            list(action_probs), probs_sum)

    def set_agent_index(self, agent_index):
        self.agent_index = agent_index

    def set_mdp(self, mdp):
        self.mdp = mdp

    def reset(self):
        """
        One should always reset agents in between trajectory rollouts, as resetting
        usually clears history or other trajectory-specific attributes.
        """
        self.agent_index = None
        self.mdp = None


class AgentGroup(object):
    """
    AgentGroup is a group of N agents used to sample 
    joint actions in the context of an OvercookedEnv instance.
    """

    def __init__(self, *agents, allow_duplicate_agents=False):
        self.agents = agents
        self.n = len(self.agents)
        self.reset()

        if not all(a0 is not a1 for a0, a1 in itertools.combinations(agents, 2)):
            assert allow_duplicate_agents, "All agents should be separate instances, unless allow_duplicate_agents is set to true"

    def joint_action(self, state):
        actions_and_probs_n = tuple(a.action(state) for a in self.agents)
        return actions_and_probs_n

    def set_mdp(self, mdp):
        for a in self.agents:
            a.set_mdp(mdp)

    def reset(self):
        """
        When resetting an agent group, we know that the agent indices will remain the same,
        but we have no guarantee about the mdp, that must be set again separately.
        """
        for i, agent in enumerate(self.agents):
            agent.reset()
            agent.set_agent_index(i)


class AgentPair(AgentGroup):
    """
    AgentPair is the N=2 case of AgentGroup. Unlike AgentGroup,
    it supports having both agents being the same instance of Agent.

    NOTE: Allowing duplicate agents (using the same instance of an agent
    for both fields can lead to problems if the agents have state / history)
    """

    def __init__(self, *agents, allow_duplicate_agents=False):
        super().__init__(*agents, allow_duplicate_agents=allow_duplicate_agents)
        assert self.n == 2
        self.a0, self.a1 = self.agents

    def joint_action(self, state):
        if self.a0 is self.a1:
            # When using the same instance of an agent for self-play,
            # reset agent index at each turn to prevent overwriting it
            self.a0.set_agent_index(0)
            action_and_infos_0 = self.a0.action(state)
            self.a1.set_agent_index(1)
            action_and_infos_1 = self.a1.action(state)
            joint_action_and_infos = (action_and_infos_0, action_and_infos_1)
            return joint_action_and_infos
        else:
            return super().joint_action(state)


class NNPolicy(object):
    """
    This is a common format for NN-based policies. Once one has wrangled the intended trained neural net
    to this format, one can then easily create an Agent with the AgentFromPolicy class.
    """

    def __init__(self):
        pass

    def multi_state_policy(self, states, agent_indices):
        """
        A function that takes in multiple OvercookedState instances and their respective agent indices and returns action probabilities.
        """
        raise NotImplementedError()

    def multi_obs_policy(self, states):
        """
        A function that takes in multiple preprocessed OvercookedState instatences and returns action probabilities.
        """
        raise NotImplementedError()


class AgentFromPolicy(Agent):
    """
    This is a useful Agent class backbone from which to subclass from NN-based agents.
    """

    def __init__(self, policy):
        """
        Takes as input an NN Policy instance
        """
        self.policy = policy
        self.reset()

    def action(self, state):
        return self.actions([state], [self.agent_index])[0]

    def actions(self, states, agent_indices):
        action_probs_n = self.policy.multi_state_policy(states, agent_indices)
        actions_and_infos_n = []
        for action_probs in action_probs_n:
            action = Action.sample(action_probs)
            actions_and_infos_n.append(
                (action, {"action_probs": action_probs}))
        return actions_and_infos_n

    def set_mdp(self, mdp):
        super().set_mdp(mdp)
        self.policy.mdp = mdp

    def reset(self):
        super(AgentFromPolicy, self).reset()
        self.policy.mdp = None


class RandomAgent(Agent):
    """
    An agent that randomly picks motion actions.
    NOTE: Does not perform interact actions, unless specified
    """

    def __init__(self, sim_threads=None, all_actions=False, custom_wait_prob=None):
        self.sim_threads = sim_threads
        self.all_actions = all_actions
        self.custom_wait_prob = custom_wait_prob
        self.motion_goal = None
        self.stuck_frames = None
        self.hl_objective_switch = None
        self.intentions = None

    def action(self, state):
        action_probs = np.zeros(Action.NUM_ACTIONS)
        legal_actions = list(Action.MOTION_ACTIONS)
        if self.all_actions:
            legal_actions = Action.ALL_ACTIONS
        legal_actions_indices = np.array(
            [Action.ACTION_TO_INDEX[motion_a] for motion_a in legal_actions])
        action_probs[legal_actions_indices] = 1 / len(legal_actions_indices)

        if self.custom_wait_prob is not None:
            stay = Action.STAY
            if np.random.random() < self.custom_wait_prob:
                return stay, {"action_probs": Agent.a_probs_from_action(stay)}
            else:
                action_probs = Action.remove_indices_and_renormalize(
                    action_probs, [Action.ACTION_TO_INDEX[stay]])

        return Action.sample(action_probs), {"action_probs": action_probs}

    def actions(self, states, agent_indices):
        return [self.action(state) for state in states]

    def direct_action(self, obs):
        return [np.random.randint(4) for _ in range(self.sim_threads)]


class StayAgent(Agent):

    def __init__(self, sim_threads=None):
        self.sim_threads = sim_threads

    def action(self, state):
        a = Action.STAY
        return a, {}

    def direct_action(self, obs):
        return [Action.ACTION_TO_INDEX[Action.STAY]] * self.sim_threads


class FixedPlanAgent(Agent):
    """
    An Agent with a fixed plan. Returns Stay actions once pre-defined plan has terminated.
    # NOTE: Assumes that calls to action are sequential (agent has history)
    """

    def __init__(self, plan):
        self.plan = plan
        self.i = 0

    def action(self, state):
        if self.i >= len(self.plan):
            return Action.STAY, {}
        curr_action = self.plan[self.i]
        self.i += 1
        return curr_action, {}

    def reset(self):
        super().reset()
        self.i = 0


class PlanningAgent(Agent):
    """
    Agent that at each step selects a medium level action corresponding
    to the most intuitively high-priority thing to do

    NOTE: MIGHT NOT WORK IN ALL ENVIRONMENTS, for example forced_coordination.layout,
    in which an individual agent cannot complete the task on their own.
    Will work only in environments where the only order is 3 onion soup.
    """

    def __init__(self, hl_boltzmann_rational=False, ll_boltzmann_rational=False, hl_temp=1, ll_temp=1,
                 auto_unstuck=True):
        #self.mdp = mdp
        self.intentions = {"recipe": None, "goal": None, "agent_name": None}
        self.motion_goal = None
        self.chosen_goal = None
        self.hl_objective_switch = 0
        self.stuck_frames = 0
        Recipe.configure({})
        self.hl_goal = Recipe(['tomato'])
        

        # Bool for perfect rationality vs Boltzmann rationality for high level and low level action selection
        # For choices among high level goals of same type
        self.hl_boltzmann_rational = hl_boltzmann_rational
        # For choices about low level motion
        self.ll_boltzmann_rational = ll_boltzmann_rational

        # Coefficient for Boltzmann rationality for high level action selection
        self.hl_temperature = hl_temp
        self.ll_temperature = ll_temp

        # Whether to automatically take an action to get the agent unstuck if it's in the same
        # state as the previous turn. If false, the agent is history-less, while if true it has history.
        self.auto_unstuck = auto_unstuck
        self.next_order_info = None
        self.reset()

    def reset(self):
        self.prev_state = None
        #self.mdp = mdp
        #self.mlam = MediumLevelActionManager.from_pickle_or_compute(self.mdp, NO_COUNTERS_PARAMS)

    def set_mdp(self, mdp):
        super().set_mdp(mdp)
        counter_params = COUNTERS_MLG_PARAMS
        if self.mdp.counter_goals:
            counter_params["counter_goals"] = self.mdp.counter_goals
            counter_params["counter_drop"] = self.mdp.counter_goals
            counter_params["counter_pickup"] = self.mdp.counter_goals
        self.mlam = MediumLevelActionManager.from_pickle_or_compute(
            self.mdp, counter_params, force_compute=False)
        a = 1

    def actions(self, states, agent_indices):
        actions_and_infos_n = []
        for state, agent_idx in zip(states, agent_indices):
            self.set_agent_index(agent_idx)
            self.reset()
            actions_and_infos_n.append(self.action(state))
        return actions_and_infos_n

    def action(self, state):
        self.motion_goal = self.ml_action(state)

        # Once we have identified the motion goals for the medium
        # level action we want to perform, select the one with lowest cost
        start_pos_and_or = state.players_pos_and_or[self.agent_index]

        chosen_goal, chosen_action, action_probs = self.choose_motion_goal(
            start_pos_and_or, self.motion_goal)
        self.chosen_goal = chosen_goal

        if self.ll_boltzmann_rational and chosen_goal[0] == start_pos_and_or[0]:
            chosen_action, action_probs = self.boltzmann_rational_ll_action(
                start_pos_and_or, chosen_goal)

        if self.auto_unstuck:
            # HACK: if two agents get stuck, select an action at random that would
            # change the player positions if the other player were not to move
            if self.prev_state is not None and state.players_pos_and_or == self.prev_state.players_pos_and_or:
                self.stuck_frames += 1
                if self.agent_index == 0:
                    joint_actions = list(itertools.product(
                        Action.ALL_ACTIONS, [Action.STAY]))
                elif self.agent_index == 1:
                    joint_actions = list(itertools.product(
                        [Action.STAY], Action.ALL_ACTIONS))
                else:
                    raise ValueError("Player index not recognized")

                unblocking_joint_actions = []
                for j_a in joint_actions:
                    new_state, _ = self.mlam.mdp.get_state_transition(
                        state, j_a)
                    if new_state.player_positions != self.prev_state.player_positions:
                        unblocking_joint_actions.append(j_a)
                # Getting stuck became a possiblity simply because the nature of a layout (having a dip in the middle)
                if len(unblocking_joint_actions) == 0:
                    unblocking_joint_actions.append([Action.STAY, Action.STAY])
                chosen_action = unblocking_joint_actions[np.random.choice(len(unblocking_joint_actions))][
                    self.agent_index]
                action_probs = self.a_probs_from_action(chosen_action)

            # NOTE: Assumes that calls to the action method are sequential
            self.prev_state = state
        return chosen_action, {"action_probs": action_probs}

    def choose_motion_goal(self, start_pos_and_or, motion_goals):
        """
        For each motion goal, consider the optimal motion plan that reaches the desired location.
        Based on the plan's cost, the method chooses a motion goal (either boltzmann rationally
        or rationally), and returns the plan and the corresponding first action on that plan.
        """
        if self.hl_boltzmann_rational:
            possible_plans = [self.mlam.motion_planner.get_plan(
                start_pos_and_or, goal) for goal in motion_goals]
            plan_costs = [plan[2] for plan in possible_plans]
            goal_idx, action_probs = self.get_boltzmann_rational_action_idx(
                plan_costs, self.hl_temperature)
            chosen_goal = motion_goals[goal_idx]
            chosen_goal_action = possible_plans[goal_idx][0][0]
        else:
            chosen_goal, chosen_goal_action = self.get_lowest_cost_action_and_goal(
                start_pos_and_or, motion_goals)
            action_probs = self.a_probs_from_action(chosen_goal_action)
        return chosen_goal, chosen_goal_action, action_probs

    def get_boltzmann_rational_action_idx(self, costs, temperature):
        """Chooses index based on softmax probabilities obtained from cost array"""
        costs = np.array(costs)
        softmax_probs = np.exp(-costs * temperature) / \
            np.sum(np.exp(-costs * temperature))
        action_idx = np.random.choice(len(costs), p=softmax_probs)
        return action_idx, softmax_probs

    def get_lowest_cost_action_and_goal(self, start_pos_and_or, motion_goals):
        """
        Chooses motion goal that has the lowest cost action plan.
        Returns the motion goal itself and the first action on the plan.
        """
        min_cost = np.Inf
        best_action, best_goal = None, None
        for goal in motion_goals:
            action_plan, _, plan_cost = self.mlam.motion_planner.get_plan(
                start_pos_and_or, goal)
            if plan_cost < min_cost:
                best_action = action_plan[0]
                min_cost = plan_cost
                best_goal = goal
        return best_goal, best_action

    def boltzmann_rational_ll_action(self, start_pos_and_or, goal, inverted_costs=False):
        """
        Computes the plan cost to reach the goal after taking each possible low level action.
        Selects a low level action boltzmann rationally based on the one-step-ahead plan costs.

        If `inverted_costs` is True, it will make a boltzmann "irrational" choice, exponentially
        favouring high cost plans rather than low cost ones.
        """
        future_costs = []
        for action in Action.ALL_ACTIONS:
            pos, orient = start_pos_and_or
            new_pos_and_or = self.mdp._move_if_direction(pos, orient, action)
            _, _, plan_cost = self.mlam.motion_planner.get_plan(
                new_pos_and_or, goal)
            sign = (-1) ** int(inverted_costs)
            future_costs.append(sign * plan_cost)

        action_idx, action_probs = self.get_boltzmann_rational_action_idx(
            future_costs, self.ll_temperature)
        return Action.ALL_ACTIONS[action_idx], action_probs

    




    def ml_action(self, state):
        """
        Selects a medium level action for the current state.
        Motion goals can be thought of instructions of the form:
            [do X] at location [Y]

        In this method, X (e.g. deliver the soup, pick up an onion, etc) is chosen based on
        a simple set of greedy heuristics based on the current state.

        Effectively, will return a list of all possible locations Y in which the selected
        medium level action X can be performed.
        """
        player = state.players[self.agent_index]
        other_player = state.players[1 - self.agent_index]
        am = self.mlam

        counter_objects = self.mlam.mdp.get_counter_objects_dict(
            state, list(self.mlam.mdp.terrain_pos_dict['X']))
        pot_states_dict = self.mlam.mdp.get_pot_states(state)

        if not player.has_object():
            ready_soups = pot_states_dict['ready']
            cooking_soups = pot_states_dict['cooking']

            soup_nearly_ready = len(ready_soups) > 0 or len(cooking_soups) > 0
            other_has_dish = other_player.has_object(
            ) and other_player.get_object().name == 'dish'

            if soup_nearly_ready and not other_has_dish:
                self.intentions['goal'] = 'D'
                motion_goals = am.pickup_dish_actions(counter_objects)
            else:
                self.next_order_info = self.hl_action(state)
                self.intentions["recipe"] = self.next_order_info["recipe"].ingredients
                soups_ready_to_cook_key = '{}_items'.format(
                    len(self.next_order_info["recipe"].ingredients))
                soups_ready_to_cook = pot_states_dict[soups_ready_to_cook_key]
                if soups_ready_to_cook:
                    only_pot_states_ready_to_cook = defaultdict(list)
                    only_pot_states_ready_to_cook[soups_ready_to_cook_key] = soups_ready_to_cook
                    # we want to cook only soups that has same len as order
                    motion_goals = am.start_cooking_actions(
                        only_pot_states_ready_to_cook)

                elif self.next_order_info["most_advanced_pot"]:
                    if 'onion' in self.next_order_info["missing_ingredients_in_MA_pot"]:
                        self.intentions['goal'] = 'O'
                        motion_goals = am.pickup_onion_actions(counter_objects)
                    elif 'tomato' in self.next_order_info["missing_ingredients_in_MA_pot"]:
                        self.intentions['goal'] = 'T'
                        motion_goals = am.pickup_tomato_actions(
                            counter_objects)
                    else:
                        #self.next_order = self.hl_action(state)
                        motion_goals = am.wait_actions(player)
                        motion_goals
                else:
                    motion_goals = am.go_to_closest_feature_actions(player)
                    motion_goals

        else:
            player_obj = player.get_object()
            all_recipes = self.hl_info(state)
            try :
                self.next_order_info["missing_ingredients_in_MA_pot"] = all_recipes[self.next_order_info["recipe"]]["missing_ingredients_in_MA_pot"]
            except KeyError:
                pass

            if player_obj.name == 'onion':
                # self.next_order_info["min_cost_to_complete"] == any([10000, 0]):
                if 'onion' not in self.next_order_info["missing_ingredients_in_MA_pot"]:
                    motion_goals = am.place_obj_on_counter_actions(state)
                    self.intentions['goal'] = 'X'
                else:
                    motion_goals = am.put_onion_in_pot_actions(
                        pot_states_dict)  # TODO : sélectionner le bon pot
                    self.intentions['goal'] = 'P'

            elif player_obj.name == 'tomato':
                # self.next_order.min_cost_to_complete == 10000 or self.next_order.min_cost_to_complete == 0 :
                if 'tomato' not in self.next_order_info["missing_ingredients_in_MA_pot"]:
                    motion_goals = am.place_obj_on_counter_actions(state)
                    self.intentions['goal'] = 'X'
                else:
                    motion_goals = am.put_tomato_in_pot_actions(
                        pot_states_dict)
                    self.intentions['goal'] = 'P'

            elif player_obj.name == 'dish':
                self.intentions['goal'] = 'P'
                motion_goals = am.pickup_soup_with_dish_actions(
                    pot_states_dict, only_nearly_ready=True)
                if motion_goals == []:
                   motion_goals = am.place_obj_on_counter_actions(state) 
                   self.intentions['goal'] = 'X'

            elif player_obj.name == 'soup':
                if player_obj.recipe not in state.all_orders :
                    motion_goals = am.place_obj_on_counter_actions(state)
                    self.intentions['goal'] = 'X'
                else : 
                    self.intentions['goal'] = 'S'
                    motion_goals = am.deliver_soup_actions()

            else:
                raise ValueError()

        motion_goals = [mg for mg in motion_goals if self.mlam.motion_planner.is_valid_motion_start_goal_pair(
            player.pos_and_or, mg)]

        if len(motion_goals) == 0:
            if player.has_object():
                motion_goals = am.place_obj_on_counter_actions(state)
                self.intentions['goal'] = 'X'                
            else:
                motion_goals = am.go_to_closest_feature_actions(player)
            motion_goals = [mg for mg in motion_goals if self.mlam.motion_planner.is_valid_motion_start_goal_pair(
                player.pos_and_or, mg)]
            if len(motion_goals) ==0:
                motion_goals = am.go_to_closest_feature_actions(player)
                motion_goals = [mg for mg in motion_goals if self.mlam.motion_planner.is_valid_motion_start_goal_pair(
                player.pos_and_or, mg)]          
            
            assert len(motion_goals) != 0

        return motion_goals

    def hl_info(self, state):

        def missing_ingredients_in_pot(recipe, pos):
            """
            computes the difference between recipe's ingredients and ingredients already present in pot
            """
            missing_ingredients = list(recipe.ingredients)
            if state.all_objects_by_type["soup"]:
                for soup in state.all_objects_by_type["soup"]:
                    if soup.position == pos and not soup.is_cooking and not soup.is_ready:
                        for ingredient in soup.ingredients:
                            if ingredient in missing_ingredients:
                                missing_ingredients.remove(ingredient)
            return missing_ingredients

        def calculate_recipe_cost(recipe, pot_pos):
            costs_dict = deepcopy(self.mlam.motion_planner.costs_dict)
            delivery_locations = self.mdp.get_serving_locations()
            onion_locations = self.mlam.mdp.get_onion_dispenser_locations()
            tomato_locations = self.mlam.mdp.get_tomato_dispenser_locations()
            onion_delivery_cost = self.mlam.motion_planner.min_cost_between_features(onion_locations, delivery_locations, manhattan_if_fail=False)
            tomato_delivery_cost = self.mlam.motion_planner.min_cost_between_features(tomato_locations, delivery_locations, manhattan_if_fail=False)
            cost = 0
            missing_ingredients = missing_ingredients_in_pot(recipe, pot_pos)
            for index, ingredient in enumerate(missing_ingredients) :
                if index ==0 :
                    if ingredient == 'onion' :
                        cost = self.mlam.motion_planner.min_cost_to_feature(state.players[self.agent_index].pos_and_or, onion_locations) + costs_dict['onion-pot'] 
                    else : 
                        cost = self.mlam.motion_planner.min_cost_to_feature(state.players[self.agent_index].pos_and_or, tomato_locations)# + costs_dict['tomato-pot']
                else :
                    if ingredient == 'onion' :
                            cost+= costs_dict['onion-pot'] * 2
                    else :
                            pass
                            #cost+= costs_dict['tomato-pot'] * 2     
            return cost + costs_dict['pot-delivery'] + min([onion_delivery_cost, tomato_delivery_cost])

        def cost_to_complete(recipe, state):
            pot_locations = self.mlam.mdp.get_pot_locations().copy()
            missing_ingredients_in_pots = {}
            costs = {}
            for pos in pot_locations:
                if pos in state.objects.keys():
                    missing_ingredients_in_pots[pos] = missing_ingredients_in_pot(recipe, pos)

                    if len(missing_ingredients_in_pots[pos]) + len(state.objects[pos].ingredients) > recipe.MAX_NUM_INGREDIENTS : #test wether ingredients already in pot are compatible with order
                        costs[pos] = 10000 #arbitrary value allowing to drop onion on counter.
                    else:
                        costs[pos] = calculate_recipe_cost(recipe, pos)
                else:
                    missing_ingredients_in_pots[pos] = list(recipe.ingredients)
                    costs[pos] = calculate_recipe_cost(recipe, pos)
            min_cost_to_complete = min(costs.values())
            return  costs, min_cost_to_complete, missing_ingredients_in_pots

        def point_time_ratio(recipe, costs):
            pot_locations = self.mlam.mdp.get_pot_locations()
            if costs :
                point_time_ratio = 1#recipe.value*10/(min(costs.values()) + recipe.time)
                most_advanced_pot = min(costs, key=costs.get) #so the min is calculated on value rather than key
            else :
                point_time_ratio = -1
                most_advanced_pot = pot_locations[0]
            return point_time_ratio, most_advanced_pot
        
        
        all_recipes = {}
        costs = {}
        cooking_or_ready_soups = [sorted(soup.ingredients) for soup in filter(lambda soup: soup.is_cooking or soup.is_ready, state.all_objects_by_type['soup'])]
        for index, recipe in enumerate(state.all_orders) :
            try:
                assert recipe.value is not None
            except AssertionError:
                recipe.configure(self.mdp.recipe_config)
            if list(recipe.ingredients) in cooking_or_ready_soups:
                continue            
            costs, min_cost_to_complete, missing_ingredients_in_pots = cost_to_complete(recipe, state)
            ratio, most_advanced_pot = point_time_ratio(recipe, costs)
            all_recipes[recipe] = {
                "recipe" : recipe,
                "costs" : costs,
                "min_cost_to_complete" : min_cost_to_complete,
                "point_time_ratio" : ratio,
                "most_advanced_pot" : most_advanced_pot,
                "value" : recipe.value,
                "missing_ingredients_in_MA_pot" : missing_ingredients_in_pot(recipe, most_advanced_pot)
                }
            
            
        
        return all_recipes
        
        

class RationalAgent(PlanningAgent):
    def __init__(self, hl_boltzmann_rational=False, ll_boltzmann_rational=False, hl_temp=1, ll_temp=1, auto_unstuck=True):
        super().__init__(hl_boltzmann_rational, ll_boltzmann_rational, hl_temp, ll_temp, auto_unstuck)
        self.intentions["agent_name"] = "rational"

    def hl_action(self, state):
        all_recipes = self.hl_info(state)
        if len(all_recipes) == 0:
            return self.next_order_info
        cheapest = max(all_recipes, key= lambda key : all_recipes.get(key)["point_time_ratio"])  
         # the cheapest recipe is the one from the all_recipes dict based on point time ratio of value dict 
        #cheapest = max(filter(lambda recipe : sorted(recipe.ingredients) not in cooking_or_ready_soups, all_recipes), key="point_time_ratio")
        if cheapest != self.hl_goal :
            #cheapest.update_cost_to_complete(state, self.mlam, self.agent_index)
            #cheapest.update_point_time_ratio(self.mlam)
            self.hl_objective_switch += 1
            self.hl_goal =cheapest
        cheapest_info = {
            "recipe" : all_recipes[cheapest]["recipe"],
            "most_advanced_pot" : all_recipes[cheapest]["most_advanced_pot"],
            "missing_ingredients_in_MA_pot" : all_recipes[cheapest]["missing_ingredients_in_MA_pot"],
            "point_time_ratio" : all_recipes[cheapest]["point_time_ratio"],
            "min_cost_to_complete" : all_recipes[cheapest]["min_cost_to_complete"]
            }
        return cheapest_info

class GreedyAgent(PlanningAgent):
    def __init__(self, hl_boltzmann_rational=False, ll_boltzmann_rational=False, hl_temp=1, ll_temp=1, auto_unstuck=True):
        super().__init__(hl_boltzmann_rational, ll_boltzmann_rational, hl_temp, ll_temp, auto_unstuck)
        self.intentions["agent_name"] = "greedy"
    def hl_action(self, state):
        all_recipes = self.hl_info(state)
        if len(all_recipes) == 0:
            return self.next_order_info
        cheapest = max(all_recipes, key= lambda key : all_recipes.get(key)["value"])  
        #cheapest = max(filter(lambda recipe : sorted(recipe.ingredients) not in cooking_or_ready_soups, all_recipes), key="point_time_ratio")
        if cheapest != self.hl_goal :
            #cheapest.update_cost_to_complete(state, self.mlam, self.agent_index)
            #cheapest.update_point_time_ratio(self.mlam)
            self.hl_objective_switch += 1
            self.hl_goal =cheapest
        cheapest_info = {
            "recipe" : all_recipes[cheapest]["recipe"],
            "most_advanced_pot" : all_recipes[cheapest]["most_advanced_pot"],
            "missing_ingredients_in_MA_pot" : all_recipes[cheapest]["missing_ingredients_in_MA_pot"],
            "point_time_ratio" : all_recipes[cheapest]["point_time_ratio"],
            "min_cost_to_complete" : all_recipes[cheapest]["min_cost_to_complete"]
            }
        return cheapest_info

class LazyAgent(PlanningAgent):
    def __init__(self, hl_boltzmann_rational=False, ll_boltzmann_rational=False, hl_temp=1, ll_temp=1, auto_unstuck=True):
        super().__init__(hl_boltzmann_rational, ll_boltzmann_rational, hl_temp, ll_temp, auto_unstuck)
        self.intentions["agent_name"] = "lazy" 
    def hl_action(self, state):
        cooking_or_ready_soups = [sorted(soup.ingredients) for soup in filter(lambda soup: soup.is_cooking or soup.is_ready, state.all_objects_by_type['soup'])]
        for recipe in state.all_orders :
            recipe.update_cost_to_complete(state, self.mlam, self.agent_index)
            recipe.update_point_time_ratio(self.mlam)

        shortest = min(filter(lambda recipe : sorted(recipe.ingredients) not in cooking_or_ready_soups, state.all_orders), key=attrgetter("min_cost_to_complete"))
        if shortest != self.hl_goal :
            self.hl_objective_switch += 1
            self.hl_goal = shortest
        return shortest







class SampleAgent(Agent):
    """ Agent that samples action using the average action_probs across multiple agents
    """

    def __init__(self, agents):
        self.agents = agents

    def action(self, state):
        action_probs = np.zeros(Action.NUM_ACTIONS)
        for agent in self.agents:
            action_probs += agent.action(state)[1]["action_probs"]
        action_probs = action_probs/len(self.agents)
        return Action.sample(action_probs), {"action_probs": action_probs}
    """
    """
# Deprecated. Need to fix Heuristic to work with the new MDP to reactivate Planning
# class CoupledPlanningAgent(Agent):
#     """
#     An agent that uses a joint planner (mlp, a MediumLevelPlanner) to find near-optimal
#     plans. At each timestep the agent re-plans under the assumption that the other agent
#     is also a CoupledPlanningAgent, and then takes the first action in the plan.
#     """
#
#     def __init__(self, mlp, delivery_horizon=2, heuristic=None):
#         self.mlp = mlp
#         self.mlp.failures = 0
#         self.heuristic = heuristic if heuristic is not None else Heuristic(mlp.mp).simple_heuristic
#         self.delivery_horizon = delivery_horizon
#
#     def action(self, state):
#         try:
#             joint_action_plan = self.mlp.get_low_level_action_plan(state, self.heuristic, delivery_horizon=self.delivery_horizon, goal_info=True)
#         except TimeoutError:
#             print("COUPLED PLANNING FAILURE")
#             self.mlp.failures += 1
#             return Direction.ALL_DIRECTIONS[np.random.randint(4)]
#         return (joint_action_plan[0][self.agent_index], {}) if len(joint_action_plan) > 0 else (Action.STAY, {})
#
#
# class EmbeddedPlanningAgent(Agent):
#     """
#     An agent that uses A* search to find an optimal action based on a model of the other agent,
#     `other_agent`. This class approximates the other agent as being deterministic even though it
#     might be stochastic in order to perform the search.
#     """
#
#     def __init__(self, other_agent, mlp, env, delivery_horizon=2, logging_level=0):
#         """mlp is a MediumLevelPlanner"""
#         self.other_agent = other_agent
#         self.delivery_horizon = delivery_horizon
#         self.mlp = mlp
#         self.env = env
#         self.h_fn = Heuristic(mlp.mp).simple_heuristic
#         self.logging_level = logging_level
#
#     def action(self, state):
#         start_state = state.deepcopy()
#         order_list = start_state.order_list if start_state.order_list is not None else ["any", "any"]
#         start_state.order_list = order_list[:self.delivery_horizon]
#         other_agent_index = 1 - self.agent_index
#         initial_env_state = self.env.state
#         self.other_agent.env = self.env
#
#         expand_fn = lambda state: self.mlp.get_successor_states_fixed_other(state, self.other_agent, other_agent_index)
#         goal_fn = lambda state: len(state.order_list) == 0
#         heuristic_fn = lambda state: self.h_fn(state)
#
#         search_problem = SearchTree(start_state, goal_fn, expand_fn, heuristic_fn, max_iter_count=50000)
#
#         try:
#             ml_s_a_plan, cost = search_problem.A_star_graph_search(info=True)
#         except TimeoutError:
#             print("A* failed, taking random action")
#             idx = np.random.randint(5)
#             return Action.ALL_ACTIONS[idx]
#
#         # Check estimated cost of the plan equals
#         # the sum of the costs of each medium-level action
#         assert sum([len(item[0]) for item in ml_s_a_plan[1:]]) == cost
#
#         # In this case medium level actions are tuples of low level actions
#         # We just care about the first low level action of the first med level action
#         first_s_a = ml_s_a_plan[1]
#
#         # Print what the agent is expecting to happen
#         if self.logging_level >= 2:
#             self.env.state = start_state
#             for joint_a in first_s_a[0]:
#                 print(self.env)
#                 print(joint_a)
#                 self.env.step(joint_a)
#             print(self.env)
#             print("======The End======")
#
#         self.env.state = initial_env_state
#
#         first_joint_action = first_s_a[0][0]
#         if self.logging_level >= 1:
#             print("expected joint action", first_joint_action)
#         action = first_joint_action[self.agent_index]
#         return action, {}
#

# Deprecated. Due to Heuristic and MLP
# class CoupledPlanningPair(AgentPair):
#     """
#     Pair of identical coupled planning agents. Enables to search for optimal
#     action once rather than repeating computation to find action of second agent
#     """
#
#     def __init__(self, agent):
#         super().__init__(agent, agent, allow_duplicate_agents=True)
#
#     def joint_action(self, state):
#         # Reduce computation by half if both agents are coupled planning agents
#         joint_action_plan = self.a0.mlp.get_low_level_action_plan(state, self.a0.heuristic, delivery_horizon=self.a0.delivery_horizon, goal_info=True)
#
#         if len(joint_action_plan) == 0:
#             return ((Action.STAY, {}), (Action.STAY, {}))
#
#         joint_action_and_infos = [(a, {}) for a in joint_action_plan[0]]
#         return joint_action_and_infos
