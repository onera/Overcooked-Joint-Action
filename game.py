from abc import ABC, abstractmethod
from email.policy import default
from threading import Lock, Thread
from queue import Queue, LifoQueue, Empty, Full
from time import time
from overcooked_ai_py.mdp.overcooked_mdp import OvercookedGridworld
from overcooked_ai_py.mdp.overcooked_env import OvercookedEnv
from overcooked_ai_py.mdp.actions import Action, Direction
from overcooked_ai_py.planning.planners import MediumLevelActionManager, MotionPlanner, NO_COUNTERS_PARAMS, COUNTERS_MLG_PARAMS
from overcooked_ai_py.agents.agent import GreedyAgent, LazyAgent, RationalAgent, RandomAgent
import random
import os
import pickle
import json
from copy import deepcopy
from overcooked_ai_py.static import LAYOUTS_DIR
from time import gmtime, asctime

# Relative path to where all static pre-trained agents are stored on server
AGENT_DIR = None

# Maximum allowable game time (in seconds)
MAX_GAME_TIME = 1000


def _configure(max_game_time, agent_dir):
    global AGENT_DIR, MAX_GAME_TIME
    MAX_GAME_TIME = max_game_time
    AGENT_DIR = agent_dir


class Game(ABC):
    """
    Class representing a game object. Coordinates the simultaneous actions of arbitrary
    number of players. Override this base class in order to use. 

    Players can post actions to a `pending_actions` queue, and driver code can call `tick` to apply these actions.


    It should be noted that most operations in this class are not on their own thread safe. Thus, client code should
    acquire `self.lock` before making any modifications to the instance. 

    One important exception to the above rule is `enqueue_actions` which is thread safe out of the box
    """

    # Possible TODO: create a static list of IDs used by the class so far to verify id uniqueness
    # This would need to be serialized, however, which might cause too great a performance hit to
    # be worth it

    EMPTY = 'EMPTY'

    class Status:
        DONE = 'done'
        ACTIVE = 'active'
        RESET = 'reset'
        INACTIVE = 'inactive'
        ERROR = 'error'
        QPT = 'qpt'

    def __init__(self, *args, **kwargs):
        """
        players (list): List of IDs of players currently in the game
        spectators (set): Collection of IDs of players that are not allowed to enqueue actions but are currently watching the game
        id (int):   Unique identifier for this game
        pending_actions List[(Queue)]: Buffer of (player_id, action) pairs have submitted that haven't been commited yet
        lock (Lock):    Used to serialize updates to the game state
        is_active(bool): Whether the game is currently being played or not
        """
        self.players = []
        self.spectators = set()
        self.pending_actions = []
        self.id = kwargs.get('id', id(self))
        self.id = id(self)
        self.lock = Lock()
        self._is_active = False

    @abstractmethod
    def is_full(self):
        """
        Returns whether there is room for additional players to join or not
        """
        pass

    @abstractmethod
    def apply_action(self, player_idx, action):
        """
        Updates the game state by applying a single (player_idx, action) tuple. Subclasses should try to override this method
        if possible
        """
        pass

    @abstractmethod
    def is_finished(self):
        """
        Returns whether the game has concluded or not
        """
        pass

    def is_ready(self):
        """
        Returns whether the game can be started. Defaults to having enough players
        """
        return self.is_full()

    @property
    def is_active(self):
        """
        Whether the game is currently being played
        """
        return self._is_active

    @property
    def reset_timeout(self):
        """
        Number of milliseconds to pause game on reset
        """
        return 3000

    def apply_actions(self):
        """
        Updates the game state by applying each of the pending actions in the buffer. Is called by the tick method. Subclasses
        should override this method if joint actions are necessary. If actions can be serialized, overriding `apply_action` is 
        preferred
        """
        for i in range(len(self.players)):
            try:
                while True:
                    action = self.pending_actions[i].get(block=False)
                    self.apply_action(i, action)
            except Empty:
                pass

    def activate(self):
        """
        Activates the game to let server know real-time updates should start. Provides little functionality but useful as
        a check for debugging
        """
        self._is_active = True

    def deactivate(self):
        """
        Deactives the game such that subsequent calls to `tick` will be no-ops. Used to handle case where game ends but 
        there is still a buffer of client pings to handle
        """
        self._is_active = False

    def reset(self):
        """
        Restarts the game while keeping all active players by resetting game stats and temporarily disabling `tick`
        """
        if not self.is_active:
            raise ValueError("Inactive Games cannot be reset")
        if self.is_finished():
            return self.Status.DONE
        self.deactivate()
        self.activate()
        return self.Status.RESET

    def needs_reset(self):
        """
        Returns whether the game should be reset on the next call to `tick`
        """
        return False

    def tick(self):
        """
        Updates the game state by applying each of the pending actions. This is done so that players cannot directly modify
        the game state, offering an additional level of safety and thread security. 

        One can think of "enqueue_action" like calling "git add" and "tick" like calling "git commit"

        Subclasses should try to override `apply_actions` if possible. Only override this method if necessary
        """
        if not self.is_active:
            return self.Status.INACTIVE
        if self.needs_reset():
            self.reset()
            return self.Status.RESET

        self.apply_actions()
        return self.Status.DONE if self.is_finished() else self.Status.ACTIVE

    def enqueue_action(self, player_id, action):
        """
        Add (player_id, action) pair to the pending action queue, without modifying underlying game state

        Note: This function IS thread safe
        """
        if not self.is_active:
            # Could run into issues with is_active not being thread safe
            return
        if player_id not in self.players:
            # Only players actively in game are allowed to enqueue actions
            return
        try:
            player_idx = self.players.index(player_id)
            self.pending_actions[player_idx].put(action)
        except Full:
            pass

    def get_state(self):
        """
        Return a JSON compatible serialized state of the game. Note that this should be as minimalistic as possible
        as the size of the game state will be the most important factor in game performance. This is sent to the client
        every frame update.
        """
        return {"players": self.players}

    def to_json(self):
        """
        Return a JSON compatible serialized state of the game. Contains all information about the game, does not need to
        be minimalistic. This is sent to the client only once, upon game creation
        """
        return self.get_state()

    def is_empty(self):
        """
        Return whether it is safe to garbage collect this game instance
        """
        return not self.num_players

    def add_player(self, player_id, idx=None, buff_size=-1):
        """
        Add player_id to the game
        """
        if self.is_full():
            raise ValueError("Cannot add players to full game")
        if self.is_active:
            raise ValueError("Cannot add players to active games")
        if not idx and self.EMPTY in self.players:
            idx = self.players.index(self.EMPTY)
        elif not idx:
            idx = len(self.players)

        padding = max(0, idx - len(self.players) + 1)
        for _ in range(padding):
            self.players.append(self.EMPTY)
            self.pending_actions.append(self.EMPTY)

        self.players[idx] = player_id
        self.pending_actions[idx] = Queue(maxsize=buff_size)

    def add_spectator(self, spectator_id):
        """
        Add spectator_id to list of spectators for this game
        """
        if spectator_id in self.players:
            raise ValueError("Cannot spectate and play at same time")
        self.spectators.add(spectator_id)

    def remove_player(self, player_id):
        """
        Remove player_id from the game
        """
        try:
            idx = self.players.index(player_id)
            self.players[idx] = self.EMPTY
            self.pending_actions[idx] = self.EMPTY
        except ValueError:
            return False
        else:
            return True

    def remove_spectator(self, spectator_id):
        """
        Removes spectator_id if they are in list of spectators. Returns True if spectator successfully removed, False otherwise
        """
        try:
            self.spectators.remove(spectator_id)
        except ValueError:
            return False
        else:
            return True

    def clear_pending_actions(self):
        """
        Remove all queued actions for all players
        """
        for i, player in enumerate(self.players):
            if player != self.EMPTY:
                queue = self.pending_actions[i]
                queue.queue.clear()

    @property
    def num_players(self):
        return len([player for player in self.players if player != self.EMPTY])

    def get_data(self):
        """
        Return any game metadata to server driver. Really only relevant for Psiturk code
        """
        return {}


class DummyGame(Game):
    """
    Standin class used to test basic server logic
    """

    def __init__(self, **kwargs):
        super(DummyGame, self).__init__(**kwargs)
        self.counter = 0

    def is_full(self):
        return self.num_players == 2

    def apply_action(self, idx, action):
        pass

    def apply_actions(self):
        self.counter += 1

    def is_finished(self):
        return self.counter >= 100

    def get_state(self):
        state = super(DummyGame, self).get_state()
        state['count'] = self.counter
        return state


class DummyInteractiveGame(Game):
    """
    Standing class used to test interactive components of the server logic
    """

    def __init__(self, **kwargs):
        super(DummyInteractiveGame, self).__init__(**kwargs)
        self.max_players = int(kwargs.get('playerZero', 'human') == 'human') + int(
            kwargs.get('playerOne', 'human') == 'human')
        self.max_count = kwargs.get('max_count', 30)
        self.counter = 0
        self.counts = [0] * self.max_players

    def is_full(self):
        return self.num_players == self.max_players

    def is_finished(self):
        return max(self.counts) >= self.max_count

    def apply_action(self, player_idx, action):
        if action.upper() == Direction.NORTH:
            self.counts[player_idx] += 1
        if action.upper() == Direction.SOUTH:
            self.counts[player_idx] -= 1

    def apply_actions(self):
        super(DummyInteractiveGame, self).apply_actions()
        self.counter += 1

    def get_state(self):
        state = super(DummyInteractiveGame, self).get_state()
        state['count'] = self.counter
        for i in range(self.num_players):
            state['player_{}_count'.format(i)] = self.counts[i]
        return state


class OvercookedGame(Game):
    """
    Class for bridging the gap between Overcooked_Env and the Game interface

    Instance variable:
        - max_players (int): Maximum number of players that can be in the game at once
        - mdp (OvercookedGridworld): Controls the underlying Overcooked game logic
        - score (int): Current reward acheived by all players
        - max_time (int): Number of seconds the game should last
        - npc_policies (dict): Maps user_id to policy (Agent) for each AI player
        - npc_state_queues (dict): Mapping of NPC user_ids to LIFO queues for the policy to process
        - curr_tick (int): How many times the game server has called this instance's `tick` method
        - ticker_per_ai_action (int): How many frames should pass in between NPC policy forward passes. 
            Note that this is a lower bound; if the policy is computationally expensive the actual frames
            per forward pass can be higher
        - action_to_overcooked_action (dict): Maps action names returned by client to action names used by OvercookedGridworld
            Note that this is an instance variable and not a static variable for efficiency reasons
        - human_players (set(str)): Collection of all player IDs that correspond to humans
        - npc_players (set(str)): Collection of all player IDs that correspond to AI
        - randomized (boolean): Whether the order of the layouts should be randomized

    Methods:
        - npc_policy_consumer: Background process that asynchronously computes NPC policy forward passes. One thread
            spawned for each NPC
        - _curr_game_over: Determines whether the game on the current mdp has ended
    """

    def __init__(self, layouts=["cramped_room"], mdp_params={}, num_players=2, gameTime=30, playerZero='human',
                 playerOne='human', showPotential=False, randomized=False, **kwargs):
        super(OvercookedGame, self).__init__(**kwargs)
        self.show_potential = showPotential
        self.mdp_params = mdp_params
        self.layouts = layouts
        self.curr_trial_in_game = kwargs.get("curr_trial_in_game",-1)
        self.curr_layout = self.layouts[self.curr_trial_in_game]
        self.max_players = int(num_players)
        self.mdp = None
        self.mp = None
        self.score = 0
        self.phi = 0
        self.max_time = min(int(gameTime), MAX_GAME_TIME)
        self.action_to_overcooked_action = {
            "STAY": Action.STAY,
            "UP": Direction.NORTH,
            "DOWN": Direction.SOUTH,
            "LEFT": Direction.WEST,
            "RIGHT": Direction.EAST,
            "SPACE": Action.INTERACT
        }
        self.ticks_per_ai_action = 4
        self.curr_tick = 0
        self.human_players = set()
        self.npc_players = set()
        self.playerZero = playerZero
        self.playerOne = playerOne
        self.npc_policies = {}
        self.npc_state_queues = {}
        try:
            self.layouts_dir = kwargs["config"]['layouts_dir']
        except KeyError:
            self.layouts_dir = LAYOUTS_DIR

        if randomized:
            random.shuffle(self.layouts)

        if self.playerZero != 'human':
            self.planning_agent_id = self.playerZero + '_0'
            player_zero_id = self.playerZero + '_0'
            self.add_player(player_zero_id, idx=0, buff_size=1, is_human=False)
            self.npc_policies[player_zero_id] = self.get_policy(
                self.playerZero, idx=0)
            self.npc_state_queues[player_zero_id] = LifoQueue()

        if self.playerOne != 'human':
            self.planning_agent_id = self.playerOne + '_1'
            player_one_id = self.playerOne + '_1'
            self.add_player(player_one_id, idx=1, buff_size=1, is_human=False)
            self.npc_policies[player_one_id] = self.get_policy(
                self.playerOne, idx=1)
            self.npc_state_queues[player_one_id] = LifoQueue()

    def needs_player_renew(self):
        return None, False

    def _curr_game_over(self):
        return time() - self.start_time >= self.max_time

    def needs_reset(self):
        return self._curr_game_over() and not self.is_finished()

    def add_player(self, player_id, idx=None, buff_size=-1, is_human=True):
        super(OvercookedGame, self).add_player(
            player_id, idx=idx, buff_size=buff_size)
        if is_human:
            self.human_players.add(player_id)
        else:
            self.npc_players.add(player_id)

    def remove_player(self, player_id):
        removed = super(OvercookedGame, self).remove_player(player_id)
        if removed:
            if player_id in self.human_players:
                self.human_players.remove(player_id)
            elif player_id in self.npc_players:
                self.npc_players.remove(player_id)
            else:
                raise ValueError("Inconsistent state")

    def npc_policy_consumer(self, policy_id):
        queue = self.npc_state_queues[policy_id]
        policy = self.npc_policies[policy_id]
        while self._is_active:
            state = queue.get()
            npc_action, _ = policy.action(state)
            super(OvercookedGame, self).enqueue_action(policy_id, npc_action)

    def is_full(self):
        return self.num_players >= self.max_players

    def is_finished(self):
        val = self.curr_trial_in_game >= len(
            self.layouts) - 1 and self._curr_game_over()
        return val

    def is_empty(self):
        """
        Game is considered safe to scrap if there are no active players or if there are no humans (spectating or playing)
        """
        return super(OvercookedGame, self).is_empty() or not self.spectators and not self.human_players

    def is_ready(self):
        """
        Game is ready to be activated if there are a sufficient number of players and at least one human (spectator or player)
        """
        return super(OvercookedGame, self).is_ready() and not self.is_empty()

    def apply_action(self, player_id, action):
        pass

    def apply_actions(self):
        # Default joint action, as NPC policies and clients probably don't enqueue actions fast
        # enough to produce one at every tick
        joint_action = [Action.STAY] * len(self.players)

        # Synchronize individual player actions into a joint-action as required by overcooked logic
        for i in range(len(self.players)):
            try:
                joint_action[i] = self.pending_actions[i].get(block=False)
            except Empty:
                pass

        # Apply overcooked game logic to get state transition
        prev_state = self.state
        self.state, info = self.mdp.get_state_transition(
            prev_state, joint_action)
        if self.show_potential:
            self.phi = self.mdp.potential_function(
                prev_state, self.mp, gamma=0.99)

        # Send next state to all background consumers if needed
        if self.curr_tick % self.ticks_per_ai_action == 0:
            for npc_id in self.npc_policies:
                self.npc_state_queues[npc_id].put(self.state, block=False)

        # Update score based on soup deliveries that might have occured
        curr_reward = sum(info['sparse_reward_by_agent'])
        self.score += curr_reward

        # Return about the current transition
        return prev_state, joint_action, info

    def enqueue_action(self, player_id, action):
        overcooked_action = self.action_to_overcooked_action[action]
        super(OvercookedGame, self).enqueue_action(
            player_id, overcooked_action)

    def reset(self):
        status = super(OvercookedGame, self).reset()
        if status == self.Status.RESET:
            # Hacky way of making sure game timer doesn't "start" until after reset timeout has passed
            self.start_time += self.reset_timeout / 1000

    def tick(self):
        if self.curr_tick == 0:
            self.start_time = time()
        self.curr_tick += 1
        return super(OvercookedGame, self).tick()

    def activate(self):
        self.curr_trial_in_game += 1
        self.curr_layout = self.layouts[self.curr_trial_in_game]
        self.mdp = OvercookedGridworld.from_layout_name(
            self.curr_layout, self.layouts_dir, **self.mdp_params)
        player_to_renew, needs_player_renew = self.needs_player_renew()
        if needs_player_renew:
            self.remove_player(player_to_renew)
            self.npc_policies = {}
            self.npc_state_queues = {}
            if self.playerZero != 'human':
                self.planning_agent_id = self.playerZero + '_0'
                player_zero_id = self.playerZero + '_0'
                self.add_player(player_zero_id, idx=0,
                                buff_size=1, is_human=False)
                self.npc_policies[player_zero_id] = self.get_policy(
                    self.playerZero, idx=0)
                self.npc_state_queues[player_zero_id] = LifoQueue()

            if self.playerOne != 'human':
                self.planning_agent_id = self.playerOne + '_1'
                player_one_id = self.playerOne + '_1'
                self.add_player(player_one_id, idx=1,
                                buff_size=1, is_human=False)
                self.npc_policies[player_one_id] = self.get_policy(
                    self.playerOne, idx=1)
                self.npc_state_queues[player_one_id] = LifoQueue()

        # Sanity check at start of each game
        if not self.npc_players.union(self.human_players) == set(self.players):
            raise ValueError("Inconsistent State")

        if self.show_potential:
            self.mp = MotionPlanner.from_pickle_or_compute(
                self.mdp, counter_goals=self.mdp.counter_goals)
        self.state = self.mdp.get_standard_start_state()
        if self.show_potential:
            self.phi = self.mdp.potential_function(
                self.state, self.mp, gamma=0.99)
        self.curr_tick = 0
        self.score = 0
        self.threads = []
        super(OvercookedGame, self).activate()
        for npc_policy in self.npc_policies:
            self.npc_policies[npc_policy].reset()
            self.npc_policies[npc_policy].set_mdp(self.mdp)
            self.npc_state_queues[npc_policy] = LifoQueue()
            self.npc_state_queues[npc_policy].put(self.state)
            t = Thread(target=self.npc_policy_consumer, args=(npc_policy,))
            self.threads.append(t)
            t.start()
        self.start_time = time()

    def deactivate(self):
        super(OvercookedGame, self).deactivate()
        # Ensure the background consumers do not hang
        for npc_policy in self.npc_policies:
            self.npc_state_queues[npc_policy].put(self.state)

        # Wait for all background threads to exit
        for t in self.threads:
            t.join()

        # Clear all action queues
        self.clear_pending_actions()

    def get_state(self):
        state_dict = {}
        state_dict['potential'] = self.phi if self.show_potential else None
        state_dict['state'] = self.state.to_dict()
        state_dict['score'] = self.score
        state_dict['time_left'] = max(
            self.max_time - (time() - self.start_time), 0)
        return state_dict

    def to_json(self):
        obj_dict = {}
        obj_dict['counter_goals'] = self.mdp.counter_goals
        obj_dict['terrain'] = self.mdp.terrain_mtx if self._is_active else None
        obj_dict['state'] = self.get_state() if self._is_active else None
        return obj_dict

    def get_policy(self, npc_id, idx=0):
        # if npc_id.lower().startswith("rllib"):
        #     try:
        #         # Loading rllib agents requires additional helpers
        #         fpath = os.path.join(AGENT_DIR, npc_id, 'agent', 'agent')
        #         agent = load_agent(fpath, agent_index=idx)
        #         return agent
        #     except Exception as e:
        #         raise IOError(
        #             "Error loading Rllib Agent\n{}".format(e.__repr__()))
        #     finally:
        #         # Always kill ray after loading agent, otherwise, ray will crash once process exits
        #         if ray.is_initialized():
        #             ray.shutdown()
        #else:
        try:
            fpath = os.path.join(AGENT_DIR, npc_id, 'agent.pickle')
            with open(fpath, 'rb') as f:
                return pickle.load(f)
        except Exception as e:
            raise IOError("Error loading agent\n{}".format(e.__repr__()))


class PlanningGame(OvercookedGame):
    """

    """

    def __init__(self, mdp_params={}, *args,  **kwargs):
        self.step = kwargs.get("step", -1)
        kwargs.get("config").pop("completion_link", None)
        self.config = kwargs.get("config")
        self.shuffle_trials = bool(self.config.get("shuffle_trials", False))
        self.layouts = self.config.get("blocs")[str(self.step)]
        self.curr_condition = self.config.get("conditions")[str(self.step)]
        self.participant_uid = kwargs.get('player_uid', '-1')
        self.mechanic = self.config.get('mechanic', 'time')
        self.qpt = self.config.get('qpt', {})
        self.qpt_length = self.config.get('qpt_length', 5)
        self.data = []
        self.mdp_params = mdp_params
        self.trajectory = []
        self.human_action_count = 0
        self.agent_action_count = 0
        self.human_interact_count = 0
        self.agent_interact_count = 0
        self.human_counter_share = 0
        self.infos = []
        kwargs.update(
            {"playerZero": self.config["agent"], "gameTime": self.config["gameTime"]})
        super(PlanningGame, self).__init__(
            mdp_params=mdp_params, layouts=self.layouts, *args, **kwargs)

    def is_finished(self):
        val = self.curr_trial_in_game >= len(
            self.layouts) - 1 and self._curr_game_over()
        return val
  
    def _curr_game_over(self):
        if self.mechanic == "recipe":
            return len(self.state.all_orders) == 0 or time() - self.start_time >= self.max_time
        else :
            return time() - self.start_time >= self.max_time

    def set_trial_id_error(self):
        self.trial_id = self.participant_uid + '_' + \
            str(self.step) + 'ERROR' + self.curr_layout[-1]

    def activate(self):
        """
        Resets trial ID at start of new "game"
        """
        self.human_action_count = 0
        self.agent_action_count = 0
        self.human_interact_count = 0
        self.agent_interact_count = 0
        self.human_counter_share = 0
        self.infos = []
        super().activate()
        self.trial_id = self.participant_uid + '_' + \
            str(self.step) + "_" + str(self.curr_trial_in_game)

    def deactivate(self):
        try:
            self.data = self.get_data()
        except IndexError:
            pass
        super(PlanningGame, self).deactivate()

    def apply_actions(self):
        """
        Applies pending actions then logs transition data
        """
        # Apply MDP logic
        prev_state, joint_action, info = super(
            PlanningGame, self).apply_actions()
        self.infos.append(info['event_infos'])
        
        if joint_action[1] != (0, 0):
            self.human_action_count += 1
            if joint_action[1] == 'interact':
                self.human_interact_count += 1
        if joint_action[0] != (0, 0):
            self.agent_action_count += 1
            if joint_action[0] == 'interact':
                self.agent_interact_count += 1

        # Log data to send to psiturk client
        curr_reward = sum(info['sparse_reward_by_agent'])
        ach_orders = len(self.mdp.start_all_orders) - \
            len(self.state.all_orders)
        transition = {
            "joint_action": json.dumps(joint_action),
            "reward": curr_reward,
            "time_left": max(self.max_time - (time() - self.start_time), 0),
            "score": self.score,
            "time_elapsed": time() - self.start_time,
            "cur_gameloop": self.curr_tick,
            "layout": json.dumps(self.mdp.terrain_mtx),
            "layout_name": self.curr_layout,
            "trial_id": self.trial_id,
            "participant_uid": self.participant_uid,
            "player_0_id": self.players[0],
            "player_1_id": self.players[1],
            "player_0_is_human": self.players[0] in self.human_players,
            "player_1_is_human": self.players[1] in self.human_players,
            "all_orders": self.state.all_orders,
            "achieved_orders_len": ach_orders,
            "human_action_count": self.human_action_count,
            "agent_action_count": self.agent_action_count,
            "agent_stuck_loop": self.npc_policies[self.planning_agent_id].stuck_frames,
            "hl_switch": self.npc_policies[self.planning_agent_id].hl_objective_switch

        }
        transition.update(prev_state.to_dict())
        self.trajectory.append(transition)

    def get_policy(self, npc_id,  idx):
        #self.mdp = OvercookedGridworld.from_layout_name(self.layouts[-1], self.layouts_dir, **self.mdp_params)
        if "Lazy" in self.planning_agent_id:
            agent = LazyAgent()
        elif "Greedy" in self.planning_agent_id:
            agent = GreedyAgent()
        elif "Rational" in self.planning_agent_id:
            agent = RationalAgent()
        else:
            agent = RandomAgent()
        agent.set_agent_index(idx)
        return agent

    def get_intentions(self, policy_id):
        #queue = self.npc_state_queues[policy_id]
        policy = self.npc_policies[policy_id]
        return policy.intentions

    def get_motion_goal(self, policy_id):
        policy = self.npc_policies[policy_id]
        if policy.motion_goal:
            return policy.chosen_goal[0]

    def get_state(self):
        state_dict = {}
        state_dict['potential'] = self.phi if self.show_potential else None
        state_dict['state'] = self.state.to_dict()
        state_dict['score'] = self.score
        state_dict['time_left'] = max(
            self.max_time - (time() - self.start_time), 0)
        state_dict['intentions'] = self.get_intentions(self.planning_agent_id)
        state_dict['state']['players'][int(
            self.planning_agent_id[-1])]['motion_goal'] = self.get_motion_goal(self.planning_agent_id)
        state_dict['state']['players'][int(
            self.planning_agent_id[-1])]['intentions'] = self.get_intentions(self.planning_agent_id)
        return state_dict

    def get_data(self):
        """
        Returns and then clears the accumulated trajectory
        """
        info_sum = deepcopy(self.infos[-1])
        for key, value in info_sum.items():
            info_sum[key] = [0,0]
        for info in self.infos:
            for key, value in info.items():
                if value[0]:
                    info_sum[key][0] +=1
                if value[1]:
                    info_sum[key][1] +=1
        data = {"uid": self.participant_uid, "trial_id": self.trial_id, "layout": self.curr_layout, "time_elapsed": self.trajectory[-1]["time_elapsed"],
                "mechanic": self.mechanic,
                "timestamp": gmtime(), "date": asctime(gmtime()), "step": self.step,
                "condition": self.curr_condition, 
                "curr_trial_in_game": self.curr_trial_in_game, 
                "score": self.trajectory[-1]["score"],
                "info_sum" : info_sum,
                "human_action_count": self.trajectory[-1]["human_action_count"],
                "agent_action_count": self.trajectory[-1]["agent_action_count"],
                "agent_interact_count": self.agent_interact_count,
                "human_interact_count": self.human_interact_count,
                "agent_stuck_loop": self.trajectory[-1]["agent_stuck_loop"],
                "hl_switch": self.trajectory[-1]["hl_switch"],
                "achieved_orders_len": self.trajectory[-1]["achieved_orders_len"],
                "bloc": self.step,
                "trajectory": self.trajectory,
                "config" : self.config}

        self.trajectory = []
        return data


# class PlanningGame(OvercookedGame):
#
#     def __init__(self, layouts=["cramped_room"], **kwargs):
#         super(PlanningGame, self).__init__(layouts=layouts)
#         super().__init__(layouts, **kwargs)
#         self.mlam = MediumLevelActionManager.from_pickle_or_compute(self.mdp, NO_COUNTERS_PARAMS)
#
#     def get_policy(self, *args, **kwargs):
#         return GreedyHumanModel(self.mlam)


class OvercookedPsiturk(OvercookedGame):
    """
    Wrapper on OvercookedGame that handles additional housekeeping for Psiturk experiments

    Instance Variables:
        - trajectory (list(dict)): list of state-action pairs in current trajectory
        - psiturk_uid (string): Unique id for each psiturk game instance (provided by Psiturk backend)
            Note, this is not the user id -- two users in the same game will have the same psiturk_uid
        - trial_id (string): Unique identifier for each psiturk trial, updated on each call to reset
            Note, one OvercookedPsiturk game handles multiple layouts. This is how we differentiate

    Methods:
        get_data: Returns the accumulated trajectory data and clears the self.trajectory instance variable

    """

    def __init__(self, *args, psiturk_uid='-1', **kwargs):
        super(OvercookedPsiturk, self).__init__(
            *args, showPotential=False, **kwargs)
        self.psiturk_uid = psiturk_uid
        self.trajectory = []

    def activate(self):
        """
        Resets trial ID at start of new "game"
        """
        super(OvercookedPsiturk, self).activate()
        self.trial_id = self.psiturk_uid + str(self.start_time)

    def apply_actions(self):
        """
        Applies pending actions then logs transition data
        """
        # Apply MDP logic
        prev_state, joint_action, info = super(
            OvercookedPsiturk, self).apply_actions()

        # Log data to send to psiturk client
        curr_reward = sum(info['sparse_reward_by_agent'])
        transition = {
            "state": json.dumps(prev_state.to_dict()),
            "joint_action": json.dumps(joint_action),
            "reward": curr_reward,
            "time_left": max(self.max_time - (time() - self.start_time), 0),
            "score": self.score,
            "time_elapsed": time() - self.start_time,
            "cur_gameloop": self.curr_tick,
            "layout": json.dumps(self.mdp.terrain_mtx),
            "layout_name": self.curr_layout,
            "trial_id": self.trial_id,
            "player_0_id": self.players[0],
            "player_1_id": self.players[1],
            "player_0_is_human": self.players[0] in self.human_players,
            "player_1_is_human": self.players[1] in self.human_players
        }

        self.trajectory.append(transition)

    def get_data(self):
        """
        Returns and then clears the accumulated trajectory
        """
        data = {"uid": self.psiturk_uid + "_" +
                str(time()), "trajectory": self.trajectory}
        self.trajectory = []
        return data


class OvercookedTutorial(OvercookedGame):
    """
    Wrapper on OvercookedGame that includes additional data for tutorial mechanics, most notably the introduction of tutorial "phases"

    Instance Variables:
        - curr_phase (int): Indicates what tutorial phase we are currently on
        - phase_two_score (float): The exact sparse reward the user must obtain to advance past phase 2
    """

    def __init__(self, layouts=["tutorial_2", "tutorial_1", "tutorial_0"], mdp_params={}, playerZero='human', playerOne='AI', phaseTwoScore=15,
                 **kwargs):
        super(OvercookedTutorial, self).__init__(layouts=layouts, mdp_params=mdp_params, playerZero=playerZero,
                                                 playerOne=playerOne, showPotential=False, **kwargs)
        self.phase_two_score = phaseTwoScore
        self.phase_two_finished = False
        self.config = kwargs.get("config")
        self.max_time = 0
        self.max_players = 2
        self.ticks_per_ai_action = 8
        self.curr_phase = 0
        self.participant_uid = kwargs.get('player_uid', '-1')
        self.trial_id = "tutorial" + str(self.curr_phase)
        self.data = []
        self.trajectory = []

    @property
    def reset_timeout(self):
        return 1

    def needs_reset(self):
        if self.curr_phase == 0:
            return self.score > 0
        elif self.curr_phase == 1:
            return self.score > 0
        elif self.curr_phase == 2:
            return self.phase_two_finished
        return False

    def is_finished(self):
        # = float('inf')
        return self.curr_trial_in_game >= len(self.layouts) - 1 and self.score > 0

    def reset(self):
        self.curr_phase += 1
        self.data = self.get_data()
        super(OvercookedTutorial, self).reset()

    def activate(self):
        self.trial_id = "tutorial" + str(self.curr_phase)
        super(OvercookedTutorial, self).activate()

    def deactivate(self):
        super(OvercookedTutorial, self).deactivate()

    def get_policy(self, *args, **kwargs):
        return TutorialAI()

    def apply_actions(self):
        """
        Apply regular MDP logic with retroactive score adjustment tutorial purposes
        """
        prev_state, joint_action, info = super(
            OvercookedTutorial, self).apply_actions()

        human_reward, ai_reward = info['sparse_reward_by_agent']

        # We only want to keep track of the human's score in the tutorial
        self.score -= ai_reward

        # Phase two requires a specific reward to complete
        if self.curr_phase == 2:
            #self.score = 0
            if human_reward == self.phase_two_score:
                self.phase_two_finished = True
        transition = {
            "joint_action": json.dumps(joint_action),
            "time_left": max(self.max_time - (time() - self.start_time), 0),
            "score": self.score,
            "time_elapsed": time() - self.start_time,
            "cur_gameloop": self.curr_tick,
            "layout": json.dumps(self.mdp.terrain_mtx),
            "layout_name": self.curr_layout,
            "trial_id": self.trial_id,
            "participant_uid": self.participant_uid,
            "player_0_id": self.players[0],
            "player_1_id": self.players[1],
            "player_0_is_human": self.players[0] in self.human_players,
            "player_1_is_human": self.players[1] in self.human_players,
            "all_orders": self.state.all_orders
        }
        transition.update(prev_state.to_dict())
        self.trajectory.append(transition)

    def get_data(self):
        """
        Returns and then clears the accumulated trajectory
        """
        data = {"uid": self.participant_uid, "trial_id": self.trial_id,
                "timestamp": gmtime(), "date": asctime(gmtime()),
                "score": self.trajectory[-1]["score"],
                "trajectory": self.trajectory,
                "config" : self.config}
        self.trajectory = []
        return data


class DummyOvercookedGame(OvercookedGame):
    """
    Class that hardcodes the AI to be random. Used for debugging
    """

    def __init__(self, layouts=["cramped_room"], **kwargs):
        super(DummyOvercookedGame, self).__init__(layouts, **kwargs)

    def get_policy(self, *args, **kwargs):
        return DummyAI()


class DummyAI():
    """
    Randomly samples actions. Used for debugging
    """

    def action(self, state):
        [action] = random.sample(
            [Action.STAY, Direction.NORTH, Direction.SOUTH, Direction.WEST, Direction.EAST, Action.INTERACT], 1)
        return action, None

    def reset(self):
        pass

    def set_mdp(self, mdp):
        pass


class DummyComputeAI(DummyAI):
    """
    Performs simulated compute before randomly sampling actions. Used for debugging
    """

    def __init__(self, compute_unit_iters=1e5):
        """
        compute_unit_iters (int): Number of for loop cycles in one "unit" of compute. Number of 
                                    units performed each time is randomly sampled
        """
        super(DummyComputeAI, self).__init__()
        self.compute_unit_iters = int(compute_unit_iters)

    def action(self, state):
        # Randomly sample amount of time to busy wait
        iters = random.randint(1, 10) * self.compute_unit_iters

        # Actually compute something (can't sleep) to avoid scheduling optimizations
        val = 0
        for i in range(iters):
            # Avoid branch prediction optimizations
            if i % 2 == 0:
                val += 1
            else:
                val += 2

        # Return randomly sampled action
        return super(DummyComputeAI, self).action(state)


class StayAI():
    """
    Always returns "stay" action. Used for debugging
    """

    def action(self, state):
        return Action.STAY, None

    def reset(self):
        pass


class TutorialAI():
    COOK_SOUP_LOOP = [
        # Grab first onion
        Direction.WEST,
        Direction.WEST,
        Direction.WEST,
        Action.INTERACT,

        # Place onion in pot
        Direction.EAST,
        Direction.NORTH,
        Action.INTERACT,

        # Grab second onion
        Direction.WEST,
        Action.INTERACT,

        # Place onion in pot
        Direction.EAST,
        Direction.NORTH,
        Action.INTERACT,

        # Grab third onion
        Direction.WEST,
        Action.INTERACT,

        # Place onion in pot
        Direction.EAST,
        Direction.NORTH,
        Action.INTERACT,

        # Cook soup
        Action.INTERACT,

        # Grab plate
        Direction.EAST,
        Direction.SOUTH,
        Action.INTERACT,
        Direction.WEST,
        Direction.NORTH,

        # Deliver soup
        Action.INTERACT,
        Direction.EAST,
        Direction.EAST,
        Direction.EAST,
        Action.INTERACT,
        Direction.WEST
    ]

    COOK_SOUP_COOP_LOOP = [
        # Grab first onion
        Direction.WEST,
        Direction.WEST,
        Direction.WEST,
        Action.INTERACT,

        # Place onion in pot
        Direction.EAST,
        Direction.SOUTH,
        Action.INTERACT,

        # Move to start so this loops
        Direction.EAST,
        Direction.EAST,

        # Pause to make cooperation more real time
        Action.STAY,
        Action.STAY,
        Action.STAY,
        Action.STAY,
        Action.STAY,
        Action.STAY,
        Action.STAY,
        Action.STAY,
        Action.STAY
    ]

    def __init__(self, *args, **kwargs):
        self.curr_phase = -1
        self.curr_tick = -1

    def action(self, state):
        self.curr_tick += 1
        if self.curr_phase == 0:
            return self.COOK_SOUP_LOOP[self.curr_tick % len(self.COOK_SOUP_LOOP)], None
        elif self.curr_phase == 2:
            return self.COOK_SOUP_COOP_LOOP[self.curr_tick % len(self.COOK_SOUP_COOP_LOOP)], None
        return Action.STAY, None

    def set_mdp(self, mdp):
        pass

    def reset(self):
        self.curr_tick = -1
        self.curr_phase += 1
