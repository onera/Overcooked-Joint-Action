import os
from pathlib import Path
from socket import socket

# Import and patch the production eventlet server if necessary

import eventlet

eventlet.monkey_patch()


# All other imports must come after patch to ensure eventlet compatibility
import random
import pickle
import queue
import atexit
from socketio.exceptions import TimeoutError as SocketIOTimeOutError
import json
import logging
import glob
from time import gmtime, asctime, sleep, time
from threading import Lock
from utils import ThreadSafeSet, ThreadSafeDict, questionnaire_to_surveyjs
from flask import Flask, render_template, jsonify, request, session
from flask_socketio import SocketIO, join_room, leave_room, emit
from flask_session import Session
from flask_login import LoginManager, UserMixin, login_user, logout_user, current_user, login_required
from flask_sqlalchemy import SQLAlchemy
from sqlalchemy import JSON
from game import OvercookedGame, OvercookedTutorial, Game, OvercookedPsiturk, PlanningGame
import game

# Thoughts -- where I'll log potential issues/ideas as they come up
# Should make game driver code more error robust -- if overcooked randomlly errors we should catch it and report it to user
# Right now, if one user 'join's before other user's 'join' finishes, they won't end up in same game
# Could use a monitor on a conditional to block all global ops during calls to _ensure_consistent_state for debugging
# Could cap number of sinlge- and multi-player games separately since the latter has much higher RAM and CPU usage

###########
# Globals #
###########

# Read in global config
CONF_PATH = os.getenv('CONF_PATH', 'config.json')
TRIALS_PATH = os.getenv('CONF_PATH', 'trials.json')
with open(CONF_PATH, 'r') as f:
    CONFIG = json.load(f)

# Where errors will be logged
LOGFILE = CONFIG['logfile']

# Available layout names
LAYOUTS = CONFIG['layouts']


# Values that are standard across layouts
LAYOUT_GLOBALS = CONFIG['layout_globals']

# Maximum allowable game length (in seconds)
MAX_GAME_LENGTH = CONFIG['MAX_GAME_LENGTH']

# Path to where pre-trained agents will be stored on server
AGENT_DIR = CONFIG['AGENT_DIR']

# Maximum number of games that can run concurrently. Contrained by available memory and CPU
MAX_GAMES = CONFIG['MAX_GAMES']

# Frames per second cap for serving to client
MAX_FPS = CONFIG['MAX_FPS']



# Default configuration for planning experiment design
PLANNING_DESIGN_CONFIG = CONFIG['planning_design']

# Default configuration for tutorial
TUTORIAL_CONFIG = json.dumps(CONFIG['tutorial'])

# Global queue of available IDs. This is how we synch game creation and keep track of how many games are in memory
#FREE_IDS = queue.Queue(maxsize=MAX_GAMES)

# Bitmap that indicates whether ID is currently in use. Game with ID=i is "freed" by setting FREE_MAP[i] = True
#FREE_MAP = ThreadSafeDict()

# Initialize our ID tracking data
#for i in range(MAX_GAMES):
 #   FREE_IDS.put(i)
  #  FREE_MAP[i] = True

# Mapping of game-id to game objects
GAMES = ThreadSafeDict()

# Set of games IDs that are currently being played
ACTIVE_GAMES = ThreadSafeSet()

# Queue of games IDs that are waiting for additional players to join. Note that some of these IDs might
# be stale (i.e. if FREE_MAP[id] = True)
#WAITING_GAMES = queue.Queue()

# Mapping of users to locks associated with the ID. Enforces user-level serialization
USERS = ThreadSafeDict()


# Mapping of user id's to the current game (room) they are in
USER_ROOMS = ThreadSafeDict()

# Mapping of string game names to corresponding classes
GAME_NAME_TO_CLS = {
    "overcooked": OvercookedGame,
    "tutorial": OvercookedTutorial,
    "psiturk": OvercookedPsiturk,
    "planning": PlanningGame
}

game._configure(MAX_GAME_LENGTH, AGENT_DIR)

#######################
# Flask Configuration #
#######################
# Create and configure flask app
app = Flask(__name__, template_folder=os.path.join('static', 'templates'))
app.config['DEBUG'] = os.getenv('FLASK_ENV', 'production') == 'development'
app.config['SECRET_KEY'] = 'c-\x9f^\x80\xd8\xd0j\xed\xc1\x15\xf7\xc9\x97J{\x97\x165Iq#\x87\x88'
app.config['SESSION_COOKIE_HTTPONLY'] = False
app.config['SESSION_COOKIE_SAMESITE'] = "Lax"
app.config['SESSION_COOKIE_SECURE'] = True
app.config['SESSION_TYPE'] = 'filesystem'
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///db.sqlite'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
#app.config.update(SECRET_KEY='osd(99092=36&462134kjKDhuIS_d23', ENV='development')
socketio = SocketIO(app, cors_allowed_origins="*", logger=app.config['DEBUG'], ping_interval=5, ping_timeout=5)
login_manager = LoginManager()
login_manager.init_app(app)
db = SQLAlchemy()
db.init_app(app)
# Attach handler for logging errors to file
handler = logging.FileHandler(LOGFILE)
handler.setLevel(logging.ERROR)
app.logger.addHandler(handler)


class User(UserMixin, db.Model):

    __tablename__ = 'user'
    uid = db.Column(db.String, primary_key=True)
    config = db.Column(JSON)
    step = db.Column(db.Integer)
    trial = db.Column(db.Integer)

    def get_id(self):
        return str(self.uid)


with app.app_context():
    db.create_all()


@login_manager.user_loader
def load_user(user_id):
    return User.query.get(user_id)


#################################
# Global Coordination Functions #
#################################

def try_create_game(game_name, **kwargs):
    """
    Tries to create a brand new Game object based on parameters in `kwargs`

    Returns (Game, Error) that represent a pointer to a game object, and error that occured
    during creation, if any. In case of error, `Game` returned in None. In case of sucess,
    `Error` returned is None

    Possible Errors:
        - Runtime error if server is at max game capacity
        - Propogate any error that occured in game __init__ function
    """
    try:
        #curr_id = FREE_IDS.get(block=False)
        #assert FREE_MAP[curr_id], "Current id is already in use"
        game_cls = GAME_NAME_TO_CLS.get(game_name, OvercookedGame)
        if game_cls == OvercookedTutorial:
            kwargs["config"]["layouts_dir"] = "overcooked_ai_py/data/layouts"
        game = game_cls(**kwargs)
    #except queue.Empty:
    #    err = RuntimeError("Server at max capacity")
    #    return None, err
    except Exception as e:
        return None, e
    else:
        GAMES[game.id] = game
        #FREE_MAP[game.id] = False
        return game, None


def cleanup_game(game):
    #if FREE_MAP[game.id]:
     #   raise ValueError("Double free on a game")

    # User tracking
    for user_id in game.players:
        leave_curr_room(user_id)

    # Socketio tracking
    socketio.close_room(game.id)

    # Game tracking
    #FREE_MAP[game.id] = True
    #FREE_IDS.put(game.id)
    del GAMES[game.id]

    if game.id in ACTIVE_GAMES:
        ACTIVE_GAMES.remove(game.id)


def get_game(game_id):
    return GAMES.get(game_id, None)


def get_curr_game(user_id):
    return get_game(get_curr_room(user_id))


def get_curr_room(user_id):
    return USER_ROOMS.get(user_id, None)


def set_curr_room(user_id, room_id):
    USER_ROOMS[user_id] = room_id


def leave_curr_room(user_id):
    del USER_ROOMS[user_id]


# def get_waiting_game():
#     """
#     Return a pointer to a waiting game, if one exists

#     Note: The use of a queue ensures that no two threads will ever receive the same pointer, unless
#     the waiting game's ID is re-added to the WAITING_GAMES queue
#     """
#     try:
#         waiting_id = WAITING_GAMES.get(block=False)
#         while FREE_MAP[waiting_id]:
#             waiting_id = WAITING_GAMES.get(block=False)
#     except queue.Empty:
#         return None
#     else:
#         return get_game(waiting_id)


##########################
# Socket Handler Helpers #
##########################

def _leave_game(user_id):
    """
    Removes `user_id` from it's current game, if it exists. Rebroadcast updated game state to all
    other users in the relevant game.

    Leaving an active game force-ends the game for all other users, if they exist

    Leaving a waiting game causes the garbage collection of game memory, if no other users are in the
    game after `user_id` is removed
    """
    # Get pointer to current game if it exists
    game = get_curr_game(user_id)

    if not game:
        # Cannot leave a game if not currently in one
        return False

    # Acquire this game's lock to ensure all global state updates are atomic
    with game.lock:
        # Update socket state maintained by socketio
        leave_room(game.id)

        # Update user data maintained by this app
        leave_curr_room(user_id)

        # Update game state maintained by game object
        if user_id in game.players:
            game.remove_player(user_id)
        else:
            game.remove_spectator(user_id)

        # Whether the game was active before the user left
        was_active = game.id in ACTIVE_GAMES

        # Rebroadcast data and handle cleanup based on the transition caused by leaving
        if was_active and game.is_empty():
            # Active -> Empty
            game.deactivate()
        elif game.is_empty():
            # Waiting -> Empty
            cleanup_game(game)
        elif not was_active:
            # Waiting -> Waiting
            emit('waiting', {"in_game": True}, room=game.id)
        elif was_active and game.is_ready():
            # Active -> Active
            pass
        elif was_active and not game.is_empty():
            # Active -> Waiting
            game.deactivate()

    return was_active


def _create_game(user_id, game_name, params={}):
    existing_game = GAMES.get(game_name, None)
    if existing_game:
        cleanup_game(existing_game)
    game, err = try_create_game(game_name, **params)
    if not game:
        emit("creation_failed", {"error": err.__repr__()}, to=current_user.uid)
        print("error:" + (err.__repr__()))
        return
    spectating = True
    with game.lock:
        if not game.is_full():
            spectating = False
            game.add_player(user_id)
        else:
            spectating = True
            game.add_spectator(user_id)
        socketio.close_room(game.id) #ensure the same client is not in the same room with two sids after connect/disconnect . Will need to be changed in case of multiplayer games
        join_room(game.id)
        set_curr_room(user_id, game.id)
        game.activate()
        ACTIVE_GAMES.add(game.id)

        emit('start_game', {"spectating": spectating,
                "start_info": game.to_json(), "trial": current_user.trial, "step": current_user.step, "config": game.config}, room=game.id)
        socketio.start_background_task(play_game, game, fps=current_user.config.get("fps",MAX_FPS))
        # else:
        #     WAITING_GAMES.put(game.id)
        #     emit('waiting', {"in_game": True}, room=game.id)


#####################
# Debugging Helpers #
#####################

def _ensure_consistent_state():
    """
    Simple sanity checks of invariants on global state data

    Let ACTIVE be the set of all active game IDs, GAMES be the set of all existing
    game IDs, and WAITING be the set of all waiting (non-stale) game IDs. Note that
    a game could be in the WAITING_GAMES queue but no longer exist (indicated by
    the FREE_MAP)

    - Intersection of WAITING and ACTIVE games must be empty set
    - Union of WAITING and ACTIVE must be equal to GAMES
    - id \in FREE_IDS => FREE_MAP[id]
    - id \in ACTIVE_GAMES => Game in active state
    - id \in WAITING_GAMES => Game in inactive state
    """
    #waiting_games = set()
    active_games = set()
    all_games = set(GAMES)

    # for game_id in list(FREE_IDS.queue):
    #     assert FREE_MAP[game_id], "Freemap in inconsistent state"

    # for game_id in list(WAITING_GAMES.queue):
    #     if not FREE_MAP[game_id]:
    #         waiting_games.add(game_id)

    for game_id in ACTIVE_GAMES:
        active_games.add(game_id)

    # assert waiting_games.union(
    #     active_games) == all_games, "WAITING union ACTIVE != ALL"

    # assert not waiting_games.intersection(
    #     active_games), "WAITING intersect ACTIVE != EMPTY"

    assert all([get_game(g_id)._is_active for g_id in active_games]
               ), "Active ID in waiting state"
    # assert all([not get_game(g_id)._id_active for g_id in waiting_games]
    #            ), "Waiting ID in active state"


def get_agent_names():
    return [d for d in os.listdir(AGENT_DIR) if os.path.isdir(os.path.join(AGENT_DIR, d))]


######################
# Application routes #
######################

# Hitting each of these endpoints creates a brand new socket that is closed
# at after the server response is received. Standard HTTP protocol

@app.route('/')
def index():
    uid = request.args.get('PROLIFIC_PID', default=None)
    user_sid = "None"
    try:
        config_id = request.args.get('CONFIG', default=None)
        config = CONFIG[config_id]
        config["config_id"] = config_id
        for bloc, value in config["conditions"].items():
            if value == "U":
                config["conditions"][bloc]={
            "recipe_head": False,
            "recipe_hud" : False,
            "asset_hud" : False,
            "motion_goal" : False
            }
            elif value =="E":
                config["conditions"][bloc]={
            "recipe_head": True,
            "recipe_hud" : True,
            "asset_hud" : True,
            "motion_goal" : True
            }

    except KeyError:
        return render_template('UID_error.html')

    if uid:
        session["type"] = "PROLIFIC"
    else:
        uid = request.args.get('TEST_UID', default=None)
        session["type"] = "TEST"
    if uid:
        user = User.query.filter_by(uid=uid).first()
        if user:
            login_user(user)
        else:
            new_user = User(uid=uid, config=config, step=0, trial=0)
            try:
                if os.path.exists("./questionnaires/post_trial/" + new_user.config["questionnaire_post_trial"]):
                    with open("./questionnaires/post_trial/" + new_user.config["questionnaire_post_trial"], 'r', encoding='utf-8') as f:
                        qpt = json.load(f)
                    f.close()
                    new_user.config["qpt"] = qpt
            except KeyError:
                new_user.config["qpt"] = {}
            if new_user.config.get("shuffle_trials", False) == True:
                for key, value in new_user.config["blocs"].items():
                    random.shuffle(value)
            try:
                if os.path.exists("./questionnaires/post_bloc/" + new_user.config["questionnaire_post_bloc"]):
                    with open("./questionnaires/post_bloc/" + new_user.config["questionnaire_post_bloc"], 'r', encoding='utf-8') as f:
                        qpb = json.load(f)
                    f.close()
                    new_user.config["qpb"] = qpb
            except KeyError:
                new_user.config["qpb"] = {}
                for key, value in new_user.config["blocs"].items():
                    random.shuffle(value)
            db.session.add(new_user)
            db.session.commit()
            login_user(new_user)
        return render_template('index.html', uid=uid, layout_conf=LAYOUT_GLOBALS)
    else:
        return render_template('UID_error.html')


@app.route('/instructions', methods=['GET', 'POST'])
@login_required
def instructions():
    uid = current_user.uid
    condition = current_user.config["conditions"]
    is_explained = False
    all_conditions = [item for sublist in [list(bloc.values()) for bloc in condition.values()] for item in sublist] #test wheter at least 1 intention is given at some point
    if any(all_conditions):
        is_explained = True
    mechanic_type =  current_user.config["mechanic"]
    isAgency =  current_user.config.get("agency", False)
    form = request.form.to_dict()
    form["timestamp"] = gmtime()
    form["date"] = asctime(form["timestamp"])
    form["useragent"] = request.headers.get('User-Agent')
    #form["IPadress"] = request.remote_addr
    if form["consentRadio"] == "accept":
        Path("trajectories/" + current_user.config["config_id"] + "/"+ uid).mkdir(parents=True, exist_ok=True)
        try:
            with open('trajectories/' + current_user.config["config_id"] + "/" +uid + '/CONSENT.json', 'w', encoding='utf-8') as f:
                json.dump(form, f, ensure_ascii=False, indent=4)
                f.close()
        except KeyError:
            pass
        if condition:
            if mechanic_type == "recipe":
                if isAgency:
                    return render_template('instructions_recipe_Agency.html', is_explained=is_explained)
                else :
                    return render_template('instructions_recipe.html', is_explained=is_explained)

        else:
            return render_template('condition_error.html')

    else:
        Path("trajectories/" + uid).mkdir(parents=True, exist_ok=True)
        try:
            with open('trajectories/' + uid + '/NOT_CONSENT.json', 'w', encoding='utf-8') as f:
                json.dump(form, f, ensure_ascii=False, indent=4)
                f.close()
        except KeyError:
            pass
        return render_template('leave.html', uid=uid, complete=False)


@app.route('/instructions_explained')
def instructions_explained():
    uid = request.args.get('UID')
    #agent_names = get_agent_names()
    return render_template('instructions_explained.html', uid=uid, layout_conf=LAYOUT_GLOBALS)


@app.route('/planning', methods=['GET', 'POST'])
@login_required
def planning():
    uid = current_user.uid
    try:
        condition = current_user.config["conditions"][str(current_user.step)]
    except KeyError:
        condition = request.args.get('CONDITION')
    agent_names = get_agent_names()

    qpt = questionnaire_to_surveyjs(current_user.config["qpt"], current_user.step, current_user.config.get("pagify_qpt", False))#{"elements" :[value for key,value in current_user.config["qpt"].items() if current_user.step in value["steps"]] }
    qpb = {"elements" :[value for key,value in current_user.config["qpb"].items() if current_user.step in value["steps"]] }
    if current_user.step >= len(current_user.config["blocs"].keys()):
        return render_template('goodbye.html', completion_link=current_user.config["completion_link"])
    else :
        return render_template('planning.html', qpb=json.dumps(qpb), qpt=json.dumps(qpt))


@app.route('/transition', methods=['GET', 'POST'])
def transition():
    uid = current_user.uid
    step = current_user.step
    condition = current_user.config["conditions"][str(current_user.step)]
    form = {}
    form["answer"] = request.form.to_dict()
    form["step"] = step
    form["user_agent"] = request.headers.get('User-Agent')
    form["condition"] = condition
    form["uid"] = uid
    form["timestamp"] = gmtime()
    form["date"] = asctime(form["timestamp"])

    Path("trajectories/" + uid).mkdir(parents=True, exist_ok=True)
    try:
        with open('trajectories/' + uid + "/" + uid + "_"  + str(step) + 'QPB.json', 'w', encoding='utf-8') as f:
            json.dump(form, f, ensure_ascii=False, indent=4)
            f.close()
    except KeyError:
        pass
    step += 1
    return render_template('goodbye.html', uid=uid, step=step, completion_link=current_user.config["completion_link"])
    # else :
    #   return render_template('bloc_transition.html', uid = uid, step = step)


@app.route('/planning_design')
def planning_design():
    uid = "design" + str(gmtime())
    new_user = User(uid=uid, config={}, step=0, trial=0)
    db.session.add(new_user)
    db.session.commit()
    login_user(new_user)
    layouts_path = "overcooked_ai_py/data/layouts"
    layouts = [f[:-7] for f in os.listdir(layouts_path)
               if os.path.isfile(os.path.join(layouts_path, f))]
    layouts.sort()
    return render_template('planning_design.html', uid="design", agent_names=["Lazy", "Greedy", "Rational", "Random"], layouts=layouts)


@app.route('/cat')
def cat():
    return render_template('cat.html')


@app.route('/tutorial')
@login_required
def tutorial():
    uid = current_user.uid
    step = 0
    psiturk = request.args.get('psiturk', False)
    return render_template('tutorial.html', uid=uid, seq_id=step, config=TUTORIAL_CONFIG)


@app.route('/debug')
def debug():
    resp = {}
    games = []
    active_games = []
    #waiting_games = []
    users = []
    # free_ids = []
    # free_map = {}
    for game_id in ACTIVE_GAMES:
        game = get_game(game_id)
        active_games.append({"id": game_id, "state": game.to_json()})

    # for game_id in list(WAITING_GAMES.queue):
    #     game = get_game(game_id)
    #     game_state = None if FREE_MAP[game_id] else game.to_json()
    #     waiting_games.append({"id": game_id, "state": game_state})

    for game_id in GAMES:
        games.append(game_id)

    for user_id in USER_ROOMS:
        users.append({user_id: get_curr_room(user_id)})

    # for game_id in list(FREE_IDS.queue):
    #     free_ids.append(game_id)

    # for game_id in FREE_MAP:
    #     free_map[game_id] = FREE_MAP[game_id]

    resp['active_games'] = active_games
    #resp['waiting_games'] = waiting_games
    resp['all_games'] = games
    resp['users'] = users
    # resp['free_ids'] = free_ids
    # resp['free_map'] = free_map
    return jsonify(resp)


#########################
# Socket Event Handlers #
#########################

# Asynchronous handling of client-side socket events. Note that the socket persists even after the
# event has been handled. This allows for more rapid data communication, as a handshake only has to
# happen once at the beginning. Thus, socket events are used for all game updates, where more rapid
# communication is needed

@socketio.on('create')
def on_create(data):
    user_id = current_user.uid


    # Retrieve current game if one exists
    curr_game = get_curr_game(user_id)
    if curr_game:
        # Cannot create if currently in a game
        return
    if data.get("planning_design", None):
        #data.pop("planning_design")
        current_user.config["mechanic"] = data["params"]["mechanic"]
        current_user.config["blocs"] = {"0": data['params']['layouts']}
        current_user.config["agent"] = data['params']["playerOne"] if data[
            'params']["playerOne"] != "human" else data['params']["playerZero"]
        current_user.config["gameTime"] = data['params']['gameTime']
        current_user.config["conditions"] = {
            "0": data['params']['condition']}
    params = data.get('params', {})
    game_name = data.get('game_name', 'overcooked')
    _create_game(user_id, game_name, {"id": current_user.uid, "player_uid": current_user.uid, "step": int(
        current_user.step), "curr_trial_in_game" : int(current_user.trial)-1, "config": current_user.config})


@socketio.on('join')
def on_join(data):
    user_id = current_user.uid
    with USERS[user_id]:
        create_if_not_found = data.get("create_if_not_found", True)

        # Retrieve current game if one exists
        curr_game = get_curr_game(user_id)
        if curr_game:
            # Cannot join if currently in a game
            return

        # Retrieve a currently open game if one exists
        #game = get_waiting_game()

        # No available game was found so create a game
        params = data.get('params', {})
        if user_id != current_user.uid:
            current_user.uid = user_id
            db.session.commit()
        params = data.get('params', {})
        game_name = data.get('game_name', 'overcooked')
        _create_game(user_id, game_name, {"player_uid": current_user.uid, "step": int(
        current_user.step), "curr_trial_in_game" : int(current_user.trial)-1, "room" : current_user.uid,"config": current_user.config})
        return
            # # Game was found so join it
            # with game.lock:

            #     join_room(game.id)
            #     set_curr_room(user_id, game.id)
            #     game.add_player(user_id)

            #     # Game is ready to begin play
            #     game.activate()
            #     ACTIVE_GAMES.add(game.id)
            #     emit('start_game', {"start_info": game.to_json(
            #     ), "trial": current_user.trial, "step": current_user.step, "config": game.config}, to=current_user.uid)
            #     socketio.start_background_task(play_game, game)
            #     # else:
            #     #     # Still need to keep waiting for players
            #     #     WAITING_GAMES.put(game.id)
            #     #     emit('waiting', {"in_game": True}, current_user.uid)


@socketio.on('leave')
def on_leave(data):
    user_id = current_user.uid
    with USERS[user_id]:
        was_active = _leave_game(user_id)

        if was_active:
            emit('end_game', {"status": Game.Status.DONE, "data": {}}, to=current_user.uid)
        else:
            emit('end_lobby', to=current_user.uid)


@socketio.on('action')
def on_action(data):
    user_id = current_user.uid
    action = data['action']

    game = get_curr_game(user_id)
    if not game:
        return

    game.enqueue_action(user_id, action)


@socketio.on('connect')
def on_connect():
    user_id = current_user.uid
    if user_id in USERS:
        return

    USERS[user_id] = Lock()



@socketio.on('disconnect')
def on_disconnect():
    # Ensure game data is properly cleaned-up in case of unexpected disconnect
    user_id = current_user.uid
    if user_id not in USERS:
        return
    with USERS[user_id]:
        _leave_game(user_id)

    del USERS[user_id]

@socketio.on("new_trial")
def on_new_trial():
    user_id = current_user.uid
    game = get_curr_game(user_id)
    if not game:
        return
    current_user.trial = game.curr_trial_in_game
    db.session.commit()
    

@socketio.on("post_qpt")
def post_qpt(data):
    uid = current_user.uid
    form = {}
    form["answer"] = {value["name"] : None for key,value in current_user.config["qpt"].items() if current_user.step in value["steps"]}
    for key, value in data["survey_data"].items():
        form["answer"][key] = value
    condition = current_user.config["conditions"][str(current_user.step)]
    form["timeout_bool"] = data["timeout_bool"]
    form["step"] = current_user.step
    form["trial"] = current_user.trial
    form["trial_id"] = uid + "_" + str(current_user.step) + 'QPT' + str(data["trial_id"])
    form["layout"] = current_user.config["blocs"][str(current_user.step)][current_user.trial]
    form["user_agent"] = request.headers.get('User-Agent')
    form["condition"] = current_user.config["conditions"][str(
        current_user.step)]
    form["uid"] = current_user.uid
    form["timestamp"] = gmtime()
    form["date"] = asctime(form["timestamp"])

    Path("trajectories/"+ current_user.config["config_id"] + "/" + uid + "/QPT").mkdir(parents=True, exist_ok=True)
    try:
        with open('trajectories/'+ current_user.config["config_id"] + "/" + uid + "/QPT/" + uid + "_" + str(current_user.step) + 'QPT' + str(data["trial_id"]) + '.json', 'w', encoding='utf-8') as f:
            json.dump(form, f, ensure_ascii=False, indent=4)
            f.close()
    except KeyError:
        pass

@socketio.on("post_qpb")
def post_qpb(data):
    sid = request.sid
    uid = current_user.uid
    #condition = current_user.config["conditions"][str(current_user.step)]
    form = {}
    form["answer"] = {value["name"] : None for key,value in current_user.config["qpb"].items() if current_user.step in value["steps"]}
    for key, value in data["survey_data"].items():
        form["answer"][key] = value
    condition = current_user.config["conditions"][str(current_user.step)]
    #form["answer"] = data
    form["step"] = current_user.step
    form["trial_id"] = uid + "_" + str(current_user.step) + 'QPB'
    form["user_agent"] = request.headers.get('User-Agent')
    form["condition"] = current_user.config["conditions"][str(
        current_user.step)]
    form["uid"] = current_user.uid
    form["timestamp"] = gmtime()
    form["date"] = asctime(form["timestamp"])

    Path("trajectories/" + current_user.config["config_id"] + "/" + uid).mkdir(parents=True, exist_ok=True)
    try:
        with open('trajectories/' + current_user.config["config_id"] + "/" + uid + "/" + uid + "_" + str(current_user.step) + 'QPB.json', 'w', encoding='utf-8') as f:
            json.dump(form, f, ensure_ascii=False, indent=4)
            f.close()
    except KeyError:
        pass
    current_user.step += 1
    current_user.trial = 0
    db.session.commit()
    socketio.emit("next_step", to=sid)

# Exit handler for server
def on_exit():

    # Force-terminate all games on server termination
    for game_id in GAMES:
        socketio.emit('end_game', {"status": Game.Status.INACTIVE, "data": get_game(
            game_id).get_data()}, room=game_id)

def trial_save_routine(data):
    try:
        Path("trajectories/" + data["config"].get("config_id")+ "/" + data["uid"]
                            ).mkdir(parents=True, exist_ok=True)
    except TypeError:
        return
    try:
        with open('trajectories/'+ data["config"].get("config_id") + "/" + data["uid"] + "/" + data['trial_id']+'.json', 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=4)
        f.close()
    except KeyError:
        pass

#############
# Game Loop #
#############

def play_game(game, fps=30):
    """
    Asynchronously apply real-time game updates and broadcast state to all clients currently active
    in the game. Note that this loop must be initiated by a parallel thread for each active game

    game (Game object):     Stores relevant game state. Note that the game id is the same as to socketio
                            room id for all clients connected to this game
    fps (int):              Number of game ticks that should happen every second
    """
    status = Game.Status.ACTIVE
    while status != Game.Status.DONE and status != Game.Status.INACTIVE:
        with game.lock:
            status = game.tick()
        if status == Game.Status.RESET:
            with game.lock:
                data = game.data  # data is updated in the reset function already triggered at this point
            try:
                trial_save_routine(data)
                if game.qpt and any(game.step in x["steps"] for x in game.qpt.values()) :
                    try:
                        socketio.call("qpt", {"qpt_length": game.qpt_length, "trial" : data["curr_trial_in_game"], "show_time": game.config.get("show_trial_time", False), "time_elapsed": data["time_elapsed"]}, room=game.id)
                    except SocketIOTimeOutError:
                        print("Player " + game.id + " is not on")
                    
                    socketio.emit('reset_game', {"state": game.to_json(), "timeout": game.reset_timeout, "trial": game.curr_trial_in_game, "step": game.step, "condition": game.curr_condition, "config": game.config},
                                    room=game.id)
                    socketio.sleep(game.reset_timeout / 1000)
                                        
                else:
                    socketio.emit('reset_game', {"state": game.to_json(), "timeout": game.reset_timeout, "trial": game.curr_trial_in_game, "step": game.step, "condition": game.curr_condition, "config": game.config},
                                    room=game.id)
                    socketio.sleep(game.reset_timeout / 1000)

            except AttributeError:
                trial_save_routine(data)
                socketio.emit('reset_game', {"state": game.to_json(), "timeout": game.reset_timeout}, room=game.id)
                socketio.sleep(game.reset_timeout / 1000)            
        
        else:
            socketio.emit(
                'state_pong', {"state": game.get_state()}, room=game.id)
        socketio.sleep(1 / fps)
    with game.lock:            
        if status != Game.Status.INACTIVE:
            game.deactivate()
        data = game.data
        trial_save_routine(data)
        if status == Game.Status.DONE:
            try:
                if game.qpt and any(game.step in x["steps"] for x in game.qpt.values()):
                        socketio.call("qpt", {"qpt_length": game.qpt_length, "trial" : data["curr_trial_in_game"]}, room=game.id)
                socketio.emit("qpb", room=game.id)
                                
            except AttributeError:
                pass
            except SocketIOTimeOutError:
                print("Player " + game.id + " is not on")
                socketio.emit("qpb", room=game.id)
            socketio.emit('end_game', {"status": status,
                                "data": data}, room=game.id) 
        

    cleanup_game(game)


if __name__ == '__main__':
    # Dynamically parse host and port from environment variables (set by docker build)
    # host = os.getenv('HOST', 'localhost')
    # port = int(os.getenv('PORT', 8080))
    # Attach exit handler to ensure graceful shutdown
    atexit.register(on_exit)
    if os.getenv('FLASK_ENV', 'production') == 'production':
        debug_env=False
    else:
        debug_env=True

    # https://localhost:80 is external facing address regardless of build environment
    socketio.run(app, host='127.0.0.1', port='5000', debug=debug_env)
