console.log("executing planning")
// Persistent network connection that will be used to transmit real-time data
var socket = io();

/* * * * * * * * * * * * * * * * 
 * Button click event handlers *
 * * * * * * * * * * * * * * * */

var experimentParams = {
    layouts : ["cramped_room", "counter_circuit"],
    gameTime : 10,
    playerZero : "DummyAI"
};

let step = 0;
var curr_trial = 0;
var condition = "U";




$(function() {
    $('#create').click(function () {
        let params = JSON.parse($('#config').text());
        let uid = $('#uid').text();
        params.player_uid = uid;
        params.bloc = bloc;
        params.condition = condition;
        data = {
            "params" : params,
            "game_name" : "planning",
            "create_if_not_found" : false
        };
        socket.emit("create", data);
        $('#waiting').show();
        $('#join').hide();
        $('#join').attr("disabled", true);
        $('#create').hide();
        $('#create').attr("disabled", true)
        $("#instructions").hide();
        $('#tutorial').hide();
    });
});

$(function() {
    $('#join').click(function() {
        socket.emit("join", {});
        $('#join').attr("disabled", true);
        $('#create').attr("disabled", true);
    });
});

$(function() {
    $('#leave').click(function() {
        socket.emit('leave', {});
        $('#leave').attr("disabled", true);
    });
});

$(function() {
    $('#answer').click(function() {
        let uid = $('#uid').text();
        let step = $('#step').text();
        $('answer').attr("disable", true);
        window.location.href = "./question";
    });
});

$(function() {
    $('#start').click(function() {;
        $('start').attr("disable", true);
        $.ajax({
            type: "POST",
            url: "/planning", 
            data: {"achieved_step" : $('#step').text()},
            success: function(){
                location.reload();
            }
        })
    });
});


/* * * * * * * * * * * * * 
 * Socket event handlers *
 * * * * * * * * * * * * */

window.intervalID = -1;
window.spectating = true;

socket.on("connect", function () {
    let params = $('#config')//JSON.parse($('#config').text());
    params.condition = $('#condition')
    data = {
        "params": params,
        "game_name": "planning",
        "create_if_not_found": false
    };
    socket.emit("create", data);
    $('#waiting').show();
    $('#join').hide();
    $('#join').attr("disabled", true);
    $('#create').hide();
    $('#create').attr("disabled", true)
    $("#instructions").hide();
    $('#tutorial').hide();
});

socket.on('waiting', function (data) {
    // Show game lobby
    $('#error-exit').hide();
    $('#waiting').hide();
    $('#game-over').hide();
    $('#instructions').hide();
    $('#tutorial').hide();
    $("#overcooked").empty();
    $('#lobby').show();
    $('#join').hide();
    $('#join').attr("disabled", true)
    $('#create').hide();
    $('#create').attr("disabled", true)
    $('#leave').show();
    $('#leave').attr("disabled", false);
    if (!data.in_game) {
        // Begin pinging to join if not currently in a game
        if (window.intervalID === -1) {
            window.intervalID = setInterval(function () {
                socket.emit('join', {});
            }, 1000);
        }
    }
});

socket.on('creation_failed', function(data) {
    // Tell user what went wrong
    let err = data['error']
    $("#overcooked").empty();
    $('#lobby').hide();
    $("#instructions").show();
    $('#tutorial').show();
    $('#waiting').hide();
    $('#join').show();
    $('#join').attr("disabled", false);
    $('#create').show();
    $('#create').attr("disabled", false);
    console.log("creation")
    $('#overcooked').append(`<h4>Sorry, game creation code failed with error: ${JSON.stringify(err)}</>`);
});

socket.on('start_game', function(data) {
    // Hide game-over and lobby, show game title header
    if (window.intervalID !== -1) {
        clearInterval(window.intervalID);
        window.intervalID = -1;
    }
    graphics_config = {
        container_id : "overcooked",
        start_info : data.start_info,
        condition : data.config.conditions[data.step],
        mechanic : data.config.mechanic,
        show_counter_drop : data.config.show_counter_drop,
    };
    window.spectating = data.spectating;
    $('#error-exit').hide();
    $("#overcooked").empty();
    $('#game-over').hide();
    $('#lobby').hide();
    $('#waiting').hide();
    $('#join').hide();
    $('#join').attr("disabled", true);
    $('#create').hide();
    $('#create').attr("disabled", true)
    $("#instructions").hide();
    $('#tutorial').hide();
    $('#leave').show();
    $('#leave').attr("disabled", false)
    curr_trial = data.trial +1;

    $('#game-title').text(`Experiment in Progress, Bloc ${data.step + 1}/${Object.keys(data.config.blocs).length}, essai ${curr_trial}/${Object.keys(data.config.blocs[data.step]).length}`);
    $('#game-title').show(); 
    
    if (!window.spectating) {
        enable_key_listener();
    }
    graphics_start(graphics_config);
});



socket.on('reset_game', function(data) {   
    step = $('#step')
    //graphics_end();
    game_config.scene.endLevel();
    if (!window.spectating) {
        disable_key_listener();
    }
    curr_trial = data.trial + 1;
    $('#game-title').text(`Experiment in Progress, Bloc ${data.step+1}/${Object.keys(data.config.blocs).length}, essai ${curr_trial}/${Object.keys(data.config.blocs[data.step]).length}`);
    $("#reset-game").show();
    setTimeout(function() {
        $("reset-game").hide();
        graphics_config = {
            container_id : "overcooked",
            start_info : data.state, 
            condition : data.condition,
        };
        if (!window.spectating) {
            enable_key_listener();
        }
        graphics_reset(graphics_config);
    }, data.timeout);
    socket.emit("new_trial");     
});

socket.on('state_pong', function(data) {
    // Draw state update
    drawState(data['state']);
});

socket.on('end_game', function(data) {
    // Hide game data and display game-over html
    graphics_end();
    if (!window.spectating) {
        disable_key_listener();
    }
    let bloc = $('#bloc').text();
    let step = $('#step').text();
    $('#overcooked-container').append(`<h4>Now we are going to ask you a few questions about your feeling during the last games</>`);
    $('#game-title').hide();
    $('#game-over').show();
    $('#overcooked').hide();
    $('#answer').attr("disabled", false);
    $("#leave").hide();
    $('#leave').attr("disabled", true)
    
    // Game ended unexpectedly
    if (data.status === 'inactive') {
        $('#error-exit').show();
    }
});

socket.on('end_lobby', function() {
    // Hide lobby
    console.log("end_lobby");
    $('#lobby').hide();
    $("#join").show();
    $('#join').attr("disabled", false);
    $("#create").show();
    $('#create').attr("disabled", false)
    $("#leave").hide();
    $('#leave').attr("disabled", true)
    $("#instructions").show();
    $('#tutorial').show();

    // Stop trying to join
    clearInterval(window.intervalID);
    window.intervalID = -1;
})



/* * * * * * * * * * * * * * 
 * Game Key Event Listener *
 * * * * * * * * * * * * * */

function enable_key_listener() {
    $(document).on('keydown', function(e) {
        let action = 'STAY'
        switch (e.which) {
            case 37: // left
                action = 'LEFT';
                break;

            case 38: // up
                action = 'UP';
                break;

            case 39: // right
                action = 'RIGHT';
                break;

            case 40: // down
                action = 'DOWN';
                break;

            case 32: //space
                action = 'SPACE';
                break;

            default: // exit this handler for other keys
                return; 
        }
        e.preventDefault();
        socket.emit('action', { 'action' : action, 'condition' : condition});
    });
};

function disable_key_listener() {
    $(document).off('keydown');
};


/* * * * * * * * * * *
 * Utility Functions *
 * * * * * * * * * * */

var arrToJSON = function(arr) {
    let retval = {}
    for (let i = 0; i < arr.length; i++) {
        elem = arr[i];
        key = elem['name'];
        value = elem['value'];
        retval[key] = value;
    }
    return retval;
};

