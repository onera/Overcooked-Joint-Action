console.log("executing survey_manager");

class CallBackTrigger {
    constructor(callback, trial_id) {
        this.callback = callback;
        this.trial_id = trial_id;
    }
    trigger() {
        this.callback("qpt")
    }
}

var timeout;
var timeleft;
var callbacktrigger;
var qpt_timeout_bool = false;
var qpt_model
var qpt_elements

socket.on("connect", function () {
    console.log("connect survey")
    Survey.StylesManager.applyTheme("defaultV2");
    qpt_elements = JSON.parse($('#qpt_elements').text());
    qpt_model = new Survey.Model(qpt_elements);

    qpt_model.onComplete.add(function (sender) {
        callbacktrigger.trigger()
        clearTimeout(timeout);
        clearInterval(timeleft);
        $('#overcooked').show();
        $('#qpt').hide();
        console.log(sender.data);
        socket.emit("post_qpt", { "survey_data": sender.data, "trial_id": callbacktrigger.trial_id, "timeout_bool": qpt_timeout_bool });
        qpt_timeout_bool = false;
    });
    $("#QptDisplay").Survey({ model: qpt_model });
    var qpb_elements = JSON.parse($('#qpb_elements').text());
    var qpb_model = new Survey.Model(qpb_elements);
    qpb_model.onComplete.add(function (sender) {
        socket.emit("post_qpb", { "survey_data": sender.data });
    });
    $("#QpbDisplay").Survey({
        model: qpb_model
    });
})

socket.on('qpt', function (data, callback) {
    qpt_model.clear();
    qpt_model.render();
    $('#overcooked').hide();
    $("#qpt").show();
    if (data.show_time) {
        $("#elapsed_time").text("You have completed the game in " + Math.round(data.time_elapsed) +" seconds !")
    }
    console.log(data.show_time);
    callbacktrigger = new CallBackTrigger(callback, data.trial)
    const timeout_start = Date.now();
    callback = callback
    timeout = setTimeout(function () {
        qpt_timeout_bool = true;
        qpt_model.doComplete(true);
    }, data.qpt_length * 1000
    );

    timeleft = setInterval(() => {
        //console.log("Time left: ", data.qpt_length- (Date.now() - timeout_start)/1000, "s");
        $('#qpt_timer').text("Remaining time  " + Math.round(data.qpt_length * 10 - (Date.now() - timeout_start) / 100) / 10 + "  Seconds");
    }, 100);
})

socket.on('qpb', function () {
    $('#overcooked').hide();
    $("#qpb").show();
})

socket.on('next_step', function () {
    location.reload();
})

