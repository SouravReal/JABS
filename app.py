from flask import Flask, render_template, request, abort
import pickle
import numpy as np

app = Flask(__name__)
model = pickle.load(open("trained_model.pkl", "rb"))

# blocked_routes = ['/predict']


# @app.before_request
# def check_blocked_routes():
#     if request.path in blocked_routes:
#         abort(403)


@app.route("/")
def homePage():
    return render_template('Landing.html')


@app.route("/ppage")
def ppage():
    return render_template('prediction.html')

# # predict function re sabu allocation haba


@app.route("/predict", methods=["POST", "GET"])
def predict():
    responses = {}

    responses["name"] = str(request.form["name"])
    responses["age"] = int(request.form["age"])
    responses["label_gender"] = request.form["label_gender"]
    responses["label_family_history"] = request.form["label_family_history"]
    responses["label_observed_consequences"] = request.form["label_observed_consequences"]
    responses["label_work_interfere"] = request.form["label_work_interfere"]
    responses["label_benefits"] = request.form["label_benefits"]
    responses["label_care_options"] = request.form["label_care_options"]
    responses["label_wellness_program"] = request.form["label_wellness_program"]
    responses["label_seek_help"] = request.form["label_seek_help"]
    responses["label_mental_health_consequence"] = request.form["label_mental_health_consequence"]
    responses["label_coworkers"] = request.form["label_coworkers"]
    responses["label_mental_health_interview"] = request.form["label_mental_health_interview"]
    responses["label_mental_vs_physical"] = request.form["label_mental_vs_physical"]

    responses2 = responses.copy()

    indexes2 = {
        "label_gender":	{"female": 0, "male": 1, "others": 2},
        "label_family_history":	{"no": 0,	"yes": 1},
        "label_observed_consequences":	{"no": 0,	"yes": 1},
        "label_work_interfere":	{"never": 0,	"often": 1,	"sometimes": 2},
        "label_benefits":	{"no": 0,	"yes": 1},
        "label_care_options":	{"no": 0,	"yes": 1},
        "label_wellness_program":	{"no": 0,	"yes": 1},
        "label_seek_help":	{"no": 0,	"yes": 1},
        "label_mental_health_consequence":	{"maybe": 0,	"no": 1,	"yes": 2},
        "label_coworkers":	{"no": 0,	"some_of_them": 1,	"yes": 2},
        "label_mental_health_interview":	{"maybe": 0,	"no": 1,	"yes": 2},
        "label_mental_vs_physical":	{"no": 0,	"yes": 1}
    }

    for x, y in indexes2.items():
        for i, j in responses.items():
            if i == x:
                for a, b in y.items():
                    if j == a:
                        responses[i] = b

    resp = []
    del responses[next(iter(responses))]
    for i, _ in responses.items():
        resp.append(responses[i])

    finalResp = np.asarray(resp)
    finalResp_reshape = finalResp.reshape(1, -1)

    prediction = model.predict(finalResp_reshape)
    if prediction == 0:
        return render_template("result.html",
                               name=responses2["name"],
                               age=responses2["age"],
                               label_gender=responses2["label_gender"],
                               label_family_history=responses2["label_family_history"],
                               label_observed_consequences=responses2["label_observed_consequences"],
                               label_work_interfere=responses2["label_work_interfere"],
                               label_benefits=responses2["label_benefits"],
                               label_care_options=responses2["label_care_options"],
                               label_wellness_program=responses2["label_wellness_program"],
                               label_seek_help=responses2["label_seek_help"],
                               label_mental_health_consequence=responses2["label_mental_health_consequence"],
                               label_coworkers=responses2["label_coworkers"],
                               label_mental_health_interview=responses2["label_mental_health_interview"],
                               label_mental_vs_physical=responses2["label_mental_vs_physical"],
                               pred="You are Fine")
    else:
        return render_template(
            "result.html",
            name=responses2["name"],
            age=responses2["age"],
            label_gender=responses2["label_gender"],
            label_family_history=responses2["label_family_history"],
            label_observed_consequences=responses2["label_observed_consequences"],
            label_work_interfere=responses2["label_work_interfere"],
            label_benefits=responses2["label_benefits"],
            label_care_options=responses2["label_care_options"],
            label_wellness_program=responses2["label_wellness_program"],
            label_seek_help=responses2["label_seek_help"],
            label_mental_health_consequence=responses2["label_mental_health_consequence"],
            label_coworkers=responses2["label_coworkers"],
            label_mental_health_interview=responses2["label_mental_health_interview"],
            label_mental_vs_physical=responses2["label_mental_vs_physical"], pred="You might have mental disorder.Better consult a doctor"
        )


if __name__ == "__main__":
    app.run(debug=True, port=6969)
