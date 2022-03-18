from datetime import timedelta

from flask import Flask, jsonify, render_template, request
import json
import numpy as np

app = Flask(__name__)

# flask的jinja2模板引擎的分隔符{{}}和vue冲突了！！！！
app.jinja_env.variable_start_string = '[['
app.jinja_env.variable_end_string = ']]'
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = timedelta(seconds=1)


@app.route('/')
def hello_world():  # put application's code here
    return render_template('index.html')


@app.route('/query', methods=['GET', 'POST'])
def query():  # open the index page
    query_traj = json.loads(request.form.get("input"))
    query_type = request.form.get("type")

    res1 = np.array(query_traj) + np.random.randn(len(query_traj), 2) * 0.001
    res2 = np.array(query_traj) + np.random.randn(len(query_traj), 2) * 0.001
    import random

    result = [{"id": "res1", "data": res1.tolist(), "sim": random.random()},
              {"id": "res2", "data": res2.tolist(), "sim": random.random()}]
    return jsonify({"code": 200, "success": True, "result": result})


if __name__ == '__main__':
    app.run(debug=True)
