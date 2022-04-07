import os
from flask import Flask, jsonify, render_template, request
import json
from datetime import timedelta
from service import Service
import hashlib

app = Flask(__name__)
app.jinja_env.variable_start_string = '[['  # 解决jinja2和vue的分隔符{{}}冲突
app.jinja_env.variable_end_string = ']]'
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = timedelta(seconds=1)  # 浏览器不缓存实时更新静态文件
service = Service()


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/query', methods=['GET', 'POST'])
def query():
    query_traj = json.loads(request.form.get("input"))
    query_type = request.form.get("type")
    time_slice = request.form.get("time_slice")
    k = request.form.get("k")
    traj_list, sim_list = service.knn_query(query_traj, query_type, k, time_slice)
    result = []
    return jsonify({"code": 200, "success": True, "result": result, "msg": "查询成功"})


if __name__ == '__main__':
    app.run(debug=True)
