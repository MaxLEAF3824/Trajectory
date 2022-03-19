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
    # open the index page
    return render_template('index.html')


@app.route('/query', methods=['GET', 'POST'])
def query():
    # handle query request
    query_traj = json.loads(request.form.get("input"))
    query_type = request.form.get("type")

    if int(query_type) == 0:
        result = service.query_efficient(query_traj)
    elif int(query_type) == 1:
        result = service.query_edr(query_traj)
    else:
        result = []

    return jsonify({"code": 200, "success": True, "result": result, "msg": "查询成功"})


@app.route('/upload', methods=['POST'])
def upload_dataset():
    # handle upload request
    json_file = request.files.get("file")

    # 简单判断是否为json文件
    if not json_file.filename.lower().endswith('.json'):
        return jsonify({"code": 400, "success": False, "msg": "文件格式错误"})

    # 用md5判断是否要保存文件
    json_file.save(f"./static/dataset/{json_file.filename}")
    md5_obj = hashlib.md5()
    md5_obj.update(open(f"./static/dataset/{json_file.filename}", 'rb').read())
    file_md5_id = md5_obj.hexdigest()
    if os.path.isfile(f"./static/dataset/{file_md5_id}.json"):
        os.remove(f"./static/dataset/{json_file.filename}")
    else:
        os.rename(f"./static/dataset/{json_file.filename}", f"./static/dataset/{file_md5_id}.json")

    # 插入数据库
    ret = service.insert_trajectories(f"./static/dataset/{file_md5_id}.json")
    if ret == 1:
        return jsonify({"code": 400, "success": False, "msg": "文件内容错误"})
    return jsonify({"code": 200, "success": True, "msg": "上传成功"})


if __name__ == '__main__':
    app.run(debug=True)
