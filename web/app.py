from datetime import timedelta
from service import *
from flask import Flask, jsonify, render_template, request
import json
import numpy as np

app = Flask(__name__)

# flask的jinja2模板引擎的分隔符{{}}和vue冲突了！！！！
app.jinja_env.variable_start_string = '[['
app.jinja_env.variable_end_string = ']]'
# 静态文件缓存时间不设置为1的话，浏览器不会实时更新静态文件，对debug不利
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = timedelta(seconds=1) 


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
        result = query_efficient(query_traj)
    elif int(query_type) == 1:
        result = query_edr(query_traj)
    else:
        result = []
    
    return jsonify({"code": 200, "success": True, "result": result})

if __name__ == '__main__':
    
    app.run(debug=True)
