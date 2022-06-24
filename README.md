# 相似轨迹检索系统
## 本科毕设
部署步骤：  
1. 下载项目到本地，按照requirements.txt安装好依赖
2. 在项目根目录新建一个model文件夹，在其中放入模型文件
3. 启动MySQL，运行/web/static/trajectory.sql，并手动导入数据
4. 根据MySQL服务器地址，修改/web/mapper.py的db_path
5. 去[高德开放平台](https://lbs.amap.com/api/jsapi-v2/guide/abc/prepare)获取你的开发者Key（使用高德地图API需要）
6. 按照[高德开放平台](https://lbs.amap.com/api/jsapi-v2/guide/abc/prepare)的教程，将你的开发者Key填入/web/templates/index.html
7. 在项目根目录下，执行`python web/app.py`启动项目，默认访问地址localhost:5000