var app = new Vue({
    el: '#app',
    data: {
        map: null,
        mouseTool: null,
        query_traj: null,
        result_trajs: [],
        input_id: '',
        select_options: [{
            value: 'efficient_bf',
            label: 'efficient brute force'
        }, {
            value: 'efficient_faiss',
            label: 'efficient faiss'
        }, {
            value: 'lcss',
            label: 'lcss'
        }, {
            value: 'discret_frechet',
            label: 'discret_frechet'
        }, {
            value: 'hausdorff',
            label: 'hausdorff'
        }, {
            value: 'sspd',
            label: 'sspd'
        }],
        value: '',
        loading: false,
    },
    mounted() {
        this.map = new AMap.Map("container", {
            center: [104.08275, 30.67225],
            zoom: 15
        });
        this.mouseTool = new AMap.MouseTool(this.map);
        this.mouseTool.on("draw", function (event) {
            // event.obj.$x[0] 为折线各个点坐标的list
            app.query_traj = event.obj.$x[0]
            app.mouseTool.close()
        });
        this.drawBorder();
    },
    methods: {
        drawPolyline() {
            app.mouseTool.polyline({
                strokeColor: "red", //线颜色
                strokeOpacity: 1,
                strokeWeight: 6,
                strokeStyle: "solid",
            })
        },
        drawBorder() {
            var pad = 0.002
            var max_lon = 104.12958 - pad
            var max_lat = 30.72775 - pad
            var min_lon = 104.04214 + pad
            var min_lat = 30.65294 + pad
            var polyline = new AMap.Polyline({
                map: this.map,
                path: [
                    [min_lon, min_lat],
                    [min_lon, max_lat],
                    [max_lon, max_lat],
                    [max_lon, min_lat],
                    [min_lon, min_lat]
                ],
                strokeColor: "#000000",
                strokeOpacity: 0.8,
                strokeWeight: 4,
                strokeStyle: "dashed",
                strokeDasharray: [10, 5],
            });
        },
        drawTrajectory(traj, id, polyline_opts) {
            var highlight_opts = JSON.parse(JSON.stringify(polyline_opts));
            highlight_opts['strokeWeight'] = 15
            highlight_opts['strokeOpacity'] = 1
            highlight_opts['zIndex'] = 99
            highlight_opts['path'] = traj
            highlight_opts['map'] = app.map
            polyline_opts['path'] = traj
            polyline_opts['map'] = app.map
            var polyline = new AMap.Polyline(polyline_opts);
            var start_marker = new AMap.Marker({
                map: app.map,
                position: new AMap.LngLat(traj[0][0], traj[0][1]),
                title: id + '起点',
            });
            var end_marker = new AMap.Marker({
                map: app.map,
                position: new AMap.LngLat(traj[traj.length - 1][0], traj[traj.length - 1][1]),
                title: id + '终点',
            });
            start_marker.on('click', function (e) {
                if (polyline._opts.zIndex !== 99) {
                    polyline.setOptions(highlight_opts)
                } else {
                    polyline.setOptions(polyline_opts)
                }
            });
            end_marker.on('click', function (e) {
                if (polyline._opts.zIndex !== 99) {
                    polyline.setOptions(highlight_opts)
                } else {
                    polyline.setOptions(polyline_opts)
                }
            });
        },
        query() {
            if (this.query_traj === null) {
                app.$message({
                    type: 'error',
                    message: '要查询的轨迹为空',
                  });
            } else {
                var selected_type = $("#type_select").val()
                const start_time = Math.round(new Date());
                $.ajax({
                    type: "POST",
                    url: "/query",
                    data: {
                        input: JSON.stringify(this.query_traj),
                        type: selected_type,
                    },
                    success: function (data) {
                        console.log(data)
                        app.loading = false;
                        if (data.success) {
                            const end_time = Math.round(new Date());
                            app.$message({
                                type: 'success',
                                message: '查询成功!\n耗时: ' + (end_time - start_time) + 'ms',
                              });
                            app.result_trajs = data.result;
                            app.drawResult();
                        } else {
                            app.$message({
                                type: 'error',
                                message: '查询失败',
                              });
                        }
                    }
                })
                app.loading = true;
            }
        },
        clearDrawTraj() {
            this.query_traj = null;
            this.map.clearMap();
            this.mouseTool.close(true)
            this.drawBorder();
        },
        drawResult() {
            this.map.clearMap();
            this.drawBorder();
            this.result_trajs.forEach(function (traj) {
                app.drawTrajectory(traj.data, traj.id, {
                    strokeColor: "#39c5bb", //线颜色
                    strokeOpacity: 0.5,
                    strokeWeight: 8,
                    strokeStyle: "solid",
                    zIndex: 9,
                })
            })
            app.drawTrajectory(app.query_traj, "query", {
                strokeColor: "red", //线颜色
                strokeOpacity: 0.8,
                strokeWeight: 10,
                strokeStyle: "solid",
                zIndex: 10,
            })
        },
        handleHighlight(index, row) {
            console.log(index)
            console.log(row)
        }
    }
})