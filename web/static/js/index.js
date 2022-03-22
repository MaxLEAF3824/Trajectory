var app = new Vue({
    el: '#app',
    data: {
        map: null,
        mouseTool: null,
        query_traj: null,
        result_trajs: [],
        default_polyline_opts: {
            strokeColor: "red", //线颜色
            strokeOpacity: 1,
            strokeWeight: 6,
            // 线样式还支持 'dashed'
            strokeStyle: "solid",
            // strokeStyle是dashed时有效
            // strokeDasharray: [10, 5],
        }
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
            console.log(app.query_traj)
            app.mouseTool.close()
        });
    },
    methods: {
        drawPolyline() {
            app.mouseTool.polyline(app.default_polyline_opts)
        },
        drawTrajectory(traj, polyline_opts) {
            var polyline_opts = polyline_opts||app.default_polyline_opts
            polyline_opts['path'] = traj
            polyline_opts['map'] = app.map
            var polyline = new AMap.Polyline(polyline_opts);
            var start_marker = new AMap.Marker({
                map: app.map,
                position: new AMap.LngLat(traj[0][0], traj[0][1]), 
                title: 'from',
            });
            var end_marker = new AMap.Marker({
                map: app.map,
                position: new AMap.LngLat(traj[traj.length - 1][0], traj[traj.length - 1][1]), 
                title: 'to',
            });
            start_marker.on('click', function (e) {
                if (polyline._opts.strokeColor == polyline_opts['strokeColor']) {
                    polyline.setOptions({
                        strokeColor: "#0D66AB", //线颜色
                    })
                } else if (polyline._opts.strokeColor == "#0D66AB") {
                    polyline.setOptions({
                        strokeColor: polyline_opts['strokeColor'], //线颜色
                    })
                }
            });
            end_marker.on('click', function (e) {
                if (polyline._opts.strokeColor == "#39c5bb") {
                    polyline.setOptions({
                        strokeColor: "#0D66AB", //线颜色
                    })
                } else if (polyline._opts.strokeColor == "#0D66AB") {
                    polyline.setOptions({
                        strokeColor: "#39c5bb", //线颜色
                    })
                }
            });
        },
        query() {
            if (this.query_traj === null) {
                alert("query_traj为空")
            } else {
                var selected_type = $("#type_select option:selected").val()
                $.ajax({
                    type: "POST",
                    url: "/query",
                    data: {
                        input: JSON.stringify(this.query_traj),
                        type: selected_type,
                    },
                    success: function (data) {
                        console.log(data)
                        if (data.success) {
                            console.log("查询成功");
                            app.result_trajs = data.result;
                            app.drawResult();
                        } else {
                            alert("查询失败")
                        }
                    }
                })
            }
        },
        clearDrawTraj() {
            this.query_traj = null;
            this.map.clearMap();
            this.mouseTool.close(true)
        },
        drawResult() {
            this.map.clearMap();
            this.result_trajs.forEach(function (traj) {
                app.drawTrajectory(traj.data, {
                    strokeColor: "#39c5bb", //线颜色
                    strokeOpacity: 1,
                    strokeWeight: 4,
                    strokeStyle: "solid",
                })
            })
            app.drawTrajectory(app.query_traj)
        },
        uploadDataset() {
            var file = $("#file")[0].files[0];
            var formData = new FormData();
            formData.append("file", file);
            $.ajax({
                type: "POST",
                url: "/upload",
                data: formData,
                processData: false,
                contentType: false,
                success: function (data) {
                    console.log(data)
                    if (data.success) {
                        alert("成功: " + data.msg);
                    } else {
                        alert("失败: " + data.msg);
                    }
                }
            })
        },
    }
})