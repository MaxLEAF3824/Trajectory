var app = new Vue({
    el: '#app',
    data: {
        map: null,
        mouseTool: null,
        query_traj: null,
        result_trajs: [],
    },
    mounted() {
        this.map = new AMap.Map("container", {
            center: [104.08275, 30.67225],
            zoom: 15
        });
        this.mouseTool = new AMap.MouseTool(this.map);
        this.mouseTool.on("draw", function (event) {
            // event.obj 为绘制出来的覆盖物对象
            // event.obj.$x[0] 为折线各个点坐标的list
            app.query_traj = event.obj.$x[0]
            console.log(app.query_traj)
            app.mouseTool.close()
        });
    },
    methods: {
        drawPolyline() {
            this.mouseTool.polyline({
                strokeColor: "#fa4141",
                strokeOpacity: 1,
                strokeWeight: 6,
                strokeStyle: "solid",
            })
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
            var idx = 0;
            this.result_trajs.forEach(function (traj) {
                idx += 1;
                var sim = traj.sim
                var polyline = new AMap.Polyline({
                    map: app.map,
                    path: traj.data,
                    strokeColor: "#36c5bb", //线颜色
                    strokeOpacity: 1,
                    strokeWeight: 4,
                    strokeStyle: "solid",
                });
                var start_marker = new AMap.Marker({
                    map: app.map,
                    position: new AMap.LngLat(traj.data[0][0], traj.data[0][1]),   // 经纬度对象，也可以是经纬度构成的一维数组[116.39, 39.9]
                    title: '起点' + idx,
                });
                var end_marker = new AMap.Marker({
                    map: app.map,
                    position: new AMap.LngLat(traj.data[traj.data.length - 1][0], traj.data[traj.data.length - 1][1]),   // 经纬度对象，也可以是经纬度构成的一维数组[116.39, 39.9]
                    title: '终点' + idx,
                });
                start_marker.on('click', function (e) {
                    end_marker
                });
            })
            var polyline = new AMap.Polyline({
                map: app.map,
                path: app.query_traj,
                strokeColor: "#fa4141", //线颜色
                strokeOpacity: 1,
                strokeWeight: 6,
                strokeStyle: "solid",
            });

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



