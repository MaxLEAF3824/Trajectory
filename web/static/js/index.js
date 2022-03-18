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
            center: [104.04275, 30.69225],
            zoom: 14
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
                // 线样式还支持 'dashed'
                strokeStyle: "solid",
                // strokeStyle是dashed时有效
                // strokeDasharray: [10, 5],
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
            // this.map.clearMap();
            this.result_trajs.forEach(function (traj) {
                var polyline = new AMap.Polyline({
                    map: app.map,
                    path: traj.data,
                    strokeColor: "#41c9fa",
                    strokeOpacity: 1,
                    strokeWeight: 6,
                    // 线样式还支持 'dashed'
                    strokeStyle: "solid",
                    // strokeStyle是dashed时有效
                    // strokeDasharray: [10, 5],
                });
            })
        },
    }
})



