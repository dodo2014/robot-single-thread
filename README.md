### config配置说明

```
{
  "plc_config": {  // plc配置
    "ip": "192.168.0.10",  // ip
    "port": 502  // port
  },
  "robot_params": {  // 机器人参数
    "l1": 850.0,     // 大臂长度
    "l2": 650.0,     // 小臂长度
    "z0": 0.0,       // z轴高度
    "nn3": 0.05       // z轴丝杆导程的倒数，即 rev/mm, 每毫米需要转几圈
  },
  "origin_params": {  // 原点配置
    "name": "初始位置原点",
    "coords": [
      1022.37,
      -114.60,
      0,
      -53.2
    ],
    "photo": 0    // 0 - 默认值；1 - 拍照；2 - 给出目标坐标
  },
  "processes": {  // 动作步骤
    "0x40082": {  // 动作编号
      "name": "首次臂去上料位拍照",  // 动作名称
      "type": "vision_trigger",    // 动作类型，vision_trigger - 拍照；其他的类型待定
      "points": [   // 坐标点
        {
          "name": "P2",  // 坐标名称
          "coords": [    // 坐标参数
            1500.0,      // x
            0.0,         // y
            -300.0,      // z
            0.0          // te, 末端电机的角度
          ],
          "photo": 0
        },
        {
          "name": "P3",
          "coords": [
            1500.0,
            0.0,
            0.0,
            45.0
          ],
          "photo": 0
        },
        {
          "name": "P4",
          "coords": [
            968.0,
            462.0,
            0.0,
            0.0
          ],
          "photo": 0
        },
        ...
      ]
    },
    ...
  },
  "trajectory_params": {   // 轨迹规划参数，暂时不使用
    "num_points": 9,
    "max_velocity": null,
    "acceleration_time": 0.2
  }
}
```