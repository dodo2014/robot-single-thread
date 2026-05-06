loop_log_rate = 100

# 动作监控起始位置
process_start_addr = 0x40082
process_num = 39

# 关节坐标地址位
# 每组包含：[关节1角度, 关节2角度, 关节3高度, 关节4角度, 速度, 加速度], 每个关节坐标（浮点数）占用2个寄存器
point_addresses = [
    [0x40000, 0x40002, 0x40004, 0x40006, 0x40008, 0x4000A],
    # 第1组坐标地址，0x40000, 0x40002, 0x40004, 0x40006, 0x40008, 0x40010
    [0x4000C, 0x4000E, 0x40010, 0x40012, 0x40014, 0x40016],
    # 第2组坐标地址，0x40012, 0x40014, 0x40016, 0x40018, 0x40020, 0x40022
    [0x40018, 0x4001A, 0x4001C, 0x4001E, 0x40020, 0x40022],
    # 第3组坐标地址，0x40024, 0x40026, 0x40028, 0x40030, 0x40032, 0x40034
    [0x40024, 0x40026, 0x40028, 0x4002A, 0x4002C, 0x4002E],
    # 第4组坐标地址，0x40036, 0x40038, 0x40040, 0x40042, 0x40044, 0x40046
    [0x40030, 0x40032, 0x40034, 0x40036, 0x40038, 0x4003A],
    # 第5组坐标地址，0x40048, 0x40050, 0x40052, 0x40054, 0x40056, 0x40058
    [0x4003C, 0x4003E, 0x40040, 0x40042, 0x40044, 0x40046],
    # 第6组坐标地址，0x40060, 0x40062, 0x40064, 0x40066, 0x40068, 0x40070
    [0x40048, 0x4004A, 0x4004C, 0x4004E, 0x40050, 0x40052],
    # 第7组坐标地址，0x40072, 0x40074, 0x40076, 0x40078, 0x40080, 0x40082
    [0x40054, 0x40056, 0x40058, 0x4005A, 0x4005C, 0x4005E],
    # 第8组坐标地址，0x40084, 0x40086, 0x40088, 0x40090, 0x40092, 0x40094
    [0x40060, 0x40062, 0x40064, 0x40066, 0x40068, 0x4006A],
    # 第9组坐标地址，0x40096, 0x40098, 0x40100, 0x40102, 0x40104, 0x40106
    [0x4006C, 0x4006E, 0x40070, 0x40072, 0x40074, 0x40076],
    # 第10组坐标地址，0x40108, 0x40110, 0x40112, 0x40114, 0x40116, 0x40118
]

point_once_address = point_addresses[7]  # 发送单个坐标的地址位

# 坐标间插值数
point_interpolated_num = 7

# 相机执行参数
photo_trigger_depth = 1  # 深度相机
photo_trigger_ccd = 2  # ccd相机
photo_trigger_laser = 3  # 点激光传感器

ADDR_FIRST = 0x40000

# 实时反馈地址定义 (Float32, 每个占用2个寄存器)
ADDR_FEEDBACK_J1 = 0x400BE
ADDR_FEEDBACK_J2 = 0x400C0
ADDR_FEEDBACK_J3 = 0x400C2
ADDR_FEEDBACK_J4 = 0x400C4

# 机械臂轴4个轴的实时数据，批量读取的起始地址和长度 (从 400BE 开始，读 4个float = 8个寄存器)
ADDR_FEEDBACK_START = 0x400BE
FEEDBACK_LEN = 8

# 急停监控地址
ADDR_ESTOP_MONITOR = 0x400A8
# 急停触发值
ESTOP_TRIGGER_VAL = 10

# 暂停/恢复控制
ADDR_PAUSE_CONTROL = 0x400A5
VAL_PAUSE_REQ = 10  # 暂停请求
VAL_RESUME_REQ = 11  # 恢复请求
VAL_RESET = 0  # 复位/空闲

# 机械臂J1活动范围
J1_LIMIT_MAX = 98
J1_LIMIT_MIN = -98

# 机械臂 J2 活动范围
J2_LIMIT_MAX = 162
J2_LIMIT_MIN = -162

# 机械臂J4活动范围
J4_LIMIT_MAX = 145
J4_LIMIT_MIN = -145

# 设定精拍阈值 (mm)
PRECISE_PHOTO_DISTANCE = 400.0

# 上一个步骤的地址位映射
last_process_addr_map = {
    0x40082: 0x400A7,
    0x40083: 0x40082,
    0x40084: 0x40083,
    0x40085: 0x40084,
    0x40086: 0x40085,
    0x40087: 0x40086,
    0x40088: 0x40087,
    0x40089: 0x40088,
    0x4008A: 0x40089,
    0x4008B: 0x4008A,
    0x4008C: 0x4008B,
    0x4008D: 0x4008C,

    0x4008F: 0x4008E,
    0x40090: 0x4008F,
    0x40091: 0x40090,
    0x40092: 0x40091,
    0x40096: 0x40093,
    0x40097: 0x40096,
    0x40099: 0x40098,
    0x4009C: 0x4008F,
}

# 拍照触发的动作类型，普通拍照(物料识别)/1，上料(空料判断)/2，下料(满料判断)/3，铝屑识别/4
photo_type_normal = 1  # 普通拍照
photo_type_loading = 2  # 上料
photo_type_find_head = 5  # 上料区找端头
photo_type_unloading = 3  # 下料
photo_type_aluminum = 4  # 铝屑识别

PHOTO_TYPE_DESC = {
    photo_type_normal: "normal",
    photo_type_loading: "loading",
    photo_type_find_head: "find head",
    photo_type_unloading: "unloading",
    photo_type_aluminum: "aluminum"
}

# 上料区域索引文件名
loading_index_file_name = "loading_index.json"
# 上料区阵列索引的key
search_index_name="current_search_index"
# 上料区域端头索引的key
head_layer_index_name = "current_head_layer_index"
# 上次成功抓取的层
last_picked_layer_name = "last_picked_layer"

# 上料区物料参数
product_height = 117  # 单个产品的高度(mm)
product_width = 123  # 单个产品的宽度(mm)
interval_height = 15  # 层与层之间间隔木条的高度（mm）
base_depth = 420  # 首层标准深度，最上层拍照位，深度相机与物料的距离,低于这个值，物料上沿的深度会丢失, (mm)
product_cols_per_layer = 5  # 每层的物料数量
product_total_layers = 2  # 物料的总层数
tolerange = 30.0  # 机械臂扫描，相机拍照深度z的安全容差，允许深度数据有 30mm 的向下波动 (mm)
product_y_offset = 1010.0  # 上料区域，端头到抓料点的偏移，即物料右侧端头偏左offset的位置
depth_interference_x_offset = 20 # 上料区域，精拍点位，为了消除精拍之后的深度干扰，在精拍点位X+方向，增加偏移(mm)
fine_photo_world_angle = -3.35  # 上料区域精拍点位的末端绝对角度 (度°)

# Y方向偏差 (1010 漂移到 1030) 补偿
loading_x_back = -427.0   # 后方(X-方向，靠近基座)的料，Y值理想情况下的X坐标值
loading_x_front = -17.0   # 前方料 X值下，Y 偏了 loading_y_error_front 长度值
loading_y_error_back = 0.0 # 后方Y刚好准(0误差)，前方Y少了20mm(需要+20补偿)
loading_y_error_front = 20.0 # 前方多出了20mm，需要减去20mm拉回来，取值-20；前方少，需要加回去，+20
loading_x_error_back = 0.0 # 假设：后方料正好(能包住)，前方料因为下垂/视角放大，需要补偿 +20mm 才能包住
loading_x_error_front = 20.0


# 上料架的状态地址，用于发送上料架状态数据
ADDR_PRODUCT_LOADING_RACK = 0x400C8  # 200
product_loading_rack_empty = 11  # 空料
product_loading_rack_wood_stick = 12 # 【新增】等待取垫木

# 上料料架复位信号监听地址
ADDR_PRODUCT_LOADING_RACK_RESET = 0x400C9  # 202
product_loading_rack_reset = 10  # 空料复位信号，plc发出
product_loading_rack_reset_ack = 13  # 上位机复位完成，上位机回复plc

# 尺寸检测udp配置
inspection_udp_ip = '192.168.0.5'
inspection_udp_port = 8501
inspection_udp_local_port = 8500
