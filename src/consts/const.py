# loop日志打印频率控制，防止刷频
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
J1_LIMIT_MIN = -105

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

    0x40092: 0x40090,
    0x40096: 0x40093,
    0x40097: 0x40096,
    0x40099: 0x40098,
    0x4009C: 0x4008F,
}

# 拍照触发的动作类型，普通拍照(物料识别)/1，上料(空料判断)/2，下料(满料判断)/3，铝屑识别/4
photo_type_normal = 1  # 普通拍照
photo_type_loading = 2  # 上料
photo_type_find_head = 5  # 上料区找料头

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

# 上料区参数
product_height = 117  # 单个产品的高度(mm)
product_width = 123  # 单个产品的宽度(mm)
interval_height = 15  # 层与层之间垫木木条的高度（mm）
base_depth = 420  # 首层标准深度，最上层拍照位，深度相机与物料的距离,低于这个值，物料上沿的深度会丢失, (mm)
depth_tolerange = 30.0  # 机械臂扫描，相机拍照深度z的安全容差，允许深度数据有 30mm 的向下波动 (mm)
depth_valid_filter = 450 # 相机距离物料的拍摄高度420mm，漂移会增加到440mm，超过阈值的深度直接过滤
product_per_layer = 5  # 每层的物料数量
product_total_layers = 6  # 物料的总层数

product_y_offset = 985.0  # 上料区域，端头到抓料点的偏移，即物料右侧端头偏左offset的位置
depth_interference_x_offset = 20 # 上料区域，精拍点位，为了消除精拍之后的深度干扰，在精拍点位X+方向，增加偏移(mm)
fine_photo_world_angle = -3.35  # 上料区域精拍点位的末端绝对角度 (度°)

loading_forward_distance = 40 # 夹爪前移距离(mm)，用于包住物料

# 抓料，系统漂移+透视视差补偿
loading_x_back = -471.94   # 后方参考 X 坐标；后方(X-方向，靠近基座)的料，Y值理想情况下的X坐标值
loading_x_front = 136.12   # 前方参考 X 坐标；
loading_x_comp_back = 0.0 # 假设：后方料正好(能包住)，前方料因为下垂/视角放大，需要补偿 +20mm 才能包住
loading_x_comp_front = 25  # 前方的料，需要前移x(mm)，才能包住物料
loading_y_comp_back = 0.0 # 后方Y刚好对准(0误差)，前方Y+少了20mm(需要+20补偿)
loading_y_comp_front = 10.0 # 前方少，需要加回去，+20; 前方多出了20mm，需要减去20mm拉回来，取值-20
loading_r_comp_back = 0.0 # 假设后方的料，r角正好对齐物料, 0°表示对齐
loading_r_comp_front = 4.38 # 假设后方的料，r角有偏差，加上之后才于物料对齐，单位度°

# 上料安全高度
loading_safe_z = 600

# 下料区域参数
unloading_safe_z = 600 # 下料安全高度
# unloding_x_list = [-442, -293, -144, -5, 154] # 5个料位的绝对X坐标, 坐标差值149-150
unloding_x_list = [-328, -179, -30, 119, 268] # 5个料位的绝对X坐标, 坐标差值149-150
unloding_y = -1297    #  固定的Y坐标
unloading_layer_0_z = -546  # 第0层的放料高度，料架+50mm厚度的垫木
unloading_layer_gap = product_height + interval_height  # 产品高度 + 垫木高度
# fine_unloading_world_angle = -3.9  # 放料末端绝对角度°
fine_unloading_world_angle = -1  # 放料末端绝对角度°

# 下料排序
unloding_x_sort_asc = 1  # 下料X坐标排序，1/X从小到大排序, 2/X从大到小排序
unloding_x_sort_desc = 2  #
unloding_x_sort = unloding_x_sort_desc

# 上料架的状态地址，用于发送上料架状态数据
ADDR_PRODUCT_LOADING_RACK = 0x400C8  # 200
product_loading_rack_empty = 11  # 空料
product_loading_rack_wood_stick = 12 # 【新增】等待取垫木

# 上料料架复位信号监听地址
ADDR_PRODUCT_LOADING_RACK_RESET = 0x400C9  # 201
product_loading_rack_reset = 11  # 空料/取垫木 复位信号，plc发出
product_loading_rack_reset_ack = 12  # 上位机复位完成，上位机回复plc

# 下料架的状态地址，用于发送下料架状态数据
ADDR_PRODUCT_UNLOADING_RACK = 0x400CA      # 202
product_unloading_rack_wood_stick = 11     # 提示放垫木
product_unloading_rack_full = 12           # 下料架满料

# 下料架复位信号监听地址
ADDR_PRODUCT_UNLOADING_RACK_RESET = 0x400CB # 203
product_unloading_rack_reset = 11          # 满料/放垫木 复位信号，plc发出
product_unloading_rack_reset_ack = 12      # 上位机复位完成，上位机回复plc


# 尺寸检测udp配置
inspection_udp_ip = '192.168.0.5'
inspection_udp_port = 8501
inspection_udp_local_port = 8500
