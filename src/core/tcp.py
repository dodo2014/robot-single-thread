# -*- coding: utf-8 -*-
import math
import traceback

from src.utils import logger
from src.core.kinematics import ScaraKinematics

"""相机坐标转换为夹爪的基座标"""

def compute_gripper_target(
        camera_data,  # [xc, yc, zc, rc] 来自 VisionSystem
        robot_state,  # [x, y, z, r] 当前PLC反馈的坐标
        elbow_config,  # 当前机械臂的elbow状态，('elbow_up' 或 'elbow_down')
        robot_joints,  # [j1, j2, j3, j4] 当前关节角

        # --- 标定参数 ---
        camera_offset,  # [dx, dy] 相机中心相对于电机中心的偏移
        gripper_offset,  # [dx, dy] 选定夹爪相对于电机中心的偏移
        z_diff,  # 夹爪指尖比相机镜头低多少 (正数)

        # --- 结构参数 ---
        robot_params,  # {l1, l2, ...}
        cam_rotation=0,  # 相机安装旋转角 (0: 图像上=机器后, 90: 图像上=机器右...)
        gripper_install_angle=0, # 如果夹爪本身装歪了，也可以传这个
        angle_offset=0, # 夹爪角度补偿值
        joint_valid=True  # 转换是否需要验证角度限位
):
    try:
        logger.info(f"camera data: {camera_data}")
        logger.info(f"robot state: {robot_state}")
        logger.info(f"elbow config: {elbow_config}")
        logger.info(f"robot joints: {robot_joints}")
        # 1. 解包
        xc, yc, zc, rc = camera_data
        curr_x, curr_y, curr_z, curr_r = robot_state
        j1, j2, j3, j4 = robot_joints
        cam_dx, cam_dy = camera_offset
        grip_dx, grip_dy = gripper_offset

        # 2. 计算当前末端绝对角度 (弧度)
        # 逆时针为正
        current_abs_angle_deg = j1 + j2 + j4
        rad_curr = math.radians(current_abs_angle_deg)
        print(f"current abs_angle: {current_abs_angle_deg}")

        # 3. 【关键修改】相机坐标 -> 法兰坐标系 (Flange Frame)
        # Orbbec: X右, Y下. Robot: X前, Y左.
        # 假设标准安装：相机正对下方，图像上方指向机器人后方(X-)
        # 图像X+ (右) -> 机器人Y- (右)
        # 图像Y+ (下) -> 机器人X- (后)

        # #########################################################################
        # # 基础映射 (未考虑额外旋转)
        # x_in_flange_raw = -yc
        # y_in_flange_raw = -xc
        #
        # # 如果相机自身还有安装旋转 theta_cam，需要再转一次
        # # 这里做简单的 2D 旋转
        # rad_cam_install = math.radians(cam_rotation)
        # x_f_rot = x_in_flange_raw * math.cos(rad_cam_install) - y_in_flange_raw * math.sin(rad_cam_install)
        # y_f_rot = x_in_flange_raw * math.sin(rad_cam_install) + y_in_flange_raw * math.cos(rad_cam_install)
        #
        # # 加上物理安装偏移
        # # 这是物料相对于电机中心的坐标（在法兰坐标系下）
        # obj_x_flange = x_f_rot + cam_dx
        # obj_y_flange = y_f_rot + cam_dy
        #
        # # 4. 法兰坐标 -> 基座坐标 (Base Frame)
        # # 将法兰坐标系旋转当前机械臂角度，并平移
        # obj_x_base = curr_x + (obj_x_flange * math.cos(rad_curr) - obj_y_flange * math.sin(rad_curr))
        # obj_y_base = curr_y + (obj_x_flange * math.sin(rad_curr) + obj_y_flange * math.cos(rad_curr))
        # #########################################################################

        # 将角度转为弧度
        # rad_cam = math.radians(cam_rotation)  # -90度 -> -1.57弧度

        # 二维旋转公式:
        # X_new = x*cos(theta) - y*sin(theta)
        # Y_new = x*sin(theta) + y*cos(theta)

        # 计算物料相对于相机中心(但在法兰坐标系方向下)的坐标
        # x_f_rot = xc * math.cos(rad_cam) - yc * math.sin(rad_cam)
        # y_f_rot = xc * math.sin(rad_cam) + yc * math.cos(rad_cam)


        # 由于相机实际安装有翻转镜像，所以直接变换；此处使用三角变换，无论如何变换，都不能实现下面的映射关系
        x_f_rot = -yc
        y_f_rot = -xc

        print(x_f_rot, y_f_rot)

        # 验证一下：
        # 如果 rot=-90: cos=0, sin=-1
        # x_new = 0 - y*(-1) = y  (相机Y+ 变成 法兰X+) -> 意味着图像下方是机器人的前方
        # y_new = x*(-1) + 0 = -x (相机X+ 变成 法兰Y-) -> 意味着图像右方是机器人的右方
        # 这与你的物理描述完美契合！

        # 加上物理安装偏移 (offset_x, offset_y)
        obj_x_flange = x_f_rot + cam_dx
        obj_y_flange = y_f_rot + cam_dy

        # 4. 法兰坐标 -> 基座坐标 (保持不变)
        obj_x_base = curr_x + (obj_x_flange * math.cos(rad_curr) - obj_y_flange * math.sin(rad_curr))
        obj_y_base = curr_y + (obj_x_flange * math.sin(rad_curr) + obj_y_flange * math.cos(rad_curr))

        # #########################################################################
        # --- 此时 obj_x_base, obj_y_base 是物料在桌子上的绝对坐标 ---

        # 5. 计算目标角度
        # 目标是让夹爪转到 rc 角度。 rc 是物料相对于相机的角度。
        # 目标绝对角度 = 当前绝对角度 + rc, 这边用-rc，因为相机给出的角度方向和基座标相反

        # target_abs_angle = current_abs_angle_deg + rc
        target_abs_angle = current_abs_angle_deg - rc  + angle_offset #

        logger.info(f"current abs_angle: {current_abs_angle_deg}, rc: {rc}, target abs_angle: {target_abs_angle}")

        # rc = rc + cam_rotation
        # phase_diff = cam_rotation - gripper_install_angle
        # target_abs_angle = current_abs_angle_deg + rc + phase_diff

        rad_target = math.radians(target_abs_angle)

        # 6. 计算电机目标坐标
        # 目标：让 "夹爪中心" 重合于 "物料中心"
        # 电机坐标 = 物料坐标 - 旋转后的夹爪偏移

        grip_off_x_world = grip_dx * math.cos(rad_target) - grip_dy * math.sin(rad_target)
        grip_off_y_world = grip_dx * math.sin(rad_target) + grip_dy * math.cos(rad_target)

        target_motor_x = obj_x_base - grip_off_x_world
        target_motor_y = obj_y_base - grip_off_y_world

        # 7. Z 轴计算
        # zc 是相机测出的深度。如果 zc=200mm, 夹爪比相机长 50mm(z_diff=50)
        # 那么还需要下降 200 - 50 = 150mm
        target_motor_z = curr_z - (zc - z_diff)

        # 8. 反算 PLC 需要的 R (J4相对角)
        # 调用逆解算 J1, J2
        ik_res = ScaraKinematics().inverse_kinematics_v2(
            target_motor_x, target_motor_y, target_motor_z, 0,
            robot_params['l1'], robot_params['l2'], robot_params['z0'], robot_params['nn3'],
            config_type=elbow_config, joint_valid=joint_valid
        )

        if not ik_res:
            return None

        new_j1 = ik_res['the1']
        new_j2 = ik_res['the2']

        print(f"new j1: {new_j1}, new j2: {new_j2}")

        # J4 = 目标绝对 - (J1 + J2)
        target_motor_r = target_abs_angle - (new_j1 + new_j2)

        logger.info(f">>>>>>>>>>>>>.target_motor_r: {target_motor_r}")
        print(f"target motor r: {target_motor_r}")
        # 归一化
        while target_motor_r > 180: target_motor_r -= 360
        while target_motor_r <= -180: target_motor_r += 360

        logger.info(f"target coord: {target_motor_x, target_motor_y, target_motor_z, target_motor_r}")

        return [target_motor_x, target_motor_y, target_motor_z, target_motor_r]
    except Exception as ex:
        logger.error(f"{ex} \n{traceback.format_exc()}")
        return None



def compute_gripper_target_v2(
        camera_data,  # [xc, yc, zc, rc] 来自 VisionSystem (相机局部坐标和物体偏转角)
        robot_state,  # [x, y, z, r] 当前PLC反馈的法兰绝对坐标 (r是法兰的转角)
        elbow_config,  # 当前机械臂的elbow状态 ('elbow_up' 或 'elbow_down')
        robot_joints,  # [j1, j2, j3, j4] 当前关节角 (本逻辑中可作为参考备用)
        # --- 标定参数 ---
        camera_offset,  # [dx, dy] 相机中心相对于电机中心的偏移 (法兰坐标系下)
        gripper_offset,  # [dx, dy] 选定夹爪相对于电机中心的偏移 (法兰坐标系下)
        z_diff,  # 夹爪指尖比相机镜头低多少 (正数)
        # --- 结构参数 ---
        robot_params  # dict: {'l1': 740, 'l2': 640, 'z0': 0, 'nn3': 丝杆参数}
):
    """
    计算基于相机视觉数据的夹爪目标基坐标及逆运动学关节角
    """
    try:
        # 1. 提取当前数据
        curr_x, curr_y, curr_z, curr_r = robot_state
        xc, yc, zc, rc = camera_data
        cam_dx, cam_dy = camera_offset
        grip_dx, grip_dy = gripper_offset

        # 将当前法兰的角度转换为弧度，用于旋转矩阵
        curr_r_rad = math.radians(curr_r)
        cos_r = math.cos(curr_r_rad)
        sin_r = math.sin(curr_r_rad)

        # ==========================================
        # 步骤 2: 计算目标物体在【未旋转法兰】坐标系下的偏移
        # ==========================================
        # 根据需求说明："相机的坐标系和基座标的坐标系关系 X=-Y，Y=-X"
        # 这意味着在法兰未旋转(r=0)时，相机看到的 xc 对应基座标的 -Y 方向，yc 对应 -X 方向
        obj_dx_local = -yc
        obj_dy_local = -xc

        # 结合相机本身的机械安装偏移量，得到物体相对法兰中心的总偏移量（局部未旋转）
        total_dx_local = cam_dx + obj_dx_local
        total_dy_local = cam_dy + obj_dy_local

        # ==========================================
        # 步骤 3: 结合当前法兰旋转角度 (r)，计算物体真实的【基座标绝对位置】
        # ==========================================
        # 使用 2D 旋转矩阵将局部偏移叠加到法兰当前绝对坐标上
        # [ cos(r)  -sin(r) ][ total_dx_local ]
        # [ sin(r)   cos(r) ][ total_dy_local ]
        obj_base_x = curr_x + (total_dx_local * cos_r - total_dy_local * sin_r)
        obj_base_y = curr_y + (total_dx_local * sin_r + total_dy_local * cos_r)

        # 计算 Z 轴物理绝对高度
        # 假设 336L (RGB-D相机) 的 zc 表示物体距离镜头的【深度】
        # 目标物体的绝对高度 = 相机镜头高度(curr_z) - 深度(zc)
        # (注：如果 zc 已经是外部视觉库换算好的绝对高度，直接改成 obj_base_z = zc 即可)
        obj_base_z = curr_z - zc

        # 计算物体的绝对旋转角度
        # 视觉 X=-Y, Y=-X 相当于做了一个关于直线 y=-x 的镜像映射。
        # 在这种映射下，相机画面中物体的角度 rc 转换到基坐标的角度公式为 -90 - rc (假设rc为逆时针正方向)
        # 加上当前法兰的旋转基础值：
        obj_base_r = curr_r  + rc

        # ==========================================
        # 步骤 4: 计算法兰的目标抓取位置 (Target Flange Position)
        # ==========================================
        # 现在我们知道了物体绝对位置 (obj_base_x, obj_base_y, obj_base_z, obj_base_r)
        # 我们要控制法兰移动，使得【夹爪的中心】正好对准这个物体。

        # 抓取时，法兰的最终角度应该与物体对齐
        target_r = obj_base_r
        target_r_rad = math.radians(target_r)
        cos_tr = math.cos(target_r_rad)
        sin_tr = math.sin(target_r_rad)

        # 反向推算：法兰目标位置 = 物体位置 - 旋转后的夹爪偏移量
        target_x = obj_base_x - (grip_dx * cos_tr - grip_dy * sin_tr)
        target_y = obj_base_y - (grip_dx * sin_tr + grip_dy * cos_tr)

        # 计算法兰的目标 Z 高度
        # 我们需要夹爪尖端的高度恰好等于物体的高度 (obj_base_z)
        # 夹爪尖端高度 = 法兰目标高度 - z_diff
        # 因此：法兰目标高度 = 物体高度 + z_diff
        target_z = obj_base_z + z_diff

        target_flange_state = {
            'x': target_x,
            'y': target_y,
            'z': target_z,
            'r': target_r
        }

        # ==========================================
        # 步骤 5: 调用逆运动学求关节角
        # ==========================================
        ik_result = ScaraKinematics().inverse_kinematics_v2(
            xe=target_x,
            ye=target_y,
            ze=target_z,
            te=target_r,
            l1=robot_params['l1'],
            l2=robot_params['l2'],
            z0=robot_params['z0'],
            nn3=robot_params['nn3'],
            config_type=elbow_config
        )

        if not ik_result:
            logger.error("目标点超出机械臂工作空间或逆解失败。")
            return target_flange_state, None

        logger.info(f"计算抓取目标成功! \n"
                    f"物体基座标: X={obj_base_x:.2f}, Y={obj_base_y:.2f}, Z={obj_base_z:.2f}, R={obj_base_r:.2f}\n"
                    f"法兰目标坐标: {target_flange_state}\n"
                    f"关节目标角度: {ik_result}")

        return target_flange_state, ik_result

    except Exception as e:
        logger.error(f"计算抓取目标坐标系转换时发生错误: {e}", exc_info=True)
        raise e


def move_forward(l1, l2, z0, nn3, xe, ye, ze, te, config_curr, distance):
    target_x, target_y, target_z, target_r = ScaraKinematics().calculate_forward_move(l1, l2, z0,
                                                                                      nn3, xe, ye, ze, te,
                                                                                      distance,
                                                                                      config_curr=config_curr)
    foward_point = {
        "name": "FP_P0",
        "coords": [
            target_x,
            target_y,
            target_z,
            target_r
        ],
        "photo": 0,
        "config": config_curr
    }
    # print(f"foward point: {foward_point}")
    return foward_point


def main():
    from src.utils.config_manager import ConfigManager

    cfg_manager = ConfigManager()
    robot_params = cfg_manager.get_robot_params()

    # 1. 获取配置
    tool_cfg = cfg_manager.get_current_tool_model()
    cam_cfg = tool_cfg.get("camera", {})
    grip_cfg = tool_cfg.get("main_gripper", {})  # 获取这组夹爪的配置

    # 2. 准备 Offset 参数
    camera_offset = [cam_cfg.get("offset_x", 0), cam_cfg.get("offset_y", 0)]

    # 【关键】这里的 Offset 是 两个夹爪的连线中点 相对于 电机中心 的偏移
    gripper_offset = [grip_cfg.get("offset_x", 0), grip_cfg.get("offset_y", 0)]
    z_diff = grip_cfg.get("z_diff", 0)
    angle_offset = grip_cfg.get("angle_offset", 0)
    # 3. 准备视觉数据和当前状态
    # vision_data: [xc, yc, zc, rc]
    # robot_state: [x, y, z, r] (PLC反馈)
    # robot_joints: [j1, j2, j3, j4] (PLC反馈)

    # vision_data = [9.17, 31.59, 749, -3.8700000000000045]
    # robot_state = [1324.2601, 7.8166, 20.0, -29.4765]
    # elbow_config = 'elbow_up'
    # robot_joints = [-14.82, 32.76, 20.00, 29.48]

    vision_data =  [28.0, 32.2, 343.0, -0.14928571428571552]
    # vision_data = [0, 0, 0, 10]
    robot_state = [227.8038, 1305.7167, -15.9826, -64.8508]
    # robot_state = [-186.7302, 1294.0801, 175.23, -116.29]
    # robot_state = [740, 640, 0, -116.29]

    elbow_config = 'elbow_down'
    # robot_joints = [97.85723114013672, -31.08187484741211, 175.23980712890625, -63.38987731933594]

    ik = ScaraKinematics().inverse_kinematics_v2(robot_state[0], robot_state[1], robot_state[2], robot_state[3],
                                                 robot_params['l1'], robot_params['l2'], robot_params['z0'],
                                                 robot_params['nn3'],
                                                 config_type=elbow_config
                                                 )
    print(f"ik: {ik}")
    robot_joints = [ik["the1"], ik["the2"], ik["the3"], ik["th4"]]
    print(f"robot joints: {robot_joints}")
    print(f'robot abs joint: {ik["the1"] + ik["the2"] + ik["th4"]}')
    # robot_joints = [94.87798309326172, -6.075453281402588, -41.95000457763672, -89.45999145507812]

    # 4. 调用计算
    target_coords = compute_gripper_target(
        camera_data=vision_data,
        robot_state=robot_state,
        elbow_config=elbow_config,
        robot_joints=robot_joints,
        camera_offset=camera_offset,
        gripper_offset=gripper_offset,  # 传入中点偏移
        z_diff=z_diff,
        robot_params=robot_params,

        angle_offset=angle_offset
        # cam_rotation=cam_cfg.get("rotation", 0)
    )

    print(f"target coords: {target_coords}")

    # target_coords_v2 = compute_gripper_target_v2(
    #     camera_data=vision_data,
    #     robot_state=robot_state,
    #     elbow_config=elbow_config,
    #     robot_joints=robot_joints,
    #     camera_offset=camera_offset,
    #     gripper_offset=gripper_offset,  # 传入中点偏移
    #     z_diff=z_diff,
    #     robot_params=robot_params,
    #     # cam_rotation=cam_cfg.get("rotation", 0)
    # )
    #
    # print(f"target coords v2: {target_coords_v2}")


    # xe, ye, ze, te = target_coords
    #
    #
    # back_coords = move_forward(
    #     robot_params['l1'], robot_params['l2'], robot_params['z0'], robot_params['nn3'],
    #     xe, ye, ze, te, elbow_config, -50)
    #
    # forward_coords = move_forward(
    #     robot_params['l1'], robot_params['l2'], robot_params['z0'], robot_params['nn3'],
    #     xe, ye, ze, te, elbow_config, 50)
    #
    # print(f"back coords: {back_coords}")
    # print(f"forward coords: {forward_coords}")

if __name__ == '__main__':
    main()
