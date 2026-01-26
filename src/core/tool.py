# -*- coding: utf-8 -*-
import math
import traceback

from src.utils import logger
from src.core.kinematics import ScaraKinematics

"""相机坐标转换为夹爪的基座标"""


def compute_gripper_target(
        camera_data,  # [xc, yc, zc, rc] 来自 VisionSystem
        robot_state,  # [x, y, z, r] 当前PLC反馈的坐标
        elbow_config,  # 当前机械臂的elbow状态
        robot_joints,  # [j1, j2, j3, j4] 当前关节角

        # --- 标定参数 ---
        camera_offset,  # [dx, dy] 相机中心相对于电机中心的偏移
        gripper_offset,  # [dx, dy] 选定夹爪相对于电机中心的偏移
        z_diff,  # 夹爪指尖比相机镜头低多少 (正数)

        # --- 结构参数 ---
        robot_params,  # {l1, l2, ...}
        cam_rotation=0,  # 相机安装旋转角 (0: 图像上=机器后, 90: 图像上=机器右...)
        gripper_install_angle=0 # 如果夹爪本身装歪了，也可以传这个
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

        # 3. 【关键修改】相机坐标 -> 法兰坐标系 (Flange Frame)
        # Orbbec: X右, Y下. Robot: X前, Y左.
        # 假设标准安装：相机正对下方，图像上方指向机器人后方(X-)
        # 图像X+ (右) -> 机器人Y- (右)
        # 图像Y+ (下) -> 机器人X- (后)

        # 基础映射 (未考虑额外旋转)
        x_in_flange_raw = -yc
        y_in_flange_raw = -xc

        # 如果相机自身还有安装旋转 theta_cam，需要再转一次
        # 这里做简单的 2D 旋转
        rad_cam_install = math.radians(cam_rotation)
        x_f_rot = x_in_flange_raw * math.cos(rad_cam_install) - y_in_flange_raw * math.sin(rad_cam_install)
        y_f_rot = x_in_flange_raw * math.sin(rad_cam_install) + y_in_flange_raw * math.cos(rad_cam_install)

        # 加上物理安装偏移
        # 这是物料相对于电机中心的坐标（在法兰坐标系下）
        obj_x_flange = x_f_rot + cam_dx
        obj_y_flange = y_f_rot + cam_dy

        # 4. 法兰坐标 -> 基座坐标 (Base Frame)
        # 将法兰坐标系旋转当前机械臂角度，并平移
        obj_x_base = curr_x + (obj_x_flange * math.cos(rad_curr) - obj_y_flange * math.sin(rad_curr))
        obj_y_base = curr_y + (obj_x_flange * math.sin(rad_curr) + obj_y_flange * math.cos(rad_curr))

        # --- 此时 obj_x_base, obj_y_base 是物料在桌子上的绝对坐标 ---

        # 5. 计算目标角度
        # 目标是让夹爪转到 rc 角度。 rc 是物料相对于相机的角度。
        # 目标绝对角度 = 当前绝对角度 + rc
        target_abs_angle = current_abs_angle_deg + rc
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
            config_type=elbow_config
        )

        if not ik_res:
            return None

        new_j1 = ik_res['the1']
        new_j2 = ik_res['the2']

        # J4 = 目标绝对 - (J1 + J2)
        target_motor_r = target_abs_angle - (new_j1 + new_j2)

        # 归一化
        while target_motor_r > 180: target_motor_r -= 360
        while target_motor_r <= -180: target_motor_r += 360

        logger.info(f"target coord: {target_motor_x, target_motor_y, target_motor_z, target_motor_r}")
        return [target_motor_x, target_motor_y, target_motor_z, target_motor_r]
    except Exception as ex:
        logger.error(f"{ex} \n{traceback.format_exc()}")
        return None


def move_forward(l1, l2, z0, nn3, xe, ye, ze, te, config_curr, j1_curr, j2_curr, distance):
    target_x, target_y, target_z, target_r = ScaraKinematics().calculate_forward_move(l1, l2, z0,
                                                                                      nn3, xe, ye, ze, te,
                                                                                      j1_curr, j2_curr,
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

    # 3. 准备视觉数据和当前状态
    # vision_data: [xc, yc, zc, rc]
    # robot_state: [x, y, z, r] (PLC反馈)
    # robot_joints: [j1, j2, j3, j4] (PLC反馈)

    # vision_data = [9.17, 31.59, 749, -3.8700000000000045]
    # robot_state = [1324.2601, 7.8166, 20.0, -29.4765]
    # elbow_config = 'elbow_up'
    # robot_joints = [-14.82, 32.76, 20.00, 29.48]


    vision_data = [6.7, -98.98, 547, -7.980000000000004]
    robot_state = [144.1203, 989.7299, 120.0, -52.12]
    elbow_config = 'elbow_up'
    robot_joints = [41.980735778808594, 87.38981628417969, 120.0, -52.119991302490234]


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
        cam_rotation=cam_cfg.get("rotation", 0)
    )
    print(f"target coords: {target_coords}")

    xe, ye, ze, te = target_coords

    ik_res = ScaraKinematics.inverse_kinematics_v2(
        xe, ye, ze, 0,
        robot_params['l1'], robot_params['l2'], robot_params['z0'], robot_params['nn3'],
        config_type=elbow_config  # 【关键】强制保持当前姿态
    )


    j1_new = ik_res['the1']
    j2_new = ik_res['the2']

    back_coords = move_forward(
        robot_params['l1'], robot_params['l2'], robot_params['z0'], robot_params['nn3'],
        xe, ye, ze, te, elbow_config, j1_new, j2_new, -50)

    forward_coords = move_forward(
        robot_params['l1'], robot_params['l2'], robot_params['z0'], robot_params['nn3'],
        xe, ye, ze, te, elbow_config, j1_new, j2_new, 50)

    print(f"back coords: {back_coords}")
    print(f"forward coords: {forward_coords}")

if __name__ == '__main__':
    main()
