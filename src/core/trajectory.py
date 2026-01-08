# -*- coding: utf-8 -*-
import threading
import numpy as np
from src.utils.logger import logger
from src.core.kinematics import ScaraKinematics


class TrajectoryThread:
    """轨迹规划线程"""

    def __init__(self, points, point_names, params, main_window=None):
        super().__init__(main_window)
        self.points = points
        self.point_names = point_names  # 路径点名称列表
        self.params = params
        self.is_running = True
        self.trajectory_segments = []  # 存储轨迹段
        self.main_window = main_window  # 保存主窗口引用
        logger.info("轨迹规划线程初始化完成")

    def generate_trapezoidal_velocity_profile(self, total_time, acceleration_time, num_points):
        """生成梯形速度曲线的时间分布"""
        try:
            if num_points <= 0:
                # 如果没有插值点，时间列表只有 [0, 总时间]
                return np.array([0.0, total_time])

            constant_time = total_time - 2 * acceleration_time
            if constant_time < 0:
                acceleration_time = total_time / 2
                constant_time = 0

            times = []
            dt = total_time / num_points
            for i in range(num_points + 1):
                t = i * dt
                times.append(t)
            return np.array(times)
        except Exception as e:
            logger.error(f"生成速度曲线错误: {e}", exc_info=True)
            raise

    def generate_trajectory_with_velocity(self, l1, l2, z0, nn3, start_joint, target_joint, start_pos, target_pos,
                                          interpolation_method='joint', num_points=10,
                                          max_velocity=None, acceleration_time=0.3):
        """生成带速度控制的轨迹"""
        try:
            cartesian_traj = []

            if num_points <= 0:
                # 直接生成 起点 -> 终点 两个点
                cartesian_traj = np.array([start_pos, target_pos])
            else:
                # 笛卡尔空间插补
                start_array = np.array(start_pos)
                target_array = np.array(target_pos)

                for i in range(num_points + 1):
                    ratio = i / num_points
                    interpolated_pos = start_array + ratio * (target_array - start_array)
                    cartesian_traj.append(interpolated_pos)
                cartesian_traj = np.array(cartesian_traj)

            trajectory = []
            for point in cartesian_traj:
                # 逆运动学求解
                ik_solution = ScaraKinematics.inverse_kinematics_v2(point[0], point[1], point[2], point[3],
                                                                 l1, l2, z0, nn3)
                if ik_solution:
                    joint_point = [ik_solution['the1'], ik_solution['the2'],
                                   ik_solution['the3'], ik_solution['th4']]
                    trajectory.append(joint_point)
            trajectory = np.array(trajectory)

            # 计算各关节运动距离和时间
            joint_distances = np.abs(trajectory[-1] - trajectory[0])
            if max_velocity is None:
                max_velocities = [180, 180, 100, 180]
            else:
                max_velocities = [max_velocity] * 4

            joint_times = []
            for i in range(4):
                if max_velocities[i] > 0:
                    time_needed = joint_distances[i] / max_velocities[i]
                    time_needed = max(time_needed, acceleration_time * 2)
                    joint_times.append(time_needed)
                else:
                    joint_times.append(1.0)

            total_time = max(joint_times)

            actual_velocities = []
            for i in range(4):
                if total_time > 0:
                    effective_time = total_time - acceleration_time
                    actual_velocity = joint_distances[i] / effective_time if effective_time > 0 else 0
                    actual_velocities.append(min(actual_velocity, max_velocities[i]))
                else:
                    actual_velocities.append(0)

            time_points = self.generate_trapezoidal_velocity_profile(total_time, acceleration_time, num_points)
            return trajectory, time_points, actual_velocities, total_time
        except Exception as e:
            logger.error(f"生成带速度控制的轨迹错误: {e}", exc_info=True)
            raise

    def run_(self):
        """执行轨迹规划"""
        threading.current_thread().name = "Calc_Thread"  # 设置线程名称
        logger.info("执行轨迹规划")

        try:
            l1 = self.params['l1']
            l2 = self.params['l2']
            z0 = self.params['z0']
            nn3 = self.params['nn3']
            num_points = self.params['num_points']
            max_velocity = self.params['max_velocity']
            acceleration_time = self.params['acceleration_time']

            logger.info("开始轨迹规划...")

            # 计算所有路径点的逆运动学解
            joint_points = []
            cartesian_points = []

            for i, (point, point_name) in enumerate(zip(self.points, self.point_names)):
                xe, ye, ze, te = point
                ik_result = ScaraKinematics.inverse_kinematics_v2(xe, ye, ze, te, l1, l2, z0, nn3)
                if ik_result:
                    joint_points.append(ik_result)
                    cartesian_points.append(point)
                    logger.info(f"点位{i + 1} [{point_name}] 计算完成")
                else:
                    raise Exception(f"点{i + 1} [{point_name}] 逆运动学求解失败")

                if not self.is_running: return

            total_segments = len(self.points) - 1
            total_interpolated_points = 0

            for seg in range(total_segments):
                if not self.is_running: return

                start_joint = joint_points[seg]
                target_joint = joint_points[seg + 1]
                start_pos = cartesian_points[seg]
                target_pos = cartesian_points[seg + 1]

                trajectory, time_points, actual_velocities, total_time = self.generate_trajectory_with_velocity(
                    l1, l2, z0, nn3, start_joint, target_joint, start_pos, target_pos,
                    interpolation_method='cartesian',
                    num_points=num_points,
                    max_velocity=max_velocity,
                    acceleration_time=acceleration_time
                )

                current_segment = []
                for i, (joint_angles, t) in enumerate(zip(trajectory, time_points)):
                    if not self.is_running: return
                    current_segment.append(joint_angles)

                    if i % 10 == 0:
                        logger.debug(f"第{seg + 1}段插补点{i + 1}: 时间={t:.2f}s")

                self.trajectory_segments.append(current_segment)
                total_interpolated_points += len(trajectory)
                logger.info(f"第{seg + 1}段轨迹完成，共{len(trajectory)}个插补点")

            # 将轨迹数据传递给主窗口
            if self.main_window:
                self.main_window.trajectory_segments = self.trajectory_segments

            logger.info(f"轨迹规划完全完成 - 总段数: {total_segments}段")

        except Exception as e:
            error_msg = f"错误: {str(e)}"
            logger.error(error_msg, exc_info=True)


    def stop(self):
        self.is_running = False


class TrajectoryV2:

    @staticmethod
    def generate_cartesian_interpolated_path(raw_points, num_inserts=7):
        """
        生成笛卡尔空间插值轨迹
        :param raw_points: 原始关键点列表 [{'coords': [x,y,z,r], 'photo': 0/1}, ...]
        :param num_inserts: 两点之间插入点的数量
        :return: 包含插值点的完整轨迹列表
        """
        if not raw_points or len(raw_points) < 2:
            return raw_points

        full_trajectory = []

        # 添加起始点
        full_trajectory.append(raw_points[0])

        # 遍历每一段 (从第0个点到倒数第2个点)
        for i in range(len(raw_points) - 1):
            p_start = raw_points[i]
            p_end = raw_points[i + 1]

            coords_s = p_start['coords']  # [x, y, z, r]
            coords_e = p_end['coords']  # [x, y, z, r]

            # 生成中间插值点
            # 总段数 = 插入数 + 1
            steps = num_inserts + 1

            for j in range(1, steps):
                ratio = j / steps

                # 线性插值公式: P = P_start + (P_end - P_start) * ratio
                # 对 x, y, z, r 四个维度同时插值
                inter_coords = []
                for k in range(4):
                    val_s = coords_s[k]
                    val_e = coords_e[k]
                    val_new = val_s + (val_e - val_s) * ratio
                    inter_coords.append(round(val_new, 2))  # 保留2位小数

                # 创建中间点对象
                # 注意：中间点通常不触发拍照 (photo=0)，也不应该是关键节点
                inter_point = {
                    "name": f"{p_start.get('name', 'P')}_{p_end.get('name', 'P')}_inter_{j}",
                    "coords": inter_coords,
                    "photo": 0,  # 插值点不拍照
                    "is_inter": True  # 标记为插值点(可选，用于日志区分)
                }
                full_trajectory.append(inter_point)

            # 添加这一段的终点 (保留原有的 photo 属性)
            full_trajectory.append(p_end)

        return full_trajectory

    @staticmethod
    def generate_joint_interpolated_path(raw_points, num_inserts=7, robot_params=None):
        """
        生成关节空间插值轨迹 (MOVJ) - 解决直线插补抖动问题

        :param raw_points: [Start_Point, End_Point]
        :param robot_params: 包含 l1, l2, z0, nn3 的字典 (必须传，用于正逆解)
        :return: 包含插值点的完整轨迹列表 (坐标依然是 XYZR 格式)
        """
        if not raw_points or len(raw_points) < 2 or not robot_params:
            return raw_points

        full_trajectory = []
        p_start = raw_points[0]
        p_end = raw_points[1]

        # 1. 获取起终点的 XYZR
        coords_s = p_start['coords']
        coords_e = p_end['coords']

        # 获取姿态配置
        # 注意：这里假设起点已经是正确的姿态，终点跟随起点或指定
        # 简单起见，我们计算逆解时需要知道当前的 config
        cfg_s = p_start.get('config', 'elbow_up')
        cfg_e = p_end.get('config', cfg_s)  # 终点默认跟随起点

        # 2. 逆解算出 Start 和 End 的关节角 (J1, J2, J3, J4)
        l1, l2 = robot_params['l1'], robot_params['l2']
        z0, nn3 = robot_params['z0'], robot_params['nn3']

        # 起点逆解
        ik_s = ScaraKinematics.inverse_kinematics_v2(
            coords_s[0], coords_s[1], coords_s[2], coords_s[3],
            l1, l2, z0, nn3, config_type=cfg_s
        )
        # 终点逆解
        ik_e = ScaraKinematics.inverse_kinematics_v2(
            coords_e[0], coords_e[1], coords_e[2], coords_e[3],
            l1, l2, z0, nn3, config_type=cfg_e
        )

        if not ik_s or not ik_e:
            # 如果逆解失败，回退到普通直线插值，防止程序崩
            return TrajectoryV2().generate_cartesian_interpolated_path(raw_points, num_inserts)

        # 提取关节角度 [J1, J2, J3, J4]
        # 注意：inverse_kinematics_v2 返回的 the3 是位移，我们需要插值这个位移
        joints_s = [ik_s['the1'], ik_s['the2'], ik_s['the3'], ik_s['th4']]
        joints_e = [ik_e['the1'], ik_e['the2'], ik_e['the3'], ik_e['th4']]

        # 添加起点
        full_trajectory.append(p_start)

        # 3. 对关节角度进行线性插值
        steps = num_inserts + 1
        for j in range(1, steps):
            ratio = j / steps
            inter_joints = []

            # 对 4 个轴进行插值
            for k in range(4):
                val = joints_s[k] + (joints_e[k] - joints_s[k]) * ratio
                inter_joints.append(val)

            # 4. 正运动学 (FK) 算回 XYZR
            # 这样生成的 XYZ 坐标，虽然连起来不是直线，但能保证电机转动最平滑
            fk_res = ScaraKinematics.forward_kinematics(
                inter_joints[0], inter_joints[1], inter_joints[2], inter_joints[3],
                l1, l2, z0, nn3
            )

            if fk_res:
                # 构造中间点
                inter_point = {
                    "name": f"J_Inter_{j}",
                    "coords": [fk_res['x'], fk_res['y'], fk_res['z'], fk_res['r']],
                    "photo": 0,
                    "config": fk_res['config'],  # FK 会自动算出是 up 还是 down
                    "is_inter": True
                }
                full_trajectory.append(inter_point)

        # 添加终点
        full_trajectory.append(p_end)

        return full_trajectory

if __name__ == "__main__":

    raw_points = [{
          "name": "P3",
          "coords": [
            1500.0,
            0.0,
            0.0,
            45.0
          ]
        },
        {
          "name": "P4",
          "coords": [
            968.0,
            462.0,
            0.0,
            0.0
          ]
        }]
    tobj = TrajectoryV2()
    full_path = tobj.generate_cartesian_interpolated_path(raw_points)
    print(full_path)
    print(len(full_path))