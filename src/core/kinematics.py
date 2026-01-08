# -*- coding: utf-8 -*-
import math
import sys

import numpy as np
# import sympy as sp
from sympy import solve, sin, cos, symbols, Eq
from src.utils.logger import logger


class ScaraKinematics:
    """SCARA机械臂运动学计算类"""

    @staticmethod
    def inverse_kinematics(xe, ye, ze, te, l1, l2, z0, nn3, config_type='elbow_up'):
        """
        计算SCARA机械臂的逆运动学解
        此方法求得的the2, 是第二根连杆相对于世界坐标系 X 轴的绝对角度
        """
        # 定义符号变量
        the1, the2, the3, th4 = symbols('the1 the2 the3 th4')

        # 弧度与角度转换
        pp = np.pi / 180
        ep = te * pp  # 转换为弧度

        try:
            # logger.debug(f"开始逆运动学求解 - 目标坐标: X={xe}, Y={ye}, Z={ze}, 角度={te}°")

            # 建立运动学方程
            eq1 = Eq(xe, l1 * cos(the1) + l2 * cos(the2))
            eq2 = Eq(ye, l1 * sin(the1) + l2 * sin(the2))
            eq3 = Eq(ze, z0 + the3 / nn3)  # 电机转动 the3 圈，Z轴下降 the3/nn3 mm，丝杆导程的倒数
            eq4 = Eq(th4, ep)

            # 求解方程
            solutions_12 = solve([eq1, eq2], (the1, the2))  # 求解关节1和2（平面二连杆）
            solution_3 = solve(eq3, the3)[0]  # 求解丝杆转动圈数
            solution_4 = solve(eq4, th4)[0]  # 求解关节4（独立方程）

            logger.info(f"solution 12 is : {solutions_12}")
            logger.info(f"solution 3 is : {solution_3}")
            logger.info(f"solution 4 is : {solution_4}")

            # 构型选择逻辑
            if len(solutions_12) > 1:
                if config_type == 'elbow_up':
                    # 选择肘部向上构型（通常θ₂ > 0）
                    sol1, sol2 = solutions_12[0] if float(solutions_12[0][1]) > 0 else solutions_12[1]
                else:  # elbow_down
                    # 选择肘部向下构型（通常θ₂ < 0）
                    sol1, sol2 = solutions_12[0] if float(solutions_12[0][1]) < 0 else solutions_12[1]
            else:
                sol1, sol2 = solutions_12[0]

            # 只返回一组解
            theta1 = float(sol1) / pp
            theta2 = float(sol2) / pp
            motor_turns = float(solution_3)
            z_displacement = motor_turns / nn3
            theta4 = float(solution_4) / pp

            results = []
            results.append({
                'config': config_type,
                'the1': theta1,
                'the2': theta2,
                'the3': z_displacement,
                'th4': theta4
            })

            return results[0] if results else None

        except Exception as e:
            logger.error(f"逆运动学求解错误: {e}", exc_info=True)
            raise Exception(f"逆运动学求解错误: {e}")

    @staticmethod
    def inverse_kinematics_v2(xe, ye, ze, te, l1, l2, z0, nn3, config_type='elbow_up'):
        """
        计算SCARA机械臂的逆运动学解 (使用几何解析法), 高性能版
        此方法求得的the2, 定义为第二根连杆相对于第一根连杆的相对角度

        解析法（及标准机器人学）的定义:
        在标准的串联 SCARA 机器人运动学（DH参数法）中，θ2定义为第二根连杆相对于第一根连杆的相对角度。
        x=L1cos(θ1)+L2cos(θ1+θ2)

        数学验证:
        如果我们的推论是正确的，那么：相对角度(θ2_relative)=绝对角度(θ2_absolute)−第一轴角度(θ1)

        假设目标坐标: X=40.0, Y=60.0, Z=40.0, 角度=95.0°
        sumpy求得：{'config': 'elbow_up', 'the1': -12.555775311193553, 'the2': 125.17564025923399, 'the3': 10.0, 'th4': 94.99999999999983}
        解析法求得：{'config': 'elbow_up', 'the1': -12.55577531119356, 'the2': 137.73141557042752, 'the3': 10.0, 'th4': 95.0}

        SymPy 的 the2 (绝对): 125.1756°
        SymPy 的 the1: -12.5557°

        计算相对差值：125.1756−(−12.5557)=125.1756+12.5557=137.7313°
        这个结果（137.73°）正好等于解析法算出来的 the2

        """
        try:
            # 1. 工作空间检查
            r_sq = xe ** 2 + ye ** 2
            r = math.sqrt(r_sq)

            # 允许微小的浮点误差 (epsilon)
            if r > (l1 + l2) + 0.001:
                error_msg = f"目标点超出工作空间! 目标距离: {r:.2f}, 最大臂展: {l1 + l2}"
                logger.error(error_msg)
                return None  # 或者 raise Exception(error_msg)

            # 避免原点奇点
            if r < 0.001:
                # 当处于原点时，角度可以是任意值，这里设为0
                return {
                    'config': config_type,
                    'the1': 0.0,
                    'the2': 0.0,
                    'the3': (z0 + ze) * nn3,
                    'th4': te
                }

            # 2. 计算关节2 (Theta 2) - 利用余弦定理
            # cos(theta2) = (x^2 + y^2 - l1^2 - l2^2) / (2 * l1 * l2)
            cos_theta2 = (r_sq - l1 ** 2 - l2 ** 2) / (2 * l1 * l2)

            # 数值稳定性处理（防止浮点误差导致略微超过1或-1）
            cos_theta2 = max(min(cos_theta2, 1.0), -1.0)

            # sin(theta2) = +/- sqrt(1 - cos^2)
            # 手系选择：elbow_up 通常对应 theta2 > 0 (或 < 0 取决于坐标系定义，通常 SCARA 左手系/右手系不同)
            # 在标准数学定义中，elbow_up (肘部突出) 通常意味着两个连杆形成夹角。
            # 这里我们定义：elbow_up 取正解，elbow_down 取负解
            if config_type == 'elbow_up':
                theta2 = math.acos(cos_theta2)  # 结果范围 [0, pi]
            else:
                theta2 = -math.acos(cos_theta2)  # 结果范围 [-pi, 0]

            # 3. 计算关节1 (Theta 1)
            # alpha = atan2(y, x)
            # beta = atan2(l2 * sin(theta2), l1 + l2 * cos(theta2))
            # theta1 = alpha - beta

            k1 = l1 + l2 * cos_theta2
            k2 = l2 * math.sin(theta2)

            alpha = math.atan2(ye, xe)
            beta = math.atan2(k2, k1)

            theta1 = alpha - beta

            # 4. 计算 Z轴 (Theta 3 - 假设是圈数)
            # z0 - (theta3 / nn3) = ze  =>  theta3 = (z0 - ze) * nn3
            theta3 = (z0 + ze) * nn3
            motor_turns = float(theta3)
            z_displacement = motor_turns / nn3

            # 5. 计算 关节4 (Theta 4 - 姿态角)
            # 这里的逻辑取决于机械臂的R轴是相对基座还是相对末端连杆
            # 如果 te 是绝对角度：
            # theta1 + theta2 + theta4_rel = te => theta4_rel = te - theta1 - theta2
            # 原始代码 eq4 = sp.Eq(th4, ep) ，是直接控制第四轴电机达到绝对角度或相对角度
            # 保持与原代码逻辑一致：直接将目标角度作为结果
            theta4 = te

            # 转换为角度制 (如果输入已经是角度，输出根据需要转换)
            # 注意：math库计算的是弧度，如果控制器需要角度，请转换
            results = {
                'config': config_type,
                'the1': math.degrees(theta1),  # 转为度
                'the2': math.degrees(theta2),  # 转为度
                'the3': z_displacement,  # 圈数/距离
                'th4': theta4  # 假设输入已经是度
            }

            # logger.info(f"逆解成功: {results}")
            return results

        except Exception as e:
            logger.error(f"几何法逆解算错误: {e}", exc_info=True)
            raise e

    def forward_kinematics(self, the1, the2, the3, th4, l1, l2, z0, nn3):
        """
        计算SCARA机械臂的正运动学解 (Forward Kinematics)
        根据关节角度/位置推算末端笛卡尔坐标。

        参数:
        the1: 关节1角度 (度)
        the2: 关节2角度 (度, 相对角度)
        the3: Z轴关节值 (对应逆解输出的 the3, 即 z0 + ze 的位移值, 或者电机圈数/脉冲，需根据实际情况定)
              *本函数默认假设输入与逆解输出单位一致，即 (z0 + ze)*
        th4 : 关节4角度 (度)
        l1, l2: 连杆长度
        z0  : Z轴基础高度
        nn3 : Z轴传动系数 (如果the3是位移值，此参数可能只用于校验；如果the3是脉冲，用于换算)

        返回:
        {
            'x': float,
            'y': float,
            'z': float,
            'r': float,
            'config': str ('elbow_up' or 'elbow_down')
        }
        """
        try:
            # 1. 角度转换为弧度
            t1_rad = math.radians(the1)
            t2_rad = math.radians(the2)

            # 2. 计算平面坐标 X, Y
            # 公式:
            # X = L1 * cos(θ1) + L2 * cos(θ1 + θ2)
            # Y = L1 * sin(θ1) + L2 * sin(θ1 + θ2)
            # 注意：因为 θ2 是相对角度，所以第二杆的绝对角度是 (θ1 + θ2)

            x = l1 * math.cos(t1_rad) + l2 * math.cos(t1_rad + t2_rad)
            y = l1 * math.sin(t1_rad) + l2 * math.sin(t1_rad + t2_rad)

            # 3. 计算 Z 轴坐标 (Ze)
            # 逆解逻辑: the3_output = (z0 + ze)
            # 所以正解逻辑: ze = the3_input - z0
            # (注：如果输入的 the3 是电机脉冲或圈数，则 ze = (the3 / nn3) - z0)

            # 这里假设输入的是与逆解输出一致的“位移量”
            z = the3 - z0

            # 如果输入的 the3 是电机圈数，请使用下面这行：
            # z = (the3 / nn3) - z0

            # 4. 计算 R 轴角度 (Te)
            # 逆解逻辑: th4 = te (直接映射)
            r = th4

            # 5. 判断当前的姿态 (Config)
            # 对于 SCARA，通常根据 θ2 的正负来判断
            # 左手系/右手系定义可能不同，但通常 θ2 > 0 为一种姿态，θ2 < 0 为另一种
            config = 'elbow_up' if the2 >= 0 else 'elbow_down'

            results = {
                'x': round(x, 4),
                'y': round(y, 4),
                'z': round(z, 4),
                'r': round(r, 4),
                'config': config
            }

            return results

        except Exception as e:
            logger.error(f"正运动学求解错误: {e}", exc_info=True)
            return None

    @staticmethod
    def calculate_best_inverse_kinematics(xe, ye, ze, te, l1, l2, z0, nn3, current_j2):
        """
        智能逆运动学求解：根据当前机械臂状态，自动选择最佳的手系姿态 (Elbow Up/Down)

        :param current_j2: 当前机械臂 J2 轴的角度 (用于判断应该保持什么姿态)
        :return: 最佳的逆解结果 dict
        """
        # 定义 J2 轴的物理软限位 (非常重要！根据你的机械实际情况修改)
        # 假设你的机械臂 J2 活动范围是 -145 到 +145 度
        J2_LIMIT_MAX = 113
        J2_LIMIT_MIN = -113

        # 1. 尝试计算 Elbow Up
        res_up = ScaraKinematics.inverse_kinematics_v2(
            xe, ye, ze, te, l1, l2, z0, nn3, config_type='elbow_up'
        )

        # 2. 尝试计算 Elbow Down
        res_down = ScaraKinematics.inverse_kinematics_v2(
            xe, ye, ze, te, l1, l2, z0, nn3, config_type='elbow_down'
        )

        # 3. 校验合法性 (是否在限位范围内)
        valid_up = False
        if res_up and (J2_LIMIT_MIN <= res_up['the2'] <= J2_LIMIT_MAX):
            valid_up = True

        valid_down = False
        if res_down and (J2_LIMIT_MIN <= res_down['the2'] <= J2_LIMIT_MAX):
            valid_down = True

        # 4. 决策逻辑
        if not valid_up and not valid_down:
            logger.error(f"目标点不可达：Up/Down 解均超出限位或无解")
            return None

        if valid_up and not valid_down:
            return res_up

        if not valid_up and valid_down:
            return res_down

        # 5. 两个都合法，选择"离当前状态最近"的那个 (避免大甩臂)
        diff_up = abs(res_up['the2'] - current_j2)
        diff_down = abs(res_down['the2'] - current_j2)

        if diff_up <= diff_down:
            logger.info(f"自动选择姿态: Elbow Up (变动 {diff_up:.2f}°)")
            return res_up
        else:
            logger.info(f"自动选择姿态: Elbow Down (变动 {diff_down:.2f}°)")
            return res_down

# ================= 验证代码 =================
if __name__ == "__main__":
    # 使用提供的逆解结果作为输入，验证是否能算回原来的坐标
    # 目标: X=40.0, Y=60.0, Z=40.0, 角度=95.0
    # 逆解结果: the1=-12.5558, the2=137.7314, the3=40.0(假设z0=0) 或 90.0(假设z0=50), th4=95.0

    # 假设参数
    L1 = 850
    L2 = 650.0
    Z0 = 0.0  # 假设 Z0 为 0
    NN3 = 0.05

    # 输入逆解算出来的关节角
    t1 = -36.75
    t2 = 44.62
    t3 = -2.72  # 对应 ze=40.0, z0=0
    t4 = 0.0

    print(f"--- 正运动学验证 ---")
    print(f"输入关节角: J1={t1:.4f}, J2={t2:.4f}, J3={t3}, J4={t4}")

    obj = ScaraKinematics()

    fk_res = obj.forward_kinematics(t1, t2, t3, t4, L1, L2, Z0, NN3)

    print(f"计算结果: {fk_res}")
    print("预期目标: {'x': 1324.9369, 'y': -419.5327, 'z': -2.72, 'r': 0.0, 'config': 'elbow_up'}")

    ik_res = obj.inverse_kinematics_v2(1324.9369, -419.5327, -2.72, 0.0, L1, L2, Z0, NN3)
    print(f"逆解结果：{ik_res}")

    sys.exit(1)