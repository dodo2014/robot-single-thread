import os
import json
import time
from datetime import datetime, timedelta

from src.utils.logger import logger
from src.utils.path_helper import get_base_path
from src.consts import const


class ProductionCounter:
    """生产 KPI 统计 (今日产量 / 当前节拍)

    目标:
      1. 统计"今日产量": 每个周期放料成功 (handle_process_0x40092 成功完成) 时 +1
      2. 统计"当前节拍": 机械臂完成一次上下料动作的耗时
         (相邻两次放料完成时间差 - 机床加工时间)
      3. 数据持久化到 kpi.json, 程序重启后今日产量仍能恢复显示, 且可事后查看

    实现方法:
      - 内存中维护 today_count / cycle_samples, 每次 record_one() 落盘
      - 落盘使用"临时文件 + os.replace"原子替换, 防止进程被杀导致 json 损坏
      - 跨日处理: 每次操作前比对日期, date 与今天不符则整体归零 (产量清零、节拍样本清空)
      - 节拍用滑动平均 (最近 cycle_window_size 个有效样本), 抗抖动
      - 异常样本过滤: 单件节拍 > cycle_sample_max_s 视为含暂停/人工干预, 剔除不参与平均
      - 全部异常仅记日志不抛出, 统计逻辑绝不能影响产线流程
    """

    def __init__(self):
        # kpi 文件路径 (项目根目录, 与 loading_index.json 同级)
        self.state_file = os.path.join(get_base_path(), const.kpi_file_name)

        # 当日日期字符串 (yyyy-mm-dd), 用于跨日归零判断
        self.date = datetime.now().strftime("%Y-%m-%d")
        # 今日累计产量
        self.today_count = 0
        # 节拍样本列表 (已扣除机床加工时间, 秒), 保存最近 cycle_window_size 个有效样本
        self.cycle_samples = []
        # 上一次放料完成的时间戳 (epoch 秒), 用于计算本次节拍
        self.last_piece_time = None
        # 每日历史统计, 结构: { "yyyy-mm-dd": {"count": int, "cycle_time": float} }
        # 保存最近 kpi_history_days 天 (默认一个月), 供事后查看/对账
        self.history = {}

        self._load()

    # ================================================================
    #  文件读写
    # ================================================================

    def _load(self):
        """启动时加载 kpi.json; 文件不存在/损坏/跨日时回退为初始状态

        跨日场景: 昨天的数据已保存在 history 中, 只归零"今日"的累计状态
        (today_count / cycle_samples / last_piece_time), 历史数据保留并按天数清理
        """
        if not os.path.exists(self.state_file):
            return
        try:
            with open(self.state_file, "r", encoding="utf-8") as f:
                data = json.load(f)

            # 历史数据 (兼容旧文件: 没有 history 字段时为 {})
            self.history = data.get("history", {})

            # 跨日判断: 文件里记录的日期不是今天, 则今日状态整体归零
            # (昨天及更早的数据已保存在 history 中, 不会被清除)
            cross_day = data.get("date") != self.date
            if cross_day:
                logger.info(f"[KPI] 跨日检测: 文件日期 {data.get('date')} != 今日 {self.date}, 今日产量归零")
                self.today_count = 0
                self.cycle_samples = []
                self.last_piece_time = None
            else:
                self.today_count = int(data.get("today_count", 0))
                self.cycle_samples = [float(x) for x in data.get("cycle_samples", [])]
                last_ts = data.get("last_piece_time")
                if last_ts:
                    try:
                        self.last_piece_time = float(last_ts)
                    except (TypeError, ValueError):
                        self.last_piece_time = None

            # 加载后统一在内存中清理过期历史 (跨日分支随后会写盘持久化)
            self._prune_history()
            if cross_day:
                self._save()

            logger.info(f"[KPI] 加载历史统计: 今日产量={self.today_count}, 节拍样本={self.cycle_samples}, "
                        f"历史天数={len(self.history)}")
        except Exception as e:
            logger.error(f"[KPI] 读取统计文件失败, 使用初始状态: {e}")

    def _save(self):
        """原子写盘: 先写临时文件再 os.replace, 避免写入中途崩溃损坏 json

        每次落盘都会同步更新 history 中当天的累计值, 保证:
          - 程序当天中途崩溃/重启, 已生产的数据也已写入历史
          - 跨日后, 昨天的数据已留在 history 中, 可事后查看
        """
        try:
            # 同步当天累计值到历史 (节拍取当前滑动平均, 无样本时为 0.0)
            cycle_time = self.get_snapshot()[1]
            self.history[self.date] = {
                "count": self.today_count,
                "cycle_time": round(cycle_time, 1),
            }
            # 清理过期历史 (只保留最近 kpi_history_days 天)
            self._prune_history()

            data = {
                "date": self.date,
                "today_count": self.today_count,
                "cycle_samples": self.cycle_samples[-const.cycle_window_size:],
                "last_piece_time": self.last_piece_time,
                "history": self.history,
            }
            tmp_file = self.state_file + ".tmp"
            with open(tmp_file, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=4, ensure_ascii=False)
            os.replace(tmp_file, self.state_file)
        except Exception as e:
            logger.error(f"[KPI] 保存统计文件失败: {e}")

    def _prune_history(self):
        """清理超过 kpi_history_days 天的历史条目

        日期为 yyyy-mm-dd 格式, 字符串字典序即时间序, 可直接按字符串比较过滤
        """
        try:
            threshold = (datetime.now() - timedelta(days=const.kpi_history_days)).strftime("%Y-%m-%d")
            before = len(self.history)
            self.history = {d: v for d, v in self.history.items() if d >= threshold}
            if len(self.history) != before:
                logger.info(f"[KPI] 历史数据清理: {before} -> {len(self.history)} 天 (保留 {const.kpi_history_days} 天)")
        except Exception as e:
            logger.error(f"[KPI] 清理历史数据失败: {e}")

    # ================================================================
    #  对外接口
    # ================================================================

    def record_one(self):
        """记录一件产品完成

        返回:
            (today_count, cycle_time)
            - today_count: 今日累计产量
            - cycle_time: 本次节拍(秒), 已完成机床加工时间扣除;
              首件或节拍样本不足时返回 0.0, 由调用方决定显示 "--"
        """
        try:
            now = time.time()
            # 跨日归零: 生产跨过午夜时, 新的一天重新计数
            today = datetime.now().strftime("%Y-%m-%d")
            if today != self.date:
                logger.info(f"[KPI] 跨日归零: {self.date} -> {today}")
                self.date = today
                self.today_count = 0
                self.cycle_samples = []
                self.last_piece_time = None

            # 产量 +1
            self.today_count += 1

            # 计算节拍样本: 本次与上一次放料完成的时间差, 扣除机床加工时间
            cycle_time = 0.0
            if self.last_piece_time is not None:
                raw_gap = now - self.last_piece_time - const.machine_process_time_s
                # 异常过滤: 差值超过合理上限(如含暂停/人工干预)则剔除该样本
                if raw_gap > const.cycle_sample_max_s:
                    logger.warning(f"[KPI] 节拍样本 {raw_gap:.1f}s 超过上限 {const.cycle_sample_max_s}s, 判定为含暂停/异常, 剔除")
                else:
                    self.cycle_samples.append(max(raw_gap, 0.0))
                    # 只保留最近 cycle_window_size 个样本
                    self.cycle_samples = self.cycle_samples[-const.cycle_window_size:]
                    # 滑动平均
                    cycle_time = sum(self.cycle_samples) / len(self.cycle_samples)

            # 更新上次放料时间并落盘
            self.last_piece_time = now
            self._save()

            logger.info(
                f"[KPI] piece+1 今日产量={self.today_count} 节拍={cycle_time:.1f}s "
                f"(样本数={len(self.cycle_samples)})"
            )
            return self.today_count, cycle_time
        except Exception as e:
            # 统计异常绝不抛出, 避免影响产线流程
            logger.error(f"[KPI] 统计异常: {e}")
            return self.today_count, 0.0

    def get_snapshot(self):
        """供 HMI 启动时读取当前状态

        返回:
            (today_count, cycle_time)
            - today_count: 今日累计产量
            - cycle_time: 最近有效节拍(秒), 无样本时为 0.0 (界面显示 "--")
        """
        if self.cycle_samples:
            cycle_time = sum(self.cycle_samples) / len(self.cycle_samples)
        else:
            cycle_time = 0.0
        return self.today_count, cycle_time

    def get_history(self):
        """返回每日历史统计 (供事后查看/对账)

        返回:
            dict: 按日期排序的 { "yyyy-mm-dd": {"count": int, "cycle_time": float} }
        """
        return dict(sorted(self.history.items()))


if __name__ == "__main__":
    # 手动测试入口: uv run python -m src.utils.kpi_counter
    counter = ProductionCounter()
    print(f"启动快照: 产量={counter.today_count}, 节拍={counter.get_snapshot()}")
    for i in range(8):
        count, ct = counter.record_one()
        time.sleep(0.5)
        print(f"第{i+1}件: 产量={count}, 节拍={ct:.2f}s")
    print(f"历史数据 ({len(counter.get_history())} 天):")
    for date, stat in counter.get_history().items():
        print(f"  {date}: 产量={stat['count']}, 节拍={stat['cycle_time']}s")
