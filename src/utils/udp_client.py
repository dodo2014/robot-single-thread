import socket
import time
from src.utils.logger import logger


class UdpClient:
    def __init__(self, local_port, remote_ip, remote_port):
        """
        初始化 UDP 通信客户端
        :param local_port: 本地监听端口
        :param remote_ip: 目标服务器 IP
        :param remote_port: 目标服务器端口
        """
        self.local_port = local_port
        self.remote_ip = remote_ip
        self.remote_port = remote_port
        self.sock = None
        self._init_socket()

    def _init_socket(self):
        try:
            self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            # 开启端口复用，防止程序异常退出后端口被占用
            self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            # 绑定本地端口用于接收回复
            self.sock.bind(('0.0.0.0', self.local_port))
            logger.info(f"UDP 客户端初始化成功，已绑定本地端口: {self.local_port}")
        except Exception as e:
            logger.error(f"初始化 UDP 失败 (端口 {self.local_port}): {e}")
            if self.sock:
                self.sock.close()
            self.sock = None

    def send_msg(self, msg: str) -> bool:
        """发送字符串消息到目标服务器"""
        if not self.sock:
            logger.error("UDP Socket 未初始化，无法发送")
            return False
        try:
            self.sock.sendto(msg.encode('utf-8'), (self.remote_ip, self.remote_port))
            logger.info(f"发送 UDP: {msg} -> {self.remote_ip}:{self.remote_port}")
            return True
        except Exception as e:
            logger.error(f"发送 UDP 失败: {e}")
            return False

    def wait_for_response(self, expected_msg="OK", timeout_sec=30.0,
                          check_estop_func=None, is_running_func=None) -> bool:
        """
        阻塞等待目标回复，并支持急停打断
        :param expected_msg: 期望收到的消息内容
        :param timeout_sec: 超时时间 (秒)
        :param check_estop_func: 急停检测回调函数
        :param is_running_func: 线程运行状态回调函数
        """
        if not self.sock:
            return False

        # 设置底层 socket 超时时间为 0.5 秒，以保证能高频检查急停
        self.sock.settimeout(0.5)
        start_time = time.time()

        while True:
            # 1. 检查主程序是否退出
            if is_running_func and not is_running_func():
                return False

            # 2. 优先检查急停
            if check_estop_func and check_estop_func():
                logger.warning("UDP 等待期间触发急停，终止等待")
                return False

            # 3. 检查总体超时
            if time.time() - start_time > timeout_sec:
                logger.error(f"等待 UDP 响应超时 (超过 {timeout_sec} 秒)")
                return False

            # 4. 尝试接收数据
            try:
                data, addr = self.sock.recvfrom(1024)
                msg = data.decode('utf-8').strip()

                if msg == expected_msg:
                    return True
                else:
                    logger.warning(f"收到非预期的 UDP 消息: {msg}，继续等待 '{expected_msg}'")

            except socket.timeout:
                # 0.5秒超时是正常的，直接 continue 进入下一轮检查急停
                continue
            except Exception as e:
                logger.error(f"UDP 接收数据异常: {e}")
                return False

    def close(self):
        """释放 Socket 资源"""
        if self.sock:
            try:
                self.sock.close()
            except Exception as e:
                logger.error(f"关闭 UDP Socket 异常: {e}")
            finally:
                self.sock = None