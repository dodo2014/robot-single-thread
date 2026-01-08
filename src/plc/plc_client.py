import socket
from pymodbus.client import ModbusTcpClient
from pymodbus.exceptions import ConnectionException
from pymodbus.payload import BinaryPayloadBuilder, BinaryPayloadDecoder
from pymodbus.constants import Endian
from src.utils.logger import logger

class PLCClient:
    def __init__(self, ip, port, timeout=1.5):
        self.ip = ip
        self.port = port
        self.timeout = timeout
        self.client = None
        self.is_connected = False
        self.is_running = True

        self.connect()

    def map_modbus_address(self, full_address):
        if 0x40000 <= full_address <= 0x4FFFF:
            return full_address - 0x40000
        return full_address

    def connect(self):
        try:
            logger.info(f"正在连接PLC - {self.ip}:{self.port}")
            self.client = ModbusTcpClient(host=self.ip, port=self.port, timeout=self.timeout, retries=0)
            if self.client.connect():
                self.is_connected = True

                # === 设置 Socket 层面的 KeepAlive ===
                # 强制 Windows/Linux 每隔几秒检测一次连接是否存活
                if hasattr(self.client, 'socket') and self.client.socket:
                    sock = self.client.socket
                    # 开启 KeepAlive
                    sock.setsockopt(socket.SOL_SOCKET, socket.SO_KEEPALIVE, 1)

                    # 针对 Windows 平台的特定设置 (开启后 1秒发一次心跳，3次失败就断开)
                    # SIO_KEEPALIVE_VALS = (On/Off, Time_ms, Interval_ms)
                    if hasattr(socket, 'SIO_KEEPALIVE_VALS'):
                        sock.ioctl(socket.SIO_KEEPALIVE_VALS, (1, 2000, 1000))

                    # 针对 Linux 平台的设置 (如果你的代码跑在 Linux 上)
                    elif hasattr(socket, 'TCP_KEEPIDLE'):
                        sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_KEEPIDLE, 2)
                        sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_KEEPINTVL, 1)
                        sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_KEEPCNT, 3)

                logger.info("PLC连接成功")
                return True
            else:
                self.is_connected = False
                logger.info("PLC连接失败")
                return False
        except Exception as e:
            self.is_connected = False
            logger.info(f"PLC连接异常: {e}")
            return False

    # def ensure_connection(self):
    #     """确保连接，断线自动重连"""
    #     if not self.client.connected:  # pymodbus check
    #         logger.warning("PLC断线，尝试重连...")
    #         return self.connect()
    #     return True

    def ensure_connection(self):
        """确保连接可用，断线则重连"""
        if self.client and self.client.is_socket_open():
            return True

        logger.warning("PLC断线，尝试重连...")

        # 【修改3】尝试立即重连，不要 sleep
        # 如果是死循环重连才需要 sleep，这里是单次检查
        if self.connect():
            return True

        # 如果重连失败，再稍微 sleep 一下防止 CPU 飙升，但在单次调用中没必要
        return False

    def read_holding_registers(self, addr, count):
        if not self.ensure_connection(): return None
        try:
            # 这里的 addr 需要根据你的 PLC 映射逻辑处理
            # 比如有些 PLC 需要 -40001
            address = self.map_modbus_address(addr)
            res = self.client.read_holding_registers(address=address, count=count, slave=1)
            if not res.isError():
                return res.registers
            else:
                logger.error(f"读取失败 Addr:{addr}: {res}")
                return None
        except Exception as e:
            logger.error(f"读取异常: {e}")
            return None

    def write_register(self, addr, value):
        if not self.ensure_connection(): return False
        try:
            address = self.map_modbus_address(addr)
            res = self.client.write_register(address=address, value=int(value), slave=1)
            if not res.isError():
                return True
            logger.error(f"写入失败 Addr:{addr}: {res}")
            return False
        except Exception as e:
            logger.error(f"写入异常: {e}")
            return False

    def write_registers(self, addr, values):
        if not self.ensure_connection(): return False
        try:
            # 确保 values 都是整数
            int_values = [int(v) for v in values]
            address = self.map_modbus_address(addr)
            res = self.client.write_registers(address=address, values=int_values, slave=1)
            if not res.isError():
                return True
            logger.error(f"批量写入失败 Addr:{addr}: {res}")
            return False
        except Exception as e:
            logger.error(f"批量写入异常: {e}")
            return False

    def write_float(self, addr, value):
        if not self.ensure_connection(): return False
        try:
            address = self.map_modbus_address(addr)
            builder = BinaryPayloadBuilder(byteorder=Endian.BIG, wordorder=Endian.LITTLE)  # 大端字节序，小端字序
            builder.add_32bit_float(value)
            registers = builder.to_registers()
            res = self.client.write_registers(
                address=address,
                values=registers,
                unit=1
            )
            if not res.isError():
                return True
            logger.error(f"写浮点失败 Addr:{addr}: {res}")
            return False
        except Exception as e:
            logger.error(f"写入浮点数异常: {e}")
            return False

    @staticmethod
    def registers_to_float(registers):
        if not registers: return 0.0

        # 【关键修改】：参数必须与你写入时完全一致！
        # byteorder=Endian.BIG, wordorder=Endian.LITTLE
        decoder = BinaryPayloadDecoder.fromRegisters(
            registers,
            byteorder=Endian.BIG,
            wordorder=Endian.LITTLE
        )
        return decoder.decode_32bit_float()

    @staticmethod
    def registers_to_int32(registers):
        if not registers: return 0

        # 参数与写入时保持一致
        decoder = BinaryPayloadDecoder.fromRegisters(
            registers,
            byteorder=Endian.BIG,
            wordorder=Endian.LITTLE
        )
        return decoder.decode_32bit_int()