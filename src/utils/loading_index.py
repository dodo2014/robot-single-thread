import os
import json
from src.utils.logger import logger
from src.utils.path_helper import get_base_path
from src.consts import const

"""上料位置索引"""
class LoadingIndex:
    def __init__(self):
        # 状态文件路径
        self.state_file = os.path.join(get_base_path(), const.loading_index_file_name)

        # 初始化时加载上次的记忆
        self.current_search_index = self.load_search_index(const.search_index_name)
        self.current_head_layer_index = self.load_search_index(const.head_layer_index_name)

    def load_search_index(self, key):
        """从文件加载搜寻进度"""
        if os.path.exists(self.state_file):
            try:
                with open(self.state_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    idx = data.get(key, 0)
                    logger.info(f"加载历史搜寻进度，当前索引: {idx}")
                    return idx
            except Exception as e:
                logger.error(f"读取状态文件失败: {e}")
        return 0

    def save_search_index(self, key, index):
        """保存搜寻进度到文件"""
        # self.current_search_index = index
        if hasattr(self, key):
            setattr(self, key, index)
        else:
            logger.warning(f"类中不存在属性 {key}，已跳过实例属性更新")

        try:
            # 读取原有数据（为了不覆盖其他可能存在的状态）
            data = {}
            if os.path.exists(self.state_file):
                with open(self.state_file, 'r', encoding='utf-8') as f:
                    try:
                        data = json.load(f)
                    except:
                        pass

            # 更新并写入
            data[key] = index
            with open(self.state_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=4)
        except Exception as e:
            logger.error(f"保存状态文件失败: {e}")

    def reset_search_index(self, key):
        """复位搜寻进度（换新料车时调用）"""
        logger.info(">>> 收到复位指令，已将料架搜寻进度归零 <<<")
        self.save_search_index(key, 0)

if __name__ == "__main__":
    lidx = LoadingIndex()
    idx = lidx.load_search_index()
    print(idx)