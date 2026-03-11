import json
from typing import Dict, Any, Type, TypeVar, get_type_hints

# 定义泛型类型，用于类方法的类型提示
T = TypeVar('T', bound='JSONMappedBase')


class JSONMappingError(Exception):
    """自定义异常：JSON映射不匹配时抛出"""
    pass


class JSONMappedBase:
    """JSON与类强制映射的基类"""

    def __init__(self, **kwargs):
        # 1. 获取当前类声明的字段类型（通过类型注解）
        self._field_types = get_type_hints(self.__class__)
        # 2. 校验并绑定字段
        self._validate_and_bind_fields(kwargs)

    def _validate_and_bind_fields(self, data: Dict[str, Any]):
        """校验JSON数据并绑定到类属性"""
        # 校验必填字段（所有带类型注解的字段都是必填）
        missing_fields = [field for field in self._field_types if field not in data]
        if missing_fields:
            raise JSONMappingError(f"缺失必填字段：{', '.join(missing_fields)}")

        # 校验未知字段（不允许JSON中有类未声明的字段）
        extra_fields = [field for field in data if field not in self._field_types]
        if extra_fields:
            raise JSONMappingError(f"存在未定义的字段：{', '.join(extra_fields)}")

        # 校验字段类型并绑定属性
        for field, expected_type in self._field_types.items():
            value = data[field]
            # 处理可选类型（如 Optional[str]）和基础类型
            if not isinstance(value, expected_type):
                raise JSONMappingError(
                    f"字段 {field} 类型错误，期望 {expected_type.__name__}，实际 {type(value).__name__}"
                )
            setattr(self, field, value)

    @classmethod
    def from_json_file(cls: Type[T], file_path: str) -> T:
        """从JSON文件加载并实例化类（核心方法）"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                json_data = json.load(f)
            # 强制映射：JSON数据必须完全匹配类的字段定义
            return cls(**json_data)
        except FileNotFoundError:
            raise JSONMappingError(f"JSON文件不存在：{file_path}")
        except json.JSONDecodeError:
            raise JSONMappingError(f"JSON文件格式错误：{file_path}")

    def to_json_file(self, file_path: str, indent: int = 4):
        """将类实例反向序列化为JSON文件（保持映射一致性）"""
        # 仅序列化类声明的字段，避免冗余数据
        json_data = {field: getattr(self, field) for field in self._field_types}
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(json_data, f, indent=indent, ensure_ascii=False)


# ------------------- 示例使用 -------------------
# 1. 定义业务类（继承基类，声明字段和类型）
class User(JSONMappedBase):
    """用户类：强制绑定JSON字段"""
    id: int  # 必填，类型int
    name: str  # 必填，类型str
    email: str  # 必填，类型str
    age: int  # 必填，类型int


# 2. 准备测试JSON文件（user.json）
# 内容示例（正确）：
# {
#     "id": 1001,
#     "name": "张三",
#     "email": "zhangsan@example.com",
#     "age": 25
# }

# 3. 测试强制映射
if __name__ == "__main__":
    # 正确场景：JSON字段完全匹配类定义
    try:
        user = User.from_json_file("user.json")
        print(f"实例化成功：id={user.id}, name={user.name}")
        # 反向序列化回JSON文件
        user.to_json_file("user_copy.json")
        print("已将类实例保存为user_copy.json")
    except JSONMappingError as e:
        print(f"映射失败：{e}")

    # 错误场景示例（模拟）：
    # - JSON缺失age字段 → 抛出"缺失必填字段：age"
    # - JSON多了一个"gender"字段 → 抛出"存在未定义的字段：gender"
    # - JSON的id是字符串"1001" → 抛出"字段id类型错误，期望int，实际str"