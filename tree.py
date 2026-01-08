import pathlib


def print_dir_tree(root_path=".", exclude_dirs=None, indent=""):
    """
    递归打印目录结构。

    :param root_path: 要遍历的起始路径
    :param exclude_dirs: 需要排除的目录名称列表 (例如 ['.git', '.venv'])
    :param indent: 递归使用的缩进字符串
    """
    if exclude_dirs is None:
        exclude_dirs = [".git", "__pycache__", ".venv", ".idea", ".vscode"]

    path = pathlib.Path(root_path)

    # 确保路径存在
    if not path.exists():
        print(f"路径不存在: {root_path}")
        return

    # 获取当前目录下所有的文件和文件夹，并排序（可选）
    items = sorted(list(path.iterdir()), key=lambda x: (x.is_file(), x.name.lower()))

    for i, item in enumerate(items):
        # 检查是否在排除名单中
        if item.is_dir() and item.name in exclude_dirs:
            continue

        # 判断是否是最后一个元素，决定画“树枝”的形状
        is_last = (i == len(items) - 1)
        connector = "└── " if is_last else "├── "

        print(f"{indent}{connector}{item.name}")

        # 如果是目录，则递归进入
        if item.is_dir():
            # 下一级的缩进逻辑：如果是最后一个元素，后面就不画垂直线了
            next_indent = indent + ("    " if is_last else "│   ")
            print_dir_tree(item, exclude_dirs, next_indent)


if __name__ == "__main__":
    # --- 使用示例 ---
    # 排除 .git, .venv 和你不想看到的目录
    exclude = [".git", ".venv", "__pycache__", "build", "dist", ".uv", ".idea"]

    print(f"当前目录结构 ({pathlib.Path('.').absolute()}):")
    print(".")
    print_dir_tree(".", exclude_dirs=exclude)