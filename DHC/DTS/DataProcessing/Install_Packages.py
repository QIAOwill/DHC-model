import subprocess

def install_package(packages_to_check):
    for package in packages_to_check:
        try:
            import importlib
            importlib.import_module(package)
            print(f"{package} 已安装")
        except ImportError:
            print(f"未找到 {package}，正在安装...")
            subprocess.run(["pip", "install", "-i", "https://pypi.tuna.tsinghua.edu.cn/simple", package])
            print(f"{package} 安装完成")

