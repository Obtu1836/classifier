import yaml
from pathlib import Path

def _load_config():
    # Path(__file__) 获取当前文件路径
    # .resolve() 获取绝对路径
    # .parent 是 utils/ 目录，再一个 .parent 就是项目根目录
    config_path = Path(__file__).resolve().parent.parent / 'config.yaml'
    
    if not config_path.exists():
        raise FileNotFoundError(f"未找到配置文件: {config_path}")
        
    # pathlib 的 open() 可以直接通过 utf-8 打开
    with config_path.open('r', encoding='utf-8') as f:
        return yaml.safe_load(f)

# 暴露给外部的全局变量
cfg = _load_config()