import logging
import json
import os
from datetime import datetime
from pathlib import Path

class JSONResearchHandler:
    def __init__(self, json_file):
        """初始化同时保存事件流和研究内容快照的 JSON 日志处理器。"""
        self.json_file = json_file
        self.research_data = {
            "timestamp": datetime.now().isoformat(),
            "events": [],
            "content": {
                "query": "",
                "sources": [],
                "context": [],
                "report": "",
                "costs": 0.0
            }
        }

    def log_event(self, event_type: str, data: dict):
        """追加带时间戳的研究事件并立即持久化。"""
        self.research_data["events"].append({
            "timestamp": datetime.now().isoformat(),
            "type": event_type,
            "data": data
        })
        self._save_json()

    def update_content(self, key: str, value):
        """更新研究内容区的一个字段并立即持久化。"""
        self.research_data["content"][key] = value
        self._save_json()

    def _save_json(self):
        """将当前完整研究日志快照写入 JSON 文件。"""
        with open(self.json_file, 'w') as f:
            json.dump(self.research_data, f, indent=2)

def setup_research_logging():
    """配置研究专用的文本与 JSON 日志处理器。"""
    # 如果日志目录不存在则创建。
    logs_dir = Path("logs")
    logs_dir.mkdir(exist_ok=True)
    
    # 为日志文件生成时间戳。
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 创建日志文件路径。
    log_file = logs_dir / f"research_{timestamp}.log"
    json_file = logs_dir / f"research_{timestamp}.json"
    
    # 配置研究日志的文件处理器。
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
    
    # 获取并配置研究 logger。
    research_logger = logging.getLogger('research')
    research_logger.setLevel(logging.INFO)
    
    # 移除已有处理器，避免重复记录。
    research_logger.handlers.clear()
    
    # 添加文件处理器。
    research_logger.addHandler(file_handler)
    
    # 添加控制台输出的流处理器。
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
    research_logger.addHandler(console_handler)
    
    # 阻止传播到根 logger，避免日志重复。
    research_logger.propagate = False
    
    # 创建 JSON 处理器。
    json_handler = JSONResearchHandler(json_file)
    
    return str(log_file), str(json_file), research_logger, json_handler

# 提供获取 logger 和 JSON 处理器的函数。
def get_research_logger():
    """返回全局研究流程使用的命名 logger。"""
    return logging.getLogger('research')

def get_json_handler():
    """返回挂载到研究 logger 的 JSON 处理器（若已配置）。"""
    return getattr(logging.getLogger('research'), 'json_handler', None)
