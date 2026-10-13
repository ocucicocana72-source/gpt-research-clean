#!/usr/bin/env python3
"""
GPT-Researcher Backend Server Startup Script

Run this to start the research API server.
"""

import uvicorn
import os
import sys

# 将 backend 目录加入 Python 路径。
backend_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, backend_dir)

if __name__ == "__main__":
    # 切换到 backend 目录。
    os.chdir(backend_dir)
    
    # 启动服务。
    uvicorn.run(
        "server.app:app",
        host="0.0.0.0", 
        port=8000, 
        reload=True,
        log_level="info"
    )


