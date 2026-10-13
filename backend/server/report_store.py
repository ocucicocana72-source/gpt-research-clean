import asyncio
import json
from pathlib import Path
from typing import Any, Dict, List


class ReportStore:
    def __init__(self, path: Path):
        """创建基于单个 JSON 文件的报告仓库，并用异步锁串行化访问。"""
        self._path = path
        self._lock = asyncio.Lock()

    async def _ensure_parent_dir(self) -> None:
        """确保报告文件的父目录存在。"""
        self._path.parent.mkdir(parents=True, exist_ok=True)

    async def _read_all_unlocked(self) -> Dict[str, Dict[str, Any]]:
        """在调用方已持锁时读取全部报告；文件无效时返回空字典。"""
        if not self._path.exists():
            return {}
        try:
            data = json.loads(self._path.read_text(encoding="utf-8"))
            if isinstance(data, dict):
                return data  # type: ignore[return-value]
        except Exception:
            return {}
        return {}

    async def _write_all_unlocked(self, data: Dict[str, Dict[str, Any]]) -> None:
        """在调用方已持锁时借助临时文件原子替换完整报告数据。"""
        await self._ensure_parent_dir()
        tmp_path = self._path.with_suffix(self._path.suffix + ".tmp")
        tmp_path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
        tmp_path.replace(self._path)

    async def list_reports(self, report_ids: List[str] | None = None) -> List[Dict[str, Any]]:
        """列出全部报告，或仅返回给定 ID 对应的报告。"""
        async with self._lock:
            data = await self._read_all_unlocked()
            if report_ids is None:
                return list(data.values())
            return [data[report_id] for report_id in report_ids if report_id in data]

    async def get_report(self, report_id: str) -> Dict[str, Any] | None:
        """按 ID 读取报告，不存在时返回 ``None``。"""
        async with self._lock:
            data = await self._read_all_unlocked()
            return data.get(report_id)

    async def upsert_report(self, report_id: str, report: Dict[str, Any]) -> None:
        """新增报告或覆盖同 ID 的已有报告。"""
        async with self._lock:
            data = await self._read_all_unlocked()
            data[report_id] = report
            await self._write_all_unlocked(data)

    async def delete_report(self, report_id: str) -> bool:
        """删除指定报告，并返回删除前是否存在。"""
        async with self._lock:
            data = await self._read_all_unlocked()
            existed = report_id in data
            if existed:
                del data[report_id]
                await self._write_all_unlocked(data)
            return existed
