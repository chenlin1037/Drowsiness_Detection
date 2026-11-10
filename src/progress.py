"""轻量级 YOLO 风格终端进度条(无第三方依赖)。

输出样式参考 ultralytics:
    1/20  2.34G  0.0489: 100%|██████████| 1862/1862 [00:42<00:00, 44.12it/s]
"""

import shutil
import sys
import time


def _format_time(seconds: float) -> str:
    seconds = max(int(seconds), 0)
    minutes, secs = divmod(seconds, 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"


class ProgressBar:
    """单行刷新的进度条,用法接近 tqdm。"""

    BAR_LENGTH = 10  # ultralytics 默认 {bar:10}

    def __init__(
        self,
        total: int,
        desc: str = "",
        ncols: int | None = None,
        mininterval: float = 0.1,
    ) -> None:
        self.total = max(int(total), 1)
        self.n = 0
        self.desc = desc
        self.postfix = ""
        self.mininterval = mininterval
        self._last_render = 0.0
        self._start = time.time()
        self._closed = False
        self.ncols = ncols or shutil.get_terminal_size((120, 24)).columns
        self.refresh(force=True)

    def set_description(self, desc: str) -> None:
        self.desc = desc
        self.refresh()

    def update(self, n: int = 1, postfix: str | None = None) -> None:
        self.n = min(self.n + n, self.total)
        if postfix is not None:
            self.postfix = postfix
        self.refresh()

    def refresh(self, force: bool = False) -> None:
        if self._closed:
            return
        now = time.time()
        if not force and now - self._last_render < self.mininterval:
            return
        self._last_render = now

        elapsed = now - self._start
        frac = min(self.n / self.total, 1.0)
        rate = self.n / elapsed if elapsed > 0 else 0.0
        remaining = (elapsed / self.n) * (self.total - self.n) if self.n else 0.0

        filled = round(frac * self.BAR_LENGTH)
        bar = "█" * filled + " " * (self.BAR_LENGTH - filled)
        line = (
            f"{self.desc}: {frac * 100:3.0f}%|{bar}| "
            f"{self.n}/{self.total} "
            f"[{_format_time(elapsed)}<{_format_time(remaining)}, "
            f"{rate:5.2f}it/s{self.postfix}]"
        )
        sys.stdout.write("\r" + line[: self.ncols - 1] + " " * 2)
        sys.stdout.flush()

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self.refresh(force=True)
        sys.stdout.write("\n")
        sys.stdout.flush()
