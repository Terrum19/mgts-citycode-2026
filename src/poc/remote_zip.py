"""Читать zip-архив по HTTP кусками (Range), не скачивая целиком. Ссылка figshare даёт подписанный
редирект, живущий 10 с, — поэтому на каждый запрос берём свежий."""
import io

import requests


class RemoteFile(io.RawIOBase):
    def __init__(self, url: str):
        self.url, self.pos = url, 0
        r = requests.get(self._real(), headers={"Range": "bytes=0-0"}, timeout=60)
        self.size = int(r.headers["content-range"].split("/")[-1])

    def _real(self) -> str:
        r = requests.head(self.url, allow_redirects=False, timeout=60)
        return r.headers.get("location", self.url)

    def seekable(self):
        return True

    def readable(self):
        return True

    def tell(self):
        return self.pos

    def seek(self, off, whence=0):
        self.pos = [off, self.pos + off, self.size + off][whence]
        return self.pos

    def read(self, n=-1):
        if n is None or n < 0:
            n = self.size - self.pos
        if n == 0 or self.pos >= self.size:
            return b""
        end = min(self.pos + n, self.size) - 1
        for _ in range(5):
            try:
                r = requests.get(self._real(), headers={"Range": f"bytes={self.pos}-{end}"}, timeout=300)
                if r.status_code == 206:
                    self.pos += len(r.content)
                    return r.content
            except requests.RequestException:
                pass
        raise IOError("range read failed")

    def readinto(self, b):
        data = self.read(len(b))
        b[:len(data)] = data
        return len(data)
