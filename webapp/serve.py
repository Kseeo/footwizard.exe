"""프로덕션 진입점 -- app.py를 브라우저 없이 waitress로 상시 대기시킨다.

systemd(footwizard.service)가 이 파일을 실행한다. 2단계 렌더링(pyglet/GLX)에는
가상 디스플레이가 필요하므로 DISPLAY가 Xvfb(footwizard-xvfb.service)를
가리켜야 한다 -- 안 되면 그 서비스가 떠 있는지 확인. HOST/PORT 환경변수로
바인딩 주소 변경 가능(기본 0.0.0.0:5050, 인증 없음).
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from waitress import serve  # noqa: E402

import app as wizard_app  # noqa: E402

HOST = os.environ.get("HOST", "0.0.0.0")
PORT = int(os.environ.get("PORT", "5050"))
THREADS = int(os.environ.get("WAITRESS_THREADS", "8"))


def main() -> int:
    print(
        f"[serve] {HOST}:{PORT} 에서 대기 시작 (DISPLAY={os.environ.get('DISPLAY') or '(없음 -- 2단계 렌더링 실패함)'})",
        flush=True,
    )
    serve(wizard_app.app, host=HOST, port=PORT, threads=THREADS)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
