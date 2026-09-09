# 리눅스에서 상시 웹 서비스로 실행

exe(`launcher.py` + PyInstaller)로 배포해 각자 PC에서 실행하는 대신,
서버 한 대에 계속 띄워두고 사내망에서 `http://<서버 IP>:5051/`으로 접속하는
방식. `webapp/serve.py`가 그 진입점이다(개발 모드 `app.py`/exe용
`launcher.py`는 그대로 두고 건드리지 않았다 -- 거긴 여전히 5050 기본값).

포트는 5050이 아니라 **5051**을 쓴다 -- GNN이 이제 이 프로세스 안에서
직접 도니(아래 "GNN 하중 변형 예측" 참고) hplAI의 독자 GNN 디버그 서버
(`api_server.py`)가 쓰던 5051이 남았고, 그 포트는 이미 방화벽에 열려
있었으므로 새로 방화벽 규칙을 추가할 필요 없이 그대로 재사용한 것 --
`api_server.py`는 그 자리를 비켜주려면 꺼둬야 한다(아래 참고).

## 왜 exe/개발 모드와 다른가

- 브라우저를 직접 안 연다(서버라 열 브라우저가 없음) -- waitress(운영용
  WSGI 서버)로 계속 대기.
- `host="127.0.0.1"` 고정이 아니라 `HOST`/`PORT` 환경변수로 바인딩 주소를
  정한다(사내망 접속이니 기본 `0.0.0.0:5051`).
- 인증 없음(내부망 신뢰 모델 -- GNN 추론 서버와 동일). 외부(인터넷)에서도
  접속해야 하면 이 구성으로는 부족하다 -- 최소 인증 + HTTPS 붙일 방법을
  먼저 정해야 한다.

## 필요한 패키지

`README.md`의 "패키지 설치" 목록에 두 가지가 더 필요하다(직접 겪은 문제라
적어둠):

- `pyglet<2` -- trimesh의 오프스크린 렌더링(`scene.save_image()`, 2단계
  방향 후보 미리보기가 씀)이 `trimesh.viewer.windowed`를 거치는데, 이게
  pyglet 2.x는 지원 안 하고 `pip install "pyglet<2"`를 요구한다.
- `fast_simplification` -- `trimesh.simplify_quadric_decimation()`(3단계
  미리보기 데시메이션)이 내부적으로 씀. 안 깔려있으면 `align_for_cut`이
  500으로 죽는다.
- `waitress` -- 이 배포 방식에서만 필요(운영용 WSGI 서버). exe/개발 모드는
  안 씀.

## Xvfb 준비(2단계 렌더링용, 리눅스만 -- Windows는 실제 화면이 있어 불필요)

2단계가 CPU 렌더링이 아니라 진짜 GLX(OpenGL) 컨텍스트를 여는 pyglet 창을
쓴다. 화면 없는 서버라 가상 디스플레이(Xvfb)가 떠 있어야 하는데, **root
권한 없이** 아래 두 conda-forge 경로는 실패했다:

- `xorg-x11-server-xvfb-cos7-x86_64` -- 오래된 `libcrypto.so.10`(OpenSSL
  1.0)을 요구하는데 최신 배포판엔 없음.
- `xorg-xvfb-server` -- 라이브러리는 다 맞물리지만 GLX 익스텐션 자체가
  안 들어있어서(`pyglet.gl.glx_info.GLXInfoException: pyglet requires an
  X server with GLX`) 못 씀.

**된 방법**: `apt-get download xvfb`(설치가 아니라 `.deb` 다운로드만이라
root 불필요) 받은 뒤 `dpkg-deb -x xvfb_*.deb <어떤 폴더>`로 풀어서 그 안의
`usr/bin/Xvfb`를 그대로 쓴다(이 호스트의 실제 라이브러리와 링크가 맞는
Ubuntu 패키지라 `ldd`로 다 풀림). 안정적인 위치에 복사해두고
`footwizard-xvfb.service`의 `<XVFB_BIN>`에 그 경로를 넣으면 된다.

```bash
apt-get download xvfb
dpkg-deb -x xvfb_*.deb /tmp/xvfb_extract
mkdir -p ~/.local/share/footwizard-xvfb
cp /tmp/xvfb_extract/usr/bin/Xvfb ~/.local/share/footwizard-xvfb/Xvfb
chmod +x ~/.local/share/footwizard-xvfb/Xvfb
```

## systemd --user 서비스 등록

root 없이 상시 실행 + 재부팅 후 자동 시작을 하려면 `loginctl
enable-linger`(본인 계정 대상이면 root 없이도 됨)로 로그아웃해도 그
사용자의 systemd 인스턴스가 안 죽게 해야 한다.

```bash
loginctl enable-linger   # 한 번만

mkdir -p ~/.config/systemd/user
# 아래 두 파일을 이 폴더에 복사하고 <XVFB_BIN>, <REPO_ROOT>, <PYTHON_BIN>을
# 실제 경로로 채운다 -- deploy/systemd/footwizard-xvfb.service,
# deploy/systemd/footwizard.service

systemctl --user daemon-reload
systemctl --user enable --now footwizard-xvfb.service footwizard.service

systemctl --user status footwizard.service   # 확인
journalctl --user -u footwizard.service -f   # 로그
```

## GNN 하중 변형 예측 -- 별도 서버 아니라 이 프로세스 안에서 직접 실행

`webapp/gnn_predict.py`가 hplAI(별도 저장소) 저장소의 `build_dataset.py ->
predict.py -> export_glb.py`를 subprocess로 그대로 이어 부른다. 예전엔 이걸
별도 Flask 서버(hplAI/glb_preprocess/api_server.py, 포트 5051)로 띄워놓고
HTTP로 불렀는데, 이제 그 서버 없이 마법사 자신의 요청 처리 안에서 바로
실행한다(코드 안 건드리고 subprocess 호출만 이 프로세스로 옮김).

GNN 스크립트는 torch/torch_geometric이 깔린 별도 conda 환경이 필요해서
그 부분만 subprocess 경계가 남아있다 -- 아래 환경변수로 그 환경/경로를
가리킨다(기본값은 이 배포 기준 경로, 다른 머신이면 맞춰야 함):

- `GNN_REPO_DIR` -- hplAI 저장소 경로(기본 `/home/hpl/ai/hplAI`)
- `GNN_PYTHON` -- torch/torch_geometric이 깔린 파이썬(기본
  `/home/hpl/miniconda3/envs/mesh/bin/python`)
- `GNN_TRAIN_DATASET_PATH`, `GNN_TRAIN_DATASET_FILE` -- 학습 통계용
  데이터셋(정규화 등에 필요, "모델"이 아니라 그 체크포인트가 학습된 조건)

모델(체크포인트) 목록은 `GNN_REPO_DIR/checkpoints_local/*.pt`를 그대로
스캔해서 뽑는다 -- 새 체크포인트를 그 폴더에 두면 재시작 없이 바로
드롭다운에 나타난다.

hplAI/glb_preprocess/api_server.py는 더는 마법사가 안 쓴다 -- 그리고
지금은 **5051 포트를 마법사가 대신 쓰므로 이 서버를 상시로 띄워두면 안
된다**(포트 충돌). 그 저장소를 독립적으로 테스트하고 싶으면 `--port`로
다른 포트를 주고 필요할 때만 켰다 끄면 된다:

```bash
conda activate mesh
python glb_preprocess/api_server.py --port 5052   # 예: 5051 대신 다른 포트
```

## 방화벽

이 구성 자체가 방화벽을 자동으로 열어주진 않는다 -- 다만 5051은 예전에
hplAI GNN 디버그 서버용으로 이미 열려 있던 포트를 그대로 재사용한
것이므로(위 참고), 이 배포에선 새로 방화벽 규칙을 추가할 필요가 없었다.
포트를 바꾸거나(`PORT` 환경변수) 다른 서버로 옮기면 그 포트가 실제로
사내망에서 열려 있는지 다시 확인해야 한다.
