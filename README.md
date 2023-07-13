# share_student

directory 구조
local:
   ./infer/infer.py
   ./train/train.py
   ./lib/digit_recognizer.py
   ./data
   ./model
   ./Dockerfile.train
   ./Dockerfile.test

build
   docker build -f ./Dockerfile.train -t trainxxx:x.x .
   docker build -f ./Dockerfile.infer -t inferxxx:x.x .

docker run
   docker run -v ./data:/data -v ./model:/model trainxxx:x.x .
   docker run -v (-d) ./model:/model -p 80:80 inferxxx:x.x .

이 상황에서의 컨테이너 폴더
inferxxx
   /app/infer.py
   /module/digit_recognizer.py
   /model                     (볼륨 공유 상황이며, 학습이 진행된 이후라면 modelfile이 들어 있음)

trainxxx
   /app/train.py
   /module/digit_recognizer.py
   /model                     (볼륨 공유 상황)
   /data/*                     (볼륨 공유 상황)


1. 수업 자료는 https://drive.google.com/drive/folders/1ndFuBjEvkukuR4xM6PlZp1bkmuLaWYx3?usp=sharing 링크를 참조하세요
2. 가상 환경에 대해서
3. 

   Python으로 작업을 하다 보면 모듈이 스파게티 상태가 되는 경우가 많습니다.
   특히 버전 충돌도 잦은 편 입니다.

   그래서 현재의 파이썬 모듈의 상태를 베이스로 하는 어떤 환경을 만들어서 작업합니다.
   이를 파이썬 가상환경이라고 하는데, 이 가상환경을 만드는 방법은 다음과 같습니다.

   python -m venv 가상환경이름 (ex: python -m venv recog)

   이렇게 하면 가상환경이름(recog)라는 이름의 디렉토리가 생깁니다.

   여기에서 source recog/bin/active 실행하면 가상환경으로 들어갑니다.
   (윈도우에서는 source없이 그냥 실행된다고 합니다.)
   deactive 명령을 입력하면 가상환경 밖으로 나옵니다.

   가상환경에서 설치한 모듈 (pip로...) 은 가상환경 밖으로 나오면 없는 녀석입니다.
   반대로 가상환경 밖에서 설치한 모듈은 가상환경 안에서 적용됩니다.
   다른 가상환경에서는 또 다른 설치 환경으로 동작하므로, 가상환경마다 모듈 격리를 할 수 있습니다.


