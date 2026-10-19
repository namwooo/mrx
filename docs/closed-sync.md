# 폐쇄망 소스코드 반입 절차 (git bundle + Git LFS + 이메일)
삼성디스플레이 기흥 캠퍼스 연구소, 인터넷이 안되는 폐쇄망으로 git 저장소를 옮기는 절차이다. 
통로는 삼성 knox 이메일이고, 한 통에 첨부할 수 있는 크기는 18MB 정도이다. (더 클수도 있음)

| 구분        | 내용                                                        |
|-------------|-------------------------------------------------------------|
| 코드이력    | git bundle로 파일 하나에 묶기                               |
| 대용량 파일 | Git LFS 원본(.git/lfs/objects)을 따로 tar로 묶기            |
| 전송        | 18MB 단위로 split → send_files.py 로 한 통에 한 파일씩 발송 |
| 무결성      | SHA-256 체크섬으로 합친 뒤 확인                             |
| 기준점      | 보내는 쪽의 closed-sync 태그 = "폐쇄망에 여기까지 반영됨"   |

- 최초 1회: 전체 소스 이력과 LFS 원본 전체를 보낸다. (크기가 큼, 분할 필요)
- 이후 매번: git tag `closed-sync` 이후 바뀐 커밋과 바뀐 LFS 파일만 보낸다.

[보내는 쪽]: git 번들 -> LFS 묶기 -> 체크섬 -> 분할 -> 메일 발송 <br>
[폐쇄망]: 합치기 -> 체크섬 확인 -> fetch/clone -> LFS 풀기 -> 적용 확인 <br>
[보내는 쪽]: closed-sync 태그 이동 <br>

## 0.사전 준비 (최초 1회)
### 0-1. 폐쇄망 git-lfs 설치
1. 보내는 쪽에서 https://github.com/git-lfs/git-lfs/releases 의 폐쇄망 PC용 파일을 받아 메일로 보낸다.
   - 폐쇄망 PC 종류 확인: uname -sm
   - Linux x86_64: git-lfs-linux-amd64-v3.x.x.tar.gz
   - macOS Apple Silicon: git-lfs-darwin-arm64-v3.x.x.zip <br>
2. 폐쇄망에서 설치한다.
```bash
tar -xzf git-lfs-linux-amd64-v3.*.tar.gz         
mkdir -p ~/.local/bin
cp git-lfs-*/git-lfs ~/.local/bin/
echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.bashrc
source ~/.bashrc
git lfs install
git lfs version
```

### 0-2. 메일 발송 준비 (보내는 쪽)
- Gmail 앱 비밀번호 발급: Google 계정 → 보안 → 2단계 인증 켜기 → https://myaccount.google.com/apppasswords
- send_files.py 상단의 FROM, PASSWORD를 채운다.
- 폐쇄망 메일 시스템의 첨부 한도가 18MB보다 작을 수 있으니, 처음엔 한 통만 보내 들어가는지 확인한다.

## 1. 최초 전체 전송 (1회)
### 1-1. 보내는 쪽: 묶기
```bash
cd ~/Projects/mrx
git status                                         # 커밋 안 된 변경이 없어야 함

OUT=~/transfer/$(date +%Y%m%d)
mkdir -p "$OUT"

# (1) 코드 이력
git bundle create "$OUT/mrx.bundle" main

# (2) main이 쓰는 LFS 원본 전체
git lfs fetch origin main
git lfs ls-files --long main | awk '{print $1}' | while read oid; do
  echo ".git/lfs/objects/${oid:0:2}/${oid:2:2}/${oid}"
done > "$OUT/lfs-files.txt"
tar -czf "$OUT/lfs.tar.gz" -T "$OUT/lfs-files.txt"
```

### 1-2. 보내는 쪽: 체크섬 -> 분할
```bash
cd "$OUT"
shasum -a 256 mrx.bundle lfs.tar.gz > SHA256SUMS

split -b 18m -a 3 -d mrx.bundle mrx.bundle.
split -b 18m -a 3 -d lfs.tar.gz lfs.tar.gz.

mkdir -p send
mv mrx.bundle.[0-9]* lfs.tar.gz.[0-9]* SHA256SUMS send/
ls send | wc -l                                    # 보낼 메일 수
```

### 1-3. 보내는 쪽: 이메일 발송
```bash
python3 send_files.py "$OUT/send" 받는주소@example.com --workers 10
```

### 1-4. 폐쇄망: 합치기 -> 확인 -> clone
받은 첨부를 모두 한 폴더에 저장한 뒤 실행합니다.

```bash
cat mrx.bundle.[0-9]* > mrx.bundle
cat lfs.tar.gz.[0-9]* > lfs.tar.gz
sha256sum -c SHA256SUMS                            
git bundle verify mrx.bundle

GIT_LFS_SKIP_SMUDGE=1 git clone mrx.bundle mrx
cd mrx
tar -xzf ../lfs.tar.gz                             # .git/lfs/objects 로 풀림
git lfs checkout                                   # 포인터 → 실제 파일
git lfs fsck                                       # 누락·손상 검사
git log -1 --oneline                               # 보낸 쪽 main과 같은 커밋인지 확인
```

### 1-5. 보내는 쪽: 기준점 표시
폐쇄망 적용을 확인한 뒤에 태그를 답니다.
```bash
git tag -f closed-sync {COMMIT}
```

## 2. 이후 업데이트 (매번)
### 2-1. 보내는 쪽: 보낼 내용 확인
```bash
cd ~/Projects/mrx
git status                                         # 커밋 안 된 변경 없는지
git log --oneline closed-sync..main                # 보낼 커밋 목록 (비어 있으면 보낼 것 없음)
```

### 2-2. 보내는 쪽: 묶기
```bash
OUT=~/transfer/$(date +%Y%m%d)
mkdir -p "$OUT"

# (1) 코드 변경분
git bundle create "$OUT/mrx-update.bundle" closed-sync..main

# (2) 바뀐 LFS 파일만
git lfs fetch origin main
git diff --name-only closed-sync main \
  | git check-attr --stdin filter | grep ': lfs$' | cut -d: -f1 > "$OUT/lfs-paths.txt"

git lfs ls-files --long main | grep -F -f "$OUT/lfs-paths.txt" | awk '{print $1}' | while read oid; do
  echo ".git/lfs/objects/${oid:0:2}/${oid:2:2}/${oid}"
done > "$OUT/lfs-files.txt"

FILES="mrx-update.bundle"
if [ -s "$OUT/lfs-files.txt" ]; then
  tar -czf "$OUT/lfs-update.tar.gz" -T "$OUT/lfs-files.txt"
  FILES="$FILES lfs-update.tar.gz"
fi

cd "$OUT"
shasum -a 256 $FILES > SHA256SUMS
ls -lh $FILES
```

- lfs-paths.txt가 비어 있으면 이번에 바뀐 LFS 파일이 없는 것이므로 번들만 보낸다.
- 파일이 18MB를 넘으면 1-2처럼 split으로 나눈다.

### 2-3. 보내는 쪽: 발송
```bash
mkdir -p send && cp $FILES SHA256SUMS send/        
python3 send_files.py "$OUT/send" 받는주소@example.com --workers 3
```

### 2-4. 폐쇄망: 적용
```bash
# 분할해서 받았다면 먼저 합치기
# cat mrx-update.bundle.[0-9]* > mrx-update.bundle

sha256sum -c SHA256SUMS
cd mrx
git bundle verify ../mrx-update.bundle

git checkout --detach                              # main 체크아웃 상태에서 main:main fetch가 거부되지 않도록
GIT_LFS_SKIP_SMUDGE=1 git fetch ../mrx-update.bundle main:main
[ -f ../lfs-update.tar.gz ] && tar -xzf ../lfs-update.tar.gz
git checkout main
git lfs checkout
git log -1 --oneline                               # 보낸 쪽 main과 같은 커밋인지 확인
```

### 2-5. 보내는 쪽: 기준점 이동
```bash
git tag -f closed-sync {COMMIT}
```