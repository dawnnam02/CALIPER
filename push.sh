#!/usr/bin/env bash
# CALIPER 를 깃허브에 올린다.
#
# 쓰는 법:
#   1) 먼저 한 번만:  gh auth login      (브라우저가 열린다)
#   2) 그다음:        bash push.sh              → 비공개
#                     bash push.sh public       → 공개
#
# 하는 일:
#   - 인증 확인
#   - pyproject.toml 의 USERNAME 자리를 실제 계정으로 교체하고 커밋
#   - 저장소를 만들고 푸시
#   - 이미 저장소가 있으면 원격만 붙이고 푸시
set -euo pipefail

REPO_NAME="CALIPER"
VIS="${1:-private}"

GH="$(command -v gh || true)"
if [ -z "$GH" ]; then
  GH="/c/Users/dawnn/AppData/Local/Microsoft/WinGet/Packages/GitHub.cli_Microsoft.Winget.Source_8wekyb3d8bbwe/bin/gh.exe"
fi
if [ ! -x "$GH" ] && ! command -v gh >/dev/null 2>&1; then
  echo "gh 를 찾을 수 없다. 새 터미널을 열면 PATH 가 잡힌다." >&2
  exit 1
fi

if [ "$VIS" != "public" ] && [ "$VIS" != "private" ]; then
  echo "두 번째 인자는 public 또는 private 여야 한다 (받은 값: $VIS)" >&2
  exit 2
fi

echo "== 1. 인증 확인 =="
if ! "$GH" auth status >/dev/null 2>&1; then
  cat >&2 <<'EOF'
로그인이 안 돼 있다. 먼저 이걸 실행해라 (브라우저가 열린다):

    gh auth login

  - What account?  GitHub.com
  - Protocol?      HTTPS
  - Authenticate Git with your GitHub credentials?  Yes
  - How to login?  Login with a web browser

끝나면 이 스크립트를 다시 실행하면 된다.
EOF
  exit 3
fi
USER_LOGIN="$("$GH" api user --jq .login)"
echo "   로그인됨: $USER_LOGIN"

echo "== 2. pyproject.toml 의 URL 을 실제 계정으로 =="
if grep -q "USERNAME" pyproject.toml; then
  python - "$USER_LOGIN" <<'PY'
import io, sys
login = sys.argv[1]
p = "pyproject.toml"
s = io.open(p, encoding="utf-8").read()
s = s.replace("https://github.com/USERNAME/CALIPER",
              f"https://github.com/{login}/CALIPER")
io.open(p, "w", encoding="utf-8").write(s)
print(f"   -> https://github.com/{login}/CALIPER")
PY
  git add pyproject.toml
  git commit -q -m "Point the project URL at the actual repository" || true
else
  echo "   이미 교체돼 있음"
fi

echo "== 3. 저장소 만들고 푸시 =="
if git remote get-url origin >/dev/null 2>&1; then
  echo "   원격이 이미 있다: $(git remote get-url origin)"
  git push -u origin "$(git branch --show-current)"
elif "$GH" repo view "$USER_LOGIN/$REPO_NAME" >/dev/null 2>&1; then
  echo "   깃허브에 $REPO_NAME 이 이미 있다. 원격만 연결한다."
  git remote add origin "https://github.com/$USER_LOGIN/$REPO_NAME.git"
  git push -u origin "$(git branch --show-current)"
else
  "$GH" repo create "$REPO_NAME" \
    --"$VIS" \
    --source=. \
    --remote=origin \
    --push \
    --description "Budget-aware allocation and calibration for de novo protein binder design. A multi-fidelity cascade helps under an equal compute budget but not on a fixed pool; the crossover is the finding."
fi

echo
echo "== 완료 =="
echo "   https://github.com/$USER_LOGIN/$REPO_NAME"
echo
echo "   공개 범위: $VIS"
if [ "$VIS" = "private" ]; then
  echo "   공개로 바꾸려면:  gh repo edit --visibility public --accept-visibility-change-consequences"
fi
