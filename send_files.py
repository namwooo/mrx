#!/usr/bin/env python3
import argparse
import smtplib
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from email.message import EmailMessage
from pathlib import Path

FROM = "youngchan.kim@makinarocks.ai"
PASSWORD = ""
SMTP_HOST, SMTP_PORT = "smtp.gmail.com", 465
TIMEOUT = 600
MAX_BYTES = 18 * 1024 * 1024
RETRIES = 2


def get_password() -> str:
    pw = PASSWORD.replace(" ", "")
    if not pw or pw.startswith("여기에"):
        sys.exit("❌ 스크립트 상단의 PASSWORD에 앱 비밀번호를 넣어주세요.")
    return pw


def send_one(path: Path, idx: int, total: int, to: str, pw: str) -> str:
    size_mb = path.stat().st_size / 1024 / 1024
    msg = EmailMessage()
    msg["From"] = FROM
    msg["To"] = to
    msg["Subject"] = f"파일 전달 [{idx}/{total}] {path.name}"
    msg.set_content(f"{path.name} 첨부합니다. ({idx}/{total})")
    msg.add_attachment(path.read_bytes(), maintype="application",
                       subtype="octet-stream", filename=path.name)

    last_err = None
    for attempt in range(1, RETRIES + 2):
        start = time.time()
        try:
            with smtplib.SMTP_SSL(SMTP_HOST, SMTP_PORT, timeout=TIMEOUT) as s:
                s.login(FROM, pw)
                s.send_message(msg)
            secs = time.time() - start
            return f"✅ [{idx}/{total}] {path.name} ({size_mb:.1f}MB, {secs:.0f}초)"
        except smtplib.SMTPAuthenticationError as e:
            return f"❌ [{idx}/{total}] {path.name} 인증 실패: {e.smtp_code} {e.smtp_error!r}"
        except Exception as e:
            last_err = e
            if attempt <= RETRIES:
                time.sleep(5 * attempt)
    return f"❌ [{idx}/{total}] {path.name} 발송 실패: {last_err}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("folder")
    ap.add_argument("to")
    ap.add_argument("--workers", type=int, default=3, help="동시에 보낼 메일 수 (기본 3)")
    args = ap.parse_args()

    folder = Path(args.folder).expanduser()
    if not folder.is_dir():
        sys.exit(f"❌ 폴더가 없습니다: {folder}")

    files = sorted(p for p in folder.iterdir() if p.is_file() and not p.name.startswith("."))
    if not files:
        sys.exit("보낼 파일이 없습니다.")

    total = len(files)
    pw = get_password()
    print(f"📂 {folder} — 파일 {total}개 → {args.to} (동시 {args.workers}통)", flush=True)

    jobs, skipped = [], 0
    for i, p in enumerate(files, 1):
        if p.stat().st_size > MAX_BYTES:
            print(f"⚠️  [{i}/{total}] {p.name} — 18MB 초과로 건너뜀", flush=True)
            skipped += 1
        else:
            jobs.append((p, i))

    ok = fail = 0
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futures = [ex.submit(send_one, p, i, total, args.to, pw) for p, i in jobs]
        for f in as_completed(futures):
            line = f.result()
            print(line, flush=True)
            if line.startswith("✅"):
                ok += 1
            else:
                fail += 1

    mins = (time.time() - t0) / 60
    print("──────────────")
    print(f"완료: 성공 {ok} / 실패 {fail} / 건너뜀 {skipped} ({mins:.1f}분)")


if __name__ == "__main__":
    main()