"""Копирует в отдельную папку только то, что можно публиковать на GitHub (белый список файлов).
Рабочий проект при этом не трогается: git инициализируется уже в новой папке.

Запуск из корня проекта:
    python tools/export_public_repo.py ../fairytale-bot-public
Скрипт можно запускать повторно после изменений кода: файлы в папке обновятся.
Перед копированием проверяет файлы на ключи API (sk-ant-...) и на слишком большой размер.
"""
import re
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MAX_BYTES = 5 * 1024 * 1024
KEY_RE = re.compile(r"sk-ant-[A-Za-z0-9_\-]{20,}")

WHITELIST_FILES = [
    "README.md", "requirements.txt",
    "data/SOURCES.md",
    "data/dataset/sample.jsonl", "data/dataset/examples.md", "data/dataset/stats.md",
    "data/processed/corpus_report.txt", "data/processed/chapters_report.txt",
    "data/processed/report/stats.md", "data/processed/report/examples.md",
    "data/processed/report/tales_list.csv", "data/processed/report/excluded_by_annotation.csv",
    "data/processed/report/manual_check_sample.csv",
]
WHITELIST_GLOBS = ["training/*.py", "server/**/*.py", "bot/**/*.py", "notebooks/*.ipynb"]
SKIP_NAMES = ("_backup.py",)

GITIGNORE = """venv/
.venv/
__pycache__/
.env
data/raw/
data/processed/*.jsonl
data/processed/*.tsv
data/dataset/train.jsonl
data/dataset/val.jsonl
data/dataset/test.jsonl
checkpoints/
*.safetensors
"""


def main():
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except AttributeError:
        pass
    if len(sys.argv) != 2:
        print("Использование: python tools/export_public_repo.py <папка-назначения>")
        sys.exit(1)
    dst = Path(sys.argv[1]).resolve()
    if dst == ROOT or ROOT in dst.parents:
        print("Папка назначения должна быть вне проекта, например ../fairytale-bot-public")
        sys.exit(1)

    files = [ROOT / f for f in WHITELIST_FILES if (ROOT / f).exists()]
    for g in WHITELIST_GLOBS:
        files += [p for p in ROOT.glob(g) if p.is_file() and not p.name.endswith(SKIP_NAMES) and "__pycache__" not in p.parts]
    files = sorted(set(files))

    problems = []
    for p in files:
        if p.stat().st_size > MAX_BYTES:
            problems.append(f"слишком большой файл ({p.stat().st_size // 1024} КБ): {p.relative_to(ROOT)}")
        try:
            if KEY_RE.search(p.read_text(encoding="utf-8", errors="ignore")):
                problems.append(f"в файле найден ключ API: {p.relative_to(ROOT)}")
        except OSError as e:
            problems.append(f"не удалось прочитать {p.relative_to(ROOT)}: {e}")
    if problems:
        print("КОПИРОВАНИЕ ОСТАНОВЛЕНО:")
        for x in problems:
            print(" -", x)
        sys.exit(2)

    dst.mkdir(parents=True, exist_ok=True)
    for p in files:
        rel = p.relative_to(ROOT)
        (dst / rel).parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(p, dst / rel)
    (dst / ".gitignore").write_text(GITIGNORE, encoding="utf-8")
    print(f"Скопировано файлов: {len(files)} -> {dst}")
    for p in files:
        print("  ", p.relative_to(ROOT).as_posix())
    print("Создан .gitignore")


if __name__ == "__main__":
    main()
