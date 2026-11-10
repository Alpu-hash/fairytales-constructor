"""Контрольная очистка текстов глав (после chunk_tales.py).

Что исправляет (найдено при контрольной проверке базы):
  1. невидимые символы направления текста (U+200E и подобные) — были в 655 главах;
  2. библиографическая шапка издания в начале первой главы (название книги, типография, иллюстратор, переводчик);
  3. сноски: метки вида [1] в тексте и абзацы-сноски в конце;
  4. символ «~» (метка конца строки из вёрстки стихов Викитеки);
  5. потерянная заглавная буква в начале сказки (Белоснеженка: «то случилось...» -> «Это случилось...»).
После очистки заново считает n_tokens токенайзером ruGPT3 (если установлен transformers и токенайзер доступен).

Запуск из корня проекта:
    python training/clean_chapters.py
Скрипт можно запускать повторно: исходник сохраняется один раз как data/processed/chapters_before_clean.jsonl,
а очистка всегда берёт его за основу.
"""
import argparse
import json
import re
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
P = ROOT / "data" / "processed"
SRC = P / "chapters_before_clean.jsonl"
DST = P / "chapters.jsonl"
MODEL = "ai-forever/rugpt3small_based_on_gpt2"

INVISIBLE = re.compile("[\u200b-\u200f\u2028\u2029\u202a-\u202e\u2060\ufeff\u00ad]")
HEADER_KW = re.compile(
    r"Типограф|Типо-лит|Иллюстраци|Издани[ея]|Подготовил|Перевод:|Печатается|Электронн|Книгопродав"
    r"|(?:Петербург|Москва|Киев|Харьков|Одесса),? \d{4}|\d{4} г\.")
FOOT_PAR = re.compile(r"^\s*\[\d{1,2}\]\s")
FOOT_MARK = re.compile(r"[ \t]*\[\d{1,2}\]")
# сказки, у которых потеряна первая буква: (начало текста после очистки) -> что приставить
LOST_FIRST = {"то случилось однажды в средине зимы": "Э"}


def strip_header(text):
    """Убирает шапку издания в начале: все абзацы до последнего (из первых шести), где есть признак шапки."""
    paras = text.split("\n\n")
    last = -1
    for i, p in enumerate(paras[:6]):
        if len(p) < 400 and HEADER_KW.search(p):
            last = i
    if last < 0:
        return text, None
    removed = "\n\n".join(paras[: last + 1])
    rest = "\n\n".join(paras[last + 1:]).lstrip()
    if len(removed) > 700 or len(rest) < 200:  # предохранитель: не режем лишнего
        return text, None
    return rest, removed


def clean(text, chapter):
    t = INVISIBLE.sub("", text)
    header = None
    if chapter == 1:
        t, header = strip_header(t)
    t = "\n\n".join(p for p in t.split("\n\n") if not FOOT_PAR.match(p))
    t = FOOT_MARK.sub("", t)
    t = t.replace("~", ";")
    t = re.sub(r"[ \t]{2,}", " ", t).strip()
    if chapter == 1:
        for start, letter in LOST_FIRST.items():
            if t.startswith(start):
                t = letter + t
    return t, header


def main():
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except AttributeError:
        pass
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(DST), help="куда записать результат (по умолчанию chapters.jsonl)")
    ap.add_argument("--no-recount", action="store_true", help="не пересчитывать n_tokens")
    args = ap.parse_args()

    if not SRC.exists():
        shutil.copyfile(DST, SRC)
        print(f"Исходник сохранён: {SRC.name}")
    rows = [json.loads(l) for l in open(SRC, encoding="utf-8")]
    changed = headers = 0
    samples = []
    for r in rows:
        new, header = clean(r["text"], r["chapter"])
        if header:
            headers += 1
            if len(samples) < 4:
                samples.append((r["title"][:35], header.replace("\n", " ")[:110]))
        if new != r["text"]:
            changed += 1
            r["text"] = new
    print(f"Глав: {len(rows)}; изменено: {changed}; убрано шапок издания: {headers}")
    for s in samples:
        print("  шапка:", s)

    if not args.no_recount:
        try:
            from transformers import AutoTokenizer
            tok = AutoTokenizer.from_pretrained(MODEL)
            old_sum = sum(r["n_tokens"] for r in rows)
            for r in rows:
                r["n_tokens"] = len(tok(r["text"]).input_ids)
            new_sum = sum(r["n_tokens"] for r in rows)
            mx = max(r["n_tokens"] for r in rows)
            print(f"Токенов было {old_sum}, стало {new_sum}; максимум в главе {mx}")
        except Exception as e:  # noqa: BLE001
            print("Токены не пересчитаны:", e)

    with open(args.out, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print("Записано:", args.out)


if __name__ == "__main__":
    main()
