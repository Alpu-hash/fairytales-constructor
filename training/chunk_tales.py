"""Нарезка сказок на главы и разбиение на train/val/test по произведениям.

Что делает:
1. Считает длину абзацев в токенах токенизатора ruGPT3.
2. Режет каждую сказку на главы по границам абзацев (при необходимости — предложений).
   Короткие сказки остаются одной главой.
3. Делит произведения на train/val/test по work_key (одна сказка в разных переводах
   всегда попадает в одну выборку, поэтому утечки между train и val нет).

Запуск из корня проекта (с активным venv):
    python training/chunk_tales.py

Параметры длины главы (в токенах): --min-tokens 300 --max-tokens 800 --hard-tokens 1100

Вход:  data/processed/tales_clean.jsonl
Выход: data/processed/chapters.jsonl, data/processed/chapters_report.txt
"""
import argparse
import collections
import hashlib
import json
import re
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
IN_FILE = ROOT / "data" / "processed" / "tales_clean.jsonl"
OUT_FILE = ROOT / "data" / "processed" / "chapters.jsonl"
REPORT = ROOT / "data" / "processed" / "chapters_report.txt"
MODEL = "ai-forever/rugpt3small_based_on_gpt2"

SENT_RE = re.compile(r"(?<=[.!?…»\"”])\s+")


def split_units(text, count, max_tokens):
    """Текст -> список (фрагмент, токены, разделитель_перед). Абзацы; длинные режутся дальше."""
    units = []
    for para in text.split("\n\n"):
        para = para.strip()
        if not para:
            continue
        n = count(para)
        if n <= max_tokens:
            units.append((para, n, "\n\n"))
            continue
        first = True
        for sent in SENT_RE.split(para):
            ns = count(sent)
            parts = [sent]
            if ns > max_tokens:  # нет знаков препинания: режем по словам пропорционально длине
                words = sent.split()
                per = max(1, int(len(words) * max_tokens / ns * 0.9))
                parts = [" ".join(words[i:i + per]) for i in range(0, len(words), per)]
            for part in parts:
                units.append((part, count(part), "\n\n" if first else " "))
                first = False
    return units


def chunk_units(units, min_tokens, max_tokens, hard_tokens):
    """Жадная нарезка на главы по границам фрагментов."""
    chunks, cur, cur_tok = [], [], 0
    for u in units:
        if cur and cur_tok + u[1] > max_tokens and cur_tok >= min_tokens:
            chunks.append(cur)
            cur, cur_tok = [], 0
        cur.append(u)
        cur_tok += u[1]
    if cur:
        chunks.append(cur)
    # слишком короткий хвост присоединяем к предыдущей главе, если влезает
    if len(chunks) > 1:
        tail = sum(u[1] for u in chunks[-1])
        both = tail + sum(u[1] for u in chunks[-2])
        if tail < min_tokens and both <= hard_tokens:
            last = chunks.pop()
            chunks[-1].extend(last)
    return chunks


def render(chunk):
    out = ""
    for i, (t, _, sep) in enumerate(chunk):
        out += t if i == 0 else sep + t
    return out


def split_of(work_key):
    h = int(hashlib.sha1(work_key.encode("utf-8")).hexdigest(), 16) % 100
    return "train" if h < 85 else "val" if h < 92 else "test"


def make_counter():
    from transformers import AutoConfig, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL)
    tok.model_max_length = 10 ** 9  # отключаем предупреждения о длине: мы режем сами
    try:
        cfg = AutoConfig.from_pretrained(MODEL)
        print(f"Контекст модели (n_positions): {getattr(cfg, 'n_positions', '?')}")
    except Exception as e:  # noqa: BLE001
        print("Не удалось прочитать конфиг модели:", e)
    return lambda s: len(tok(s, add_special_tokens=False)["input_ids"])


def main():
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except AttributeError:
        pass
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--min-tokens", type=int, default=300)
    ap.add_argument("--max-tokens", type=int, default=800)
    ap.add_argument("--hard-tokens", type=int, default=1100)
    ap.add_argument("--limit", type=int, default=0, help="обработать только первые N сказок (для проверки)")
    args = ap.parse_args()

    count = make_counter()
    tales = [json.loads(l) for l in open(IN_FILE, encoding="utf-8")]
    if args.limit:
        tales = tales[: args.limit]

    n_chapters_per_tale, chapter_tokens = [], []
    tales_by_split, chapters_by_split = collections.Counter(), collections.Counter()
    with open(OUT_FILE, "w", encoding="utf-8") as out:
        for i, t in enumerate(tales, 1):
            units = split_units(t["text"], count, args.max_tokens)
            chunks = chunk_units(units, args.min_tokens, args.max_tokens, args.hard_tokens)
            split = split_of(t["work_key"])
            tales_by_split[split] += 1
            for j, ch in enumerate(chunks, 1):
                n_tok = sum(u[1] for u in ch)
                rec = {
                    "tale_id": t["id"], "work_key": t["work_key"], "title": t["title"],
                    "author": t["author"], "split": split,
                    "chapter": j, "n_chapters": len(chunks), "n_tokens": n_tok,
                    "text": render(ch),
                }
                out.write(json.dumps(rec, ensure_ascii=False) + "\n")
                chapter_tokens.append(n_tok)
                chapters_by_split[split] += 1
            n_chapters_per_tale.append(len(chunks))
            if i % 200 == 0:
                print(f"   обработано {i}/{len(tales)} сказок")

    q = statistics.quantiles(chapter_tokens, n=20)
    dist = collections.Counter(min(n, 10) for n in n_chapters_per_tale)
    lines = [
        f"Сказок: {len(tales)}; глав: {len(chapter_tokens)}; токенов всего: {sum(chapter_tokens)}",
        f"Токенов в главе: мин {min(chapter_tokens)}, медиана {int(statistics.median(chapter_tokens))}, "
        f"95-й процентиль {int(q[18])}, макс {max(chapter_tokens)}",
        "Глав в сказке (10 = десять и больше): " + ", ".join(f"{k}: {dist[k]}" for k in sorted(dist)),
        "Выборки (сказок / глав): " + "; ".join(
            f"{s} {tales_by_split[s]} / {chapters_by_split[s]}" for s in ("train", "val", "test")),
    ]
    report = "\n".join(lines)
    REPORT.write_text(report, encoding="utf-8")
    print(report)
    print(f"\nГотово: {OUT_FILE}")


if __name__ == "__main__":
    main()
