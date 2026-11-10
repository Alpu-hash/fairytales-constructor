"""Очистка корпуса сказок.

Что делает:
1. Убирает библиографические заголовки и подвалы (источник, издание, типография, ссылки,
   строка с автором или названием, повторяющая начало).
2. Нормализует метаданные (HTML-сущности, годы жизни в скобках).
3. Отсеивает слишком короткие и слишком длинные тексты.
4. Ставит каждому тексту work_key — ключ произведения (одна сказка в разных переводах
   получает один ключ). Он нужен, чтобы при разбиении на train/val не было утечки.

Запуск из корня проекта:
    python training/prepare_corpus.py
    python training/prepare_corpus.py --max-words 6000

Вход:  data/raw/wikisource_tales.jsonl
Выход: data/processed/tales_clean_unfiltered.jsonl, data/processed/corpus_report.txt
Дальше: python training/filter_corpus.py (отбор не-сказок и взрослых текстов) создаёт tales_clean.jsonl
"""
import argparse
import collections
import hashlib
import html
import json
import re
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw" / "wikisource_tales.jsonl"
OUT = ROOT / "data" / "processed" / "tales_clean_unfiltered.jsonl"
REPORT = ROOT / "data" / "processed" / "corpus_report.txt"

URL_RE = re.compile(r"https?://|www\.|WWW:", re.I)
YEAR_END_RE = re.compile(r"\b(1[5-9]\d\d|20\d\d)\s*(г\.?|года?)?\s*[.,]?\s*$")
HEADER_RES = [re.compile(p, re.I) for p in [
    r"^источник( текста)?\b", r"^издани[ея]\b", r"^изд-?во\b", r"^издательство\b",
    r"^переводчик\b", r"^перевод\b", r"^пер\.", r"\bocr\b", r"spell check",
    r"типо-?лит", r"типографи", r"товарищество", r"скоропечатни",
    r"^исходник", r"^оригинал находится", r"^версия \d", r"^составител", r"^сост\.",
    r"^печатается по", r"^текст приводится", r"^публикуется", r"московский рабочий",
]]


def norm_key(s):
    return re.sub(r"[\W_]+", " ", s.lower().replace("ё", "е")).strip()


def base_title(title):
    t = title.split("/")[0]
    return norm_key(re.sub(r"\s*\(.*?\)", "", t))


def clean_meta(s):
    s = html.unescape(s).replace("\xa0", " ")
    s = re.sub(r"\s*\(\d{3,4}\s*[—–-]\s*\d{0,4}\)", "", s)
    return re.sub(r"\s+", " ", s).strip()


def is_header(p, bt, au, first=False):
    if len(p) > 400:
        return False
    # короткая первая строка без знаков препинания: имя автора, подзаголовок и т. п.
    if (first and len(p) <= 45 and len(p.split()) <= 6
            and not re.search(r"[.,!?:;—–…«»()\d]", p)):
        return True
    if URL_RE.search(p) or any(r.search(p) for r in HEADER_RES):
        return True
    if len(p) < 200 and YEAR_END_RE.search(p) and not re.search(r"[!?]\s*$", p):
        return True
    k = norm_key(p)
    if k and (k == bt or k == au):
        return True
    if (len(p) < 160 and not re.search(r"[.!?…»“\"]$", p)
            and re.match(r"^[А-ЯЁ][^—]{3,70}\s—\s[А-ЯЁ]", p)):
        return True
    return False


def clean_text(text, title, author):
    bt, au = base_title(title), norm_key(author)
    paras = text.split("\n\n")
    n_head = n_foot = 0
    while paras and n_head < 12 and is_header(paras[0], bt, au, first=(n_head == 0)):
        paras.pop(0)
        n_head += 1
    while paras and n_foot < 8 and is_header(paras[-1], bt, au):
        paras.pop()
        n_foot += 1
    before = len(paras)
    paras = [p for p in paras if not (URL_RE.search(p) and len(p) < 300)]
    n_url = before - len(paras)
    return "\n\n".join(paras).strip(), n_head, n_foot, n_url


def main():
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except AttributeError:
        pass
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--min-words", type=int, default=150)
    ap.add_argument("--max-words", type=int, default=8000)
    args = ap.parse_args()

    rows = [json.loads(l) for l in open(RAW, encoding="utf-8")]
    OUT.parent.mkdir(parents=True, exist_ok=True)

    kept, dropped_short, dropped_long, dropped_url = [], [], [], []
    with_head = with_foot = 0
    for r in rows:
        author = clean_meta(r.get("author_raw", ""))
        translator = clean_meta(r.get("translator_raw", ""))
        text, nh, nf, nu = clean_text(r["text"], r["title"], author)
        with_head += nh > 0
        with_foot += nf > 0
        words = len(text.split())
        if words < args.min_words:
            dropped_short.append((r["title"], words))
            continue
        if words > args.max_words:
            dropped_long.append((r["title"], words))
            continue
        if URL_RE.search(text):
            dropped_url.append(r["title"])
            continue
        kept.append({
            "id": hashlib.sha1(r["title"].encode("utf-8")).hexdigest()[:10],
            "title": r["title"],
            "work_key": base_title(r["title"]),
            "author": author,
            "translator": translator,
            "url": r["url"],
            "revid": r["revid"],
            "n_words": words,
            "text": text,
        })

    with open(OUT, "w", encoding="utf-8") as f:
        for r in kept:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    w = [r["n_words"] for r in kept]
    groups = collections.Counter(r["work_key"] for r in kept)
    multi = {k: v for k, v in groups.items() if v > 1}
    authors = collections.Counter(r["author"] or "— (народные/не указан)" for r in kept)

    lines = [
        f"Исходных текстов: {len(rows)}",
        f"Заголовок удалён у: {with_head} текстов; подвал удалён у: {with_foot} текстов",
        f"Отсеяно: короче {args.min_words} слов — {len(dropped_short)}; длиннее {args.max_words} слов — "
        f"{len(dropped_long)}; с остаточными ссылками — {len(dropped_url)}",
        f"Осталось: {len(kept)} текстов, {sum(w)} слов, медиана {int(statistics.median(w))} слов",
        f"Уникальных произведений (work_key): {len(groups)}; "
        f"произведений в нескольких версиях/переводах: {len(multi)}",
        "", "Топ авторов:",
    ]
    lines += [f"  {c:4d} {a}" for a, c in authors.most_common(20)]
    lines += ["", "Отсеяны как слишком длинные (первые 20):"]
    lines += [f"  {n:7d} {t}" for t, n in sorted(dropped_long, key=lambda x: -x[1])[:20]]
    if dropped_url:
        lines += ["", "С остаточными ссылками:"] + [f"  {t}" for t in dropped_url[:20]]
    report = "\n".join(lines)
    REPORT.write_text(report, encoding="utf-8")
    print(report)
    print(f"\nГотово: {OUT}")


if __name__ == "__main__":
    main()
