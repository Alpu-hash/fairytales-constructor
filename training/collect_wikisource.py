"""Сбор сказок на русском языке из русской Викитеки (ru.wikisource.org).

Как работает:
1. Находит в Викитеке категории, названия которых начинаются с заданных префиксов
   (по умолчанию «Сказки», «Русские народные сказки», «Литературные сказки»).
2. Собирает страницы этих категорий (и подкатегорий до заданной глубины).
3. Скачивает каждую страницу, очищает от служебной разметки и сохраняет в JSONL.

Запуск из корня проекта (папка Fairytale_bot):
    python training/collect_wikisource.py --dry-run     # только посмотреть, что нашлось
    python training/collect_wikisource.py               # скачать тексты

Скрипт можно прерывать и запускать снова: уже скачанные страницы пропускаются.
"""
import argparse
import json
import re
import sys
import time
from datetime import date
from pathlib import Path

import requests
from bs4 import BeautifulSoup

API = "https://ru.wikisource.org/w/api.php"
# При желании допишите сюда свой контакт: так просит поступать Wikimedia.
USER_AGENT = "FairytaleBotDiploma/0.1 (student diploma project)"
PAUSE = 1.0  # пауза между запросами, секунды

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUT = ROOT / "data" / "raw" / "wikisource_tales.jsonl"
SKIPPED_FILE = ROOT / "data" / "raw" / "wikisource_skipped.tsv"
SOURCES_MD = ROOT / "data" / "SOURCES.md"

# Категории, которые не подходят для детской аудитории или не являются сказками
DEFAULT_EXCLUDE = r"заветные|Мопассан|Дорошевич|Базар поэта|бекаса|[Лл]егенды|Предания"
# Дореформенная орфография (/ДО), «Версия N» и издания по годам (/1899) — дубли одних и тех же текстов
VERSION_PATTERN = r"/ДО\b|\(ДО\)|/Версия|/\d{4}\b"

DROP_SELECTORS = [
    "style", "script", "table.infobox", ".ws-noexport", ".noprint",
    ".mw-editsection", ".reference", "sup.reference", "#headertemplate",
    ".headertemplate", ".mw-references-wrap", ".toc", "#toc", ".navbox",
    ".prp-page-qualityheader", ".mw-empty-elt",
]

session = requests.Session()
session.headers.update({"User-Agent": USER_AGENT})


def api_get(params, retries=6):
    """GET-запрос к API с повторами; при 429/503 ждём столько, сколько просит сервер."""
    params = {**params, "format": "json", "formatversion": "2"}
    for attempt in range(retries):
        try:
            r = session.get(API, params=params, timeout=30)
            if r.status_code in (429, 503):
                try:
                    wait = int(r.headers.get("Retry-After", ""))
                except ValueError:
                    wait = 10 * 2 ** attempt  # 10, 20, 40, 80, ...
                wait = min(wait, 300)
                print(f"  сервер просит подождать (код {r.status_code}); пауза {wait} с "
                      f"(попытка {attempt + 1}/{retries})", file=sys.stderr)
                time.sleep(wait)
                continue
            r.raise_for_status()
            data = r.json()
            time.sleep(PAUSE)
            return data
        except (requests.RequestException, ValueError) as e:
            wait = 5 * 2 ** attempt
            print(f"  ошибка запроса ({type(e).__name__}); повтор через {wait} с", file=sys.stderr)
            time.sleep(wait)
    raise RuntimeError("API Викитеки недоступен после нескольких попыток")


def find_categories(prefix):
    """Названия категорий, начинающиеся с prefix."""
    cats = []
    params = {"action": "query", "list": "allcategories",
              "acprefix": prefix, "aclimit": "500"}
    while True:
        data = api_get(params)
        cats += [c["category"] for c in data["query"]["allcategories"]]
        if "continue" in data:
            params.update(data["continue"])
        else:
            break
    return cats


def category_members(cat_title):
    """Страницы (ns 0) и подкатегории (ns 14) одной категории."""
    pages, subcats = [], []
    params = {"action": "query", "list": "categorymembers",
              "cmtitle": cat_title, "cmlimit": "500", "cmtype": "page|subcat"}
    while True:
        data = api_get(params)
        for m in data["query"]["categorymembers"]:
            if m["ns"] == 0:
                pages.append(m["title"])
            elif m["ns"] == 14:
                subcats.append(m["title"])
        if "continue" in data:
            params.update(data["continue"])
        else:
            break
    return pages, subcats


def collect_titles(category_names, depth, cache_file):
    """Все страницы из категорий и подкатегорий (до глубины depth).

    Результат кэшируется в cache_file: при повторном запуске уже обойденные
    категории не запрашиваются заново.
    """
    cache = {}
    if cache_file.exists():
        try:
            cache = json.loads(cache_file.read_text(encoding="utf-8"))
        except ValueError:
            cache = {}
    titles = {}  # title -> категория, откуда страница взята
    seen = set()
    frontier = [("Категория:" + c, 0) for c in category_names]
    n = 0
    while frontier:
        cat, level = frontier.pop(0)
        if cat in seen:
            continue
        seen.add(cat)
        n += 1
        if cat in cache:
            pages, subcats = cache[cat]["pages"], cache[cat]["subcats"]
        else:
            try:
                pages, subcats = category_members(cat)
            except RuntimeError as e:
                print(f"   ! пропускаю «{cat}»: {e}", file=sys.stderr)
                continue
            cache[cat] = {"pages": pages, "subcats": subcats}
            cache_file.parent.mkdir(parents=True, exist_ok=True)
            cache_file.write_text(json.dumps(cache, ensure_ascii=False), encoding="utf-8")
        for p in pages:
            titles.setdefault(p, cat)
        if n % 10 == 0:
            print(f"   обойдено категорий: {n}, в очереди: {len(frontier)}, страниц: {len(titles)}")
        if level < depth:
            frontier += [(s, level + 1) for s in subcats]
    return titles


def fetch_page(title):
    """HTML, категории и вики-текст страницы."""
    data = api_get({
        "action": "parse", "page": title, "prop": "text|categories|revid|wikitext",
        "redirects": "1", "disablelimitreport": "1", "disableeditsection": "1",
    })
    if "error" in data:
        return None
    p = data["parse"]
    return {
        "title": p["title"],
        "revid": p["revid"],
        "html": p["text"],
        "wikitext": p.get("wikitext", ""),
        "categories": [c["category"] for c in p.get("categories", []) if not c.get("hidden")],
    }


_INVISIBLE = re.compile("[­​‌‍⁠﻿]")
_SPACES = re.compile("[   ]")


def normalize_text(s):
    """Убирает невидимые символы (мягкие переносы и т. п.), редкие пробелы -> обычный пробел."""
    return _SPACES.sub(" ", _INVISIBLE.sub("", s))


def html_to_text(html):
    """Чистый текст из HTML страницы. Возвращает (text, link_ratio)."""
    soup = BeautifulSoup(html, "html.parser")
    for sel in DROP_SELECTORS:
        for el in soup.select(sel):
            el.decompose()
    total = len(soup.get_text())
    link_chars = sum(len(a.get_text()) for a in soup.find_all("a"))
    link_ratio = link_chars / total if total else 0.0
    for br in soup.find_all("br"):
        br.replace_with("\n")
    blocks = []
    for p in soup.find_all("p"):
        lines = [re.sub(r"[ \t]+", " ", ln).strip() for ln in normalize_text(p.get_text()).split("\n")]
        text = "\n".join(ln for ln in lines if ln)
        if text:
            blocks.append(text)
    return "\n\n".join(blocks), link_ratio


def wiki_field(wikitext, name):
    """Поле шаблона «Отексте» (АВТОР, ПЕРЕВОДЧИК, ...). Best effort."""
    m = re.search(r"\|\s*" + name + r"\s*=\s*([^\n]*)", wikitext)
    if not m:
        return ""
    # сначала раскрываем вики-ссылки [[Страница|Текст]] -> Текст, потом режем по следующему параметру
    value = re.sub(r"\[\[(?:[^\]|]*\|)?([^\]]*)\]\]", r"\1", m.group(1))
    value = value.split("|")[0]
    return re.sub(r"<[^>]+>|\{\{|\}\}", "", value).strip()


def load_done(path):
    done = set()
    if path.exists():
        with open(path, encoding="utf-8") as f:
            for line in f:
                try:
                    done.add(json.loads(line)["title"])
                except (ValueError, KeyError):
                    pass
    return done


def load_skipped(path):
    skipped = set()
    if path.exists():
        with open(path, encoding="utf-8") as f:
            for line in f:
                skipped.add(line.split("\t")[0])
    return skipped


def main():
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except AttributeError:
        pass

    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--prefix", nargs="+",
                    default=["Сказки", "Русские народные сказки", "Литературные сказки"],
                    help="префиксы названий категорий")
    ap.add_argument("--depth", type=int, default=1, help="глубина обхода подкатегорий")
    ap.add_argument("--min-words", type=int, default=150, help="минимум слов в тексте")
    ap.add_argument("--max-pages", type=int, default=0, help="ограничить число страниц (0 = все)")
    ap.add_argument("--exclude-categories", default=DEFAULT_EXCLUDE,
                    help="регулярное выражение: категории с такими названиями пропускаются "
                         "(по умолчанию — заветные сказки, Мопассан, Дорошевич, легенды и т. п.)")
    ap.add_argument("--keep-variants", action="store_true",
                    help="не отсеивать дореформенные версии (/ДО) и дубли (/Версия N, /год)")
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--contact", default="",
                    help="ваш контакт для User-Agent (почта или GitHub), по желанию: "
                         "Wikimedia просит указывать контакт и реже ограничивает такие запросы")
    ap.add_argument("--dry-run", action="store_true",
                    help="только найти категории и посчитать страницы, ничего не скачивая")
    args = ap.parse_args()
    if args.contact:
        session.headers["User-Agent"] = f"FairytaleBotDiploma/0.1 (student diploma project; {args.contact})"

    print("1) Ищу категории по префиксам:", ", ".join(args.prefix))
    cats = []
    for pref in args.prefix:
        found = find_categories(pref)
        print(f"   «{pref}»: найдено категорий: {len(found)}")
        cats += found
    cats = sorted(set(cats))
    if not cats:
        print("Категории не найдены. Проверьте подключение к интернету или префиксы.")
        return
    for c in cats[:40]:
        print("   -", c)
    if len(cats) > 40:
        print(f"   ... и ещё {len(cats) - 40}")

    print("2) Собираю страницы из категорий (может занять несколько минут)...")
    titles = collect_titles(cats, args.depth, ROOT / "data" / "raw" / "wikisource_categories_cache.json")
    print(f"   Уникальных страниц-кандидатов: {len(titles)}")

    # Фильтры: категории, неподходящие для детей, и дубли в других версиях/орфографии
    exclude_cat = re.compile(args.exclude_categories, re.I) if args.exclude_categories else None
    version_re = re.compile(VERSION_PATTERN)
    before = len(titles)
    titles = {t: c for t, c in titles.items()
              if not (exclude_cat and exclude_cat.search(c))}
    after_cats = len(titles)
    if not args.keep_variants:
        titles = {t: c for t, c in titles.items() if not version_re.search(t)}
    print(f"   После исключения категорий: {after_cats} (было {before}); "
          f"после отсева дореформенных версий и дублей: {len(titles)}")

    if args.dry_run:
        report = ROOT / "data" / "raw" / "wikisource_dryrun.txt"
        by_cat = {}
        for t, c in titles.items():
            by_cat.setdefault(c, []).append(t)
        with open(report, "w", encoding="utf-8") as f:
            f.write(f"Категорий найдено: {len(cats)}\n")
            f.write(f"Страниц-кандидатов: {len(titles)}\n\n")
            for c in sorted(by_cat):
                f.write(f"## {c} ({len(by_cat[c])})\n")
                for t in sorted(by_cat[c]):
                    f.write(f"   {t}\n")
                f.write("\n")
        print("Пробный режим: тексты не скачивались.")
        print(f"Полный список записан в файл: {report}")
        return

    args.out.parent.mkdir(parents=True, exist_ok=True)
    done = load_done(args.out)
    skipped = load_skipped(SKIPPED_FILE)
    todo = [t for t in sorted(titles) if t not in done and t not in skipped]
    if args.max_pages:
        todo = todo[: args.max_pages]
    print(f"3) Скачиваю: {len(todo)} страниц (уже есть: {len(done)}, пропущено ранее: {len(skipped)})")

    saved = 0
    with open(args.out, "a", encoding="utf-8") as out, \
            open(SKIPPED_FILE, "a", encoding="utf-8") as skip:
        for i, title in enumerate(todo, 1):
            page = fetch_page(title)
            if page is None:
                skip.write(f"{title}\tошибка загрузки\n")
                continue
            text, link_ratio = html_to_text(page["html"])
            words = len(text.split())
            if link_ratio > 0.4:
                skip.write(f"{title}\tстраница-оглавление (много ссылок)\n")
                continue
            if words < args.min_words:
                skip.write(f"{title}\tкороткий текст ({words} слов)\n")
                continue
            record = {
                "title": page["title"],
                "url": "https://ru.wikisource.org/wiki/" + page["title"].replace(" ", "_"),
                "revid": page["revid"],
                "author_raw": wiki_field(page["wikitext"], "АВТОР"),
                "translator_raw": wiki_field(page["wikitext"], "ПЕРЕВОДЧИК"),
                "source_category": titles[title],
                "categories": page["categories"],
                "n_words": words,
                "text": text,
            }
            out.write(json.dumps(record, ensure_ascii=False) + "\n")
            out.flush()
            saved += 1
            if i % 25 == 0:
                print(f"   обработано {i}/{len(todo)}, сохранено {saved}")

    print(f"Готово. Сохранено новых текстов: {saved}. Файл: {args.out}")
    if saved and SOURCES_MD.exists():
        with open(SOURCES_MD, "a", encoding="utf-8") as f:
            f.write(
                f"| Русская Викитека (категории «{'», «'.join(args.prefix)}») | "
                f"https://ru.wikisource.org | общественное достояние / свободные лицензии "
                f"(по правилам Викитеки; проверить выборочно) | {date.today().isoformat()} | {saved} |\n"
            )
        print("Строка добавлена в data/SOURCES.md")


if __name__ == "__main__":
    main()
