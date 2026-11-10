"""Собирает файлы для отчёта по этапу 2 (база данных) из tales_clean.jsonl, chapters.jsonl и cards.jsonl.

Создаёт в data/processed/report/:
  tales_list.csv            — все сказки с разметкой (открывается в Excel)
  excluded_by_annotation.csv — сказки, отсеянные разметкой, с причиной (для проверки)
  stats.md                  — цифры для текста отчёта
  examples.md               — примеры «фрагмент сказки → разметка»
  manual_check_sample.csv   — 30 случайных сказок для ручной проверки разметки

Запуск из корня проекта:  python training/make_report_files.py
"""
import collections
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
P = ROOT / "data" / "processed"
OUT = P / "report"
OUT.mkdir(exist_ok=True)


def jl(path):
    return [json.loads(l) for l in open(path, encoding="utf-8")]


def short(title):
    return title.split("(")[0].split("/")[-1].strip()


def main():
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except AttributeError:
        pass
    tales = {t["id"]: t for t in jl(P / "tales_clean.jsonl")}
    cards = {c["tale_id"]: c for c in jl(P / "cards.jsonl")}
    chapters = jl(P / "chapters.jsonl")
    split = {c["tale_id"]: c["split"] for c in chapters}
    nch = collections.Counter(c["tale_id"] for c in chapters)
    ntok = collections.Counter()
    for c in chapters:
        ntok[c["tale_id"]] += c["n_tokens"]

    # 1. таблица сказок
    rows = []
    for i, t in tales.items():
        c = cards[i]
        rows.append([t["title"], t.get("author", ""), t.get("translator", ""), t["n_words"], nch[i], split[i],
                     c["kind"], c["violence"], c["mood"], "; ".join(c["characters"]), c["moral"], c["mood_reason"], t.get("url", "")])
    rows.sort(key=lambda r: r[0].lower())
    head = ["Название", "Автор", "Переводчик", "Слов", "Глав", "Выборка", "Тип", "Жестокость", "Настроение",
            "Герои", "Мораль", "Почему такая метка", "Ссылка"]
    with open(OUT / "tales_list.csv", "w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f, delimiter=";")
        w.writerow(["№"] + head)
        for n, r in enumerate(rows, 1):
            w.writerow([n] + r)
    with open(OUT / "excluded_by_annotation.csv", "w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f, delimiter=";")
        w.writerow(["№", "Название", "Автор", "Тип", "Жестокость", "Причина", "Ссылка"])
        n = 0
        for r in rows:
            if r[8] == "exclude":
                n += 1
                w.writerow([n, r[0], r[1], r[6], r[7], r[11], r[12]])

    # 1б. выборка для ручной проверки разметки (30 случайных сказок, фиксированное зерно)
    import random
    first = {c["tale_id"]: c["text"] for c in chapters if c["chapter"] == 1}
    pick_ids = random.Random(42).sample(sorted(tales), 30)
    with open(OUT / "manual_check_sample.csv", "w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f, delimiter=";")
        w.writerow(["№", "Название", "Автор", "Начало сказки", "Настроение", "Жестокость", "Почему так",
                    "Верно? (да/нет)", "Комментарий", "Ссылка"])
        for n, i in enumerate(pick_ids, 1):
            c = cards[i]
            w.writerow([n, tales[i]["title"], tales[i].get("author", ""), first[i][:300].replace("\n", " "),
                        c["mood"], c["violence"], c["mood_reason"], "", "", tales[i].get("url", "")])

    # 2. статистика
    keep = [i for i in tales if cards[i]["mood"] != "exclude"]
    k_ch = [c for c in chapters if cards[c["tale_id"]]["mood"] != "exclude"]
    L = []
    A = L.append
    A("# Статистика базы\n")
    fmt = lambda n: f"{n:,}".replace(",", " ")
    A(f"- Исходно собрано из Викитеки: 1714 текстов. После очистки и отбора: **{len(tales)}**, "
      f"это {len({t['work_key'] for t in tales.values()})} уникальных произведений.")
    A(f"- Разбито на главы: **{len(chapters)}**, всего {fmt(sum(c['n_tokens'] for c in chapters))} токенов (токенайзер ruGPT3).")
    A(f"- Размечено: все {len(cards)} сказок и все главы (краткое содержание и действие для каждой главы).\n")
    A("## Результат разметки\n")
    A("| Настроение | Сказок | Глав |\n|---|---|---|")
    for m in ("calm", "adventure", "scary", "exclude"):
        ids = [i for i in tales if cards[i]["mood"] == m]
        A(f"| {m} | {len(ids)} | {sum(nch[i] for i in ids)} |")
    A("\n| Жестокость | Сказок |\n|---|---|")
    for v, n in collections.Counter(cards[i]["violence"] for i in tales).most_common():
        A(f"| {v} | {n} |")
    A(f"\n- Тип: tale — {sum(cards[i]['kind'] == 'tale' for i in tales)}, not_tale — {sum(cards[i]['kind'] == 'not_tale' for i in tales)}.\n")
    A("## Итоговая база для обучения (без exclude)\n")
    A(f"- Сказок: **{len(keep)}**; глав (= обучающих примеров): **{len(k_ch)}**; "
      f"токенов: **{fmt(sum(c['n_tokens'] for c in k_ch))}**.")
    A("\n| Выборка | Сказок | Глав | calm | adventure | scary |\n|---|---|---|---|---|---|")
    for s in ("train", "val", "test"):
        ids = [i for i in keep if split[i] == s]
        mc = collections.Counter(cards[i]["mood"] for i in ids)
        A(f"| {s} | {len(ids)} | {sum(nch[i] for i in ids)} | {mc['calm']} | {mc['adventure']} | {mc['scary']} |")
    chs = sorted(c["n_tokens"] for c in k_ch)
    A(f"\n- Длина главы: от {chs[0]} до {chs[-1]} токенов, медиана {chs[len(chs) // 2]}; контекст модели 2048.")
    A("\n## Источники\n")
    src = collections.Counter()
    for i in keep:
        t = tales[i]
        s = t["title"].split("/")[0]
        src["Народные русские сказки (Афанасьев)" if "Афанасьев" in s else
            "Старая погудка на новый лад" if s.startswith("Старая погудка") else
            "Русские сказки в ранних записях" if s.startswith("Русские сказки в ранних") else
            (t.get("author") or "автор не указан")] += 1
    A("| Источник / автор | Сказок в базе |\n|---|---|")
    for s, n in src.most_common(15):
        A(f"| {s} | {n} |")
    A(f"\nВсе тексты — русская Викитека, общественное достояние (список в data/SOURCES.md).")
    (OUT / "stats.md").write_text("\n".join(L) + "\n", encoding="utf-8")

    # 3. примеры
    def pick(sub, mood):
        for i, c in cards.items():
            if sub in tales[i]["title"] and c["mood"] == mood:
                return i
    want = [("Дюймовочка", "adventure"), ("Аленький цветочек", "scary"), ("Морозко", "scary"), ("Белоснеж", "scary"),
            ("Котома", "exclude"), ("Чайник", "calm")]
    E = ["# Примеры из базы\n", "Для каждого примера: фрагмент текста (начало первой главы) и разметка, "
         "полученная от модели-разметчика.\n"]
    for sub, mood in want:
        i = pick(sub, mood)
        if i is None:
            continue
        t, c = tales[i], cards[i]
        first = next(x for x in chapters if x["tale_id"] == i and x["chapter"] == 1)
        E.append(f"## {short(t['title'])} ({t.get('author') or 'народная'}) — {mood}\n")
        E.append(f"Фрагмент: «{first['text'][:450].strip()}…»\n")
        E.append(f"- Герои: {', '.join(c['characters'])}\n- Место: {c['setting']}\n- Цель: {c['goal']}\n- Мораль: {c['moral']}")
        E.append(f"- Тип: {c['kind']}; жестокость: {c['violence']}; настроение: {c['mood']} ({c['mood_reason']})")
        ch1 = c["chapters"][0]
        E.append(f"- Глава 1: {ch1['summary']} Действие-кнопка: «{ch1['action']}»")
        E.append(f"- Выборка: {split[i]}; глав в сказке: {nch[i]}\n")
    (OUT / "examples.md").write_text("\n".join(E) + "\n", encoding="utf-8")
    print("Готово. Файлы в", OUT)
    print((OUT / "stats.md").read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
