"""Сборка обучающих примеров для дообучения ruGPT3 из глав и карточек разметки.

Один пример = одна глава сказки:
    prompt     — карточка сказки + краткое содержание предыдущих глав + выбор ребёнка (действие-кнопка)
    completion — текст главы + конец последовательности (</s>)
При обучении потери считаются только на completion (подсказка маскируется).

Берутся только сказки, пригодные детям: kind = tale и mood in (calm, adventure, scary); exclude не берётся.
Длина prompt + completion не больше 2048 токенов (контекст ruGPT3): если не помещается, число пересказов
предыдущих глав уменьшается, а если и так не помещается, пример отбрасывается (и это попадает в статистику).

Выход (папка data/dataset/): train.jsonl, val.jsonl, test.jsonl, sample.jsonl (30 примеров для просмотра),
stats.md (цифры для отчёта), examples.md (примеры «было -> стало»).

Запуск из корня проекта:
    python training/build_dataset.py            # настоящий токенайзер ruGPT3
    python training/build_dataset.py --approx   # приближённый подсчёт токенов (без transformers)
"""
import argparse
import collections
import json
import random
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
P = ROOT / "data" / "processed"
OUT = ROOT / "data" / "dataset"
MODEL = "ai-forever/rugpt3small_based_on_gpt2"
MAX_LEN = 2048          # контекст ruGPT3
MAX_PREV = 4            # сколько последних пересказов предыдущих глав кладём в подсказку
GOOD_MOODS = ("calm", "adventure", "scary")
EOS = "</s>"


def jl(path):
    return [json.loads(l) for l in open(path, encoding="utf-8")]


def make_prompt(card, k, n, prev):
    moral = card["moral"].strip()
    if not moral or "не выражена" in moral:
        moral = "не указана"
    lines = [
        f"Настроение: {card['mood']}",
        f"Герои: {', '.join(card['characters'])}",
        f"Место: {card['setting']}",
        f"Цель: {card['goal']}",
        f"Мораль: {moral}",
    ]
    if prev:
        lines.append("Предыдущие главы:")
        lines += [f"{i}. {s}" for i, s in prev]
    else:
        lines.append("Предыдущие главы: нет")
    lines.append(f"Выбор ребёнка: {card['chapters'][k - 1]['action']}")
    lines.append(f"Глава {k} из {n}:")
    return "\n".join(lines) + "\n"


def get_counter(approx):
    if approx:
        return (lambda s: int(len(s) / 3.2) + 1), EOS, "приближённо: символов / 3.2"
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL)
    eos = tok.eos_token or EOS
    return (lambda s: len(tok(s, add_special_tokens=False)["input_ids"])), eos, f"токенайзер {MODEL}"


def main():
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except AttributeError:
        pass
    ap = argparse.ArgumentParser()
    ap.add_argument("--approx", action="store_true", help="приближённый подсчёт токенов, без transformers")
    args = ap.parse_args()
    count, eos, how = get_counter(args.approx)
    OUT.mkdir(parents=True, exist_ok=True)

    cards = {c["tale_id"]: c for c in jl(P / "cards.jsonl")}
    chapters = jl(P / "chapters.jsonl")
    by_tale = collections.defaultdict(list)
    for c in chapters:
        by_tale[c["tale_id"]].append(c)

    examples = {"train": [], "val": [], "test": []}
    skipped = collections.Counter()
    shrunk = 0
    for tid, chs in by_tale.items():
        card = cards.get(tid)
        if card is None:
            skipped["нет карточки"] += len(chs)
            continue
        if card["kind"] != "tale" or card["mood"] not in GOOD_MOODS:
            skipped["исключено разметкой (exclude / not_tale)"] += len(chs)
            continue
        chs.sort(key=lambda c: c["chapter"])
        n = len(chs)
        if len(card["chapters"]) != n:
            skipped["число глав в карточке не совпало"] += n
            continue
        summaries = [(i + 1, x["summary"]) for i, x in enumerate(card["chapters"])]
        for ch in chs:
            k = ch["chapter"]
            completion = ch["text"].strip() + eos
            n_comp = count(completion)
            prev_all = summaries[:k - 1]
            used = None
            for keep in range(min(MAX_PREV, len(prev_all)), -1, -1):
                prev = prev_all[len(prev_all) - keep:] if keep else []
                prompt = make_prompt(card, k, n, prev)
                n_prompt = count(prompt)
                if n_prompt + n_comp <= MAX_LEN:
                    used = (prompt, n_prompt, keep)
                    break
                shrunk += 1 if keep == min(MAX_PREV, len(prev_all)) else 0
            if used is None:
                skipped["не помещается в 2048 токенов"] += 1
                continue
            prompt, n_prompt, keep = used
            examples[ch["split"]].append({
                "prompt": prompt, "completion": completion,
                "meta": {"tale_id": tid, "title": ch["title"], "work_key": ch["work_key"], "chapter": k,
                         "n_chapters": n, "mood": card["mood"], "violence": card["violence"],
                         "prev_summaries": keep, "n_prompt_tokens": n_prompt, "n_completion_tokens": n_comp},
            })

    # проверка: ни одно произведение не в двух выборках
    wk = collections.defaultdict(set)
    for s, ex in examples.items():
        for e in ex:
            wk[e["meta"]["work_key"]].add(s)
    leak = sum(len(v) > 1 for v in wk.values())

    for s, ex in examples.items():
        with open(OUT / f"{s}.jsonl", "w", encoding="utf-8") as f:
            for e in ex:
                f.write(json.dumps(e, ensure_ascii=False) + "\n")
    allex = [e for ex in examples.values() for e in ex]
    rnd = random.Random(42)
    with open(OUT / "sample.jsonl", "w", encoding="utf-8") as f:
        for e in rnd.sample(allex, min(30, len(allex))):
            f.write(json.dumps(e, ensure_ascii=False) + "\n")

    # статистика
    fmt = lambda x: f"{x:,}".replace(",", " ")
    tot = [e["meta"]["n_prompt_tokens"] + e["meta"]["n_completion_tokens"] for e in allex]
    pr = [e["meta"]["n_prompt_tokens"] for e in allex]
    co = [e["meta"]["n_completion_tokens"] for e in allex]
    L = ["# Статистика обучающих примеров\n", f"Подсчёт токенов: {how}. Максимальная длина примера: {MAX_LEN}.\n"]
    L.append("| Выборка | Примеров (глав) | Сказок | из них calm | из них adventure | из них scary |\n|---|---|---|---|---|---|")
    for s, ex in examples.items():
        mc = collections.Counter(e["meta"]["mood"] for e in ex)
        L.append(f"| {s} | {len(ex)} | {len({e['meta']['tale_id'] for e in ex})} | {mc['calm']} | {mc['adventure']} | {mc['scary']} |")
    L.append(f"| **всего** | **{len(allex)}** | {len({e['meta']['tale_id'] for e in allex})} | | | |\n")
    L.append(f"- Токенов всего (подсказка + глава): {fmt(sum(tot))}; из них в тексте глав (на них считается ошибка): {fmt(sum(co))}.")
    L.append(f"- Длина подсказки: медиана {statistics.median(pr)}, максимум {max(pr)}.")
    L.append(f"- Длина главы: медиана {statistics.median(co)}, максимум {max(co)}.")
    L.append(f"- Длина примера целиком: медиана {statistics.median(tot)}, максимум {max(tot)}.")
    pv = collections.Counter(e["meta"]["prev_summaries"] for e in allex)
    L.append("- Сколько пересказов предыдущих глав в подсказке: " + ", ".join(f"{k}: {v}" for k, v in sorted(pv.items())) + ".")
    L.append(f"- Примеров, где пересказы пришлось сократить, чтобы уложиться в {MAX_LEN} токенов: {shrunk}.")
    L.append("- Не вошло в обучающие примеры: " + (", ".join(f"{k} — {v}" for k, v in skipped.items()) or "ничего") + ".")
    L.append(f"- Произведений, попавших сразу в несколько выборок: {leak}.")
    (OUT / "stats.md").write_text("\n".join(L) + "\n", encoding="utf-8")

    # примеры «было -> стало»
    E = ["# Примеры обучающих примеров\n"]
    picks = []
    for want in (1, 2, "last"):
        for e in rnd.sample(examples["train"], len(examples["train"])):
            m = e["meta"]
            if (want == "last" and m["chapter"] == m["n_chapters"] and m["n_chapters"] > 2) or \
               (want == 2 and m["chapter"] == 2 and m["n_chapters"] > 2) or (want == 1 and m["chapter"] == 1 and m["n_chapters"] > 2):
                picks.append(e)
                break
    for e in picks:
        m = e["meta"]
        E.append(f"## {m['title'].split('(')[0].split('/')[-1].strip()}, глава {m['chapter']} из {m['n_chapters']} "
                 f"({m['mood']}, {m['n_prompt_tokens']} + {m['n_completion_tokens']} токенов)\n")
        E.append("**Подсказка (вход, ошибка на ней не считается):**\n\n```\n" + e["prompt"] + "```\n")
        comp = e["completion"]
        E.append("**Ответ (текст главы, на нём учится модель):**\n\n```\n" + comp[:500].rstrip() + " …" + comp[-60:].replace("\n", " ") + "\n```\n")
    (OUT / "examples.md").write_text("\n".join(E) + "\n", encoding="utf-8")

    print("\n".join(L))


if __name__ == "__main__":
    main()
