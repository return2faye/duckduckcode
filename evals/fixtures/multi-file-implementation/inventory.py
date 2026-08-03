from pathlib import Path


def load_totals(path):
    totals = {}
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        name, quantity = line.split(",")
        totals[name] = totals.get(name, 0) + int(quantity)
    return totals
