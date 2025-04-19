import re
import csv
from pathlib import Path

logs_dir = Path("backtests")
out_csv = Path("all_sweeps_sorted.csv")

rows = []
for log_file in sorted(logs_dir.glob("*.log")):
    text = log_file.read_text()

    # 1) find the params from the first "→ Running …" line
    m = re.search(r"→ Running ([^\n]+)", text)
    if not m:
        continue
    parts = [p.strip() for p in m.group(1).split(",")]
    params = {}
    for part in parts:
        k, v = part.split("=", 1)
        params[k] = float(v) if "." in v else int(v)

    # 2) find the final "Total profit: X"
    m2 = re.findall(r"Total profit:\s*([-\d,]+)", text)
    pnl = None
    if m2:
        # take the last occurrence in the log
        pnl = int(m2[-1].replace(",", ""))

    rows.append({
        "log_file":    log_file.name,
        "grad":        params.get("grad"),
        "z_high":      params.get("z_high"),
        "z_low":       params.get("z_low"),
        "profit_margin": params.get("profit_margin"),
        "PnL":         pnl,
    })

# 3) sort by PnL descending (None at the bottom)
rows.sort(key=lambda r: (r["PnL"] is None, -(r["PnL"] or 0)))

# 4) write CSV
with out_csv.open("w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=["log_file","grad","z_high","z_low","profit_margin","PnL"])
    writer.writeheader()
    writer.writerows(rows)

print(f"Wrote {out_csv} with {len(rows)} entries, sorted by descending PnL.")
