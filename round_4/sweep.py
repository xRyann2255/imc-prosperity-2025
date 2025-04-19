import itertools
import subprocess
import re
import time
from pathlib import Path
import shutil
from concurrent.futures import ThreadPoolExecutor, as_completed

def sanitize(num):
    """Turn a numeric value into a safe string for filenames."""
    s = str(num)
    s = s.replace('-', 'neg')   # no minus signs
    s = s.replace('.', 'p')     # no decimal points
    return s

# make sure temp folder exists
temp_dir = Path("temp")
temp_dir.mkdir(parents=True, exist_ok=True)

# 1) load your original script
base = Path("jackson_r4.py").read_text()

# 2) parameter grid
grad_vals             = [-2]
z_high_vals           = [2.0]
z_low_vals            = [0.2]
profit_margin_vals    = [17]
entering_agro_vals    = [1.5]   # ← your new sweep dimension

combos = list(itertools.product(
    grad_vals,
    z_high_vals,
    z_low_vals,
    profit_margin_vals,
    entering_agro_vals
))
total = len(combos)

def run_backtest(params):
    gv, zh, zl, pm, ea = params

    # patch literals
    code = base
    code = re.sub(r"grad\s*=\s*[-\d\.]+",
                  f"grad={gv}", code)
    code = re.sub(r"z_high\s*=\s*[-\d\.]+",
                  f"z_high={zh}", code)
    code = re.sub(r"z_low\s*=\s*[-\d\.]+",
                  f"z_low={zl}", code)
    code = re.sub(r"profit_margin\s*=\s*[\d]+",
                  f"profit_margin={pm}", code)
    code = re.sub(r"entering_aggression\s*=\s*[\d\.]+",
                  f"entering_aggression={ea}", code)

    # sandbox (include entering_agro in the dir-name)
    run_dir = temp_dir / f"{sanitize(gv)}_{sanitize(zh)}_{sanitize(zl)}_{pm}_{sanitize(ea)}"
    run_dir.mkdir(exist_ok=True)
    (run_dir / "jackson_tmp.py").write_text(code)
    shutil.copy("datamodel.py", run_dir / "datamodel.py")

    # backtest
    proc = subprocess.run(
        ["prosperity3bt", "jackson_tmp.py", "4"],
        cwd=run_dir,
        capture_output=True, text=True, check=False
    )
    out = proc.stdout + proc.stderr

    # parse PnL
    if proc.returncode != 0:
        pnl = float("-inf")
    else:
        m = re.search(r"Total profit:\s*([-\d,]+)", out)
        pnl = int(m.group(1).replace(",", "")) if m else 0

    # cleanup
    shutil.rmtree(run_dir, ignore_errors=True)
    return (gv, zh, zl, pm, ea, pnl)

# 4 at a time, batch by batch
results = []
start = time.time()
done = 0

for batch_start in range(0, total, 4):
    batch = combos[batch_start:batch_start+4]
    with ThreadPoolExecutor(max_workers=4) as ex:
        futures = { ex.submit(run_backtest, p): p for p in batch }
        for fut in as_completed(futures):
            gv, zh, zl, pm, ea, pnl = fut.result()
            done += 1
            results.append((gv, zh, zl, pm, ea, pnl))

            # ETA
            elapsed = time.time() - start
            avg = elapsed / done
            rem = total - done
            eta = avg * rem
            h, rem2 = divmod(eta, 3600)
            m, s = divmod(rem2, 60)
            if h >= 1:
                eta_str = f"{int(h)}h{int(m)}m{int(s)}s"
            elif m >= 1:
                eta_str = f"{int(m)}m{int(s)}s"
            else:
                eta_str = f"{int(s)}s"

            print(f"[{done}/{total}] ETA {eta_str} ✓ "
                  f"grad={gv}, z_high={zh}, z_low={zl}, "
                  f"profit_margin={pm}, entering_agro={ea} → PnL={pnl}")

# final top 5
top5 = sorted(results, key=lambda x: x[5], reverse=True)[:5]
print("\n=== Top 5 parameter sets by total PnL ===")
for gv, zh, zl, pm, ea, pnl in top5:
    print(f"grad={gv:>4}, z_high={zh:>3}, z_low={zl:>4}, "
          f"profit_margin={pm:>2}, entering_agro={ea:.2f} → PnL={pnl}")
