"""
Clean-rebuild SIGNAL_SCRIPTS in run_all_local_tw_to_excel.py:
  - Collects ALL {'file':..,'name':..} entries found anywhere in the file
  - Deduplicates (keeps first occurrence)
  - Replaces everything between SIGNAL_SCRIPTS = [ and def get_ticker_from_code
"""
import re, sys
sys.stdout.reconfigure(encoding='utf-8')

with open('run_all_local_tw_to_excel.py', encoding='utf-8') as f:
    txt = f.read()

# ── 1. Grab every entry from the whole file ──────────────────────────────────
pattern = re.compile(r"\{'file':\s*'([^']+)',\s*'name':\s*'([^']+)'\}")
all_entries = pattern.findall(txt)          # list of (file, name)

seen = set()
unique_entries = []
for fpath, name in all_entries:
    if fpath not in seen:
        seen.add(fpath)
        unique_entries.append((fpath, name))

print(f"Total entries found : {len(all_entries)}")
print(f"Unique entries kept : {len(unique_entries)}")

# ── 2. Build the replacement SIGNAL_SCRIPTS block ────────────────────────────
lines = ["SIGNAL_SCRIPTS = ["]
for fpath, name in unique_entries:
    lines.append(f"    {{'file': '{fpath}', 'name': '{name}'}},")
lines.append("]")
new_block = "\n".join(lines) + "\n"

# ── 3. Locate the section to replace: from SIGNAL_SCRIPTS = [ up to def get_ticker_from_code ──
start_match = re.search(r'^SIGNAL_SCRIPTS\s*=\s*\[', txt, re.MULTILINE)
end_match   = re.search(r'^def get_ticker_from_code', txt, re.MULTILINE)

if not start_match or not end_match:
    print("ERROR: Could not locate markers in file")
    sys.exit(1)

start_pos = start_match.start()
end_pos   = end_match.start()

# Everything between the two markers is replaced with new_block + blank line
new_txt = txt[:start_pos] + new_block + "\n" + txt[end_pos:]

# ── 4. Write back ─────────────────────────────────────────────────────────────
with open('run_all_local_tw_to_excel.py', 'w', encoding='utf-8') as f:
    f.write(new_txt)

# ── 5. Verify with Python parser ─────────────────────────────────────────────
import py_compile, os, tempfile
tmp = tempfile.mktemp(suffix='.py')
with open(tmp, 'w', encoding='utf-8') as f:
    f.write(new_txt)
try:
    py_compile.compile(tmp, doraise=True)
    print("Syntax check: PASSED")
except py_compile.PyCompileError as e:
    print(f"Syntax check: FAILED — {e}")
finally:
    os.unlink(tmp)

# Count entries in final file
final_entries = re.findall(r"'file':\s*'(get_trading_signal_[^']+\.py)'", new_txt)
dupes = len(final_entries) - len(set(final_entries))
print(f"Final SIGNAL_SCRIPTS entries: {len(final_entries)}, duplicates: {dupes}")
