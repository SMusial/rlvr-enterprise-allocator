#!/bin/bash
# RLVR Chapter 05 — Apply script
# Run from project root: bash apply_ch05.sh

set -e
echo "Applying Chapter 05..."

# 1 — Rust core
cp ch05_package/rlvr-core/src/ch05_mc.rs rlvr-core/src/ch05_mc.rs
echo "pub mod ch05_mc;" >> rlvr-core/src/lib.rs

# 2 — Bridge
cp ch05_package/rlvr-py/src/lib.rs rlvr-py/src/lib.rs

# 3 — UI
cp ch05_package/gui/chapters/ch05.py gui/chapters/ch05.py

# 4 — README
cp ch05_package/README.md README.md

# 5 — Routing in app.py
python3 -c "
content = open('gui/app.py').read()
content = content.replace(
    'elif ch_num == 4:\n    from chapters.ch04 import render\n    render()',
    'elif ch_num == 4:\n    from chapters.ch04 import render\n    render()\nelif ch_num == 5:\n    from chapters.ch05 import render\n    render()'
)
open('gui/app.py', 'w').write(content)
print('app.py routing updated')
"

# 6 — Remove duplicate lib.rs module if needed
python3 -c "
lines = open('rlvr-core/src/lib.rs').readlines()
seen = set()
out = []
for line in lines:
    if line.strip().startswith('pub mod') and line.strip() in seen:
        continue
    seen.add(line.strip())
    out.append(line)
open('rlvr-core/src/lib.rs', 'w').writelines(out)
print('lib.rs deduped')
"

# 7 — Build and test
echo "Building..."
cd rlvr-py && maturin develop && cd ..
echo "Testing..."
cargo test --workspace

echo ""
echo "Chapter 05 applied successfully!"
echo "Run: streamlit run gui/app.py"
