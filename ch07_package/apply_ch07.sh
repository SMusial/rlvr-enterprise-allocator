#!/bin/bash
set -e
echo "Applying Chapter 07..."

cp ch07_package/rlvr-core/src/ch07_nstep.rs rlvr-core/src/ch07_nstep.rs
echo "pub mod ch07_nstep;" >> rlvr-core/src/lib.rs

cp ch07_package/rlvr-py/src/lib.rs rlvr-py/src/lib.rs
cp ch07_package/gui/chapters/ch07.py gui/chapters/ch07.py
cp ch07_package/README.md README.md

python3 -c "
content = open('gui/app.py').read()
old = 'elif ch_num == 6:\n    from chapters.ch06 import render\n    render()'
new = old + '\nelif ch_num == 7:\n    from chapters.ch07 import render\n    render()'
if old in content:
    content = content.replace(old, new)
    open('gui/app.py', 'w').write(content)
    print('app.py routing updated')
else:
    print('WARNING: ch06 routing not found in app.py — add ch07 routing manually')
"

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

cd rlvr-py && maturin develop && cd ..
cargo test --workspace

echo ""
echo "Chapter 07 applied! Run: streamlit run gui/app.py"
