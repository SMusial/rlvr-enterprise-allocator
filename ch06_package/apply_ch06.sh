#!/bin/bash
set -e
echo "Applying Chapter 06..."

cp ch06_package/rlvr-core/src/ch06_td.rs rlvr-core/src/ch06_td.rs
echo "pub mod ch06_td;" >> rlvr-core/src/lib.rs

cp ch06_package/rlvr-py/src/lib.rs rlvr-py/src/lib.rs
cp ch06_package/gui/chapters/ch06.py gui/chapters/ch06.py
cp ch06_package/README.md README.md

python3 -c "
content = open('gui/app.py').read()
content = content.replace(
    'elif ch_num == 5:\n    from chapters.ch05 import render\n    render()',
    'elif ch_num == 5:\n    from chapters.ch05 import render\n    render()\nelif ch_num == 6:\n    from chapters.ch06 import render\n    render()'
)
open('gui/app.py', 'w').write(content)
print('app.py routing updated')
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
echo "Chapter 06 applied! Run: streamlit run gui/app.py"
