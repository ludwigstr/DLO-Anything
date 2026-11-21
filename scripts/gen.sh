#!/bin/bash
cd "$(dirname "$0")/.."
W=36
python scripts/make_u3.py
python -m dlo.generate_u_ood --families random --per-family 1500 --slim --out data/u3/broad_1.npz --seed 710000 --workers $W; python scripts/make_u3.py
python -m dlo.generate_u --episodes 3100 --out data/u3/narrow_1.npz --seed 600000 --workers $W | tail -2; python scripts/make_u3.py
python -m dlo.generate_u_ood --families random --per-family 1600 --slim --out data/u3/broad_2.npz --seed 712000 --workers $W; python scripts/make_u3.py
python -m dlo.generate_u --episodes 6100 --out data/u3/narrow_2.npz --seed 604000 --workers $W | tail -2; python scripts/make_u3.py
python -m dlo.generate_u_ood --families random --per-family 3000 --slim --out data/u3/broad_3.npz --seed 714000 --workers $W; python scripts/make_u3.py
python -m dlo.generate_u_ood --families random --per-family 6100 --slim --out data/u3/broad_4.npz --seed 718000 --workers $W; python scripts/make_u3.py
python -m dlo.generate_u --episodes 12300 --out data/u3/narrow_3.npz --seed 612000 --workers $W | tail -2; python scripts/make_u3.py
echo generation finished
