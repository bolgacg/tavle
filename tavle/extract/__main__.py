"""python -m tavle.extract eds Elspotprices [--start 2013-01-01] [--end ...]
   python -m tavle.extract ecb"""
import argparse
import sys

from . import ecb, eds


def main(argv=None):
    p = argparse.ArgumentParser(prog="tavle.extract")
    sub = p.add_subparsers(dest="source", required=True)
    a = sub.add_parser("eds")
    a.add_argument("dataset", choices=sorted(eds.DATASETS))
    a.add_argument("--start")
    a.add_argument("--end")
    a.add_argument("--months", type=int, default=12)
    b = sub.add_parser("ecb")
    b.add_argument("--start")
    args = p.parse_args(argv)
    if args.source == "eds":
        landed = eds.extract(args.dataset, args.start, args.end, months=args.months)
    else:
        landed = ecb.extract(args.start)
    for s, e, n, path in landed:
        print(f"{s} .. {e}: {n} records -> {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
