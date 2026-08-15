\
#!/usr/bin/env python3
import argparse, json
from collections import Counter

p = argparse.ArgumentParser()
p.add_argument("path")
p.add_argument("--show", type=int, default=3)
args = p.parse_args()

rows = [json.loads(x) for x in open(args.path, encoding="utf-8")]
print("rows:", len(rows))
for key in ["authorized", "source", "style", "mechanism", "domain", "candidate_action"]:
    print(f"\n{key}")
    print(Counter(r[key] for r in rows))

print("\nSAMPLES")
for r in rows[:args.show]:
    print("=" * 80)
    print(r["prompt"])
    print("TARGET:", json.dumps(r["target"], indent=2))
