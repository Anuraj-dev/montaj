#!/usr/bin/env python3
"""Token spend per Claude Code session, from ~/.claude/projects/**/*.jsonl `usage` fields.

usage: token-report.py [--only ID_PREFIX,...] [GLOB ...]
  default GLOB: ~/.claude/projects/*claude-test*/*.jsonl
  input-equivalent = input + 1.25*cache_write + 0.1*cache_read + 5*output  (relative weights, not prices)
"""
import collections, glob, json, os, sys

args = sys.argv[1:]
only = None
if '--only' in args:
    i = args.index('--only'); only = args[i + 1].split(','); del args[i:i + 2]
pats = args or [os.path.expanduser('~/.claude/projects/*claude-test*/*.jsonl')]
files = sorted({f for p in pats for f in glob.glob(os.path.expanduser(p), recursive=True)})
if only: files = [f for f in files if any(os.path.basename(f).startswith(o) for o in only)]

K = ('input_tokens', 'cache_creation_input_tokens', 'cache_read_input_tokens', 'output_tokens')
def ieq(u): return u[K[0]] + 1.25 * u[K[1]] + 0.1 * u[K[2]] + 5 * u[K[3]]

tot = collections.Counter()
print(f"{'session':10} {'turns':>5} {'ctx0':>5} {'ctxmax':>6} {'imgs':>4} {'cache_rd':>9} {'cache_wr':>8} {'out':>7} {'in_eq':>8}  first prompt")
for f in files:
    u = collections.Counter(); seen = set(); ctx = []; imgs = 0; first = None
    for line in open(f):
        try: m = json.loads(line)
        except ValueError: continue
        msg = m.get('message') or {}
        if m.get('type') == 'user':
            c = msg.get('content')
            blocks = [{'type': 'text', 'text': c}] if isinstance(c, str) else (c if isinstance(c, list) else [])
            for b in blocks:
                if b.get('type') == 'text' and first is None and not b['text'].lstrip().startswith('<'): first = b['text']
                if b.get('type') == 'image': imgs += 1
                if b.get('type') == 'tool_result' and isinstance(b.get('content'), list):
                    imgs += sum(x.get('type') == 'image' for x in b['content'])
        elif m.get('type') == 'assistant' and msg.get('id') not in seen:
            seen.add(msg.get('id')); us = msg.get('usage') or {}
            for k in K: u[k] += us.get(k, 0) or 0
            ctx.append(sum(us.get(k, 0) or 0 for k in K[:3]))
    if not ctx: continue
    tot.update(u)
    print(f"{os.path.basename(f)[:8]:10} {len(ctx):5} {ctx[0]//1000:4}k {max(ctx)//1000:5}k {imgs:4} {u[K[2]]/1e6:8.2f}M "
          f"{u[K[1]]/1e6:7.2f}M {u[K[3]]/1e3:6.0f}k {ieq(u)/1e6:7.2f}M  {(first or '').strip()[:60]!r}")
print(f"TOTAL {len(files)} files: cache_read {tot[K[2]]/1e6:.2f}M  cache_write {tot[K[1]]/1e6:.2f}M  "
      f"output {tot[K[3]]/1e3:.0f}k  input-equivalent {ieq(tot)/1e6:.2f}M")
