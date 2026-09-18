#!/usr/bin/env python3
"""Tabulate the BENCH lines emitted by bench_poisson_mpi.

    python3 tabulate_bench.py logfile

One table per grid size, rows (P, strategy).  Parallel efficiency is
T(1)/(P*T(P)) against the P=1 run of the SAME size and strategy (at
P=1 both strategies run identical code -- transpose_mpi returns right
after the local diagonal block -- so the two baselines differ only by
run-to-run noise, which is itself worth reading as the noise floor).

`imbal%` is (max_rank_total - min_rank_total)/max_rank_total: the load
spread the block decomposition leaves when P does not divide nx.
"""
import sys, re

# Args are  [label=]logfile ...  -- the BENCH line records P but not the
# node count, so the configuration label comes from the file it was
# found in (e.g. 1node=logfile  2node=logfile.2node).
rows=[]; skipped=[]; labels=[]
for arg in sys.argv[1:]:
    label, _, path = arg.partition('=')
    if not path: label, path = '', label
    if label and label not in labels: labels.append(label)
    for line in open(path, errors='replace'):
        if 'BENCH' not in line: continue
        if 'SKIPPED' in line:
            f=line.split()
            skipped.append((int(f[1]),int(f[3]),label)); continue
        f=line.split()
        try: nx,ny,P,si = int(f[1]),int(f[2]),int(f[3]),int(f[4])
        except ValueError: continue
        kv={}
        toks=f[6:]
        for i in range(0,len(toks)-1,2):
            try: kv[toks[i]]=float(toks[i+1])
            except ValueError: pass
        rows.append(dict(N=nx,P=P,s=si,name=f[5],cfg=label,**kv))
if not labels: labels=['']

SN={1:'dealing-staggered',2:'crystal-router'}
# Efficiency baseline: P=1 of the same size, strategy AND configuration.
# P=1 cannot be split over two nodes, so a 2-node sweep's baseline is
# still a single-node serial run -- stated here rather than hidden.
base={}
for r in rows:
    if r['P']==1: base[(r['N'],r['s'],r['cfg'])]=r['total']

sizes=sorted({r['N'] for r in rows})
hdr=(f"{'P':>4} {'cfg':<6} {'strategy':<18} {'total(s)':>11} {'fst_y':>10} {'fst_x':>10} "
     f"{'transp':>10} {'..msg':>10} {'..pack':>10} {'..local':>10} "
     f"{'divide':>10} {'copy':>10} {'GFLOPS':>8} {'eff%':>7} {'imbal%':>7} {'max|err|':>11}")
for N in sizes:
    print(f"\n### Nx = Ny = {N}   (nx = ny = {N-1})")
    print(hdr); print('-'*len(hdr))
    for P in sorted({r['P'] for r in rows if r['N']==N}):
      for cfg in labels:
        for s in (1,2):
            m=[r for r in rows if r['N']==N and r['P']==P and r['s']==s
               and r['cfg']==cfg]
            if not m: continue
            r=m[0]
            b=base.get((N,s,cfg))
            eff = 100.0*b/(P*r['total']) if b else float('nan')
            imb = 100.0*(r['total']-r['tmin'])/r['total'] if r['total'] else 0.0
            print(f"{P:>4} {cfg:<6} {SN[s]:<18} {r['total']:>11.4e} {r['fst_y']:>10.3e} "
                  f"{r['fst_x']:>10.3e} {r['transpose']:>10.3e} {r['tr_msg']:>10.3e} "
                  f"{r['tr_pack']:>10.3e} {r['tr_local']:>10.3e} {r['divide']:>10.3e} "
                  f"{r['copy']:>10.3e} {r['gflops']:>8.2f} {eff:>7.1f} {imb:>7.1f} "
                  f"{r['err']:>11.4e}")
    seen=set()
    for n,P,cfg in skipped:
        if n==N and (P,cfg) not in seen:
            seen.add((P,cfg))
            print(f"{P:>4} {cfg:<6} {'(both)':<18} {'skipped: P > nx':>11}")
print(f"\n{len(rows)} runs, {len(skipped)} skipped")
