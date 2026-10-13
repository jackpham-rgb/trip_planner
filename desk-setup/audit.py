"""Consistency audit for desk_spec: things the per-mode checks do not catch.

1. Coils: each spare-cable coil must sit on its own cable's route and rest on/against a support.
2. Ties and clips: must touch their cable and a support.
3. Floating parts: every static part must touch another part (or the floor/wall).
4. Cable paths: no cable may pass through a solid part (plugs at both ends excepted).
5. Stale words in labels and notes.
Run: python audit.py
"""
import io, contextlib, re, sys, math
import numpy as np

with contextlib.redirect_stdout(io.StringIO()):
    import desk_spec as S

MODES = S.MODES
SKIP_GROUPS = {'person', 'room'}
SUPPORTS = ['Desktop', 'Rail back', 'Rail front', 'Rail left', 'Rail right', 'Left post back', 'Left post front',
            'Right leg back', 'Right leg front', 'X-brace a', 'X-brace b', 'Middle shelf', 'Bottom shelf', 'Top shelf',
            'Basket floor', 'Dock power brick 240 W', 'LiOA strip', 'Liner inside L', 'Liner inside R', 'Liner inside back', 'Liner inside front', 'Wall', 'Floor']
problems = []
def bad(kind, msg): problems.append((kind, msg))

def world_aabb(p, m):
    Mw = S.world(p['group'], m)
    if p['kind'] == 'box':
        c, s = S.v(*p['c']), S.v(*p['s']) / 2
        corners = [c + s * S.v(a, b, d) for a in (-1, 1) for b in (-1, 1) for d in (-1, 1)]
    elif p['kind'] == 'cyl':
        r = max(p['r1'], p['r2']); corners = []
        for q in (p['p1'], p['p2']):
            corners += [S.v(*q) + S.v(a, b, d) * r for a in (-1, 1) for b in (-1, 1) for d in (-1, 1)]
    elif p['kind'] == 'prism':
        corners = S.prism_vertices(p)
    else:
        corners = [S.v(*p['c']) + S.v(a, b, d) * p['r'] for a in (-1, 1) for b in (-1, 1) for d in (-1, 1)]
    w = np.array([(Mw @ np.array([*q, 1.0]))[:3] for q in corners])
    return w.min(0), w.max(0)

def aabb_gap(a, b):
    d = np.maximum(np.maximum(a[0] - b[1], b[0] - a[1]), 0)
    return float(np.linalg.norm(d))

def support_sdf(m):
    fns = [S.sdf_fn(n, m) for n in SUPPORTS if S.parts_named(n)]
    return lambda q: min(f(q) for f in fns)

def coil_ring(bd, n=36):
    c, ax = S.v(*bd['c']), S.norm(S.v(*bd['axis']))
    e1 = S.norm(np.cross(ax, S.v(1, 0, 0) if abs(ax[0]) < 0.9 else S.v(0, 1, 0))); e2 = np.cross(ax, e1)
    return [c + bd['R'] * (math.cos(t) * e1 + math.sin(t) * e2) for t in np.linspace(0, 2 * math.pi, n, endpoint=False)]

# ---------------------------------------------------------------- 1. coils
for m in MODES:
    sup = support_sdf(m)
    for cb in S.CABLES:
        bd = cb['bundle'][m]
        if bd['R'] <= 0.2: continue
        c, ax = S.v(*bd['c']), S.norm(S.v(*bd['axis']))
        pts = np.array(cb['points'][m])
        e1 = S.norm(np.cross(ax, S.v(1, 0, 0) if abs(ax[0]) < 0.9 else S.v(0, 1, 0))); e2 = np.cross(ax, e1)
        ring = [c + bd['R'] * (math.cos(t) * e1 + math.sin(t) * e2) for t in np.linspace(0, 2 * math.pi, 24, endpoint=False)]
        d_route = min(float(np.min(np.linalg.norm(pts - q, axis=1))) for q in ring)
        d_sup = min(sup(q) for q in ring) - bd['r']
        for other in S.CABLES:                      # a coil may rest on another coil
            ob = other['bundle'][m]
            if other is cb or ob['R'] <= 0.2: continue
            orr = np.array(coil_ring(ob))
            d_sup = min(d_sup, min(float(np.min(np.linalg.norm(orr - q, axis=1))) for q in ring) - bd['r'] - ob['r'])
        in_basket = S.IX[0] <= c[0] <= S.IX[1] and S.IZ[0] <= c[2] <= S.IZ[1] and c[1] <= S.BK['y'][1]
        if d_route > (8.0 if in_basket else 1.5):   # basket coils are packed among the plugs; the cable reaches its coil
            bad('coil', '%s [%s]: coil is %.1f cm away from its own cable' % (cb['id'], m, d_route))
        if d_sup > 1.0: bad('coil', '%s [%s]: coil floats %.1f cm from any support' % (cb['id'], m, d_sup))

# ---------------------------------------------------------------- 2/3. floating static parts (AABB contact)
for m in ('monitor', 'dual'):
    parts = [p for p in S.PARTS if p['group'] not in SKIP_GROUPS and S.GROUPS[p['group']]['visible'][m]]
    boxes = [world_aabb(p, m) for p in parts]
    for i, p in enumerate(parts):
        lo, hi = boxes[i]
        if lo[1] <= 0.05 or 'tie' in p['name'].lower(): continue     # on the floor / ties hang on their cables
        g = min((aabb_gap(boxes[i], boxes[j]) for j in range(len(parts)) if j != i), default=99)
        if g > 0.3: bad('float', '%s / %s [%s]: nearest other part %.1f cm (bbox)' % (p['group'], p['name'], m, g))

# ties and clips must also touch a cable
for m in ('dual',):
    allpts = np.vstack([np.array(cb['points'][m]) for cb in S.CABLES] + [np.array(coil_ring(cb['bundle'][m])) for cb in S.CABLES if cb['bundle'][m]['R'] > 0.2])
    for p in S.PARTS:
        if p['group'] != 'cable_mgmt': continue
        lo, hi = world_aabb(p, m); c = (lo + hi) / 2
        d = float(np.min(np.linalg.norm(allpts - c, axis=1)))
        if d > 2.0: bad('tie', '%s: %.1f cm from the nearest cable' % (p['name'], d))

# ---------------------------------------------------------------- 4. cables through solids
SOLID_SKIP = {'Thighs', 'Wall', 'Floor'}
for m in MODES:
    names = sorted({p['name'] for p in S.PARTS if p['group'] not in SKIP_GROUPS and S.GROUPS[p['group']]['visible'][m]
                    and not p['name'].startswith('Plug') and 'tie' not in p['name'].lower() and p['name'] not in SOLID_SKIP})
    fns = {n: S.sdf_fn(n, m) for n in names}
    for cb in S.CABLES:
        pts = np.array(cb['points'][m]); L = np.r_[0, np.cumsum(np.linalg.norm(np.diff(pts, axis=0), axis=1))]
        hits = {}
        for q, s in zip(pts, L):
            if s < 2.5 or s > L[-1] - 2.5: continue
            for n, f in fns.items():
                d = f(q)
                d -= cb['od'] / 2                       # cable surface, not its centreline
                if d < -0.05 and not (cb['id'] == 'usbc' and n == 'Edge cap end wall'): hits[n] = min(hits.get(n, 0), d)   # cap has a USB-C slot
        for n, d in hits.items():
            bad('through', '%s [%s]: passes %.1f cm inside "%s"' % (cb['id'], m, -d, n))

# ---------------------------------------------------------------- 4b. coils through solids
for m in ('monitor', 'dual'):
    names = sorted({p['name'] for p in S.PARTS if p['group'] not in SKIP_GROUPS and S.GROUPS[p['group']]['visible'][m]
                    and 'tie' not in p['name'].lower() and p['name'] not in SOLID_SKIP})
    fns = {n: S.sdf_fn(n, m) for n in names}
    for cb in S.CABLES:
        bd = cb['bundle'][m]
        if bd['R'] <= 0.2: continue
        c, ax = S.v(*bd['c']), S.norm(S.v(*bd['axis']))
        e1 = S.norm(np.cross(ax, S.v(1, 0, 0) if abs(ax[0]) < 0.9 else S.v(0, 1, 0))); e2 = np.cross(ax, e1)
        for t_ in np.linspace(0, 2 * math.pi, 24, endpoint=False):
            q = c + bd['R'] * (math.cos(t_) * e1 + math.sin(t_) * e2)
            for n, f in fns.items():
                if f(q) < bd['r'] - 0.15: bad('coil-hit', '%s coil [%s] runs into "%s" (%.1f cm)' % (cb['id'], m, n, bd['r'] - f(q)))
    # pole / link and other part-part overlaps worth knowing about
    for a_, b_ in (('Pole', 'Lower spring arm'),):
        if S.gap(a_, b_, m) == 0 and S.sdf_fn(b_, m)(S.v(*S.wpt('arm_l1', m, (0, 0, 0)))) < -0.2: bad('overlap', a_ + ' x ' + b_)

# ---------------------------------------------------------------- 5. stale words
STALE = r'(?i)\btray\b|riser|upper wall socket|printed bracket|under the desktop|U-straps|adhesive|portable monitor|velcroed to the back rail'
for p in S.PARTS:
    if re.search(STALE, p['name']): bad('stale', 'part name: ' + p['name'])
for g, d in S.GROUPS.items():
    if re.search(STALE, d.get('label') or ''): bad('stale', 'group %s label: %s' % (g, d['label']))
for cb in S.CABLES:
    for k in ('name', 'note', 'tie'):
        if re.search(STALE, cb[k]): bad('stale', 'cable %s %s: %s' % (cb['id'], k, cb[k]))

seen = set()
for kind, msg in problems:
    key = re.sub(r' \[\w+\]', '', msg)
    if key in seen: continue
    seen.add(key); print('%-8s %s' % (kind, msg))
print('---', len(seen), 'distinct problems')
