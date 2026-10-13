"""Fusion 360 builder + animator for desk_spec.json (run through the Fusion MCP script runner).

Every spec group becomes its own component (nested like the spec), so a mode is just a set of
occurrence transforms. Steps are separate calls to stay inside the runner's time limit:
  init            new design document 'Desk setup v2' (direct modelling)
  parts           build all components + bodies, pose = 'monitor'
  cables:<mode>   pipe-swept cables + spare-length coils for one mode (own component)
  pose:<mode>     move everything to a mode instantly
  animate:<a>:<b> move from mode a to mode b live in the viewport (optionally saving frames)
  check:<mode>    Fusion interference analysis for a mode
"""
import adsk.core, adsk.fusion, json, math, os, time

APP = adsk.core.Application.get()
def design(): return adsk.fusion.Design.cast(APP.activeProduct)
P3 = adsk.core.Point3D.create; V3 = adsk.core.Vector3D.create

# ------------------------------------------------------------------ math
def qmat(q):
    x, y, z, w = q
    return [[1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)], [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
            [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)]]
def slerp(a, b, t):
    d = sum(i * j for i, j in zip(a, b))
    if d < 0: b = [-i for i in b]; d = -d
    if d > 0.9995:
        r = [i + (j - i) * t for i, j in zip(a, b)]
    else:
        th = math.acos(d); s = math.sin(th)
        r = [(math.sin((1 - t) * th) * i + math.sin(t * th) * j) / s for i, j in zip(a, b)]
    n = math.sqrt(sum(i * i for i in r)); return [i / n for i in r]
def matrix(pos, quat):
    n = math.sqrt(sum(q * q for q in quat)); quat = [q / n for q in quat]
    R = qmat(quat); m = adsk.core.Matrix3D.create()
    m.setWithArray([R[0][0], R[0][1], R[0][2], pos[0], R[1][0], R[1][1], R[1][2], pos[1],
                    R[2][0], R[2][1], R[2][2], pos[2], 0, 0, 0, 1])
    return m
def ease(t): return 4 * t * t * t if t < 0.5 else 1 - (-2 * t + 2) ** 3 / 2

# ------------------------------------------------------------------ appearances
_APP_CACHE = {}
def appearance(mat, color):
    d = design(); key = mat + '|' + (color if mat == 'plain' or mat.startswith('Rubber') or mat.startswith('Glass') else '')
    if key in _APP_CACHE: return _APP_CACHE[key]
    lib = APP.materialLibraries.itemByName('Fusion Appearance Library')
    if mat != 'plain' and not key.endswith('|'):
        base = lib.appearances.itemByName(mat)
    else:
        base = lib.appearances.itemByName(mat if mat != 'plain' else 'Plastic - Matte (White)')
    if key.endswith('|'):                       # textured / library look, use as is
        a = d.appearances.itemByName(mat) or d.appearances.addByCopy(base, mat)
    else:                                       # tinted copy
        nm = 'desk ' + key
        a = d.appearances.itemByName(nm)
        if a is None:
            a = d.appearances.addByCopy(base, nm)
            r, g, b = int(color[1:3], 16), int(color[3:5], 16), int(color[5:7], 16)
            for p in a.appearanceProperties:
                cp = adsk.core.ColorProperty.cast(p)
                if cp and not cp.isReadOnly:
                    try: cp.value = adsk.core.Color.create(r, g, b, 255)
                    except Exception: pass
    _APP_CACHE[key] = a
    return a

# ------------------------------------------------------------------ helpers
def load(path): return json.load(open(path, encoding='utf-8'))
def occ_map():
    """group id -> occurrence proxy in the root context (needed to move nested parts)."""
    out = {}
    allo = design().rootComponent.allOccurrences
    for i in range(allo.count):
        o = allo.item(i); out[o.component.name.split(' :: ')[0]] = o
    return out

def step_init():
    doc = APP.documents.add(adsk.core.DocumentTypes.FusionDesignDocumentType)
    doc.name = 'Desk setup v2'
    d = design(); d.designType = adsk.fusion.DesignTypes.DirectDesignType
    return {'doc': doc.name}

def build_part(comp, p):
    tbm = adsk.fusion.TemporaryBRepManager.get()
    if p['kind'] == 'box':
        c, s = p['c'], p['s']
        tb = tbm.createBox(adsk.core.OrientedBoundingBox3D.create(P3(*c), V3(1, 0, 0), V3(0, 1, 0), s[0], s[1], s[2]))
        body = comp.bRepBodies.add(tb)
    elif p['kind'] == 'cyl':
        body = comp.bRepBodies.add(tbm.createCylinderOrCone(P3(*p['p1']), p['r1'], P3(*p['p2']), p['r2']))
    elif p['kind'] == 'sphere':
        body = comp.bRepBodies.add(tbm.createSphere(P3(*p['c']), p['r']))
    else:  # prism: sketch on an arbitrary plane + extrude
        o, u, v = p['o'], p['u'], p['v']
        pl = adsk.core.Plane.createUsingDirections(P3(*o), V3(*u), V3(*v))
        ci = comp.constructionPlanes.createInput(); ci.setByPlane(pl)
        cp = comp.constructionPlanes.add(ci); cp.isLightBulbOn = False
        sk = comp.sketches.add(cp); sk.isVisible = False
        pts = [P3(o[0] + a * u[0] + b * v[0], o[1] + a * u[1] + b * v[1], o[2] + a * u[2] + b * v[2]) for a, b in p['pts']]
        sp = [sk.modelToSketchSpace(q) for q in pts]
        lines = sk.sketchCurves.sketchLines
        for i in range(len(sp)):
            lines.addByTwoPoints(sp[i], sp[(i + 1) % len(sp)])
        prof = sk.profiles.item(0)
        ext = comp.features.extrudeFeatures.addSimple(prof, adsk.core.ValueInput.createByReal(p['depth']),
                                                     adsk.fusion.FeatureOperations.NewBodyFeatureOperation)
        body = ext.bodies.item(0)
        # make sure it grew towards u x v
        n = (u[1]*v[2]-u[2]*v[1], u[2]*v[0]-u[0]*v[2], u[0]*v[1]-u[1]*v[0])
        bb = body.boundingBox; cen = [(bb.minPoint.asArray()[i] + bb.maxPoint.asArray()[i]) / 2 for i in range(3)]
        if sum((cen[i] - o[i]) * n[i] for i in range(3)) < 0:
            body.deleteMe()
            ext = comp.features.extrudeFeatures.addSimple(prof, adsk.core.ValueInput.createByReal(-p['depth']),
                                                         adsk.fusion.FeatureOperations.NewBodyFeatureOperation)
            body = ext.bodies.item(0)
    body.name = p['name']
    emis = (p.get('emissive') or {}).get('monitor')
    if emis: body.appearance = appearance('plain', emis)
    else: body.appearance = appearance(p['mat'], p['color'])
    if p['opacity'] < 1:
        try: body.opacity = p['opacity']
        except Exception: pass
    return body

def step_parts(S):
    d = design(); root = d.rootComponent
    for i in range(root.occurrences.count - 1, -1, -1): root.occurrences.item(i).deleteMe()
    G = S['groups']; made = {}
    order = []
    def add(gid):
        if gid in made: return
        g = G[gid]
        if g['parent']: add(g['parent'])
        parent = made[g['parent']].component if g['parent'] else root
        ps = g['poses']['monitor']
        o = parent.occurrences.addNewComponent(matrix(ps['pos'], ps['quat']))
        o.component.name = gid + ' :: ' + g['label']
        made[gid] = o; order.append(gid)
    for gid in G: add(gid)
    n = 0
    for p in S['parts']:
        build_part(made[p['group']].component, p); n += 1
    for gid in G:
        if not G[gid]['visible']['monitor']: made[gid].isLightBulbOn = False
    return {'components': len(made), 'bodies': n}


def step_groups(S, gids):
    """Build only these groups (creating parent components as needed); safe to re-run."""
    d = design(); root = d.rootComponent; G = S['groups']; om = occ_map(); n = 0
    def ensure(gid):
        if gid in om: return om[gid]
        g = G[gid]
        parent = ensure(g['parent']).component if g['parent'] else root
        ps = g['poses']['monitor']
        o = parent.occurrences.addNewComponent(matrix(ps['pos'], ps['quat']))
        o.component.name = gid + ' :: ' + g['label']; om[gid] = o
        return o
    for gid in gids:
        o = ensure(gid); comp = o.component
        for i in range(comp.bRepBodies.count - 1, -1, -1): comp.bRepBodies.item(i).deleteMe()
        for p in S['parts']:
            if p['group'] == gid: build_part(comp, p); n += 1
        if not G[gid]['visible']['monitor']: o.isLightBulbOn = False
    return {'groups': gids, 'bodies': n}


def step_part(S, gid, idx):
    """Add the idx-th part of a group (no deletion); for slow sketch-based bodies."""
    om = occ_map()
    if gid not in om: step_groups_empty(S, gid); om = occ_map()
    parts = [p for p in S['parts'] if p['group'] == gid]
    b = build_part(om[gid].component, parts[idx])
    return {'group': gid, 'part': parts[idx]['name'], 'of': len(parts)}

def step_groups_empty(S, gid):
    d = design(); root = d.rootComponent; G = S['groups']; g = G[gid]
    parent = occ_map()[g['parent']].component if g['parent'] else root
    ps = g['poses']['monitor']
    o = parent.occurrences.addNewComponent(matrix(ps['pos'], ps['quat'])); o.component.name = gid + ' :: ' + g['label']

def step_cables(S, mode, n_seg=36):
    """Cables as chains of short cylinders + joint spheres (fast, no sweeps or booleans).
    One sub-component per cable so the browser stays tidy."""
    d = design(); root = d.rootComponent
    name = 'cables_' + mode
    for i in range(root.occurrences.count - 1, -1, -1):
        if root.occurrences.item(i).component.name.startswith(name + ' '): root.occurrences.item(i).deleteMe()
    occ = root.occurrences.addNewComponent(adsk.core.Matrix3D.create())
    occ.component.name = name + ' :: Cables - ' + S['mode_label'][mode]
    tbm = adsk.fusion.TemporaryBRepManager.get(); total = 0
    for c in S['cables']:
        co = occ.component.occurrences.addNewComponent(adsk.core.Matrix3D.create())
        comp = co.component; comp.name = 'cable %s (%.0f of %.0f cm)' % (c['id'], c['length'][mode], c['rated'])
        ap = appearance('Rubber - Soft', c['code']); r = c['od'] / 2
        pts = c['points'][mode]; step = max(1, len(pts) // n_seg)
        pts = pts[::step] + ([pts[-1]] if (len(pts) - 1) % step else [])
        for i in range(len(pts) - 1):
            b = comp.bRepBodies.add(tbm.createCylinderOrCone(P3(*pts[i]), r, P3(*pts[i + 1]), r)); b.appearance = ap; total += 1
            if 0 < i:
                s_ = comp.bRepBodies.add(tbm.createSphere(P3(*pts[i]), r)); s_.appearance = ap; total += 1
        bd = c['bundle'][mode]
        if bd['R'] > 0.2:
            ax, cc = bd['axis'], bd['c']
            e1 = [1, 0, 0] if abs(ax[0]) < 0.9 else [0, 1, 0]
            u = [ax[1]*e1[2]-ax[2]*e1[1], ax[2]*e1[0]-ax[0]*e1[2], ax[0]*e1[1]-ax[1]*e1[0]]
            L = math.sqrt(sum(x * x for x in u)); u = [x / L for x in u]
            w = [ax[1]*u[2]-ax[2]*u[1], ax[2]*u[0]-ax[0]*u[2], ax[0]*u[1]-ax[1]*u[0]]
            for k in range(14):
                a0 = 2 * math.pi * k / 14; a1 = 2 * math.pi * (k + 1) / 14
                q0 = [cc[i] + bd['R'] * (math.cos(a0) * u[i] + math.sin(a0) * w[i]) for i in range(3)]
                q1 = [cc[i] + bd['R'] * (math.cos(a1) * u[i] + math.sin(a1) * w[i]) for i in range(3)]
                b = comp.bRepBodies.add(tbm.createCylinderOrCone(P3(*q0), bd['r'], P3(*q1), bd['r'])); b.appearance = ap
                b.name = 'spare coil'; total += 1
    show_cables(mode)
    return {'mode': mode, 'bodies': total}

def step_save(desc='desk setup', out_dir=None, name='Desk setup v2.f3d'):
    """Export a local .f3d next to the spec (cloud save is unreliable from the runner)."""
    d = design(); em = d.exportManager
    fn = os.path.join(out_dir or SPEC_DIR, name)
    opts = em.createFusionArchiveExportOptions(fn)
    ok = em.execute(opts)
    return {'saved_local': fn, 'ok': bool(ok), 'bytes': os.path.getsize(fn) if os.path.exists(fn) else 0}

def show_cables(mode):
    root = design().rootComponent
    for i in range(root.occurrences.count):
        o = root.occurrences.item(i)
        if o.component.name.startswith('cables_'):
            o.isLightBulbOn = o.component.name.startswith('cables_' + mode + ' ')

def set_screens(S, mode):
    om = occ_map()
    for p in S['parts']:
        if not p.get('emissive'): continue
        comp = om[p['group']].component
        for i in range(comp.bRepBodies.count):
            b = comp.bRepBodies.item(i)
            if b.name == p['name']:
                e = p['emissive'].get(mode)
                b.appearance = appearance('plain', e) if e else appearance(p['mat'], p['color'])

def _mul(A, B):
    return [[sum(A[i][k] * B[k][j] for k in range(4)) for j in range(4)] for i in range(4)]
def _local(g, a, b, t):
    pa, pb = g['poses'][a], g['poses'][b]
    pos = [pa['pos'][i] + (pb['pos'][i] - pa['pos'][i]) * t for i in range(3)]
    q = slerp(pa['quat'], pb['quat'], t); n = math.sqrt(sum(x * x for x in q)); q = [x / n for x in q]
    R = qmat(q)
    return [R[0] + [pos[0]], R[1] + [pos[1]], R[2] + [pos[2]], [0, 0, 0, 1]]
def apply_pose(S, a, b, t, om):
    G = S['groups']; W = {}
    def world(gid):
        if gid not in W:
            L = _local(G[gid], a, b, t)
            W[gid] = _mul(world(G[gid]['parent']), L) if G[gid]['parent'] else L
        return W[gid]
    for gid, g in G.items():
        if gid not in om: continue
        if g['poses'][a] == g['poses'][b] and not (g['parent'] and G[g['parent']]['poses'][a] != G[g['parent']]['poses'][b]) and t not in (0, 1):
            continue
        M = world(gid); m = adsk.core.Matrix3D.create()
        m.setWithArray([M[0][0], M[0][1], M[0][2], M[0][3], M[1][0], M[1][1], M[1][2], M[1][3],
                        M[2][0], M[2][1], M[2][2], M[2][3], 0, 0, 0, 1])
        om[gid].transform2 = m
        va, vb = g['visible'][a], g['visible'][b]
        if va != vb: om[gid].isLightBulbOn = (vb if t > 0.5 else va)

def step_pose(S, mode):
    om = occ_map(); apply_pose(S, mode, mode, 1, om)
    for gid, g in S['groups'].items():
        if gid in om: om[gid].isLightBulbOn = g['visible'][mode]
    set_screens(S, mode); show_cables(mode)
    APP.activeViewport.refresh()
    return {'mode': mode}

def step_animate(S, a, b, frames=36, seconds=3.0, out_dir=None):
    om = occ_map(); vp = APP.activeViewport
    show_cables('none')
    saved = []
    for k in range(frames + 1):
        t = ease(k / frames)
        apply_pose(S, a, b, t, om)
        vp.refresh(); adsk.doEvents()
        if out_dir:
            fn = os.path.join(out_dir, '%s_%s_%03d.png' % (a, b, k)); vp.saveAsImageFile(fn, 960, 600); saved.append(fn)
        else:
            time.sleep(seconds / frames)
    set_screens(S, b); show_cables(b); vp.refresh()
    return {'from': a, 'to': b, 'frames': len(saved)}

def _key(b):
    bb = b.boundingBox
    return b.name + '|' + ','.join('%.2f' % x for x in bb.minPoint.asArray() + bb.maxPoint.asArray())

def step_check(S, mode):
    """Interference between different parts in one mode; parts touching by design within one
    component, and cable-to-cable contacts, are not reported."""
    d = design(); step_pose(S, mode)
    root = d.rootComponent; allo = root.allOccurrences
    coll = adsk.core.ObjectCollection.create(); OWN = {}
    for i in range(allo.count):
        o = allo.item(i); path = o.fullPathName
        top = path.split('+')[0].split(' :: ')[0]
        if top in ('room', 'person'): continue
        if top.startswith('cables_') and not top == 'cables_' + mode: continue
        if not o.isVisible: continue
        for j in range(o.component.bRepBodies.count):
            b = o.component.bRepBodies.item(j)
            if b.isLightBulbOn:
                px = b.createForAssemblyContext(o); coll.add(px); OWN[_key(px)] = o.fullPathName
    ii = d.createInterferenceInput(coll); ii.areCoincidentFacesIncluded = False
    agg = {}; raw = 0; unknown = 0; same = 0
    for r in d.analyzeInterference(ii):
        raw += 1
        p1, p2 = OWN.get(_key(r.entityOne), '?'), OWN.get(_key(r.entityTwo), '?')
        if '?' in (p1, p2): unknown += 1
        if p1 == p2: same += 1; continue
        n1 = p1.split('+')[-1]; n2 = p2.split('+')[-1]
        if n1.startswith('cable ') and n2.startswith('cable '): continue
        vol = r.interferenceBody.volume if r.interferenceBody else 0
        if vol < 0.05: continue
        k = ' x '.join(sorted([n1.split(' :: ')[-1], n2.split(' :: ')[-1]]))
        e = agg.setdefault(k, [0, 0.0, '']); e[0] += 1; e[1] += vol; e[2] = e[2] or (r.entityOne.name + ' / ' + r.entityTwo.name)
    return {'mode': mode, 'bodies': coll.count, 'raw': raw, 'unknown': unknown, 'same_part': same,
            'pairs': sorted([[k, v[0], round(v[1], 2), v[2]] for k, v in agg.items()], key=lambda x: -x[2])[:40]}


VIEWS = {'over': ((260, 260, 420), (48, 70, 30)), 'anim': ((170, 190, 255), (50, 84, 30)), 'front': ((62, 150, 330), (55, 80, 25)),
         'back': ((10, 160, -260), (50, 70, 10)), 'seat': ((62, 132.7, 82.2), (62, 98, 25)),
         'basket': ((-140, 80, 200), (14, 22, 30)), 'dock': ((95, 35, 110), (44, 70, 12)),
         'dockside': ((-60, 55, 60), (40, 70, 12))}
def step_view(name):
    vp = APP.activeViewport; cam = vp.camera
    eye, tgt = VIEWS[name]
    cam.isSmoothTransition = False
    cam.cameraType = adsk.core.CameraTypes.PerspectiveCameraType
    cam.target = P3(*tgt); cam.eye = P3(*eye); cam.upVector = V3(0, 1, 0)
    cam.isFitView = False
    vp.camera = cam
    om = occ_map()
    if 'room' in om:   # hide the wall for the back view
        comp = om['room'].component
        for i in range(comp.bRepBodies.count):
            b = comp.bRepBodies.item(i)
            if b.name == 'Wall': b.isLightBulbOn = (name != 'back')
    vp.refresh()
    return {'view': name}

# ------------------------------------------------------------------ joints (drag in Fusion)
# As-built joints from the current pose, one subsystem per call. Pose to a mode BEFORE adding
# joints; afterwards the pose:/animate: steps are not used on that file (joints own the motion).
MOVERS = {'laptop', 'lid', 'keyboard', 'mouse', 'notebook', 'panel', 'arm_l1', 'arm_l1b', 'arm_l2',
          'arm_head', 'arm_yoke', 'lamp_yaw', 'lamp_l1', 'lamp_l2', 'lamp_head'}
def _frame(o):
    """world origin + local x/y/z axes of an occurrence proxy."""
    m = o.transform2; a = m.asArray()
    return [a[3], a[7], a[11]], [a[0], a[4], a[8]], [a[1], a[5], a[9]], [a[2], a[6], a[10]]
def _cpt(p, name):
    root = design().rootComponent; ci = root.constructionPoints.createInput()
    ci.setByPoint(P3(*p)); c = root.constructionPoints.add(ci); c.name = name; c.isLightBulbOn = False; return c
def _cax(p, d, name):
    root = design().rootComponent; ci = root.constructionAxes.createInput()
    ci.setByLine(adsk.core.InfiniteLine3D.create(P3(*p), V3(*d))); c = root.constructionAxes.add(ci)
    c.name = name; c.isLightBulbOn = False; return c
def _revolute(name, moving, fixed, p, d, lim=None):
    """lim = (min_deg, max_deg) relative to the current pose."""
    root = design().rootComponent
    old = root.asBuiltJoints.itemByName(name)
    if old: old.deleteMe()
    ji = root.asBuiltJoints.createInput(moving, fixed, adsk.fusion.JointGeometry.createByPoint(_cpt(p, name + ' pt')))
    ji.setAsRevoluteJointMotion(adsk.fusion.JointDirections.CustomJointDirection, _cax(p, d, name + ' axis'))
    j = root.asBuiltJoints.add(ji); j.name = name
    if lim:
        rl = j.jointMotion.rotationLimits
        rl.isMinimumValueEnabled = True; rl.minimumValue = math.radians(lim[0])
        rl.isMaximumValueEnabled = True; rl.maximumValue = math.radians(lim[1])
    return j
def _rigid(name, moving, fixed, p):
    root = design().rootComponent
    old = root.asBuiltJoints.itemByName(name)
    if old: old.deleteMe()
    ji = root.asBuiltJoints.createInput(moving, fixed, adsk.fusion.JointGeometry.createByPoint(_cpt(p, name + ' pt')))
    ji.setAsRigidJointMotion(); j = root.asBuiltJoints.add(ji); j.name = name; return j

def step_ground():
    """Ground everything that does not move by hand (desk, room, dock, cables...)."""
    root = design().rootComponent; out = []
    for i in range(root.occurrences.count):
        o = root.occurrences.item(i); g = o.component.name.split(' :: ')[0]
        o.isGrounded = g not in MOVERS and not g.startswith('lamp_') or g == 'lamp_base'
        if o.isGrounded: out.append(g)
    return {'grounded': len(out)}

def step_joints(S, which):
    om = occ_map(); root = design().rootComponent; made = []
    if which == 'arm':
        A = {'pitch_min': -35.0, 'pitch_max': 29.5}          # NB A5 / FlowLift Pro upper-arm range
        p1, _, y1, _ = _frame(om['arm_l1'])
        made.append(_revolute('Arm swivel at pole', om['arm_l1'], om['arm'], p1, y1).name)
        # elbow yoke: same place as the upper arm but yaw only, so yaw and pitch are separate joints
        E, x2, y2, z2 = _frame(om['arm_l2'])
        pitch = math.degrees(math.asin(max(-1, min(1, x2[1]))))
        h = math.hypot(x2[0], x2[2]); xy = [x2[0] / h, 0, x2[2] / h]; zy = [-xy[2], 0, xy[0]]   # x cross y
        if 'arm_yoke' not in om:
            occ = root.occurrences.addNewComponent(adsk.core.Matrix3D.create())
            occ.component.name = 'arm_yoke :: Arm elbow yoke'
            tb = adsk.fusion.TemporaryBRepManager.get()
            cyl = tb.createCylinderOrCone(P3(0, -1.2, 0), 2.0, P3(0, 1.2, 0), 2.0)
            bb = occ.component.bRepBodies.add(cyl); bb.name = 'Elbow cap'
            bb.appearance = appearance('Plastic - Matte (Black)', '#1b1b1b')
            om = occ_map()
        m = adsk.core.Matrix3D.create()
        m.setWithArray([xy[0], 0, zy[0], E[0], xy[1], 1, zy[1], E[1], xy[2], 0, zy[2], E[2], 0, 0, 0, 1])
        om['arm_yoke'].transform2 = m
        made.append(_revolute('Arm elbow swivel', om['arm_yoke'], om['arm_l1'], E, [0, 1, 0]).name)
        made.append(_revolute('Arm height (spring)', om['arm_l2'], om['arm_yoke'], E, zy,
                              (A['pitch_min'] - pitch, A['pitch_max'] - pitch)).name)
        H, _, yh, _ = _frame(om['arm_head'])
        jt = _revolute('Arm head level', om['arm_head'], om['arm_l2'], H, zy)
        made.append(jt.name)
        # spring arms are parallelograms: the head stays level when the arm goes up or down
        try:
            jl = root.asBuiltJoints.itemByName('Arm height (spring)')
            mi = root.motionLinks.createInput(jl, jt)
            mi.valueOne = adsk.core.ValueInput.createByString('10 deg'); mi.valueTwo = adsk.core.ValueInput.createByString('-10 deg')
            root.motionLinks.add(mi); made.append('motion link: head stays level')
        except Exception as e:
            made.append('motion link failed: %s' % e)
        made.append(_revolute('Monitor swivel', om['panel'], om['arm_head'], H, [0, 1, 0], (-90, 90)).name)
        made.append(_rigid('Arm link rigid', om['arm_l1b'], om['arm_l1'], p1).name)
    elif which == 'lid':
        P, x, _, _ = _frame(om['lid'])
        made.append(_revolute('Laptop hinge', om['lid'], om['laptop'], P, x).name)
    elif which == 'lamp':
        p, _, y, _ = _frame(om['lamp_yaw'])
        made.append(_revolute('Lamp base swivel', om['lamp_yaw'], om['lamp_base'], p, y).name)
        p, x, _, _ = _frame(om['lamp_l1'])
        made.append(_revolute('Lamp shoulder', om['lamp_l1'], om['lamp_yaw'], p, x).name)
        p, x, _, _ = _frame(om['lamp_l2'])
        made.append(_revolute('Lamp elbow', om['lamp_l2'], om['lamp_l1'], p, x).name)
        p, _, _, _ = _frame(om['lamp_head']); _, x, _, _ = _frame(om['lamp_l2'])
        made.append(_revolute('Lamp head tilt', om['lamp_head'], om['lamp_l2'], p, x).name)
    else:
        raise ValueError(which)
    APP.activeViewport.refresh()
    return {'joints': which, 'made': made}

def step_joint_list():
    root = design().rootComponent; out = []
    for i in range(root.asBuiltJoints.count):
        j = root.asBuiltJoints.item(i); out.append([j.name, j.jointMotion.jointType, bool(j.healthState == 0)])
    return {'as_built': out, 'links': root.motionLinks.count if hasattr(root, 'motionLinks') else '-'}

SPEC_DIR = None
def run_step(spec_path, step, out_dir=None):
    global SPEC_DIR
    SPEC_DIR = os.path.dirname(spec_path)
    S = load(spec_path)
    if step == 'init': return step_init()
    if step == 'save': return step_save()
    if step == 'parts': return step_parts(S)
    kind, *args = step.split(':')
    if kind == 'groups': return step_groups(S, args[0].split(','))
    if kind == 'part': return step_part(S, args[0], int(args[1]))
    if kind == 'cables': return step_cables(S, args[0])
    if kind == 'view': return step_view(args[0])
    if kind == 'save': return step_save(args[0] if args else 'desk setup')
    if kind == 'pose': return step_pose(S, args[0])
    if kind == 'animate': return step_animate(S, args[0], args[1], out_dir=(os.path.join(SPEC_DIR, 'frames') if len(args) > 2 and args[2] == 'save' else None))
    if kind == 'check': return step_check(S, args[0])
    if kind == 'ground': return step_ground()
    if kind == 'joints': return step_joints(S, args[0])
    if kind == 'jointlist': return step_joint_list()
    if kind == 'savejoints': return step_save(out_dir=None, name='Desk setup v2 joints.f3d')
    raise ValueError(step)
