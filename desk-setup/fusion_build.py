"""Build one mode of desk_spec.json into the active Fusion design (direct modelling).

Run inside Fusion with globals: SPEC_PATH, MODE ('monitor' | 'dual' | 'laptop' | 'notebook' | 'clear').
Each mode becomes its own component so modes can be toggled and checked separately.
"""
import adsk.core, adsk.fusion, json, math

def qmat(q):
    x, y, z, w = q
    return [[1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
            [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
            [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)]]

def compose(A, B):  # 3x4 affine
    R = [[sum(A[i][k]*B[k][j] for k in range(3)) for j in range(3)] for i in range(3)]
    t = [sum(A[i][k]*B[k][3] for k in range(3)) + A[i][3] for i in range(3)]
    return [R[i] + [t[i]] for i in range(3)]

def xf(M, p): return [sum(M[i][k]*p[k] for k in range(3)) + M[i][3] for i in range(3)]
def col(M, j): return [M[0][j], M[1][j], M[2][j]]

def build(spec_path, mode):
    app = adsk.core.Application.get()
    design = adsk.fusion.Design.cast(app.activeProduct)
    root = design.rootComponent
    S = json.load(open(spec_path))
    label = S['mode_label'].get(mode, mode)
    cname = 'Mode %d - %s' % (S['modes'].index(mode) + 1, label) if mode in S['modes'] else None

    if mode == 'clear':
        for i in range(root.occurrences.count - 1, -1, -1):
            root.occurrences.item(i).deleteMe()
        for i in range(root.bRepBodies.count - 1, -1, -1):
            root.bRepBodies.item(i).deleteMe()
        return {'cleared': True}
    for i in range(root.occurrences.count - 1, -1, -1):
        if root.occurrences.item(i).component.name == cname:
            root.occurrences.item(i).deleteMe()

    occ = root.occurrences.addNewComponent(adsk.core.Matrix3D.create())
    comp = occ.component; comp.name = cname
    tbm = adsk.fusion.TemporaryBRepManager.get()
    P = adsk.core.Point3D.create; V = adsk.core.Vector3D.create

    lib = app.materialLibraries.itemByName('Fusion Appearance Library')
    base = None
    for n in ['Plastic - Matte (Black)', 'Paint - Enamel Glossy (Black)', 'Plastic - Glossy (Black)']:
        base = lib.appearances.itemByName(n)
        if base: break
    cache = {}
    def appear(hexc):
        if hexc in cache: return cache[hexc]
        nm = 'desk ' + hexc
        a = design.appearances.itemByName(nm) or design.appearances.addByCopy(base, nm)
        r, g, b = int(hexc[1:3], 16), int(hexc[3:5], 16), int(hexc[5:7], 16)
        for p in a.appearanceProperties:
            cp = adsk.core.ColorProperty.cast(p)
            if cp and not cp.isReadOnly:
                try: cp.value = adsk.core.Color.create(r, g, b, 255)
                except Exception: pass
        cache[hexc] = a
        return a

    W = {}
    def world(gid):
        if gid in W: return W[gid]
        g = S['groups'][gid]; ps = g['poses'][mode]
        R = qmat(ps['quat']); M = [R[i] + [ps['pos'][i]] for i in range(3)]
        if g['parent']: M = compose(world(g['parent']), M)
        W[gid] = M
        return M

    made = []
    for p in S['parts']:
        g = S['groups'][p['group']]
        if not g['visible'][mode]: continue
        M = world(p['group'])
        if p['kind'] == 'box':
            c = xf(M, p['c']); a0, a1 = col(M, 0), col(M, 1)
            bb = adsk.core.OrientedBoundingBox3D.create(P(*c), V(*a0), V(*a1), p['s'][0], p['s'][1], p['s'][2])
            tb = tbm.createBox(bb)
        elif p['kind'] == 'cyl':
            tb = tbm.createCylinderOrCone(P(*xf(M, p['p1'])), p['r1'], P(*xf(M, p['p2'])), p['r2'])
        else:
            tb = tbm.createSphere(P(*xf(M, p['c'])), p['r'])
        b = comp.bRepBodies.add(tb)
        b.name = '%s / %s' % (g['label'], p['name'])
        hexc = (p.get('emissive') or {}).get(mode) or p['color']
        b.appearance = appear(hexc)
        if p['opacity'] < 1:
            try: b.opacity = p['opacity']
            except Exception: pass
        made.append((b, p['group']))

    # cables: fitted 3D spline + pipe; spare length as a torus coil (pipe on a circle)
    sk = comp.sketches.add(comp.xYConstructionPlane); sk.name = 'cable paths'
    sk.isComputeDeferred = True
    paths = []
    for c in S['cables']:
        pts = c['points'][mode][::4] + [c['points'][mode][-1]]
        coll = adsk.core.ObjectCollection.create()
        for q in pts: coll.add(P(*q))
        paths.append((c, sk.sketchCurves.sketchFittedSplines.add(coll)))
    sk.isComputeDeferred = False
    cables = []
    for c, spl in paths:
        try:
            path = comp.features.createPath(spl, False)
            pi = comp.features.pipeFeatures.createInput(path, adsk.fusion.FeatureOperations.NewBodyFeatureOperation)
            pi.sectionSize = adsk.core.ValueInput.createByReal(c['od'])
            pf = comp.features.pipeFeatures.add(pi)
            for i in range(pf.bodies.count):
                b = pf.bodies.item(i); b.name = 'Cable / ' + c['name']; b.appearance = appear(c['code'])
                cables.append((b, c))
        except Exception as e:
            cables.append((None, c))
        bd = c['bundle'][mode]
        if bd['R'] > 0.2:
            ax = bd['axis']; cc = bd['c']
            # circle in the plane normal to ax via a 3D sketch arc of points
            n = [ax[0], ax[1], ax[2]]
            e1 = [1, 0, 0] if abs(n[0]) < 0.9 else [0, 1, 0]
            u = [n[1]*e1[2]-n[2]*e1[1], n[2]*e1[0]-n[0]*e1[2], n[0]*e1[1]-n[1]*e1[0]]
            L = math.sqrt(sum(x*x for x in u)); u = [x / L for x in u]
            w = [n[1]*u[2]-n[2]*u[1], n[2]*u[0]-n[0]*u[2], n[0]*u[1]-n[1]*u[0]]
            tor = None
            for k in range(24):
                a0 = 2 * math.pi * k / 24; a1 = 2 * math.pi * (k + 1) / 24
                q0 = [cc[i] + bd['R'] * (math.cos(a0) * u[i] + math.sin(a0) * w[i]) for i in range(3)]
                q1 = [cc[i] + bd['R'] * (math.cos(a1) * u[i] + math.sin(a1) * w[i]) for i in range(3)]
                seg = tbm.createCylinderOrCone(P(*q0), bd['r'], P(*q1), bd['r'])
                if tor is None: tor = seg
                else: tbm.booleanOperation(tor, seg, adsk.fusion.BooleanTypes.UnionBooleanType)
            b = comp.bRepBodies.add(tor); b.name = 'Coil / %s (%.0f cm, %.1f turns)' % (c['id'], c['excess'][mode], bd['turns'])
            b.appearance = appear(c['code'])
    sk.isVisible = False

    # interference between solid parts (room + body envelope excluded)
    skip = ('room', 'person')
    coll = adsk.core.ObjectCollection.create()
    for b, gid in made:
        if gid not in skip: coll.add(b)
    ii = design.createInterferenceInput(coll); ii.areCoincidentFacesIncluded = False
    hits = []
    for r in design.analyzeInterference(ii):
        v = r.interferenceBody.volume if r.interferenceBody else 0
        if v > 0.05: hits.append('%s  x  %s  (%.2f cm3)' % (r.entityOne.name, r.entityTwo.name, v))
    # cables against solids (ends plug into ports, so small overlaps there are expected)
    coll2 = adsk.core.ObjectCollection.create()
    for b, gid in made:
        if gid not in skip: coll2.add(b)
    for b, c in cables:
        if b: coll2.add(b)
    ii2 = design.createInterferenceInput(coll2); ii2.areCoincidentFacesIncluded = False
    chits = []
    for r in design.analyzeInterference(ii2):
        n1, n2 = r.entityOne.name, r.entityTwo.name
        if ('Cable /' in n1) == ('Cable /' in n2): continue
        v = r.interferenceBody.volume if r.interferenceBody else 0
        if v > 0.05: chits.append('%s  x  %s  (%.2f cm3)' % (n1, n2, v))
    return {'mode': mode, 'bodies': comp.bRepBodies.count, 'pipes_ok': sum(1 for b, c in cables if b),
            'interference': hits, 'cable_contacts': chits}
