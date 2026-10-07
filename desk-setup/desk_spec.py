"""Desk setup scene spec (v2, monitor on an NB A5 arm): single source of truth for the
Fusion model, its mode animation and the web viewer.

Units: cm. Axes: x = left -> right (seated view), y = up, z = wall (0) -> user.
Run: python desk_spec.py  -> writes desk_spec.json and prints the check report.
"""
import json, math
import numpy as np

IN = 2.54
MODES = ['monitor', 'dual', 'laptop', 'notebook']
MODE_LABEL = {'monitor': 'Monitor only', 'dual': 'Laptop + monitor', 'laptop': 'Laptop only', 'notebook': 'Notebook + pens'}

# ---------------------------------------------------------------- parameters (tagged)
P = dict(
    wall_gap=2.5 * IN,          # decided with user (2-3 in)
    desk_w=36.0 * IN, desk_d=21.2 * IN,           # listing
    desk_top=30.0 * IN,         # ASSUMPTION: listing photo ratio 30.1 in; please tape
    desk_thk=1.8, rail=2.0,     # photo estimate
    shelf_w=9.0 * IN, shelf_d=20.0 * IN, top_shelf_d=16.5 * IN,   # user tape
    shelf_thk=1.6, top_shelf_top=36.0 * IN, mid_shelf_top=51.2, bot_shelf_under=3.5 * IN,
    post_x=4.0 * IN,
    baseboard_h=9.6, baseboard_t=1.5, outlet_plate=(18.6, 30.0),  # photo estimate
    eye_above_seat=78.0, elbow_above_seat=23.5, popliteal=44.5, thigh_thk=16.0,  # anthropometry 175 cm
)
TOP = P['desk_top']; Z0 = P['wall_gap']; Z1 = Z0 + P['desk_d']; W = P['desk_w']
UND = TOP - P['desk_thk']; RAIL_Y0 = UND - P['rail']
SH_W = P['shelf_w']; SH_Z0 = Z0 + (P['desk_d'] - P['shelf_d']) / 2; SH_Z1 = SH_Z0 + P['shelf_d']
POST_X0 = P['post_x']; POST_X1 = POST_X0 + 2.0
POST_ZB = (SH_Z0 + 3.0, SH_Z0 + 5.0); POST_ZF = (SH_Z1 - 5.0, SH_Z1 - 3.0)
TS_Y1 = P['top_shelf_top']; TS_Y0 = TS_Y1 - P['shelf_thk']
TS_ZC = (POST_ZB[0] + POST_ZF[1]) / 2; TS_Z0 = TS_ZC - P['top_shelf_d'] / 2; TS_Z1 = TS_ZC + P['top_shelf_d'] / 2
MS_Y1 = P['mid_shelf_top']; MS_Y0 = MS_Y1 - P['shelf_thk']
BS_Y0 = P['bot_shelf_under']; BS_Y1 = BS_Y0 + P['shelf_thk']
SEAT = TOP + 1.5 - P['elbow_above_seat'] + 0.5
EYE_Y = SEAT + P['eye_above_seat']
FOOTREST = round(SEAT - P['popliteal'], 1)
CX = 64.0                       # monitor centre line (arm geometry, see ARM)
UX = 62.0                       # where the user sits
LX = 60.0                       # laptop centre in dual / laptop modes

# ---------------------------------------------------------------- math
def v(*a): return np.array(a, dtype=float)
def norm(a): return a / np.linalg.norm(a)
def q_axis(axis, deg):
    axis = norm(v(*axis)); h = math.radians(deg) / 2
    return np.array([*(axis * math.sin(h)), math.cos(h)])
def q_mul(a, b):
    x1, y1, z1, w1 = a; x2, y2, z2, w2 = b
    return np.array([w1*x2 + x1*w2 + y1*z2 - z1*y2, w1*y2 - x1*z2 + y1*w2 + z1*x2,
                     w1*z2 + x1*y2 - y1*x2 + z1*w2, w1*w2 - x1*x2 - y1*y2 - z1*z2])
def q_from_to(a, b):
    a = norm(a); b = norm(b); c = np.cross(a, b); d = float(np.dot(a, b))
    if d < -0.999999:
        ax = np.cross(a, v(1, 0, 0))
        if np.linalg.norm(ax) < 1e-6: ax = np.cross(a, v(0, 1, 0))
        return q_axis(ax, 180)
    q = np.array([*c, 1 + d]); return q / np.linalg.norm(q)
def q_mat(q):
    x, y, z, w = q
    return np.array([[1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
                     [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
                     [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)]])
QI = np.array([0, 0, 0, 1.0])
def q_basis(xa, ya):
    """Quaternion whose local x,y axes map to the given world directions."""
    xa = norm(v(*xa)); ya = norm(v(*ya) - np.dot(v(*ya), xa) * xa); za = np.cross(xa, ya)
    m = np.column_stack([xa, ya, za]); t = np.trace(m)
    if t > 0:
        s = math.sqrt(t + 1) * 2; return np.array([(m[2,1]-m[1,2])/s, (m[0,2]-m[2,0])/s, (m[1,0]-m[0,1])/s, 0.25*s])
    i = int(np.argmax([m[0,0], m[1,1], m[2,2]])); j, k = (i+1) % 3, (i+2) % 3
    s = math.sqrt(1 + m[i,i] - m[j,j] - m[k,k]) * 2; q = np.zeros(4)
    q[i] = 0.25*s; q[j] = (m[j,i]+m[i,j])/s; q[k] = (m[k,i]+m[i,k])/s; q[3] = (m[k,j]-m[j,k])/s
    return q

# ---------------------------------------------------------------- scene graph
GROUPS, PARTS = {}, []
def group(gid, parent=None, pos=(0, 0, 0), quat=QI, label=None, moves=False):
    GROUPS[gid] = dict(parent=parent, label=label or gid, moves=moves,
                       poses={m: (v(*pos), np.array(quat, float)) for m in MODES},
                       visible={m: True for m in MODES})
def pose(gid, mode, pos=None, quat=None):
    p, q = GROUPS[gid]['poses'][mode]
    GROUPS[gid]['poses'][mode] = (v(*pos) if pos is not None else p, np.array(quat, float) if quat is not None else q)
def _f(val): return list(map(float, val)) if isinstance(val, (tuple, list, np.ndarray)) else float(val)
def part(gid, kind, name, color, mat='plain', opacity=1.0, emissive=None, **geo):
    d = dict(group=gid, kind=kind, name=name, color=color, mat=mat, opacity=opacity, emissive=emissive)
    for k, val in geo.items():
        d[k] = [_f(p) for p in val] if k == 'pts' else _f(val)
    PARTS.append(d)
def box(gid, name, color, x0, x1, y0, y1, z0, z1, **kw):
    part(gid, 'box', name, color, c=((x0+x1)/2, (y0+y1)/2, (z0+z1)/2), s=(x1-x0, y1-y0, z1-z0), **kw)
def cyl(gid, name, color, p1, p2, r1, r2=None, **kw):
    part(gid, 'cyl', name, color, p1=p1, p2=p2, r1=r1, r2=r1 if r2 is None else r2, **kw)
def prism(gid, name, color, o, u, vv, pts, depth, **kw):
    """Polygon pts (in u,v) in the plane through o, extruded by depth along u x v."""
    part(gid, 'prism', name, color, o=o, u=u, v=vv, pts=pts, depth=depth, **kw)
def world(gid, mode):
    g = GROUPS[gid]; p, q = g['poses'][mode]
    m = np.eye(4); m[:3, :3] = q_mat(q); m[:3, 3] = p
    return world(g['parent'], mode) @ m if g['parent'] else m
def wpt(gid, mode, pt): return (world(gid, mode) @ np.array([*pt, 1.0]))[:3]
def wdir(gid, mode, d): return world(gid, mode)[:3, :3] @ v(*d)

# materials: Fusion library appearances (textured where the library has them)
M = dict(oak='Oak - Semigloss', floor='Oak', black='Plastic - Matte (Black)', grey='Plastic - Matte (Gray)',
         paint='Paint - Enamel Glossy (Black)', chrome='Aluminum - Polished', steel='Steel - Satin',
         fabric='Fabric (Grey)', liner='Fabric (White)', paper='Paper (White)', rubber='Rubber - Soft',
         glass='Glass (Grey)', plain='plain')
COL = dict(wood='#9b9184', frame='#1c1c1c', wall='#efe7d6', floor='#c58a4c', base='#f5f4ef', black='#161616',
           rear='#2a2a2c', facet='#3a3a3d', dock='#2a2a2d', laptop='#2b2d31', screen_off='#0d1117', lamp='#141414',
           chrome='#c9c9c9', kb='#3a3d42', basket='#b8b0a2', liner='#ece6d9', strip='#202020', brick='#1b1b1b',
           paper='#fbf9f1', zone='#5aa9e6', print='#f08a24', arm='#1a1a1a')

# ---------------------------------------------------------------- room + desk
group('room', label='Room')
box('room', 'Wall', COL['wall'], -40, 135, 0, 180, -1, 0)
box('room', 'Floor', COL['floor'], -40, 135, -1, 0, -1, 130, mat=M['floor'])
box('room', 'Baseboard', COL['base'], -40, 135, 0, P['baseboard_h'], 0, P['baseboard_t'])
OX = 85.5
oy0, oy1 = P['outlet_plate']
box('room', 'Outlet plate', '#fbfbf8', OX - 3.5, OX + 3.5, oy0, oy1, 0, 0.5)
SOCK_UP = v(OX, oy0 + 8.3, 0.5); SOCK_LO = v(OX, oy0 + 3.1, 0.5)
for nm, s in (('Upper socket (monitor)', SOCK_UP), ('Lower socket (strip)', SOCK_LO)):
    box('room', nm, '#e9e9e4', s[0] - 1.8, s[0] + 1.8, s[1] - 1.6, s[1] + 1.6, 0.5, 0.75)

group('desk', label='Desk - Mainstays side storage')
box('desk', 'Desktop', COL['wood'], 0, W, UND, TOP, Z0, Z1, mat=M['oak'])
for nm, z in (('back', (Z0, Z0 + 2)), ('front', (Z1 - 2, Z1))):
    box('desk', 'Right leg ' + nm, COL['frame'], W - 2, W, 0, UND, *z, mat=M['black'])
    box('desk', 'Left post ' + nm, COL['frame'], POST_X0, POST_X1, 0, TS_Y0, *(POST_ZB if nm == 'back' else POST_ZF), mat=M['black'])
    box('desk', 'Rail ' + nm, COL['frame'], POST_X1, W - 2, RAIL_Y0, UND, *z, mat=M['black'])
box('desk', 'Rail right', COL['frame'], W - 2, W, RAIL_Y0, UND, Z0 + 2, Z1 - 2, mat=M['black'])
box('desk', 'Rail left', COL['frame'], POST_X0, POST_X1, RAIL_Y0, UND, POST_ZB[1], POST_ZF[0], mat=M['black'])
box('desk', 'Foot bar right', COL['frame'], W - 2, W, 3.0, 5.0, Z0 + 2, Z1 - 2, mat=M['black'])
box('desk', 'Foot bar left', COL['frame'], POST_X0, POST_X1, 3.0, 5.0, POST_ZB[1], POST_ZF[0], mat=M['black'])
XB_A = (v((POST_X0 + POST_X1) / 2, 12, Z0 + 1), v(W - 1, RAIL_Y0, Z0 + 1))
XB_B = (v((POST_X0 + POST_X1) / 2, RAIL_Y0, Z0 + 1), v(W - 1, 12, Z0 + 1))
cyl('desk', 'X-brace a', COL['frame'], *XB_A, 0.3, mat=M['black'])
cyl('desk', 'X-brace b', COL['frame'], *XB_B, 0.3, mat=M['black'])
box('desk', 'Top shelf', COL['wood'], 0, SH_W, TS_Y0, TS_Y1, TS_Z0, TS_Z1, mat=M['oak'])
box('desk', 'Middle shelf', COL['wood'], 0, SH_W, MS_Y0, MS_Y1, SH_Z0, SH_Z1, mat=M['oak'])
box('desk', 'Bottom shelf', COL['wood'], 0, SH_W, BS_Y0, BS_Y1, SH_Z0, SH_Z1, mat=M['oak'])

# ---------------------------------------------------------------- AOC Q27G4XN panel (faceted rear, tapered side)
PW, PH = 61.39, 36.63
group('panel', label='AOC Q27G4XN')
# thin full-width shell; back face tapers 1.4 cm (bottom) -> 1.0 cm (top)
prism('panel', 'Panel shell', COL['black'], (-PW/2, 0, 0), (0, 1, 0), (0, 0, 1),
      [(0, 0), (PH, 0), (PH, -1.0), (0, -1.4)], PW, mat=M['black'])
# rear body: side profile from the product side photo (slopes out below the top third)
prism('panel', 'Rear body', COL['rear'], (-25.0, 0, 0), (0, 1, 0), (0, 0, 1),
      [(1.0, -1.0), (28.0, -1.0), (19.5, -5.13), (5.0, -5.13), (1.6, -2.5)], 50.0, mat=M['black'])
# side facets (rear view: angled planes running out to the panel edges)
for s in (1, -1):
    prism('panel', 'Rear facet', COL['facet'], (0, 5.0, 0), (0, 0, 1), (1, 0, 0),
          [(-1.0, s * 25.0), (-5.13, s * 25.0), (-1.3, s * 29.8)], 14.5, mat=M['black'])
prism('panel', 'Rear plateau', COL['facet'], (0, 0, -5.43), (1, 0, 0), (0, 1, 0),
      [(-20.0, 6.0), (20.0, 6.0), (15.0, 18.5), (-15.0, 18.5)], 0.3, mat=M['grey'])
box('panel', 'Bottom chin', COL['black'], -PW/2, PW/2, 0, 2.18, -0.2, 0.02, mat=M['black'])
part('panel', 'box', 'Screen', COL['screen_off'], mat=M['glass'], c=(0, 2.18 + 33.57/2, 0.03), s=(59.67, 33.57, 0.06),
     emissive={'monitor': '#3b6fb6', 'dual': '#3b6fb6', 'laptop': None, 'notebook': None})
PORT_DIR = norm(v(0, -0.8, -0.6))
PORTS_LOCAL = {'DP': 8.0, 'HDMI1': 5.5, 'HDMI2': 3.0, 'Earphone': 1.0, 'AC': -13.0}
for n, x in PORTS_LOCAL.items():
    box('panel', 'Port ' + n, '#3a3a3a', x - 0.9, x + 0.9, 3.4, 3.9, -4.3, -3.0)
VESA_LOCAL = v(0, 16.0, -5.13)

# ---------------------------------------------------------------- NB SmooVex A5 / HUANUO FlowLift Pro arm
# Same factory design (TUV report lists A5). Published: max height 16.5 in, max extension 18.7 in,
# tilt +85/-25, swivel +-90, desk 15-90 mm, 2-9 kg. Link geometry measured off HUANUO's drawing:
ARM = dict(post_pivot=5.6, l1_run=18.3, l1_rise=20.4, elbow_top=32.0, l2=22.6, head=8.0,
           pitch_min=-35.0, pitch_max=29.5)
FACE = {'monitor': 23.0, 'dual': 17.4, 'laptop': 17.4, 'notebook': 17.4}   # screen face z per mode
B_MON = 19.0                                                               # panel bottom above desk
for m in MODES:
    pose('panel', m, (CX, TOP + B_MON, FACE[m]))
GROUPS['panel']['moves'] = True
AXP = v(25.4, TOP, Z0 + 5.5)                  # pole axis: ~39 cm left of the screen centre so the arm folds flat
group('arm', label='Arm clamp + pole (static)')
box('arm', 'Clamp base', COL['arm'], AXP[0] - 5.5, AXP[0] + 5.5, TOP, TOP + 1.4, Z0 + 0.25, Z0 + 11.25, mat=M['black'])
box('arm', 'Clamp body', COL['arm'], AXP[0] - 3.0, AXP[0] + 3.0, RAIL_Y0 - 1.4, TOP + 1.4, Z0 - 1.95, Z0, mat=M['black'])
box('arm', 'Clamp jaw', COL['arm'], AXP[0] - 3.0, AXP[0] + 3.0, RAIL_Y0 - 1.4, RAIL_Y0, Z0 - 1.95, Z0 + 5.5, mat=M['black'])
cyl('arm', 'Clamp screw', '#8a8a8a', (AXP[0], RAIL_Y0 - 1.4, Z0 + 3.0), (AXP[0], RAIL_Y0 - 7.0, Z0 + 3.0), 0.5, mat=M['steel'])
cyl('arm', 'Pole', COL['arm'], AXP + v(0, 1.4, 0), AXP + v(0, ARM['post_pivot'] + 1.0, 0), 1.9, mat=M['black'])
S1 = AXP + v(0, ARM['post_pivot'], 0)
L1_LEN = math.hypot(ARM['l1_run'], ARM['l1_rise'])
group('arm_l1', pos=S1, label='Arm lower link (fixed incline, swivels)', moves=True)
group('arm_l1b', parent='arm_l1', quat=q_axis((0, 0, 1), math.degrees(math.atan2(ARM['l1_rise'], ARM['l1_run']))))
box('arm_l1b', 'Lower spring arm', COL['arm'], -1.0, L1_LEN + 0.8, -2.1, 2.1, -1.9, 1.9, mat=M['black'])
cyl('arm_l1', 'Elbow post', COL['arm'], (ARM['l1_run'], ARM['l1_rise'] - 1.0, 0), (ARM['l1_run'], ARM['elbow_top'] - ARM['post_pivot'], 0), 1.7, mat=M['black'])
group('arm_l2', label='Arm upper link (spring, height)', moves=True)
box('arm_l2', 'Upper arm', COL['arm'], -1.8, ARM['l2'] + 1.8, -2.2, 2.2, -2.0, 2.0, mat=M['black'])
group('arm_head', label='Arm head + quick-release VESA', moves=True)
box('arm_head', 'Tilt head', COL['arm'], -2.0, 2.0, -2.0, 2.0, -1.5, ARM['head'] - 2.6, mat=M['black'])
cyl('arm_head', 'Quick-release hub', COL['arm'], (0, 0, ARM['head'] - 2.6), (0, 0, ARM['head'] - 0.4), 4.6, mat=M['black'])
box('arm_head', 'VESA plate', COL['arm'], -6, 6, -6, 6, ARM['head'] - 0.4, ARM['head'], mat=M['black'])
ARM_IK = {}
def arm_ik(m):
    vesa = wpt('panel', m, VESA_LOCAL); H = vesa - v(0, 0, ARM['head'])
    ey = TOP + ARM['elbow_top']
    s = (H[1] - ey) / ARM['l2']; assert abs(s) <= 1
    pitch = math.degrees(math.asin(s)); assert ARM['pitch_min'] <= pitch <= ARM['pitch_max'], ('arm height out of range', m, pitch)
    r1, r2 = ARM['l1_run'], ARM['l2'] * math.cos(math.radians(pitch))
    A = v(AXP[0], AXP[2]); B = v(H[0], H[2]); d = float(np.linalg.norm(B - A))
    assert abs(r1 - r2) <= d <= r1 + r2, ('arm reach out of range', m, d)
    a = (r1 * r1 - r2 * r2 + d * d) / (2 * d); h = math.sqrt(max(r1 * r1 - a * a, 0))
    u = (B - A) / d; p = A + a * u; perp = v(-u[1], u[0])
    E = min((p + h * perp, p - h * perp), key=lambda e: e[1])          # elbow on the wall side
    yaw1 = math.degrees(math.atan2(-(E[1] - A[1]), E[0] - A[0]))
    yaw2 = math.degrees(math.atan2(-(B[1] - E[1]), B[0] - E[0]))
    pose('arm_l1', m, quat=q_axis((0, 1, 0), yaw1))
    pose('arm_l2', m, pos=(E[0], ey, E[1]), quat=q_mul(q_axis((0, 1, 0), yaw2), q_axis((0, 0, 1), pitch)))
    pose('arm_head', m, pos=H)
    ARM_IK[m] = dict(pitch=round(pitch, 1), elbow=[round(E[0], 1), round(E[1], 1)], reach=round(d + ARM['head'], 1),
                     vesa_above_desk=round(vesa[1] - TOP, 1))
for m in MODES: arm_ik(m)

# ---------------------------------------------------------------- dock under the desktop (printed bracket)
group('dock', label='Dell WD19DCS on printed bracket')
DK = dict(x=(34.0, 54.5), y=(UND - 5.4, UND - 2.5), z=(Z0 + 6.7, Z0 + 15.7))
box('dock', 'Dock', COL['dock'], DK['x'][0], DK['x'][1], *DK['y'], *DK['z'], mat=M['black'])
for xs in (35.0, 48.0):
    box('dock', 'Bracket strap', COL['print'], xs, xs + 2, DK['y'][0] - 0.4, DK['y'][0], DK['z'][0] - 0.4, DK['z'][1] + 0.4)
    box('dock', 'Bracket post', COL['print'], xs, xs + 2, DK['y'][0] - 0.4, UND, DK['z'][1], DK['z'][1] + 0.4)
    box('dock', 'Bracket post', COL['print'], xs, xs + 2, DK['y'][0] - 0.4, UND, DK['z'][0] - 0.4, DK['z'][0])
DKY = (DK['y'][0] + DK['y'][1]) / 2
DOCK_DP1 = v(42.0, DKY, DK['z'][0]); DOCK_DC = v(52.0, DKY, DK['z'][0]); DOCK_HOST = v(DK['x'][0], DKY, DK['z'][0] + 4.5)

# ---------------------------------------------------------------- basket, strip, brick (bottom shelf)
group('basket', label='Basket (wicker, linen liner)')
BK = dict(x=(POST_X1 + 0.3, POST_X1 + 0.3 + 6.0 * IN), y=(BS_Y1, BS_Y1 + 4.5 * IN), z=(SH_Z0 + 1.0, SH_Z0 + 1.0 + 17 * IN))
t = 0.6
box('basket', 'Basket floor', COL['basket'], BK['x'][0], BK['x'][1], BK['y'][0], BK['y'][0] + t, *BK['z'], mat=M['fabric'])
for nm, xs, zs in (('L', (BK['x'][0], BK['x'][0] + t), BK['z']), ('R', (BK['x'][1] - t, BK['x'][1]), BK['z']),
                   ('back', BK['x'], (BK['z'][0], BK['z'][0] + t)), ('front', BK['x'], (BK['z'][1] - t, BK['z'][1]))):
    box('basket', 'Basket wall ' + nm, COL['basket'], xs[0], xs[1], BK['y'][0], BK['y'][1], zs[0], zs[1], mat=M['fabric'])
box('basket', 'Liner rim', COL['liner'], BK['x'][0], BK['x'][1], BK['y'][1] - 2.0, BK['y'][1] + 0.3, BK['z'][0] - 0.3, BK['z'][0] + 0.9, mat=M['liner'])
FLOOR_Y = BK['y'][0] + t
IX = (BK['x'][0] + t, BK['x'][1] - t); IZ = (BK['z'][0] + t, BK['z'][1] - t)
PSI = 0.0; SL, SWD, SHT = 35.5, 5.5, 4.0
sdir = v(-math.sin(math.radians(PSI)), 0, math.cos(math.radians(PSI)))
back_c = v(IX[1] - SWD / 2 - 0.2, FLOOR_Y, IZ[0] + 6.25)
group('strip', pos=back_c + sdir * SL / 2, quat=q_axis((0, 1, 0), -PSI), label='LiOA 6-outlet strip')
box('strip', 'LiOA strip', COL['strip'], -SWD / 2, SWD / 2, 0, SHT, -SL / 2, SL / 2, mat=M['black'])
OUTLET_COLORS = ['#1a1a1a', '#2f9e5b', '#2e5fb3', '#e6c229', '#c0392b', '#7a4a2a']
OUT_Z = [-14.75, -9.75, -4.75, 5.25, 10.15, 15.05]
for i, (cz, c) in enumerate(zip(OUT_Z, OUTLET_COLORS)):
    box('strip', 'Outlet %d' % (i + 1), c, -2.1, 2.1, SHT, SHT + 0.05, cz - 2.1, cz + 2.1)
for cz in (-0.5, 2.3):
    box('strip', 'Strip switch', '#b22222', -0.9, 0.9, SHT, SHT + 0.4, cz - 1.0, cz + 1.0)
outlets = [wpt('strip', 'monitor', (0, SHT + 3.0, cz)) for cz in OUT_Z]
STRIP_CORD = wpt('strip', 'monitor', (0, 2.0, -SL / 2)); STRIP_CORD_DIR = -sdir
for n_, i_ in (('brick plug', 0), ('lamp plug + switch', 1)):
    box('strip', 'Plug: ' + n_, '#2a2a2a', -1.6, 1.6, SHT, SHT + 3.0, OUT_Z[i_] - 1.8, OUT_Z[i_] + 1.8, mat=M['black'])
BR = dict(x=(IX[0] + 0.05, IX[0] + 0.05 + 2.54), y=(FLOOR_Y, FLOOR_Y + 10.0), z=(IZ[0] + 15.5, IZ[0] + 35.5))
box('basket', 'Dock power brick 240 W', COL['brick'], BR['x'][0], BR['x'][1], *BR['y'], *BR['z'], mat=M['black'])
BRX = (BR['x'][0] + BR['x'][1]) / 2; BRY = (BR['y'][0] + BR['y'][1]) / 2
BRICK_AC = v(BRX, BRY, BR['z'][0]); BRICK_DC = v(BRX, BRY, BR['z'][1])   # AC end faces the outlets

# ---------------------------------------------------------------- laptop (Dell 16, carbon black)
LW, LD, LB, LL = 35.73, 25.06, 1.39, 0.60
group('laptop', label='Dell 16 DC16250', moves=True)
group('lid', parent='laptop', pos=(0, LB, 0), label='Dell 16 lid', moves=True)
box('laptop', 'Laptop base', COL['laptop'], 0, LW, 0, LB, 0, LD, mat=M['black'])
box('laptop', 'Keyboard deck', '#1f2024', 2.5, LW - 2.5, LB - 0.03, LB - 0.005, 2.0, 13.0, mat=M['black'])
box('laptop', 'Touchpad', '#26282c', LW / 2 - 6.5, LW / 2 + 6.5, LB - 0.03, LB - 0.005, 14.5, 22.5, mat=M['grey'])
box('lid', 'Laptop lid', COL['laptop'], 0, LW, 0, 24.3, -LL, 0, mat=M['black'])
part('lid', 'box', 'Laptop screen', COL['screen_off'], mat=M['glass'], c=(LW / 2, 12.6, -0.03), s=(34.2, 21.4, 0.04),
     emissive={'monitor': None, 'dual': '#4a7cc7', 'laptop': '#4a7cc7', 'notebook': None})
USBC_LOCAL = (0.0, 0.7, 4.0)
def lid_open(deg): return q_axis((1, 0, 0), 90 - deg)
CUBBY = (25.56, TOP + 1.5, 17.3)
FRONT = (LX - LW / 2, TOP, Z1 - LD - 0.3)
for m in ('monitor', 'notebook'):
    pose('laptop', m, CUBBY, q_axis((0, 1, 0), -90)); pose('lid', m, quat=lid_open(0))
for m in ('dual', 'laptop'):
    pose('laptop', m, FRONT, QI); pose('lid', m, quat=lid_open(108))
group('laptop_feet', label='Printed vented laptop feet (cubby)')
for zc in (19.5, 51.0):
    box('laptop_feet', 'Laptop foot', COL['print'], 1.5, 24.5, TOP, TOP + 1.5, zc - 1.0, zc + 1.0)

# ---------------------------------------------------------------- lamp (articulated, on the top shelf)
L1L, L2L = 26.0, 28.0
LB_C = v(SH_W / 2, TS_Y1, TS_Z0 + 8.8)
group('lamp_base', pos=LB_C, label='Mainstays LED architect lamp')
cyl('lamp_base', 'Lamp base', COL['lamp'], (0, 0, 0), (0, 2.5, 0), 8.6, mat=M['paint'])
cyl('lamp_base', 'Lamp post', COL['chrome'], (0, 2.5, 0), (0, 7.0, 0), 0.9, 0.7, mat=M['chrome'])
group('lamp_yaw', parent='lamp_base', pos=(0, 7.5, 0), moves=True)
group('lamp_l1', parent='lamp_yaw', moves=True)
group('lamp_l2', parent='lamp_l1', pos=(0, L1L, 0), moves=True)
group('lamp_head', parent='lamp_l2', pos=(0, L2L, 0), moves=True)
for s in (-0.9, 0.9):
    cyl('lamp_l1', 'Lamp lower arm', COL['lamp'], (s, 0, 0), (s, L1L, 0), 0.3, mat=M['paint'])
    cyl('lamp_l2', 'Lamp upper arm', COL['lamp'], (s, 0, 0), (s, L2L, 0), 0.3, mat=M['paint'])
cyl('lamp_l1', 'Lamp spring', '#555555', (0, 4, 0), (0, 15, 0), 0.45, mat=M['steel'])
cyl('lamp_head', 'Lamp neck', COL['chrome'], (0, 0, 0), (0, 2.5, 0), 0.6, mat=M['chrome'])
cyl('lamp_head', 'Lamp shade', COL['lamp'], (0, 2.5, 0), (0, 13.5, 0), 3.5, 7.15, mat=M['paint'])
part('lamp_head', 'cyl', 'Lamp LED', '#f4efe2', p1=(0, 13.3, 0), p2=(0, 13.45, 0), r1=6.6, r2=6.6,
     emissive={'monitor': '#fff3d6', 'dual': '#fff3d6', 'laptop': '#fff3d6', 'notebook': '#fff3d6'})
SHOULDER = LB_C + v(0, 7.5, 0)
LAMP_TARGETS = {
    'monitor': (v(27.0, 110.0, 55.0), v(UX - 4, TOP, 44.0)),
    'dual': (v(25.0, 110.0, 57.0), v(LX - 6, TOP + 2, 48.0)),
    'laptop': (v(25.0, 110.0, 57.0), v(LX - 6, TOP + 2, 48.0)),
    'notebook': (v(39.0, 112.0, 43.0), v(UX - 2, TOP, 43.0)),
}
LAMP_ON = {m: True for m in MODES}
LAMP_POSE = {}
def lamp_ik(mode):
    H, A = LAMP_TARGETS[mode]
    d = H - SHOULDER; yaw = math.degrees(math.atan2(d[0], d[2]))
    fwd = math.hypot(d[0], d[2]); up = d[1]; D = math.hypot(fwd, up)
    assert abs(L1L - L2L) < D < L1L + L2L, ('lamp target unreachable', mode, D)
    phi = math.atan2(fwd, up); gam = math.acos((L1L**2 + D**2 - L2L**2) / (2 * L1L * D))
    p1 = phi - gam; e = v(L1L * math.sin(p1), L1L * math.cos(p1)); p2 = math.atan2(fwd - e[0], up - e[1])
    LAMP_POSE[mode] = dict(lower_lean=round(math.degrees(p1), 1), upper_below_horizontal=round(math.degrees(p2) - 90, 1))
    pose('lamp_yaw', mode, quat=q_axis((0, 1, 0), yaw))
    pose('lamp_l1', mode, quat=q_axis((1, 0, 0), math.degrees(p1)))
    pose('lamp_l2', mode, quat=q_axis((1, 0, 0), math.degrees(p2 - p1)))
    m3 = world('lamp_l2', mode)[:3, :3]
    pose('lamp_head', mode, quat=q_from_to(v(0, 1, 0), m3.T @ norm(A - H)))
for m in MODES: lamp_ik(m)

# ---------------------------------------------------------------- keyboard, mouse, notebook
group('keyboard', label='Generic BT keyboard 44 x 13.5 x 2.5', moves=True)
box('keyboard', 'Keyboard', COL['kb'], -22, 22, 0, 2.5, -6.75, 6.75, mat=M['grey'])
for r in range(5):
    box('keyboard', 'Key row', '#2a2c30', -20.5, 20.5, 2.5, 2.7, -5.6 + r * 2.3, -4.2 + r * 2.3, mat=M['black'])
group('mouse', label='Generic BT mouse', moves=True)
box('mouse', 'Mouse', COL['kb'], -3.25, 3.25, 0, 3.8, -5.75, 5.75, mat=M['grey'])
KB_DESK = (UX - 2.0, TOP, 44.0); MS_DESK = (UX + 25.0, TOP, 46.0)
KB_PARK = (CX - 2.0, TOP, Z0 + 0.4 + 6.75); MS_PARK = (CX + 24.0, TOP, Z0 + 0.4 + 5.75)
MS_SIDE = (LX + 25.0, TOP, 46.0)
for m in MODES:
    pose('keyboard', m, KB_DESK if m == 'monitor' else KB_PARK)
    pose('mouse', m, MS_PARK if m == 'notebook' else (MS_SIDE if m in ('dual', 'laptop') else MS_DESK))
group('notebook', label='A4 notebook (open) + pens', moves=True)
box('notebook', 'Notebook', COL['paper'], -21, 21, 0, 0.8, -14.85, 14.85, mat=M['paper'])
cyl('notebook', 'Pen', '#1d4fa3', (24, 0.5, -8), (24.5, 0.5, 6), 0.45)
cyl('notebook', 'Pencil', '#e2b33c', (26, 0.5, -8), (26.5, 0.5, 6), 0.4)
NB_POS = (UX - 2.0, TOP, Z1 - 14.85 - 3.0)
for m in MODES:
    pose('notebook', m, NB_POS if m == 'notebook' else (CX - 3.0, TOP - 0.79, NB_POS[2]))
    GROUPS['notebook']['visible'][m] = (m == 'notebook')

# ---------------------------------------------------------------- user envelope (175 cm)
group('person', label='175 cm user envelope')
EYE_Z = Z1 + 22.0
box('person', 'Seat (gas lift) at %.1f cm' % SEAT, '#4b5563', UX - 22, UX + 22, SEAT - 6, SEAT, Z1 - 4, Z1 + 42, opacity=0.35)
box('person', 'Thighs', COL['zone'], UX - 13, UX + 13, SEAT, SEAT + P['thigh_thk'], Z1 - 15, Z1 + 30, opacity=0.25)
box('person', 'Lower legs', COL['zone'], UX - 12, UX + 12, FOOTREST, SEAT, Z1 - 20, Z1 - 6, opacity=0.25)
box('person', 'Footrest %.1f cm' % FOOTREST, '#7b6a58', UX - 20, UX + 20, 0, FOOTREST, Z1 - 30, Z1 + 5, mat=M['oak'])
part('person', 'sphere', 'Eye point', '#d62828', c=(UX, EYE_Y, EYE_Z), r=1.0)

# ---------------------------------------------------------------- cables
TIGHT = []
def fillet_path(pts, radius):
    pts = [v(*p) for p in pts]; clean = [pts[0]]
    for p in pts[1:]:
        if np.linalg.norm(p - clean[-1]) > 1e-6: clean.append(p)
    pts = clean; n = len(pts)
    seg = [np.linalg.norm(pts[j + 1] - pts[j]) for j in range(n - 1)]
    th = [math.pi] * n; t_ = [0.0] * n
    for i in range(1, n - 1):
        u = norm(pts[i - 1] - pts[i]); w = norm(pts[i + 1] - pts[i])
        th[i] = math.acos(max(-1.0, min(1.0, float(np.dot(u, w)))))
        if th[i] < math.pi - 1e-3: t_[i] = radius / math.tan(th[i] / 2)
    for _ in range(50):
        ch = False
        for j in range(n - 1):
            s = t_[j] + t_[j + 1]
            if s > seg[j] + 1e-9: k = seg[j] / s; t_[j] *= k; t_[j + 1] *= k; ch = True
        if not ch: break
    out = [pts[0]]; rmin = math.inf; TIGHT.clear()
    for i in range(1, n - 1):
        B = pts[i]
        if t_[i] < 1e-6: out.append(B); continue
        u = norm(pts[i - 1] - B); w = norm(pts[i + 1] - B); r = t_[i] * math.tan(th[i] / 2)
        if th[i] < math.radians(170):
            rmin = min(rmin, r)
            if r < radius * 0.97: TIGHT.append((np.round(B, 1).tolist(), round(180 - math.degrees(th[i])), round(r, 1)))
        P1, P2 = B + u * t_[i], B + w * t_[i]; C0 = B + norm(u + w) * (r / math.sin(th[i] / 2))
        a, b = P1 - C0, P2 - C0; ang = math.acos(max(-1.0, min(1.0, float(np.dot(norm(a), norm(b))))))
        kn = max(2, int(math.degrees(ang) / 5))
        for k in range(kn + 1):
            s = k / kn
            q = a if ang < 1e-6 else (math.sin((1 - s) * ang) * a + math.sin(s * ang) * b) / math.sin(ang)
            out.append(C0 + q)
    out.append(pts[-1])
    return out, (rmin if rmin < math.inf else 99.0)
def resample(pts, n):
    pts = np.array(pts); seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    s = np.concatenate([[0], np.cumsum(seg)]); L = s[-1]; out = []
    for tt in np.linspace(0, L, n):
        i = min(np.searchsorted(s, tt, side='right') - 1, len(seg) - 1)
        f = 0 if seg[i] == 0 else (tt - s[i]) / seg[i]; out.append(pts[i] + f * (pts[i + 1] - pts[i]))
    return np.array(out), L
CABLES = []; NPTS = 120
BUNDLE_D = {'dp': 10.0, 'ac': 12.0, 'usbc': 10.0, 'dc': 8.0, 'brac': 8.0, 'lamp': 8.0, 'strip': 9.0}
def cable(cid, name, od, rated, k_bend, lead0, lead1, routes, bundle_at, bundle_axis, code, note, tie):
    R = k_bend * od
    e = dict(id=cid, name=name, od=od, rated=rated, rmin_req=round(R, 2), code=code, note=note, tie=tie,
             points={}, length={}, rmin={}, excess={}, bundle={}, tight={})
    for m in MODES:
        p0, d0, way, p1, d1 = routes[m]
        p0, p1 = v(*p0), v(*p1); d0, d1 = norm(v(*d0)), norm(v(*d1))
        def lead_pts(p, d, lead, nxt):
            q = p + d * (lead + R)
            for _ in range(3):
                th = math.acos(max(-1.0, min(1.0, float(np.dot(-d, norm(v(*nxt) - q))))))
                q = p + d * (lead + max(R, R / max(math.tan(th / 2), 1e-3)))
            return [p, p + d * lead, q]
        a0 = lead_pts(p0, d0, lead0, way[0] if way else p1); a1 = lead_pts(p1, d1, lead1, way[-1] if way else p0)
        dense, rmin = fillet_path(a0 + [v(*w) for w in way] + a1[::-1], R)
        pts, L = resample(dense, NPTS); ex = rated - L
        e['points'][m] = np.round(pts, 2).tolist(); e['length'][m] = round(L, 1); e['rmin'][m] = round(rmin, 2)
        e['excess'][m] = round(ex, 1); e['tight'][m] = list(TIGHT)
        D = BUNDLE_D.get(cid, 10.0); turns = max(ex, 0) / (math.pi * D)
        e['bundle'][m] = dict(c=list(map(float, bundle_at)), axis=list(map(float, bundle_axis)),
                              R=D / 2 if turns > 0.2 else 0.0, r=round(od / 2 * max(1.0, math.sqrt(turns) * 1.15), 2),
                              turns=round(turns, 1))
    CABLES.append(e)

G = Z0 - 1.1                      # cables hug the back face of the frame (clipped to it)
def mon_port(n, m): return wpt('panel', m, (PORTS_LOCAL[n], 3.4, -3.65))
dp_r, ac_r, usb_r, dc_r, brac_r, lamp_r, strip_r = {}, {}, {}, {}, {}, {}, {}
for m in MODES:
    dpp, acp = mon_port('DP', m), mon_port('AC', m)
    zv = DOCK_DP1[2] - 5.0 - 4.8
    # up the back of the frame, then tucked under the monitor's rear body (hidden by the panel's bottom edge)
    dp_r[m] = (DOCK_DP1, (0, 0, -1), [(dpp[0] - 6.0, 78.0, G)] + ([(dpp[0], 90.5, G - 0.3)] if FACE[m] > 20 else []), dpp, PORT_DIR)
    ac_r[m] = (SOCK_UP, (0, 0, 1), [(OX, 40.0, Z0 + 1.9), (76.0, 58.0, G), (acp[0] + 5.0, 78.0, G)] + ([(acp[0], 90.0, G - 0.3)] if FACE[m] > 20 else []), acp, PORT_DIR)
    lp = wpt('laptop', m, USBC_LOCAL); ld = wdir('laptop', m, (-1, 0, 0))
    if m in ('monitor', 'notebook'):
        way = [(DOCK_HOST[0] - 8.0, 63.0, 10.0), (lp[0] + 0.4, 63.0, 1.6), (lp[0] + 0.1, lp[1], 1.6)]
    else:
        way = [(DOCK_HOST[0] - 7.5, DKY - 2.0, G + 0.6), (34.4, TOP + 1.4, G + 0.8), (34.6, TOP + 0.55, Z0 + 9.0), (lp[0] - 10.6, TOP + 0.55, lp[2] - 9.0)]
    usb_r[m] = (DOCK_HOST, (-1, 0, 0), way, lp, ld)
    zdc = DOCK_DC[2] - 5.5 - 2.75
    dc_r[m] = (BRICK_DC, (0, 0, 1), [(BRX, BK['y'][1] + 0.8, BRICK_DC[2] + 5.3), (BRX + 1.0, BK['y'][1] + 1.0, IZ[0] + 1.5), (BRX + 1.0, BK['y'][1] + 1.7, IZ[0] - 1.6), (BRX + 1.0, 28.0, G), (30.0, 56.0, G), (DOCK_DC[0], DKY - 7.0, zdc)], DOCK_DC, (0, 0, -1))
    brac_r[m] = (BRICK_AC, (0, 0, -1), [(outlets[0][0] - 6.2, outlets[0][1] + 4.75, outlets[0][2] - 0.5)], outlets[0], (0, 1, 0))
    lamp_r[m] = (LB_C + v(0, 0.8, -8.6), (0, 0, -1), [(LB_C[0], 84.0, 7.0), (10.0, 70.0, G), (19.5, 30.0, G),
                 (20.0, BK['y'][1] + 2.6, IZ[0] + 0.2)], outlets[1], (0, 1, 0))
    sv = STRIP_CORD + STRIP_CORD_DIR * (2.0 + 3.75)
    strip_r[m] = (STRIP_CORD, STRIP_CORD_DIR, [(sv[0], BK['y'][1] + 4.4, sv[2]), (sv[0], BK['y'][1] + 4.4, sv[2] - 7.6),
                  (sv[0] + 0.2, 4.0, sv[2] - 7.6), (80.0, 0.45, 6.0), (OX, 0.45, SOCK_LO[2] + 8.0 + 3.75)], SOCK_LO, (0, 0, 1))
cable('dp', 'DP 1.4, dock -> monitor', 0.6, 150, 8, 5.0, 5.0, dp_r, (50.0, 64.0, G - 0.6), (0, 0, 1), '#e63946',
      'length assumed 1.5 m', 'coil velcroed to the back rail')
cable('ac', 'Monitor AC (grounded), upper wall socket -> monitor', 0.75, 150, 5, 4.0, 5.5, ac_r, (72.0, 63.0, G), (0, 0, 1), '#f77f00',
      'length assumed 1.5 m', 'coil velcroed to the X-brace')
cable('usbc', 'WD19DCS USB-C (0.8 m, fixed)', 0.7, 80, 8, 2.0, 4.5, usb_r, (16.5, 66.0, G + 0.4), (0, 0, 1), '#3a86ff',
      'spec 0.8 m', 'loose slack loop hangs behind the desk; it feeds out when the laptop moves')
cable('dc', '240 W brick DC -> dock', 0.55, 180, 5, 2.5, 5.5, dc_r, (20.2, FLOOR_Y + 8.5, 44.5), (0, 1, 0), '#ffbe0b', 'length assumed 1.8 m', 'coil in the basket')
cable('brac', '240 W brick AC -> strip outlet 1', 0.75, 100, 5, 4.0, 1.0, brac_r, (20.2, FLOOR_Y + 9.7, 44.5), (0, 1, 0), '#ffd166', 'length assumed 1.0 m', 'coil in the basket')
cable('lamp', 'Lamp cord (5 ft) -> strip outlet 2', 0.55, 152.4, 5, 1.5, 1.0, lamp_r, (20.2, FLOOR_Y + 7.3, 44.5), (0, 1, 0), '#2a9d8f', 'switch is on the plug', 'coil in the basket')
cable('strip', 'LiOA strip cord (5 m) -> lower wall socket', 0.75, 500, 5, 2.0, 8.0, strip_r, (20.2, FLOOR_Y + 5.5, 44.5), (0, 1, 0), '#8338ec', '5 m cord', 'coil in the basket')

# ---------------------------------------------------------------- cable management hardware
group('cable_mgmt', label='Cable management: adhesive clips + velcro ties')
def clip(name, p, axis='x'):
    p = v(*p)
    box('cable_mgmt', name, '#2b2b2b', p[0] - 1.2, p[0] + 1.2, p[1] - 0.9, p[1] + 0.9, p[2] - 0.9, p[2] + 0.9, mat=M['black'])
def tie(name, c, r, axis):
    cyl('cable_mgmt', name, '#202020', v(*c) - v(*axis) * 0.6, v(*c) + v(*axis) * 0.6, r, mat=M['fabric'])
RAIL_MID = (RAIL_Y0 + UND) / 2
for cid in ('dp', 'ac', 'dc'):
    cb = next(c for c in CABLES if c['id'] == cid)
    pts = np.array(cb['points']['dual'])
    near = pts[np.argmin(np.abs(pts[:, 1] - RAIL_MID) + 5 * (np.abs(pts[:, 2] - G) > 1.5))]
    clip('Clip on back rail (%s)' % cid, (near[0], RAIL_MID, Z0 - 0.4))
cbu = next(c for c in CABLES if c['id'] == 'usbc'); pu = np.array(cbu['points']['monitor'])
clip('Clip under desktop (usbc)', pu[np.argmin(np.abs(pu[:, 1] - 70.5) + np.abs(pu[:, 2] - 10.0))] + v(0, 0.9, 0))
for cb in CABLES:
    bd = cb['bundle']['dual']
    if bd['R'] > 0.2:
        tie('Velcro tie (%s coil)' % cb['id'], v(*bd['c']) + v(*norm(np.cross(v(*bd['axis']), v(0, 0, 1) if abs(bd['axis'][2]) < 0.9 else v(1, 0, 0)))) * bd['R'],
            bd['r'] + 0.25, (1, 0, 0) if abs(bd['axis'][0]) < 0.9 else (0, 1, 0))

# ---------------------------------------------------------------- geometry queries for checks
def parts_named(name): return [p for p in PARTS if p['name'] == name]
def prism_vertices(p):
    o, u, vv = v(*p['o']), v(*p['u']), v(*p['v']); n = np.cross(u, vv)
    base = [o + a * u + b * vv for a, b in p['pts']]
    return base + [q + n * p['depth'] for q in base]
def sdf_fn(name, m):
    fns = []
    for pr in parts_named(name):
        Mw = world(pr['group'], m); Mi = np.linalg.inv(Mw)
        if pr['kind'] == 'box':
            c, s = v(*pr['c']), v(*pr['s']) / 2
            def f(p, Mi=Mi, c=c, s=s):
                q = np.abs((Mi @ np.array([*p, 1]))[:3] - c) - s
                return float(np.linalg.norm(np.maximum(q, 0)) + min(q.max(), 0))
        elif pr['kind'] == 'cyl':
            a, b = (Mw @ np.array([*pr['p1'], 1]))[:3], (Mw @ np.array([*pr['p2'], 1]))[:3]
            ab = b - a; L2 = float(ab @ ab); r1, r2 = pr['r1'], pr['r2']
            def f(p, a=a, ab=ab, L2=L2, r1=r1, r2=r2):
                h = min(1.0, max(0.0, float((p - a) @ ab) / L2))
                return float(np.linalg.norm(p - (a + ab * h)) - (r1 + (r2 - r1) * h))
        elif pr['kind'] == 'prism':   # conservative: oriented bounding box of the prism
            vs = np.array([(Mw @ np.array([*q, 1]))[:3] for q in prism_vertices(pr)])
            lo, hi = vs.min(0), vs.max(0)
            def f(p, lo=lo, hi=hi):
                q = np.maximum(lo - p, p - hi)
                return float(np.linalg.norm(np.maximum(q, 0)) + min(q.max(), 0))
        else:
            c = (Mw @ np.array([*pr['c'], 1]))[:3]; r = pr['r']
            def f(p, c=c, r=r): return float(np.linalg.norm(p - c) - r)
        fns.append(f)
    return lambda p: min(fn(p) for fn in fns)
def surface_samples(name, m, n=7):
    pts = []
    for pr in parts_named(name):
        Mw = world(pr['group'], m)
        if pr['kind'] == 'box':
            c, s = v(*pr['c']), v(*pr['s']) / 2; g = np.linspace(-1, 1, n)
            for ax in range(3):
                for sg in (-1, 1):
                    for a_ in g:
                        for b_ in g:
                            q = np.zeros(3); q[ax] = sg; o = [i for i in range(3) if i != ax]; q[o[0]] = a_; q[o[1]] = b_
                            pts.append((Mw @ np.array([*(c + q * s), 1]))[:3])
        elif pr['kind'] == 'cyl':
            a, b = v(*pr['p1']), v(*pr['p2']); ax_ = norm(b - a)
            e1 = norm(np.cross(ax_, v(1, 0, 0) if abs(ax_[0]) < 0.9 else v(0, 1, 0))); e2 = np.cross(ax_, e1)
            for h in np.linspace(0, 1, n * 2):
                r = pr['r1'] + (pr['r2'] - pr['r1']) * h
                for ang in np.linspace(0, 2 * math.pi, 16, endpoint=False):
                    pts.append((Mw @ np.array([*(a + (b - a) * h + r * (math.cos(ang) * e1 + math.sin(ang) * e2)), 1]))[:3])
        elif pr['kind'] == 'prism':
            vs = prism_vertices(pr); k = len(pr['pts'])
            for i in range(k):
                for j in range(k):
                    for f in np.linspace(0, 1, n):
                        pts.append((Mw @ np.array([*(vs[i] * (1 - f) + vs[k + j] * f), 1]))[:3])
        else:
            c = v(*pr['c']); r = pr['r']
            for i in range(40):
                z_ = 1 - 2 * (i + 0.5) / 40; rr = math.sqrt(1 - z_ * z_); ph = i * 2.39996
                pts.append((Mw @ np.array([*(c + r * v(rr * math.cos(ph), z_, rr * math.sin(ph))), 1]))[:3])
    return pts
def gap(a, b, m):
    fb, fa = sdf_fn(b, m), sdf_fn(a, m)
    d = min(min(fb(p) for p in surface_samples(a, m)), min(fa(p) for p in surface_samples(b, m)))
    return round(max(0.0, d), 1)
def occlusion(m, screen_part, blockers, eye):
    pr = parts_named(screen_part)[0]; Mw = world(pr['group'], m); c, s = v(*pr['c']), v(*pr['s'])
    f = [sdf_fn(b, m) for b in blockers if parts_named(b) and GROUPS[parts_named(b)[0]['group']]['visible'][m]]
    sd = (lambda p: min(fn(p) for fn in f)) if f else (lambda p: 99.0)
    hit = tot = 0
    for i in np.linspace(-0.47, 0.47, 17):
        for j in np.linspace(-0.47, 0.47, 9):
            loc = c + v(i * s[0], j * s[1] if s[1] > s[2] else 0, 0 if s[1] > s[2] else j * s[2])
            q = (Mw @ np.array([*loc, 1]))[:3]; d = q - eye; L = np.linalg.norm(d); d = d / L; tt = 0.0; tot += 1
            while tt < L - 0.6:
                h = sd(eye + d * tt)
                if h < 0.03: hit += 1; break
                tt += max(h, 0.15)
    return 100.0 * hit / tot

# ---------------------------------------------------------------- cable visibility from the seated eye
BLOCK = ['Desktop', 'Top shelf', 'Middle shelf', 'Bottom shelf', 'Panel shell', 'Rear body', 'Laptop base', 'Laptop lid',
         'Basket wall front', 'Basket wall L', 'Basket wall R', 'Rail front', 'Rail left', 'Left post front', 'Right leg front',
         'Dock', 'Keyboard', 'Notebook', 'Lamp base', 'LiOA strip', 'Dock power brick 240 W', 'Thighs']
def visible_cable(m, n=40):
    f = [sdf_fn(b, m) for b in BLOCK if parts_named(b) and GROUPS[parts_named(b)[0]['group']]['visible'][m]]
    sd = lambda p: min(fn(p) for fn in f)
    by = {}
    for cb in CABLES:
        pts = np.array(cb['points'][m]); idx = np.linspace(0, len(pts) - 1, n).astype(int)
        seg = cb['length'][m] / (n - 1); vis = 0.0
        for i in idx:
            q = pts[i]; d = EYE - q; L = np.linalg.norm(d); d = d / L; tt = cb['od'] / 2 + 0.3; hit = False
            while tt < L - 1.0:
                h = sd(q + d * tt)
                if h < 0.05: hit = True; break
                tt += max(h, 0.3)
            if not hit: vis += seg
        by[cb['id']] = round(vis, 1)
    return dict(total=round(sum(by.values()), 1), by=by)

# ---------------------------------------------------------------- checks
CHECKS = {}
EYE = v(UX, EYE_Y, EYE_Z)
for m in MODES:
    c = []
    def add(name, val, ok, unit='cm', note=''):
        c.append(dict(name=name, value=val if isinstance(val, str) else round(float(val), 1), unit=unit, ok=bool(ok), note=note))
    if m in ('monitor', 'dual'):
        scr = wpt('panel', m, (0, 2.18 + 33.57 / 2, 0)); dist = np.linalg.norm(scr - EYE)
        add('Eye to screen centre', dist, 50 <= dist <= 100, note='Mayo 50-100 cm')
        topa = wpt('panel', m, (0, 2.18 + 33.57, 0))[1]
        add('Top of screen below eye', EYE_Y - topa, 0 <= EYE_Y - topa <= 6, note='Mayo: at or slightly below eye')
        occ = occlusion(m, 'Screen', ['Laptop lid', 'Laptop base', 'Lamp shade', 'Lamp upper arm', 'Lamp lower arm', 'Keyboard', 'Mouse'], EYE)
        add('Monitor screen hidden from eye', occ, occ < 1, unit='%')
    if m in ('dual', 'laptop'):
        occ = occlusion(m, 'Laptop screen', ['Lamp shade', 'Lamp upper arm', 'Lamp lower arm', 'Mouse'], EYE)
        add('Laptop screen hidden from eye', occ, occ < 1, unit='%')
        add('Laptop lid to monitor face', gap('Laptop lid', 'Panel shell', m), gap('Laptop lid', 'Panel shell', m) >= 1.0)
        add('Laptop to mouse', gap('Laptop base', 'Mouse', m), gap('Laptop base', 'Mouse', m) >= 1.0)
    if m in ('monitor', 'notebook'):
        add('Laptop in cubby: air above lid', TS_Y0 - (TOP + 1.0 + 1.99), True)
        add('Laptop vs shelf posts', min(gap('Laptop base', 'Left post back', m), gap('Laptop base', 'Left post front', m)),
            min(gap('Laptop base', 'Left post back', m), gap('Laptop base', 'Left post front', m)) >= 0.5)
    if m != 'monitor':
        add('Parked keyboard under monitor: headroom', gap('Keyboard', 'Panel shell', m), gap('Keyboard', 'Panel shell', m) >= 2.0)
        add('Parked keyboard vs arm clamp', gap('Keyboard', 'Clamp base', m), gap('Keyboard', 'Clamp base', m) >= 1.0)
    if m == 'notebook':
        kb_front = wpt('keyboard', m, (0, 0, 6.75))[2]
        add('Clear writing depth', Z1 - kb_front, Z1 - kb_front >= 35, note='keyboard parked at the back')
        occ = occlusion(m, 'Notebook', ['Lamp shade', 'Lamp upper arm', 'Lamp lower arm'], EYE)
        add('Notebook hidden by lamp', occ, occ < 3, unit='%')
    add('Lamp shade vs monitor', gap('Lamp shade', 'Panel shell', m), gap('Lamp shade', 'Panel shell', m) >= 1.0)
    add('Keyboard to mouse', gap('Keyboard', 'Mouse', m), gap('Keyboard', 'Mouse', m) >= 0.5)
    add('Basket vs shelf posts', min(gap('Basket wall L', 'Left post back', m), gap('Basket floor', 'Left post back', m)), min(gap('Basket wall L', 'Left post back', m), gap('Basket floor', 'Left post back', m)) >= 0.2)
    add('Brick vs strip in basket', gap('Dock power brick 240 W', 'LiOA strip', m), gap('Dock power brick 240 W', 'LiOA strip', m) >= 0.5)
    lp = LAMP_POSE[m]
    add('Lamp pose: lower arm lean / upper arm drop', '%.0f / %.0f' % (lp['lower_lean'], lp['upper_below_horizontal']),
        15 <= lp['lower_lean'] <= 45 and -5 <= lp['upper_below_horizontal'] <= 40, unit='deg', note='natural: lean 15-45, drop -5..40')
    vis = visible_cable(m)
    add('Cable visible from the seat', vis['total'], True, note=', '.join('%s %.0f' % (k, v_) for k, v_ in vis['by'].items() if v_ >= 1) or 'none')
    add('Strip inside basket', min(gap('LiOA strip', 'Basket wall front', m), gap('LiOA strip', 'Basket wall R', m), gap('LiOA strip', 'Basket wall back', m)),
        min(gap('LiOA strip', 'Basket wall front', m), gap('LiOA strip', 'Basket wall R', m), gap('LiOA strip', 'Basket wall back', m)) >= 0.2)
    add('Arm vs monitor back', min(gap('Upper arm', 'Rear body', m), gap('Lower spring arm', 'Rear body', m), gap('Lower spring arm', 'Panel shell', m)),
        min(gap('Upper arm', 'Rear body', m), gap('Lower spring arm', 'Rear body', m), gap('Lower spring arm', 'Panel shell', m)) >= 1.0)
    add('Arm vs top shelf', gap('Lower spring arm', 'Top shelf', m), gap('Lower spring arm', 'Top shelf', m) >= 1.0)
    add('Arm vs parked keyboard', gap('Lower spring arm', 'Keyboard', m), gap('Lower spring arm', 'Keyboard', m) >= 1.0)
    add('Arm vs wall', min(gap('Upper arm', 'Wall', m), gap('Lower spring arm', 'Wall', m)), min(gap('Upper arm', 'Wall', m), gap('Lower spring arm', 'Wall', m)) >= 0.5)
    add('Thigh top to frame rail', RAIL_Y0 - (SEAT + P['thigh_thk']), RAIL_Y0 - (SEAT + P['thigh_thk']) >= 1.0, note='tight: crossed legs will not fit')
    add('Thighs to dock', gap('Thighs', 'Dock', m), gap('Thighs', 'Dock', m) >= 2.0)
    for cb in CABLES:
        L, ex, rm = cb['length'][m], cb['excess'][m], cb['rmin'][m]
        add('Cable %s: routed / rated' % cb['id'], '%.0f / %.0f' % (L, cb['rated']), ex >= 8, note='%.0f cm spare (%s)' % (ex, cb['tie']))
        add('Cable %s: tightest bend' % cb['id'], rm, rm >= cb['rmin_req'] * 0.97, note='min %.1f' % cb['rmin_req'])
    CHECKS[m] = c

# ---------------------------------------------------------------- stand vs arm (sight line to the monitor's bottom edge)
lid_top = wpt('lid', 'dual', (0, 24.3, -LL))
def b_required(face_z):
    E = EYE_Y - TOP; h = lid_top[1] - TOP
    return E - (E - h) * (EYE_Z - face_z) / (EYE_Z - lid_top[2])
COMPARE = dict(
    lid_top_above_desk=round(lid_top[1] - TOP, 1), lid_top_z=round(lid_top[2], 1), eye_above_desk=round(EYE_Y - TOP, 1),
    stand_riser=dict(face_z=25.6, bottom_needed=round(b_required(25.6), 1), top_vs_eye=round(b_required(25.6) + PH - (EYE_Y - TOP), 1)),
    arm=dict(face_z=FACE['dual'], bottom_needed=round(b_required(FACE['dual']), 1), bottom_set=B_MON,
             top_vs_eye=round(B_MON + PH - (EYE_Y - TOP), 1)),
)

# ---------------------------------------------------------------- transitions (what physically moves)
STEPS = {
    ('monitor', 'dual'): ['Push the monitor back to the wall (arm, 5.6 cm)', 'Slide the keyboard back under the monitor',
                          'Take the laptop out of the cubby, open it at the front', 'Mouse stays on the right'],
    ('dual', 'laptop'): ['Nothing moves: turn the monitor off'],
    ('laptop', 'notebook'): ['Close the laptop, slide it into the cubby', 'Park the mouse under the monitor', 'Open the notebook'],
    ('notebook', 'monitor'): ['Put the notebook away', 'Pull the keyboard and mouse forward',
                              'Pull the monitor forward (arm, 5.6 cm)', 'Wake the laptop with the dock button'],
}
def moved(a, b):
    out = []
    for gid, g in GROUPS.items():
        if not g['moves'] or g['parent']: continue
        pa = wpt(gid, a, (0, 0, 0)); pb = wpt(gid, b, (0, 0, 0)); d = float(np.linalg.norm(pb - pa))
        if d > 0.5 or (g['visible'][a] != g['visible'][b]): out.append(dict(item=g['label'], cm=round(d, 1)))
    return out
TRANS = []
for i, a in enumerate(MODES):
    for b in MODES[i + 1:] + MODES[:i]:
        if a == b: continue
        key = (a, b) if (a, b) in STEPS else None
        rkey = (b, a) if (b, a) in STEPS else None
        TRANS.append(dict(frm=a, to=b, moves=moved(a, b),
                          steps=STEPS.get(key) or (['Reverse of: ' + '; '.join(STEPS[rkey])] if rkey else [])))

LOADS = dict(seat_cm=round(SEAT, 1), eye_cm=round(EYE_Y, 1), footrest_cm=FOOTREST,
             monitor_bottom_above_desk_cm=B_MON, vesa_centre_above_desk_cm=ARM_IK['dual']['vesa_above_desk'],
             arm_load_kg='3.24 of 2.0-9.0', lower_arm_length_cm=round(L1_LEN, 1),
             desktop_load_kg=round(3.24 + 2.5 + 0.6 + 1.98 + 0.69, 1), desktop_rating_kg=22.7,
             strip_current_A_at_120V=round((240 / 0.9 + 4) / 120, 2))

# ---------------------------------------------------------------- export
def jgroup(gid, g):
    return dict(parent=g['parent'], label=g['label'], moves=g['moves'], visible=g['visible'],
                poses={m: dict(pos=np.round(p, 4).tolist(), quat=np.round(q, 7).tolist()) for m, (p, q) in g['poses'].items()})
spec = dict(arm=dict(ARM, ik=ARM_IK, pole=AXP.tolist(), face=FACE), units='cm', modes=MODES, mode_label=MODE_LABEL, groups={k: jgroup(k, g) for k, g in GROUPS.items()},
            parts=PARTS, cables=CABLES, lamp_on=LAMP_ON, checks=CHECKS, loads=LOADS, compare=COMPARE, transitions=TRANS,
            camera=dict(target=[45, 85, 30], pos=[150, 175, 230]))

if __name__ == '__main__':
    import os
    here = os.path.dirname(os.path.abspath(__file__))
    with open(os.path.join(here, 'desk_spec.json'), 'w') as f:
        json.dump(spec, f, separators=(',', ':'))
    print('seat %.1f eye %.1f footrest %.1f' % (SEAT, EYE_Y, FOOTREST))
    print('compare', json.dumps(COMPARE))
    print('arm', json.dumps(ARM_IK))
    for m in MODES:
        bad = [c for c in CHECKS[m] if not c['ok']]
        print('\n==', MODE_LABEL[m], '- %d checks, %d fail' % (len(CHECKS[m]), len(bad)))
        for c in CHECKS[m]:
            if not c['ok'] or not c['name'].startswith('Cable'):
                print('  %s %-42s %s %s  %s' % ('OK ' if c['ok'] else 'XX ', c['name'], c['value'], c['unit'], c['note']))
    for cb in CABLES:
        print('%-6s' % cb['id'], ' '.join('%s %.0f/%.0f r%.1f' % (m[:3], cb['length'][m], cb['rated'], cb['rmin'][m]) for m in MODES),
              '' if not any(cb['tight'].values()) else 'TIGHT ' + str(cb['tight']['monitor'] or cb['tight']['dual']))
    for tr in TRANS[:4]:
        print(tr['frm'], '->', tr['to'], tr['moves'])
