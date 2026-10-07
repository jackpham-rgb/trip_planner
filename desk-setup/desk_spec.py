"""Desk setup scene spec: single source of truth for the Fusion model and the 3D viewer.

Units: cm. Axes: x = left -> right (seated view), y = up, z = wall (0) -> user.
Every number is tagged in SOURCES as measured / spec / photo-estimate / assumption.
Run: python desk_spec.py  -> writes desk_spec.json and prints the check report.
"""
import json, math
import numpy as np

IN = 2.54
MODES = ['monitor', 'dual', 'laptop', 'notebook']
MODE_LABEL = {
    'monitor': 'Monitor only',
    'dual': 'Laptop + monitor',
    'laptop': 'Laptop only',
    'notebook': 'Notebook + pens',
}

# ---------------------------------------------------------------- parameters
P = dict(
    wall_gap=2.5 * IN,          # measured (user: 2-3 in, decided 2.5)
    desk_w=36.0 * IN,           # spec
    desk_d=21.2 * IN,           # spec
    desk_top=30.0 * IN,         # ASSUMPTION (listing photo ratio 30.1 in) - confirm
    desk_thk=1.8,               # photo estimate (0.7 in)
    rail=2.0,                   # frame tube 2 x 2 cm, photo estimate
    shelf_w=9.0 * IN,           # measured
    shelf_d=20.0 * IN,          # measured (bottom, middle)
    top_shelf_d=16.5 * IN,      # measured
    shelf_thk=1.6,              # photo estimate
    top_shelf_top=36.0 * IN,    # spec (overall height)
    mid_shelf_top=51.2,         # measured chain scaled to 36 in total
    bot_shelf_under=3.5 * IN,   # measured
    post_x=4.0 * IN,            # measured: left post outer face from left edge
    baseboard_h=9.6, baseboard_t=1.5,   # photo estimate
    outlet_plate=(18.6, 30.0),  # photo estimate (standard 11.4 cm plate as scale)
    eye_above_seat=78.0,        # anthropometry, 175 cm adult (50th pct ~79)
    elbow_above_seat=23.5,      # anthropometry
    popliteal=44.5,             # anthropometry incl. shoes
    thigh_thk=16.0,             # anthropometry
    riser_h=8.0,                # design (DIY)
)
SOURCES = {
    'AOC Q27G4XN': '613.9 x 366.3 x 51.3 mm, 3.24 kg; stand: top of panel 386-516 mm above base, base depth 207.7 mm, 5.11 kg total (displayspecifications.com); active 596.7 x 335.7 mm; ports DP, HDMI1, HDMI2, earphone, AC (AOC manual)',
    'Dell 16 DC16250': '357.3 x 250.6 x 16.74-19.90 mm, 1.98 kg (dell.com)',
    'Dell WD19DCS': '205 x 90 x 29 mm, 685 g, USB-C cable 0.8 m (Dell spec)',
    'Dell 240 W adapter': '199.89 x 100.07 x 25.4 mm, 0.769 kg (Dell spec)',
    'Mainstays LED architect lamp': '6.77 in base, 18.15 in tall, 1.48 kg, 5 ft cord (Walmart); arm lengths photo-estimated',
    'LiOA 6-outlet strip': '10 A, 5 m cord, no ground on 5 m version (lioa.com.vn); body size estimated 40 x 5.5 x 4 cm',
    'Mainstays desk': '36 W x 21.2 D x 36 H in (listing); shelf 9 x 20 in, top shelf 16.5 in deep, bottom shelf 3.5 in off floor (user tape)',
    'Basket': '17 x 6(?) x 4.5 in (user tape); width modelled 10.5 in from top-down photo - confirm',
}

TOP = P['desk_top']
Z0 = P['wall_gap']                 # desk back edge
Z1 = Z0 + P['desk_d']              # desk front edge
W = P['desk_w']
UND = TOP - P['desk_thk']          # desktop underside
RAIL_Y0 = UND - P['rail']          # frame rail bottom
SH_W = P['shelf_w']
SH_Z0 = Z0 + (P['desk_d'] - P['shelf_d']) / 2
SH_Z1 = SH_Z0 + P['shelf_d']
POST_X0 = P['post_x']
POST_X1 = POST_X0 + 2.0
POST_ZB = (SH_Z0 + 3.0, SH_Z0 + 5.0)       # back left post
POST_ZF = (SH_Z1 - 5.0, SH_Z1 - 3.0)       # front left post
TS_Y1 = P['top_shelf_top']; TS_Y0 = TS_Y1 - P['shelf_thk']
TS_ZC = (POST_ZB[0] + POST_ZF[1]) / 2
TS_Z0 = TS_ZC - P['top_shelf_d'] / 2; TS_Z1 = TS_ZC + P['top_shelf_d'] / 2
MS_Y1 = P['mid_shelf_top']; MS_Y0 = MS_Y1 - P['shelf_thk']
BS_Y0 = P['bot_shelf_under']; BS_Y1 = BS_Y0 + P['shelf_thk']

# person (Mayo: hands at/below elbows, feet flat -> footrest)
SEAT = TOP + 1.5 - P['elbow_above_seat'] + 0.5     # keyboard home row ~1.5 above desk
EYE_Y = SEAT + P['eye_above_seat']
FOOTREST = round(SEAT - P['popliteal'], 1)

# ---------------------------------------------------------------- math helpers
def v(*a): return np.array(a, dtype=float)
def norm(a): return a / np.linalg.norm(a)

def q_axis(axis, deg):
    axis = norm(v(*axis)); h = math.radians(deg) / 2
    return np.array([*(axis * math.sin(h)), math.cos(h)])

def q_mul(a, b):
    x1, y1, z1, w1 = a; x2, y2, z2, w2 = b
    return np.array([w1*x2 + x1*w2 + y1*z2 - z1*y2,
                     w1*y2 - x1*z2 + y1*w2 + z1*x2,
                     w1*z2 + x1*y2 - y1*x2 + z1*w2,
                     w1*w2 - x1*x2 - y1*y2 - z1*z2])

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

# ---------------------------------------------------------------- scene graph
GROUPS = {}   # id -> dict(parent, poses{mode:(pos,quat)}, visible{mode:bool})
PARTS = []

def group(gid, parent=None, pos=(0, 0, 0), quat=QI, label=None):
    GROUPS[gid] = dict(parent=parent, label=label or gid,
                       poses={m: (v(*pos), np.array(quat, float)) for m in MODES},
                       visible={m: True for m in MODES})

def pose(gid, mode, pos=None, quat=None):
    p, q = GROUPS[gid]['poses'][mode]
    GROUPS[gid]['poses'][mode] = (v(*pos) if pos is not None else p,
                                  np.array(quat, float) if quat is not None else q)

def part(gid, kind, name, color, code=None, opacity=1.0, emissive=None, **geo):
    PARTS.append(dict(group=gid, kind=kind, name=name, color=color, code=code or color,
                      opacity=opacity, emissive=emissive,
                      **{k: (list(map(float, val)) if isinstance(val, (tuple, list, np.ndarray)) else float(val))
                         for k, val in geo.items()}))

def box(gid, name, color, x0, x1, y0, y1, z0, z1, **kw):
    part(gid, 'box', name, color, c=((x0+x1)/2, (y0+y1)/2, (z0+z1)/2),
         s=(x1-x0, y1-y0, z1-z0), **kw)

def cyl(gid, name, color, p1, p2, r1, r2=None, **kw):
    part(gid, 'cyl', name, color, p1=p1, p2=p2, r1=r1, r2=r1 if r2 is None else r2, **kw)

def world(gid, mode):
    """4x4 world matrix of a group in a mode."""
    g = GROUPS[gid]; p, q = g['poses'][mode]
    m = np.eye(4); m[:3, :3] = q_mat(q); m[:3, 3] = p
    return world(g['parent'], mode) @ m if g['parent'] else m

def wpt(gid, mode, pt):
    return (world(gid, mode) @ np.array([*pt, 1.0]))[:3]

def wdir(gid, mode, d):
    return world(gid, mode)[:3, :3] @ v(*d)

# ---------------------------------------------------------------- colours
COL = dict(wood='#9b9184', frame='#1c1c1c', wall='#efe7d6', floor='#c58a4c', base='#f5f4ef',
           black='#161616', dock='#2a2a2d', laptop='#2b2d31', screen_off='#0d1117',
           riser='#c7a27a', lamp='#141414', chrome='#c9c9c9', kb='#2f3237', basket='#b8b0a2',
           liner='#ece6d9', strip='#202020', brick='#1b1b1b', paper='#fbf9f1', zone='#5aa9e6',
           print='#f08a24', cable='#111111')

# ---------------------------------------------------------------- room + desk (static)
group('room', label='Room')
box('room', 'Wall', COL['wall'], -40, 135, 0, 180, -1, 0)
box('room', 'Floor', COL['floor'], -40, 135, -1, 0, -1, 130)
box('room', 'Baseboard', COL['base'], -40, 135, 0, P['baseboard_h'], 0, P['baseboard_t'])
OX = 85.5   # outlet centre x, right next to right back leg (user: above right-back corner)
oy0, oy1 = P['outlet_plate']
box('room', 'Outlet plate', '#fbfbf8', OX - 3.5, OX + 3.5, oy0, oy1, 0, 0.5)
SOCK_UP = v(OX, oy0 + 8.3, 0.5); SOCK_LO = v(OX, oy0 + 3.1, 0.5)
for nm, s in (('Upper socket (monitor)', SOCK_UP), ('Lower socket (strip)', SOCK_LO)):
    box('room', nm, '#e9e9e4', s[0] - 1.8, s[0] + 1.8, s[1] - 1.6, s[1] + 1.6, 0.5, 0.75)

group('desk', label='Desk - Mainstays side storage')
box('desk', 'Desktop', COL['wood'], 0, W, UND, TOP, Z0, Z1)
for nm, z in (('back', (Z0, Z0 + 2)), ('front', (Z1 - 2, Z1))):
    box('desk', 'Right leg ' + nm, COL['frame'], W - 2, W, 0, UND, *z)
    box('desk', 'Left post ' + nm, COL['frame'], POST_X0, POST_X1, 0, TS_Y0, *(POST_ZB if nm == 'back' else POST_ZF))
    box('desk', 'Rail ' + nm, COL['frame'], POST_X1, W - 2, RAIL_Y0, UND, *z)
box('desk', 'Rail right', COL['frame'], W - 2, W, RAIL_Y0, UND, Z0 + 2, Z1 - 2)
box('desk', 'Rail left', COL['frame'], POST_X0, POST_X1, RAIL_Y0, UND, POST_ZB[1], POST_ZF[0])
box('desk', 'Foot bar right', COL['frame'], W - 2, W, 3.0, 5.0, Z0 + 2, Z1 - 2)
box('desk', 'Foot bar left', COL['frame'], POST_X0, POST_X1, 3.0, 5.0, POST_ZB[1], POST_ZF[0])
xb = (POST_X0 + POST_X1) / 2
cyl('desk', 'X-brace a', COL['frame'], (xb, 12, Z0 + 1), (W - 1, RAIL_Y0, Z0 + 1), 0.3)
cyl('desk', 'X-brace b', COL['frame'], (xb, RAIL_Y0, Z0 + 1), (W - 1, 12, Z0 + 1), 0.3)
box('desk', 'Top shelf', COL['wood'], 0, SH_W, TS_Y0, TS_Y1, TS_Z0, TS_Z1)
box('desk', 'Middle shelf', COL['wood'], 0, SH_W, MS_Y0, MS_Y1, SH_Z0, SH_Z1)
box('desk', 'Bottom shelf', COL['wood'], 0, SH_W, BS_Y0, BS_Y1, SH_Z0, SH_Z1)

# ---------------------------------------------------------------- DIY riser (design)
RX0, RX1 = 29.5, 89.5
RZ0, RZ1 = Z0, Z0 + 23.0
RH = P['riser_h']; RT = 1.8
group('riser', label='DIY monitor riser (18 mm ply)')
box('riser', 'Riser top', COL['riser'], RX0, RX1, TOP + RH - RT, TOP + RH, RZ0, RZ1)
box('riser', 'Riser side L', COL['riser'], RX0, RX0 + RT, TOP, TOP + RH - RT, RZ0, RZ1)
box('riser', 'Riser side R', COL['riser'], RX1 - RT, RX1, TOP, TOP + RH - RT, RZ0, RZ1)
box('riser', 'Riser back rail', COL['riser'], RX0 + RT, RX1 - RT, TOP, TOP + 2.5, RZ0, RZ0 + RT)
GARAGE = dict(x=(RX0 + RT, RX1 - RT), y=(TOP, TOP + RH - RT), z=(RZ0 + RT, RZ1))

# ---------------------------------------------------------------- monitor on stock stand
MXC = (RX0 + RX1) / 2
BASE_Y = TOP + RH
PANEL_W, PANEL_H, PANEL_D = 61.39, 36.63, 5.13
TOP_TARGET = EYE_Y - 2.0                               # Mayo: top at/slightly below eye
H_SET = TOP_TARGET - BASE_Y                            # must be within 38.6..51.6
group('stand', label='AOC stock stand')
pose_stand = (MXC, BASE_Y, RZ0 + 1.0)                  # origin: base rear centre
GROUPS['stand']['poses'] = {m: (v(*pose_stand), QI.copy()) for m in MODES}
box('stand', 'Stand base rear', COL['black'], -12, 12, 0, 1.6, 0, 10)
box('stand', 'Stand base front', COL['black'], -6, 6, 0, 1.6, 10, 20.77)
box('stand', 'Stand neck', COL['black'], -3.5, 3.5, 1.6, 37.0, 1.2, 5.2)
FACE_Z = 20.77 - 2.5                                   # panel face rel. to base rear
group('panel', parent='stand', label='AOC Q27G4XN panel')
GROUPS['panel']['poses'] = {m: (v(0, H_SET - PANEL_H, FACE_Z), QI.copy()) for m in MODES}
box('panel', 'Panel front shell', COL['black'], -PANEL_W/2, PANEL_W/2, 0, PANEL_H, -1.2, 0)
box('panel', 'Panel rear housing', '#262628', -19, 19, 2.6, 25.6, -PANEL_D, -1.2)
part('panel', 'box', 'Screen', COL['screen_off'], c=(0, 2.18 + 33.57/2, 0.03), s=(59.67, 33.57, 0.06),
     emissive={'monitor': '#3b6fb6', 'dual': '#3b6fb6', 'laptop': None, 'notebook': None})
box('panel', 'VESA bracket', COL['black'], -3.0, 3.0, 10.1, 18.1, -(FACE_Z - 5.2), -PANEL_D)
PORTS_LOCAL = {'DP': 8.0, 'HDMI1': 5.5, 'HDMI2': 3.0, 'Earphone': 1.0, 'AC': -13.0}
for n, x in PORTS_LOCAL.items():
    box('panel', 'Port ' + n, '#3a3a3a', x - 0.9, x + 0.9, 2.55, 2.6, -3.7, -2.7)

# ---------------------------------------------------------------- dock on middle shelf
group('dock', label='Dell WD19DCS')
DK = dict(x=(1.0, 21.5), y=(MS_Y1, MS_Y1 + 2.9), z=(POST_ZB[1] + 0.6, POST_ZB[1] + 9.6))  # rear 13 cm off wall: plug + bend
box('dock', 'Dock', COL['dock'], DK['x'][0], DK['x'][1], *DK['y'], *DK['z'])
DKY = DK['y'][0] + 1.45
DOCK_DP1 = v(9.0, DKY, DK['z'][0]); DOCK_DC = v(19.0, DKY, DK['z'][0])
DOCK_HOST = v(DK['x'][0], DKY, DK['z'][0] + 4.5)

# ---------------------------------------------------------------- basket, strip, brick
group('basket', label='Basket + LiOA strip + 240 W brick')
BK = dict(x=(0.3, 0.3 + 10.5 * IN), y=(BS_Y1, BS_Y1 + 4.5 * IN), z=(SH_Z0 + 1.0, SH_Z0 + 1.0 + 17 * IN))
t = 0.6
box('basket', 'Basket floor', COL['basket'], BK['x'][0], BK['x'][1], BK['y'][0], BK['y'][0] + t, *BK['z'])
for nm, xs, zs in (('L', (BK['x'][0], BK['x'][0] + t), BK['z']), ('R', (BK['x'][1] - t, BK['x'][1]), BK['z']),
                   ('back', BK['x'], (BK['z'][0], BK['z'][0] + t)), ('front', BK['x'], (BK['z'][1] - t, BK['z'][1]))):
    box('basket', 'Basket wall ' + nm, COL['basket'], xs[0], xs[1], BK['y'][0], BK['y'][1], zs[0], zs[1])
FLOOR_Y = BK['y'][0] + t
IX = (BK['x'][0] + t, BK['x'][1] - t); IZ = (BK['z'][0] + t, BK['z'][1] - t)   # basket inner
PSI = 28.0                                              # strip yaw, diagonal like the photo
SL, SWD, SHT = 40.0, 5.5, 4.0                           # strip body (estimate)
sdir = v(-math.sin(math.radians(PSI)), 0, math.cos(math.radians(PSI)))   # back end -> front end
back_c = v(IX[1] - SWD / 2 * math.cos(math.radians(PSI)) - 0.8, FLOOR_Y,
           IZ[0] + SWD / 2 * math.sin(math.radians(PSI)) + 5.0)
S_C = back_c + sdir * SL / 2
group('strip', pos=S_C, quat=q_axis((0, 1, 0), -PSI), label='LiOA 6-outlet strip (diagonal)')
box('strip', 'LiOA strip', COL['strip'], -SWD / 2, SWD / 2, 0, SHT, -SL / 2, SL / 2)
OUTLET_COLORS = ['#1a1a1a', '#2f9e5b', '#2e5fb3', '#e6c229', '#c0392b', '#7a4a2a']
OUT_Z = [-14.6, -9.4, -4.2, 7.8, 13.0, 18.2]
for i, (cz, col) in enumerate(zip(OUT_Z, OUTLET_COLORS)):
    box('strip', 'Outlet %d' % (i + 1), col, -2.1, 2.1, SHT, SHT + 0.05, cz - 2.1, cz + 2.1)
for cz in (0.4, 3.2):
    box('strip', 'Strip switch', '#b22222', -0.9, 0.9, SHT, SHT + 0.4, cz - 1.0, cz + 1.0)
outlets = [wpt('strip', 'monitor', (0, SHT + 3.0, cz)) for cz in OUT_Z]   # top of plug bodies
STRIP_CORD = wpt('strip', 'monitor', (0, 2.0, -SL / 2)); STRIP_CORD_DIR = -sdir
for n_, i_ in (('brick plug', 0), ('lamp plug + switch', 1)):
    box('strip', 'Plug: ' + n_, '#2a2a2a', -1.6, 1.6, SHT, SHT + 3.0, OUT_Z[i_] - 1.8, OUT_Z[i_] + 1.8)
# 240 W brick stands on its long edge along the left basket wall, DC end toward the wall
BR = dict(x=(IX[0] + 0.05, IX[0] + 0.05 + 2.54), y=(FLOOR_Y, FLOOR_Y + 10.0), z=(IZ[0] + 5.8, IZ[0] + 25.8))
box('basket', 'Dock power brick 240 W', COL['brick'], BR['x'][0], BR['x'][1], *BR['y'], *BR['z'])
BRX = (BR['x'][0] + BR['x'][1]) / 2; BRY = (BR['y'][0] + BR['y'][1]) / 2
BRICK_DC = v(BRX, BRY, BR['z'][0]); BRICK_AC = v(BRX, BRY, BR['z'][1])

# ---------------------------------------------------------------- laptop (Dell 16)
LW, LD, LB, LL = 35.73, 25.06, 1.39, 0.60   # base thickness + lid = 1.99 max
group('laptop', label='Dell 16 DC16250')
group('lid', parent='laptop', pos=(0, LB, 0), label='Dell 16 lid')
box('laptop', 'Laptop base', COL['laptop'], 0, LW, 0, LB, 0, LD)
box('laptop', 'Keyboard deck', '#1f2024', 2.5, LW - 2.5, LB, LB + 0.02, 2.0, 13.0)
box('lid', 'Laptop lid', COL['laptop'], 0, LW, 0, 24.3, -LL, 0)
part('lid', 'box', 'Laptop screen', COL['screen_off'], c=(LW / 2, 12.6, 0.02), s=(34.2, 21.4, 0.04),
     emissive={'monitor': None, 'dual': '#4a7cc7', 'laptop': '#4a7cc7', 'notebook': None})
USBC_LOCAL = (0.0, 0.7, 4.0)
def lid_open(deg): return q_axis((1, 0, 0), 90 - deg)
# cubby (monitor, notebook): closed, rotated -90 about y so USB-C faces the wall, hinge vents to open desk
cub_pos = (25.56, TOP + 1.0, 15.0)
for m in ('monitor', 'notebook'):
    pose('laptop', m, cub_pos, q_axis((0, 1, 0), -90)); pose('lid', m, quat=lid_open(0))
pose('laptop', 'dual', (SH_W / 2 - LW / 2, TS_Y1, TS_Z0 + 17.4), QI); pose('lid', 'dual', quat=lid_open(105))
pose('laptop', 'laptop', (25.0, TOP, Z1 - LD - 0.2), QI); pose('lid', 'laptop', quat=lid_open(105))
group('laptop_feet', label='Printed vented laptop feet (cubby)')
for zc in (17.0, 48.5):
    box('laptop_feet', 'Laptop foot', COL['print'], 1.5, 24.5, TOP, TOP + 1.0, zc - 1.0, zc + 1.0)

# ---------------------------------------------------------------- lamp (articulated)
L1, L2 = 26.0, 28.0
LB_C = v(SH_W / 2, TS_Y1, TS_Z0 + 8.6 + 0.2)
group('lamp_base', pos=LB_C, label='Mainstays LED architect lamp')
cyl('lamp_base', 'Lamp base', COL['lamp'], (0, 0, 0), (0, 2.5, 0), 8.6)
cyl('lamp_base', 'Lamp post', COL['chrome'], (0, 2.5, 0), (0, 7.0, 0), 0.9, 0.7)
group('lamp_yaw', parent='lamp_base', pos=(0, 7.5, 0))
group('lamp_l1', parent='lamp_yaw')
group('lamp_l2', parent='lamp_l1', pos=(0, L1, 0))
group('lamp_head', parent='lamp_l2', pos=(0, L2, 0))
for s in (-0.9, 0.9):
    cyl('lamp_l1', 'Lamp lower arm', COL['lamp'], (s, 0, 0), (s, L1, 0), 0.3)
    cyl('lamp_l2', 'Lamp upper arm', COL['lamp'], (s, 0, 0), (s, L2, 0), 0.3)
cyl('lamp_head', 'Lamp neck', COL['chrome'], (0, 0, 0), (0, 2.5, 0), 0.6)
cyl('lamp_head', 'Lamp shade', COL['lamp'], (0, 2.5, 0), (0, 13.5, 0), 3.5, 7.15)
part('lamp_head', 'cyl', 'Lamp LED', '#f4efe2', p1=(0, 13.3, 0), p2=(0, 13.45, 0), r1=6.6, r2=6.6,
     emissive={'monitor': '#fff3d6', 'dual': None, 'laptop': '#fff3d6', 'notebook': '#fff3d6'})
SHOULDER = LB_C + v(0, 7.5, 0)
LAMP_TARGETS = {   # head joint, aim point
    'monitor': (v(24.0, 104.0, 44.0), v(MXC, TOP, 46.0)),
    'dual': (v(SH_W / 2 - 1.0, 146.0, TS_Z0 + 2.0), v(SH_W / 2, 160.0, 0.0)),   # folded upright behind the laptop, off
    'laptop': (v(13.0, 113.0, 40.0), v(42.0, TOP + 2, 50.0)),
    'notebook': (v(34.0, 110.0, 44.0), v(MXC - 2, TOP, 46.0)),
}
LAMP_ON = {'monitor': True, 'dual': False, 'laptop': True, 'notebook': True}
LAMP_ELBOW = {'monitor': 1, 'dual': -1, 'laptop': 1, 'notebook': 1}

def lamp_ik(mode):
    H, A = LAMP_TARGETS[mode]
    d = H - SHOULDER
    yaw = math.degrees(math.atan2(d[0], d[2]))
    fwd = math.hypot(d[0], d[2]); up = d[1]
    D = math.hypot(fwd, up)
    assert abs(L1 - L2) < D < L1 + L2, ('lamp target unreachable', mode, D)
    phi = math.atan2(fwd, up)
    gam = math.acos((L1 * L1 + D * D - L2 * L2) / (2 * L1 * D))
    p1 = phi - gam if LAMP_ELBOW[mode] > 0 else phi + gam
    e = v(L1 * math.sin(p1), L1 * math.cos(p1))
    p2 = math.atan2(fwd - e[0], up - e[1])
    pose('lamp_yaw', mode, quat=q_axis((0, 1, 0), yaw))
    pose('lamp_l1', mode, quat=q_axis((1, 0, 0), math.degrees(p1)))
    pose('lamp_l2', mode, quat=q_axis((1, 0, 0), math.degrees(p2 - p1)))
    # head: rotate local +y onto aim direction expressed in l2 frame
    m = world('lamp_l2', mode)[:3, :3]
    aim_local = m.T @ norm(A - H)
    pose('lamp_head', mode, quat=q_from_to(v(0, 1, 0), aim_local))
for m in MODES:
    lamp_ik(m)

# ---------------------------------------------------------------- keyboard + mouse (generic)
group('keyboard', label='Generic BT keyboard 44 x 13.5 x 2.5')
box('keyboard', 'Keyboard', COL['kb'], -22, 22, 0, 2.5, -6.75, 6.75)
group('mouse', label='Generic BT mouse')
box('mouse', 'Mouse', COL['kb'], -3.25, 3.25, 0, 3.8, -5.75, 5.75)
KB_DESK = (MXC, TOP, 44.0); MS_DESK = (MXC + 28.0, TOP, 46.0)
KB_GAR = (GARAGE['x'][0] + 22.5, TOP, GARAGE['z'][0] + 7.5); MS_GAR = (GARAGE['x'][1] - 4.5, TOP, GARAGE['z'][0] + 7.0)
for m in MODES:
    stowed = m in ('laptop', 'notebook')
    pose('keyboard', m, KB_GAR if stowed else KB_DESK)
    pose('mouse', m, MS_GAR if stowed else MS_DESK)
pose('keyboard', 'dual', (MXC - 6, TOP, 44.0)); pose('mouse', 'dual', (MXC + 22, TOP, 46.0))

# ---------------------------------------------------------------- notebook + pens
group('notebook', label='A4 notebook (open) + pens')
box('notebook', 'Notebook', COL['paper'], -21, 21, 0, 0.8, -14.85, 14.85)
cyl('notebook', 'Pen', '#1d4fa3', (24, 0.5, -8), (24.5, 0.5, 6), 0.45)
cyl('notebook', 'Pencil', '#e2b33c', (26, 0.5, -8), (26.5, 0.5, 6), 0.4)
NB_POS = (MXC - 2.0, TOP, Z1 - 14.85 - 0.3)
for m in MODES:
    pose('notebook', m, NB_POS)
    GROUPS['notebook']['visible'][m] = (m == 'notebook')
pose('notebook', 'monitor', (MXC - 2, TOP - 0.79, NB_POS[2]))  # hidden: tucked flat in place

# ---------------------------------------------------------------- person (ergonomic envelope)
group('person', label='175 cm user envelope')
PX = {'monitor': MXC, 'dual': MXC - 8.0, 'laptop': 25.0 + LW / 2, 'notebook': MXC - 2}
EYE_Z = Z1 + 22.0
for m in MODES:
    pose('person', m, (PX[m], 0, 0))
box('person', 'Seat (gas lift) at %.1f cm' % SEAT, '#4b5563', -22, 22, SEAT - 6, SEAT, Z1 - 4, Z1 + 42, opacity=0.35)
box('person', 'Thighs', COL['zone'], -13, 13, SEAT, SEAT + P['thigh_thk'], Z1 - 15, Z1 + 30, opacity=0.25)
box('person', 'Lower legs', COL['zone'], -12, 12, FOOTREST, SEAT, Z1 - 20, Z1 - 6, opacity=0.25)
box('person', 'Footrest %.1f cm' % FOOTREST, '#7b6a58', -20, 20, 0, FOOTREST, Z1 - 30, Z1 + 5)
part('person', 'sphere', 'Eye point', '#d62828', c=(0, EYE_Y, EYE_Z), r=1.0)
for s in (-1, 1):
    part('person', 'sphere', 'Elbow', '#d62828', c=(s * 21, SEAT + P['elbow_above_seat'], Z1 + 18), r=1.0)

# ---------------------------------------------------------------- cables
TIGHT = []
def fillet_path(pts, radius):
    """Round every interior corner with the bend radius. Tangent lengths are shared between
    neighbouring corners so no segment is over-used; where a segment is too short the radius
    shrinks and is reported. Returns dense points and the smallest radius achieved."""
    pts = [v(*p) for p in pts]
    clean = [pts[0]]
    for p in pts[1:]:
        if np.linalg.norm(p - clean[-1]) > 1e-6: clean.append(p)
    pts = clean; n = len(pts)
    seg = [np.linalg.norm(pts[j + 1] - pts[j]) for j in range(n - 1)]
    th = [math.pi] * n; t = [0.0] * n
    for i in range(1, n - 1):
        u = norm(pts[i - 1] - pts[i]); w = norm(pts[i + 1] - pts[i])
        th[i] = math.acos(max(-1.0, min(1.0, float(np.dot(u, w)))))
        if th[i] < math.pi - 1e-3:
            t[i] = radius / math.tan(th[i] / 2)
    for _ in range(50):
        changed = False
        for j in range(n - 1):
            s = t[j] + t[j + 1]
            if s > seg[j] + 1e-9:
                k = seg[j] / s; t[j] *= k; t[j + 1] *= k; changed = True
        if not changed: break
    out = [pts[0]]; rmin = math.inf; TIGHT.clear()
    for i in range(1, n - 1):
        B = pts[i]
        if t[i] < 1e-6:
            out.append(B); continue
        u = norm(pts[i - 1] - B); w = norm(pts[i + 1] - B)
        r = t[i] * math.tan(th[i] / 2)
        if th[i] < math.radians(170):
            rmin = min(rmin, r)
            if r < radius * 0.98:
                TIGHT.append((np.round(B, 1).tolist(), round(180 - math.degrees(th[i])), round(r, 1),
                              round(seg[i - 1], 1), round(seg[i], 1)))
        P1, P2 = B + u * t[i], B + w * t[i]
        C0 = B + norm(u + w) * (r / math.sin(th[i] / 2))
        a, b = P1 - C0, P2 - C0
        ang = math.acos(max(-1.0, min(1.0, float(np.dot(norm(a), norm(b))))))
        k_n = max(2, int(math.degrees(ang) / 5))
        for k in range(k_n + 1):
            s = k / k_n
            q = a if ang < 1e-6 else (math.sin((1 - s) * ang) * a + math.sin(s * ang) * b) / math.sin(ang)
            out.append(C0 + q)
    out.append(pts[-1])
    return out, (rmin if rmin < math.inf else 99.0)

def resample(pts, n):
    pts = np.array(pts); seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    s = np.concatenate([[0], np.cumsum(seg)]); L = s[-1]
    out = []
    for t in np.linspace(0, L, n):
        i = min(np.searchsorted(s, t, side='right') - 1, len(seg) - 1)
        f = 0 if seg[i] == 0 else (t - s[i]) / seg[i]
        out.append(pts[i] + f * (pts[i + 1] - pts[i]))
    return np.array(out), L

def sag(a, b, k=0.05):
    """Midpoint of a free span dropped by k * span (gravity)."""
    a, b = v(*a), v(*b); m = (a + b) / 2
    return m - v(0, k * np.linalg.norm(b - a), 0)

CABLES = []
NPTS = 120
BUNDLE_D = {'dp': 10.0, 'ac': 12.0, 'usbc': 12.0, 'dc': 8.0, 'brac': 8.0, 'lamp': 8.0, 'strip': 9.0}
DEBUG_MODES = ('monitor', 'dual', 'laptop')

def cable(cid, name, od, rated, k_bend, lead0, lead1, routes, bundle_at, bundle_axis, code, note):
    """routes[mode] = (port0, dir0, [waypoints], port1, dir1). Leads = straight plug length."""
    R = k_bend * od
    entry = dict(id=cid, name=name, od=od, rated=rated, rmin_req=round(R, 1), code=code, note=note,
                 points={}, length={}, rmin={}, excess={}, bundle={})
    for m in MODES:
        p0, d0, way, p1, d1 = routes[m]
        p0, p1 = v(*p0), v(*p1); d0, d1 = norm(v(*d0)), norm(v(*d1))
        def lead_pts(p, d, lead, nxt):
            # straight plug, then the bend vertex far enough out for the turn it has to make
            q = p + d * (lead + R)
            for _ in range(3):
                th = math.acos(max(-1.0, min(1.0, float(np.dot(-d, norm(v(*nxt) - q))))))
                q = p + d * (lead + max(R, R / max(math.tan(th / 2), 1e-3)))
            return [p, p + d * lead, q]
        a0 = lead_pts(p0, d0, lead0, way[0] if way else p1)
        a1 = lead_pts(p1, d1, lead1, way[-1] if way else p0)
        poly = a0 + [v(*w) for w in way] + a1[::-1]
        dense, rmin = fillet_path(poly, R)
        pts, L = resample(dense, NPTS)
        ex = rated - L
        entry['points'][m] = np.round(pts, 2).tolist()
        entry['length'][m] = round(L, 1)
        entry['rmin'][m] = round(rmin, 1)
        if TIGHT and m in DEBUG_MODES: print('TIGHT', cid, m, TIGHT)
        entry['excess'][m] = round(ex, 1)
        D = BUNDLE_D.get(cid, 10.0)
        turns = max(ex, 0) / (math.pi * D)
        entry['bundle'][m] = dict(c=list(map(float, bundle_at)), axis=list(map(float, bundle_axis)),
                                  R=D / 2 if turns > 0.2 else 0.0,
                                  r=round(od / 2 * max(1.0, math.sqrt(turns) * 1.15), 2),
                                  turns=round(turns, 1))
    CABLES.append(entry)

G = Z0 - 2.8          # cable run plane in the 6.35 cm wall gap (behind X-brace and rails)
def port(gid, m, local): return wpt(gid, m, local)
def pdir(gid, m, local): return wdir(gid, m, local)
def mon_port(n, m): return port('panel', m, (PORTS_LOCAL[n], 2.6, -3.2))

# Data cables use 8 x OD, power cords 5 x OD as the minimum static bend radius.
dp_routes, ac_routes = {}, {}
for m in MODES:
    dpp = mon_port('DP', m); acp = mon_port('AC', m)
    y_dp = dpp[1] - 5.0 - 4.8; y_ac = acp[1] - 5.5 - 3.75
    dp_routes[m] = (DOCK_DP1, (0, 0, -1), [
        (DOCK_DP1[0], 64.0, G), (30.0, 70.0, G), (dpp[0], 77.0, G), (dpp[0], y_dp, G)],
        dpp, (0, -1, 0))
    ac_routes[m] = (SOCK_UP, (0, 0, 1), [
        (OX, 40.0, Z0 + 1.9), (78.0, 56.0, G), (acp[0], 79.0, G), (acp[0], y_ac, G)],
        acp, (0, -1, 0))
cable('dp', 'DP 1.4 (monitor box), dock rear DP1 -> monitor DP', 0.6, 150, 8, 5.0, 5.0, dp_routes,
      (36.0, 70.0, G), (0, 0, 1), '#e63946', 'length assumed 1.5 m (AOC does not publish it)')
cable('ac', 'Monitor AC cord (3-prong), upper wall socket -> monitor', 0.75, 150, 5, 4.0, 5.5, ac_routes,
      (70.0, 52.0, G), (0, 0, 1), '#f77f00', 'length assumed 1.5 m; grounded, so it stays on the wall socket')

usb_routes = {}
for m in MODES:
    lp = port('laptop', m, USBC_LOCAL); ld = pdir('laptop', m, (-1, 0, 0))
    if m in ('monitor', 'notebook'):        # closed in the cubby, port faces the wall
        way = [(-6.8, 64.0, 14.0), (-4.5, 74.0, 7.5), (-2.5, 76.9, 4.9), (lp[0] - 6.0, 76.9, 4.9)]
    elif m == 'dual':                       # open on the top shelf, port faces left
        way = [(-8.0, 66.0, 22.0), (-16.5, 84.0, lp[2])]
    else:                                   # open at the front of the desk, port faces left
        way = [(-6.8, 64.0, 22.0), (-3.5, 74.0, 28.0), (-1.0, 77.2, 33.0), (6.0, 76.75, lp[2])]
    usb_routes[m] = (DOCK_HOST, (-1, 0, 0), way, lp, ld)
cable('usbc', 'WD19DCS USB-C host cable (0.8 m, fixed)', 0.7, 80, 8, 2.0, 4.5, usb_routes,
      (-8.0, 60.0, 19.0), (1, 0, 0), '#3a86ff', 'spec length 0.8 m; Dual mode needs ~17 cm free left of the desk')

lamp_cord_start = LB_C + v(0, 0.8, -8.6)
dc_routes, brac_routes, lamp_routes, strip_routes = {}, {}, {}, {}
ob, ol = outlets[0], outlets[1]
for m in MODES:
    dc_routes[m] = (BRICK_DC, (0, 0, -1), [
        (BRX, BK['y'][1] + 1.7, IZ[0] + 0.2), (4.0, 28.0, G), (DOCK_DC[0], 44.0, 5.0)],
        DOCK_DC, (0, 0, -1))
    brac_routes[m] = (BRICK_AC, (0, 0, 1), [(6.0, 21.0, 47.0), (17.0, 22.6, 30.0)], ob, (0, 1, 0))
    lamp_routes[m] = (lamp_cord_start, (0, 0, -1), [
        (LB_C[0], 84.0, 7.0), (10.0, 70.0, G), (12.0, 26.0, G), (16.0, BK['y'][1] + 2.6, IZ[0] + 0.2)],
        ol, (0, 1, 0))
    sv = STRIP_CORD + STRIP_CORD_DIR * (2.0 + 3.75)        # cord leaves the strip end, then rises
    strip_routes[m] = (STRIP_CORD, STRIP_CORD_DIR, [
        (sv[0], BK['y'][1] + 1.6, sv[2]), (sv[0], BK['y'][1] + 1.6, sv[2] - 7.6), (sv[0] + 0.2, 4.0, sv[2] - 7.6), (80.0, 0.45, 6.0),
        (OX, 0.45, SOCK_LO[2] + 8.0 + 3.75)],
        SOCK_LO, (0, 0, 1))
cable('dc', '240 W brick DC (7.4 mm barrel) -> dock DC-in', 0.55, 180, 5, 2.5, 5.5, dc_routes,
      (19.5, 15.4, 44.0), (0, 1, 0), '#ffbe0b', 'length assumed 1.8 m (Dell 240 W typical)')
cable('brac', '240 W brick AC cord -> strip outlet 1 (black)', 0.75, 100, 5, 4.0, 1.0, brac_routes,
      (19.5, 17.0, 44.0), (0, 1, 0), '#ffd166', 'length assumed 1.0 m')
cable('lamp', 'Lamp cord (5 ft) -> strip outlet 2 (green)', 0.55, 152.4, 5, 1.5, 1.0, lamp_routes,
      (19.5, 13.8, 44.0), (0, 1, 0), '#2a9d8f', 'the on/off switch is on the plug, i.e. in the basket')
cable('strip', 'LiOA strip cord (5 m) -> lower wall socket', 0.75, 500, 5, 2.0, 8.0, strip_routes,
      (19.5, 12.0, 44.0), (0, 1, 0), '#8338ec', 'about 3.7 m of the 5 m cord ends up coiled in the basket')

# ---------------------------------------------------------------- checks
def aabb(name, m):
    pr = next(p for p in PARTS if p['name'] == name)
    M = world(pr['group'], m)
    if pr['kind'] == 'box':
        c, s = v(*pr['c']), v(*pr['s']) / 2
        corners = [c + s * v(i, j, k) for i in (-1, 1) for j in (-1, 1) for k in (-1, 1)]
    elif pr['kind'] == 'cyl':
        r = max(pr['r1'], pr['r2']); corners = []
        for p in (pr['p1'], pr['p2']):
            corners += [v(*p) + v(i * r, j * r, k * r) for i in (-1, 1) for j in (-1, 1) for k in (-1, 1)]
    else:
        c, r = v(*pr['c']), pr['r']; corners = [c + v(i * r, j * r, k * r) for i in (-1, 1) for j in (-1, 1) for k in (-1, 1)]
    w = np.array([(M @ np.array([*q, 1]))[:3] for q in corners])
    return w.min(0), w.max(0)

def parts_named(name):
    return [p for p in PARTS if p['name'] == name]

def sdf_fn(name, m):
    """Signed distance (world) to all parts with this name, exact for boxes/spheres,
    capped-cone approximation for cylinders."""
    fns = []
    for pr in parts_named(name):
        M = world(pr['group'], m); Mi = np.linalg.inv(M)
        if pr['kind'] == 'box':
            c, s = v(*pr['c']), v(*pr['s']) / 2
            def f(p, Mi=Mi, c=c, s=s):
                q = np.abs((Mi @ np.array([*p, 1]))[:3] - c) - s
                return float(np.linalg.norm(np.maximum(q, 0)) + min(q.max(), 0))
        elif pr['kind'] == 'cyl':
            a, b = (M @ np.array([*pr['p1'], 1]))[:3], (M @ np.array([*pr['p2'], 1]))[:3]
            r1, r2 = pr['r1'], pr['r2']; ab = b - a; L2 = float(ab @ ab)
            def f(p, a=a, ab=ab, L2=L2, r1=r1, r2=r2):
                h = min(1.0, max(0.0, float((p - a) @ ab) / L2))
                return float(np.linalg.norm(p - (a + ab * h)) - (r1 + (r2 - r1) * h))
        else:
            c = (M @ np.array([*pr['c'], 1]))[:3]; r = pr['r']
            def f(p, c=c, r=r): return float(np.linalg.norm(p - c) - r)
        fns.append(f)
    return lambda p: min(fn(p) for fn in fns)

def surface_samples(name, m, n=7):
    pts = []
    for pr in parts_named(name):
        M = world(pr['group'], m)
        if pr['kind'] == 'box':
            c, s = v(*pr['c']), v(*pr['s']) / 2
            g = np.linspace(-1, 1, n)
            for ax in range(3):
                for sign in (-1, 1):
                    for a_ in g:
                        for b_ in g:
                            q = np.zeros(3); q[ax] = sign
                            o = [i for i in range(3) if i != ax]; q[o[0]] = a_; q[o[1]] = b_
                            pts.append((M @ np.array([*(c + q * s), 1]))[:3])
        elif pr['kind'] == 'cyl':
            a, b = v(*pr['p1']), v(*pr['p2']); ax_ = norm(b - a)
            e1 = norm(np.cross(ax_, v(1, 0, 0) if abs(ax_[0]) < 0.9 else v(0, 1, 0))); e2 = np.cross(ax_, e1)
            for h in np.linspace(0, 1, n * 2):
                r = pr['r1'] + (pr['r2'] - pr['r1']) * h
                for ang in np.linspace(0, 2 * math.pi, 16, endpoint=False):
                    q = a + (b - a) * h + r * (math.cos(ang) * e1 + math.sin(ang) * e2)
                    pts.append((M @ np.array([*q, 1]))[:3])
        else:
            c = v(*pr['c']); r = pr['r']
            for i in range(60):
                z_ = 1 - 2 * (i + 0.5) / 60; rr = math.sqrt(1 - z_ * z_); ph = i * 2.39996
                pts.append((M @ np.array([*(c + r * v(rr * math.cos(ph), z_, rr * math.sin(ph))), 1]))[:3])
    return pts

def gap(a, b, m):
    fb, fa = sdf_fn(b, m), sdf_fn(a, m)
    d = min(min(fb(p) for p in surface_samples(a, m)), min(fa(p) for p in surface_samples(b, m)))
    return round(max(0.0, d), 1)

def occlusion(m, screen_part, blockers, eye):
    """% of the visible surface hidden from the eye (sphere-traced rays)."""
    pr = parts_named(screen_part)[0]
    M = world(pr['group'], m); c, s = v(*pr['c']), v(*pr['s'])
    f = [sdf_fn(b, m) for b in blockers if parts_named(b)]
    sd = lambda p: min(fn(p) for fn in f)
    hit = tot = 0
    for i in np.linspace(-0.47, 0.47, 17):
        for j in np.linspace(-0.47, 0.47, 9):
            loc = c + v(i * s[0], j * s[1] if s[1] > s[2] else 0, 0 if s[1] > s[2] else j * s[2])
            q = (M @ np.array([*loc, 1]))[:3]
            d = q - eye; L = np.linalg.norm(d); d = d / L; t = 0.0; tot += 1
            while t < L - 0.6:
                h = sd(eye + d * t)
                if h < 0.03: hit += 1; break
                t += max(h, 0.15)
    return 100.0 * hit / tot

CHECKS = {}
for m in MODES:
    eye = v(PX[m], EYE_Y, EYE_Z)
    c = []
    def add(name, val, ok, unit='cm', note=''):
        c.append(dict(name=name, value=val if isinstance(val, str) else round(float(val), 1), unit=unit, ok=bool(ok), note=note))
    if m in ('monitor', 'dual'):
        scr = wpt('panel', m, (0, 2.18 + 33.57 / 2, 0))
        dist = np.linalg.norm(scr - eye)
        add('Eye to screen centre', dist, 50 <= dist <= 100, note='Mayo 50-100 cm')
        topa = wpt('panel', m, (0, 2.18 + 33.57, 0))[1]
        add('Top of screen below eye', EYE_Y - topa, 0 <= EYE_Y - topa <= 6, note='Mayo: at or slightly below eye')
        add('Stand height setting (38.6-51.6)', H_SET, 38.6 <= H_SET <= 51.6, note='mid-range leaves adjustment both ways')
        occ = occlusion(m, 'Screen', ['Laptop lid', 'Lamp shade', 'Lamp upper arm', 'Lamp lower arm', 'Keyboard'], eye)
        add('Monitor screen hidden from eye', occ, occ < 2, unit='%')
    if m in ('dual', 'laptop'):
        occ = occlusion(m, 'Laptop screen', ['Lamp shade', 'Lamp upper arm', 'Lamp lower arm', 'Keyboard'], eye)
        add('Laptop screen hidden from eye', occ, occ < 2, unit='%')
    if m == 'dual':
        add('Laptop lid vs lamp (parked)', gap('Laptop lid', 'Lamp upper arm', m), gap('Laptop lid', 'Lamp upper arm', m) >= 1.0)
        add('Laptop lid vs lamp shade', gap('Laptop lid', 'Lamp shade', m), gap('Laptop lid', 'Lamp shade', m) >= 1.0)
        lo, hi = aabb('Laptop base', m)
        add('Laptop overhang past top-shelf front', max(0, hi[2] - TS_Z1), hi[2] - TS_Z1 <= 1.0, note='shelf 16.5 in deep, lamp base behind')
        add('Laptop overhang left of desk edge', -lo[0], True, note='needs this much clear space left of the desk')
    if m == 'laptop':
        add('Lid vs monitor face', gap('Laptop lid', 'Panel front shell', m), gap('Laptop lid', 'Panel front shell', m) >= 1.0)
    if m in ('monitor', 'notebook'):
        add('Laptop in cubby: air above lid', TS_Y0 - aabb('Laptop lid', m)[1][1], TS_Y0 - aabb('Laptop lid', m)[1][1] >= 5)
        add('Laptop vs back post', gap('Laptop base', 'Left post back', m), gap('Laptop base', 'Left post back', m) >= 0.5)
        add('Laptop vs front post', gap('Laptop base', 'Left post front', m), gap('Laptop base', 'Left post front', m) >= 0.5)
    if m in ('laptop', 'notebook'):
        add('Keyboard in garage: headroom', (TOP + RH - RT) - (TOP + 2.5), True)
        add('Mouse in garage: headroom', (TOP + RH - RT) - (TOP + 3.8), (TOP + RH - RT) - (TOP + 3.8) >= 1.0)
    if m == 'notebook':
        add('Clear writing depth in front of riser', Z1 - RZ1, Z1 - RZ1 >= 29.7, note='A4 open landscape needs 29.7')
        occ = occlusion(m, 'Notebook', ['Lamp shade', 'Lamp upper arm', 'Lamp lower arm'], eye)
        add('Notebook hidden by lamp', occ, occ < 2, unit='%')
    if m in ('monitor', 'dual', 'notebook'):
        add('Lamp shade vs monitor', gap('Lamp shade', 'Panel front shell', m), gap('Lamp shade', 'Panel front shell', m) >= 1.0)
    add('Thigh top to frame rail', RAIL_Y0 - (SEAT + P['thigh_thk']), RAIL_Y0 - (SEAT + P['thigh_thk']) >= 1.0,
        note='tight: legs fit, crossed legs will not')
    for cb in CABLES:
        L, ex, rm = cb['length'][m], cb['excess'][m], cb['rmin'][m]
        add('Cable %s: routed / rated' % cb['id'], '%.0f / %.0f' % (L, cb['rated']), ex >= 0,
            unit='cm', note='%.0f cm spare%s' % (ex, ', bundled' if ex > 15 else ''))
        add('Cable %s: tightest bend' % cb['id'], rm, rm >= cb['rmin_req'] * 0.97, note='min %.1f' % cb['rmin_req'])
    CHECKS[m] = c

# loads (engineering notes)
E_PLY = 7000.0e6; b, h, Lspan = 0.23, 0.018, (RX1 - RX0 - 2 * RT) / 100
I = b * h ** 3 / 12
defl = lambda F: F * Lspan ** 3 / (48 * E_PLY * I) * 1000
LOADS = dict(
    riser_deflection_monitor_mm=round(defl(5.11 * 9.81), 2),
    riser_deflection_250N_mm=round(defl(250), 2),
    desktop_load_kg=round(5.11 + 2.6 + 0.6 + 1.98, 1), desktop_rating_kg=22.7,
    top_shelf_load_dual_kg=round(1.48 + 1.98, 2), shelf_rating_kg=6.8,
    middle_shelf_load_kg=0.69, bottom_shelf_load_kg=round(0.77 + 0.6 + 0.9, 2),
    strip_current_A_at_120V=round((240 / 0.9 + 4) / 120, 2), strip_rating_A=10,
    seat_cm=round(SEAT, 1), eye_cm=round(EYE_Y, 1), footrest_cm=FOOTREST,
    stand_height_setting_cm=round(H_SET, 1),
)

# ---------------------------------------------------------------- export
def jgroup(gid, g):
    return dict(parent=g['parent'], label=g['label'],
                poses={m: dict(pos=np.round(p, 3).tolist(), quat=np.round(q, 6).tolist()) for m, (p, q) in g['poses'].items()},
                visible=g['visible'])

spec = dict(units='cm', modes=MODES, mode_label=MODE_LABEL, params={k: (list(v_) if isinstance(v_, tuple) else v_) for k, v_ in P.items()},
            sources=SOURCES, groups={k: jgroup(k, g) for k, g in GROUPS.items()}, parts=PARTS, cables=CABLES,
            lamp_on=LAMP_ON, checks=CHECKS, loads=LOADS,
            camera=dict(target=[45, 85, 30], pos=[150, 175, 230]))

if __name__ == '__main__':
    import os
    here = os.path.dirname(os.path.abspath(__file__))
    with open(os.path.join(here, 'desk_spec.json'), 'w') as f:
        json.dump(spec, f, separators=(',', ':'))
    print('seat %.1f  eye %.1f  footrest %.1f  stand setting %.1f (38.6-51.6)' % (SEAT, EYE_Y, FOOTREST, H_SET))
    for m in MODES:
        bad = [c for c in CHECKS[m] if not c['ok']]
        print('\n==', MODE_LABEL[m], '- %d checks, %d fail' % (len(CHECKS[m]), len(bad)))
        for c in CHECKS[m]:
            print('  %s %-45s %s %s  %s' % ('OK ' if c['ok'] else 'XX ', c['name'], c['value'], c['unit'], c['note']))
    print('\nloads', LOADS)
