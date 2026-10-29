"""Write the overview figure (Fig. 1) as a draw.io file.

Three panels: calibration on simulated anatomical shift and the external test;
deployment at a receiving site; and what BIRS adapts. Radiograph thumbnails from
make_overview_thumbnails.py are embedded in the file. The ImageNet schedule and the cell
counts shown are read from paper_numbers.json and asserted.

Geometry is in millimetres on a 190 mm page; the figure is placed at 165 mm, so
8 pt and 9 pt type prints at about 7 and 7.8 pt.

Output:
    output/fig1_overview.drawio   (export with draw.io: File > Export as > PDF, crop)

Usage:
    python make_overview_thumbnails.py      # first: needs the datasets, see its docstring
    python make_overview_figure.py
"""
import base64
import hashlib
import html
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
NUMBERS = HERE.parent / "code" / "paper" / "paper_numbers.json"
OUT = HERE / "output"
if hashlib.sha256(NUMBERS.read_bytes()).hexdigest() != "9a7f55f08ce68668bae0722c743ba9de751f23d5091bc73b7552c506cd17b297":
    raise SystemExit("!! paper_numbers.json is not the canonical 9a7f55f0 file")
N = {k: v["value"] for k, v in json.loads(NUMBERS.read_text(encoding="utf-8"))["numbers"].items()}
BUDGETS = [10, 25, 50, 100, 200, 500]
lam = {b: N[f"lambda_schedule_at_{b}"] for b in BUDGETS}
assert lam == {10: 0, 25: 0, 50: 0, 100: 0.1, 200: 0.75, 500: 1}
n_ext = int(sum(N[f"rr_heldout_n_cells_at_{b}"] for b in BUDGETS))
assert n_ext == 1345
assert {N[f"b28_{a}_{e}_n_cells_ext"] for e in ("skelex", "raddino", "medsiglip", "dinov3") for a in "ab"} == {449}

K = 2.835                                          # points per mm
INK, SEC = "#0b0b0b", "#52514e"
KIND = {"reuse": ("#efeeea", "#8a8984"), "fit": ("#ddf3ea", "#1baf7a"), "cal": ("#e1edfb", "#2a78d6"),
        "site": ("#fde8de", "#eb6834")}
FONT = "fontFamily=Helvetica;fontColor=#0b0b0b;"
cells = []


def mm(v):
    return round(v * K, 1)


def box(cid, kind, x, y, w, h, title, body="", valign="middle", shape="rounded=1;absoluteArcSize=1;arcSize=9;"):
    fc, ec = KIND[kind]
    label = f"<b>{title}</b>" + (f"<br>{body}" if body else "")
    style = (f"{shape}whiteSpace=wrap;html=1;fillColor={fc};strokeColor={ec};strokeWidth=0.8;{FONT}fontSize=8;"
             f"align=center;verticalAlign={valign};spacing=3;spacingBottom={4 if valign == 'bottom' else 3};")
    cells.append(("v", cid, style, label, (mm(x), mm(y), mm(w), mm(h))))
    return cid


def text(cid, x, y, w, h, label, size=9, bold=True, align="left"):
    style = (f"text;html=1;whiteSpace=wrap;strokeColor=none;fillColor=none;{FONT}fontSize={size};align={align};"
             f"verticalAlign=middle;spacing=0;" + ("fontStyle=1;" if bold else ""))
    cells.append(("v", cid, style, label, (mm(x), mm(y), mm(w), mm(h))))


def image(cid, key, x, y, s):
    b64 = base64.b64encode((OUT / "thumbs" / f"{key}.jpg").read_bytes()).decode()
    style = (f"shape=image;html=1;imageAspect=0;aspect=fixed;image=data:image/jpeg,{b64};"
             "imageBorder=#52514e;strokeWidth=0.5;")
    cells.append(("v", cid, style, "", (mm(x), mm(y), mm(s), mm(s))))
    return cid


def swatch(cid, kind, x, y):
    fc, ec = KIND[kind]
    cells.append(("v", cid, f"rounded=0;html=1;fillColor={fc};strokeColor={ec};strokeWidth=0.8;", "",
                  (mm(x), mm(y), mm(4.2), mm(3))))


EDGE = ("edgeStyle=orthogonalEdgeStyle;rounded=0;html=1;endArrow=block;endFill=1;endSize=4;strokeWidth=0.8;"
        f"strokeColor={SEC};exitX=1;exitY=0.5;exitDx=0;exitDy=0;entryX=0;entryY=0.5;entryDx=0;entryDy=0;")


def edge(cid, s, t, style=EDGE):
    cells.append(("e", cid, style, "", (s, t)))


# The encoder is a trapezoid shape of size 0 (a plain box) drawn with direction=south, which rotates its
# connection points by 90 degrees:
# unrotated (0.5, 1) lands on its left side and (0.5, 0) on its right side.
INTO_TRAP = EDGE.replace("entryX=0;entryY=0.5;", "entryX=0.5;entryY=1;")
OUT_OF_TRAP = EDGE.replace("exitX=1;exitY=0.5;", "exitX=0.5;exitY=0;")


# (a) calibration, external test, comparison -------------------------------------------------------------
text("ta", 0.5, 1, 189, 5, "(a) Calibration on simulated anatomical shifts, then a test on external data")
box("a1", "reuse", 0.5, 8, 44, 47, "Source models (MURA wrist)",
    "ImageNet: DenseNet-121,<br>ConvNeXt-T, Swin-T<br>Foundation: SKELEX,<br>RAD-DINO, MedSigLIP-448,<br>DINOv3-B<br>"
    "fine-tuned or frozen<br>5 seeds each", valign="bottom")
image("i_wrist", "mura_wrist", 15.75, 10, 13.5)
box("a2", "cal", 50, 8, 33, 47, "Calibration", "6 MURA anatomies<br>10–500 labelled<br>studies per target",
    valign="bottom")
for i, k in enumerate(("elbow", "finger", "forearm", "hand", "humerus", "shoulder")):
    image(f"i_{k}", f"mura_{k}", 52 + (i % 3) * 9.9, 10 + (i // 3) * 9.9, 9.1)
box("a3", "cal", 88.5, 8, 31, 47, "Fit λ*(<i>n</i>)",
    "max. mean AUC gain<br>s.t. harm ≤ τ = 5%;<br>one schedule<br>per encoder<br>(ImageNet pooled)")
box("a4", "site", 125, 8, 33, 47, "External test", "GRAZPEDWRI-DX,<br>MFXR, FracAtlas hand<br>λ*(<i>n</i>) unchanged",
    valign="bottom")
for i, k in enumerate(("graz", "mfxr", "fracatlas_hand")):
    image(f"i_{k}", k, 126.6 + i * 10.1, 13, 9.6)
box("a5", "site", 163.5, 8, 26, 47, "Compare policies",
    f"on the same cells:<br>{n_ext:,} ImageNet;<br>449 per encoder<br>and arm")
for i, (s, t) in enumerate((("a1", "a2"), ("a2", "a3"), ("a3", "a4"), ("a4", "a5"))):
    edge(f"ea{i}", s, t)

# (b) deployment --------------------------------------------------------------------------------------------
text("tb", 0.5, 59, 189, 5,
     "(b) Deployment at a receiving site (Algorithm 1): only the count of labelled studies is needed")
sched = f"10, 25, 50 → {lam[10]:g}<br>100 → {lam[100]:g} · 200 → {lam[200]:g} · 500 → {lam[500]:g}"
box("b1", "site", 0.5, 66, 30, 30, "<i>n</i> labelled<br>local studies", valign="bottom")
for i, k in enumerate(("graz_local1", "graz_local2", "graz_local3")):
    image(f"i_l{i}", k, 6.5 + 3 * i, 67.5 + 1.6 * i, 12)
box("b2", "site", 36, 72, 30, 18, "Round down", "to the grid budget <i>n</i>′")
box("b3", "site", 71.5, 66, 50, 30, "Look up λ*(<i>n</i>′)",
    f"in the deployed encoder's schedule<br>ImageNet: {sched}<br>foundation encoders:<br>their own (Fig. 4b)")
box("b4", "site", 129, 66, 60.5, 12.5, "λ* = 0:", "keep the transferred model")
box("b5", "site", 129, 83.5, 60.5, 12.5, "λ* &gt; 0:", "refit the readout, then blend (c)")
edge("eb0", "b1", "b2")
edge("eb1", "b2", "b3")
edge("eb2", "b3", "b4")
edge("eb3", "b3", "b5")

# (c) what is adapted ---------------------------------------------------------------------------------------
text("tc", 0.5, 100, 189, 5, "(c) What BIRS adapts: the encoder stays fixed and only the readout moves")
image("i_in", "graz", 0.5, 114.5, 13)
box("c1", "reuse", 19, 107, 32, 28, "Encoder φ",
    "fixed at the site:<br>ImageNet backbone<br>or foundation encoder",
    shape="shape=trapezoid;perimeter=trapezoidPerimeter;direction=south;size=0;")
box("c2", "reuse", 58, 105, 38, 13, "Source readout", "(<i>w</i><sub>s</sub>, <i>b</i><sub>s</sub>), inherited")
box("c3", "fit", 58, 124, 38, 15, "Refitted readout",
    "(<i>w</i><sub><i>n</i></sub>, <i>b</i><sub><i>n</i></sub>),<br>on the <i>n</i> local studies")
box("c4", "site", 104, 111.5, 50, 21, "Deployed readout",
    "(1 − λ*) · source + λ* · refitted<br>λ* from the schedule in (a)")
box("c5", "reuse", 160, 111.5, 29.5, 21, "Study score", "mean of image<br>probabilities")
edge("ec_in", "i_in", "c1", INTO_TRAP)
edge("ec0", "c1", "c2", OUT_OF_TRAP)
edge("ec1", "c1", "c3", OUT_OF_TRAP)
edge("ec2", "c2", "c4")
edge("ec3", "c3", "c4")
edge("ec4", "c4", "c5")

# key -------------------------------------------------------------------------------------------------------
x = 0.5
for i, (kind, lab) in enumerate((("reuse", "source model, reused unchanged"), ("fit", "fitted on target labels"),
                                 ("cal", "calibration"), ("site", "receiving site or external evaluation"))):
    swatch(f"k{i}", kind, x, 145.5)
    w = 1.45 * len(lab) + 2
    text(f"kt{i}", x + 5.8, 145, w, 4, lab, size=8, bold=False)
    x += 5.8 + w + 3

# write ------------------------------------------------------------------------------------------------------
out = ['<mxfile host="make_overview_figure.py">',
       '  <diagram name="Fig1_overview" id="cbm-fig1">',
       f'    <mxGraphModel dx="900" dy="600" grid="1" gridSize="1" guides="1" tooltips="1" connect="1" arrows="1" '
       f'fold="1" page="1" pageScale="1" pageWidth="{mm(190)}" pageHeight="{mm(151)}" math="0" shadow="0">',
       '      <root>', '        <mxCell id="0" />', '        <mxCell id="1" parent="0" />']
q = lambda s: html.escape(s, quote=True)
for kind, cid, style, label, geo in cells:
    if kind == "v":
        x0, y0, w0, h0 = geo
        out += [f'        <mxCell id="{cid}" value="{q(label)}" style="{q(style)}" vertex="1" parent="1">',
                f'          <mxGeometry x="{x0}" y="{y0}" width="{w0}" height="{h0}" as="geometry" />',
                '        </mxCell>']
    else:
        s, t = geo
        out += [f'        <mxCell id="{cid}" style="{q(style)}" edge="1" parent="1" source="{s}" target="{t}">',
                '          <mxGeometry relative="1" as="geometry" />', '        </mxCell>']
out += ['      </root>', '    </mxGraphModel>', '  </diagram>', '</mxfile>']
(OUT / "fig1_overview.drawio").write_text("\n".join(out) + "\n", encoding="utf-8")
print(f"wrote fig1_overview.drawio ({len(cells)} cells)")
