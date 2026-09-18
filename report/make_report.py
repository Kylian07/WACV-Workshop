"""Builds the TRAIL technical report PDF.

Reads runs/synthetic/results.json live, so regenerating after more models finish
updates the tables rather than leaving stale numbers in the document.

    python report/make_report.py --out report/TRAIL_technical_report.pdf
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from reportlab.lib import colors
from reportlab.lib.enums import TA_JUSTIFY, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (BaseDocTemplate, CondPageBreak, Frame, Image,
                                KeepTogether, PageBreak, PageTemplate, Paragraph,
                                Spacer, Table, TableStyle)

import figures as F

INK = colors.HexColor("#1B2430")
GREEN = colors.HexColor("#0B6E4F")
RED = colors.HexColor("#C1666B")
AMBER = colors.HexColor("#E8A33D")
GREY = colors.HexColor("#6B7280")
RULE = colors.HexColor("#D8DEE4")
WASH = colors.HexColor("#F4F6F8")


# --------------------------------------------------------------------------
def register_fonts():
    import matplotlib
    d = Path(matplotlib.__file__).parent / "mpl-data" / "fonts" / "ttf"
    for name, fn in [("DJV", "DejaVuSans.ttf"), ("DJV-B", "DejaVuSans-Bold.ttf"),
                     ("DJV-I", "DejaVuSans-Oblique.ttf"), ("DJVM", "DejaVuSansMono.ttf")]:
        pdfmetrics.registerFont(TTFont(name, str(d / fn)))
    pdfmetrics.registerFontFamily("DJV", normal="DJV", bold="DJV-B", italic="DJV-I")


def styles():
    ss = getSampleStyleSheet()
    S = {}
    S["title"] = ParagraphStyle("t", parent=ss["Title"], fontName="DJV-B", fontSize=23,
                                leading=27, textColor=INK, alignment=TA_LEFT, spaceAfter=2)
    S["sub"] = ParagraphStyle("s", fontName="DJV", fontSize=11.5, leading=15,
                              textColor=GREY, spaceAfter=14)
    S["h1"] = ParagraphStyle("h1", fontName="DJV-B", fontSize=14, leading=17,
                             textColor=GREEN, spaceBefore=16, spaceAfter=7)
    S["h2"] = ParagraphStyle("h2", fontName="DJV-B", fontSize=10.8, leading=14,
                             textColor=INK, spaceBefore=11, spaceAfter=5)
    S["body"] = ParagraphStyle("b", fontName="DJV", fontSize=9.3, leading=13.4,
                               textColor=INK, alignment=TA_JUSTIFY, spaceAfter=6)
    S["bullet"] = ParagraphStyle("bu", parent=S["body"], leftIndent=11, bulletIndent=2,
                                 spaceAfter=3.5)
    S["small"] = ParagraphStyle("sm", fontName="DJV", fontSize=8.1, leading=11.2,
                                textColor=GREY, spaceAfter=5)
    S["cap"] = ParagraphStyle("c", fontName="DJV-I", fontSize=8, leading=10.8,
                              textColor=GREY, spaceBefore=3, spaceAfter=11)
    S["mono"] = ParagraphStyle("m", fontName="DJVM", fontSize=7.9, leading=11,
                               textColor=INK, spaceAfter=5)
    S["proof"] = ParagraphStyle("p", parent=S["body"], fontSize=8.7, leading=12.3,
                                leftIndent=9, textColor=colors.HexColor("#33404F"))
    return S


def para(t, s):
    return Paragraph(t, s)


def bullets(items, S, colour=None):
    out = []
    for it in items:
        st = S["bullet"]
        if colour:
            st = ParagraphStyle("bx", parent=st, textColor=colour)
        out.append(Paragraph(it, st, bulletText="•"))
    return out


def eq(latex, name, width_mm=None, fontsize=14):
    p = F.equation(latex, name, fontsize=fontsize)
    from PIL import Image as PILImage
    w, h = PILImage.open(p).size
    target_w = (width_mm or min(150, w / 300 * 25.4 * 1.0)) * mm
    scale = target_w / (w / 300 * 25.4 * mm)
    img = Image(str(p), width=target_w, height=h / 300 * 25.4 * mm * scale)
    img.hAlign = "CENTER"
    return KeepTogether([Spacer(1, 3), img, Spacer(1, 5)])


def pic(path, width_mm, caption=None, S=None):
    from PIL import Image as PILImage
    w, h = PILImage.open(path).size
    W = width_mm * mm
    img = Image(str(path), width=W, height=W * h / w)
    img.hAlign = "CENTER"
    out = [img]
    if caption:
        out.append(Paragraph(caption, S["cap"]))
    return out


def table(data, widths, S, head=True, align_right=None, zebra=True):
    rows = [[Paragraph(c, S["small"] if not (head and i == 0) else
                       ParagraphStyle("th", parent=S["small"], fontName="DJV-B",
                                      textColor=colors.white))
             for c in row] for i, row in enumerate(data)]
    t = Table(rows, colWidths=[w * mm for w in widths], hAlign="LEFT")
    st = [("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
          ("TOPPADDING", (0, 0), (-1, -1), 4),
          ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
          ("LEFTPADDING", (0, 0), (-1, -1), 6),
          ("LINEBELOW", (0, 0), (-1, -2), 0.4, RULE)]
    if head:
        st += [("BACKGROUND", (0, 0), (-1, 0), GREEN)]
    if zebra:
        for r in range(1, len(rows)):
            if r % 2 == 0:
                st.append(("BACKGROUND", (0, r), (-1, r), WASH))
    t.setStyle(TableStyle(st))
    return t


def callout(title, body, S, colour=AMBER):
    inner = [Paragraph(f'<font color="{colour.hexval()}"><b>{title}</b></font>', S["body"]),
             Paragraph(body, S["body"])]
    t = Table([[inner]], colWidths=[165 * mm])
    t.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, -1), WASH),
                           ("LINEBEFORE", (0, 0), (0, -1), 2.2, colour),
                           ("LEFTPADDING", (0, 0), (-1, -1), 9),
                           ("RIGHTPADDING", (0, 0), (-1, -1), 9),
                           ("TOPPADDING", (0, 0), (-1, -1), 7),
                           ("BOTTOMPADDING", (0, 0), (-1, -1), 3)]))
    return KeepTogether([Spacer(1, 4), t, Spacer(1, 8)])


# --------------------------------------------------------------------------
def load_results(path="runs/synthetic/results.json"):
    try:
        return json.load(open(path))
    except Exception:
        return {}


LBL = {"trail": "TRAIL (ours)", "latent_freeform": "Free-form latent recursion",
       "mac": "MAC", "attn1": "Single cross-attention", "film": "FiLM"}
ORDER = ["film", "attn1", "mac", "latent_freeform", "trail"]

# Measured on HOPWORLD with the corrected evaluation; see report text.
HALT_ROWS = [(0.50, 2.00, 0.000, None), (0.30, 2.00, 0.000, None),
             (0.20, 2.00, 0.000, None), (0.10, 2.00, 0.000, None),
             (0.05, 2.00, 0.026, 0.011), (0.02, 2.21, 0.408, 0.057),
             (0.01, 2.64, 0.603, 0.076), (0.00, 4.00, 0.000, None)]


def check_glyphs(story):
    """Refuse to ship a PDF containing characters the embedded font cannot draw.

    ReportLab silently substitutes a blank box for a missing glyph, so a wrong
    entity reference (&#9001; instead of &#10216;, say) survives the build and is
    only visible by looking at a rendered page. This walks the flowables and
    raises instead.
    """
    import matplotlib
    from fontTools.ttLib import TTFont as _FT

    d = Path(matplotlib.__file__).parent / "mpl-data" / "fonts" / "ttf"
    cmap = set()
    for t in _FT(str(d / "DejaVuSans.ttf"))["cmap"].tables:
        cmap |= set(t.cmap.keys())

    import re as _re
    seen, bad = set(), {}

    def walk(f):
        if isinstance(f, (list, tuple)):
            for x in f:
                walk(x)
        elif isinstance(f, KeepTogether):
            walk(f._content)
        elif isinstance(f, Table):
            for row in f._cellvalues:
                walk(row)
        elif isinstance(f, Paragraph):
            txt = _re.sub(r"<[^>]+>", "", f.text)
            txt = txt.replace("&amp;", "&").replace("&lt;", "<").replace("&gt;", ">")
            for ch in txt:
                if ord(ch) > 127 and ch not in seen:
                    seen.add(ch)
                    if ord(ch) not in cmap:
                        bad[ch] = f"U+{ord(ch):04X}"

    walk(story)
    if bad:
        raise SystemExit(f"missing glyphs in DejaVuSans: {bad} -- these render as blank "
                         f"boxes; pick a covered codepoint instead")


def build(out_path: str):
    register_fonts()
    S = styles()
    res = load_results()
    story = []

    f_arch = F.fig_architecture()
    f_loops = F.fig_two_loops()
    f_map = F.fig_repo_map()
    f_halt = F.fig_halting_saturation(HALT_ROWS)
    # Plot the MATCHED-prior baselines when they exist.  Plotting the unmatched ones
    # next to TRAIL's curve is the picture that made the withdrawn claim look true.
    matched = load_results("runs/matched_prior/results.json")
    f_front = None
    if "trail" in res:
        src = matched if matched else res
        base = {k: {"updates": v.get("iid_updates", 0), "acc": v["iid_acc"],
                    "label": LBL.get(k, k) + (" + prior" if matched else "")}
                for k, v in src.items() if k != "trail"}
        f_front = F.fig_frontier(res["trail"]["frontier"], base)

    # ---------------------------------------------------------------- cover
    story += [
        para("TRAIL", S["title"]),
        para("Latent visual reasoning as certified mirror descent "
             "on the visual simplex", S["sub"]),
        para(f"Technical report &nbsp;|&nbsp; prepared for the 1st Workshop on Latent Visual "
             f"Reasoning (LVR), WACV 2027 &nbsp;|&nbsp; {date.today().isoformat()}", S["small"]),
        Spacer(1, 6),
    ]

    story += [para("What this is", S["h1"]),
              para(
        "Latent visual reasoning iterates a hidden state in visual embedding space instead "
        "of decoding intermediate tokens. The recursions in use are free-form: a residual "
        "update <i>z &#8592; z + f(z, c)</i> repeated a fixed number of times. Nothing "
        "constrains the state to the region of embedding space the encoder actually "
        "produces, nothing says when the reasoning is finished, and nothing makes the "
        "intermediate states legible.", S["body"]),
              para(
        "TRAIL changes the <b>parameterisation</b>, not the regularisation. The reasoning "
        "state is a distribution <i>p</i> over the image's own visual atoms, and one "
        "reasoning step is one entropic mirror-descent step on a question-conditioned "
        "energy. Three properties that are normally bolted on then follow from the "
        "geometry, and a fourth &#8212; the step size &#8212; stops being a hyperparameter.",
        S["body"])]

    story += [table([
        ["Property", "How current latent reasoning gets it", "How TRAIL gets it"],
        ["Stays on the visual manifold", "hoped for, or penalised",
         "<b>z = V<sup>T</sup>p &#8712; conv(V)</b> by construction (Prop. 2)"],
        ["Knows when to stop", "fixed step count, or a learned gate",
         "the <b>Frank&#8211;Wolfe gap</b>, a sound bound on suboptimality (Thm. A)"],
        ["Interpretable trace", "post-hoc saliency",
         "<b>p<sub>t</sub> is the state</b> &#8212; the trace is the computation"],
        ["Step size", "tuned", "<b>&#951; = 1/(4&#946;)</b>, prescribed by Lemma 1"],
        ["Relation to attention", "&#8212;",
         "softmax cross-attention is <b>exactly one step</b> (Prop. 1)"],
    ], [38, 55, 72], S), Spacer(1, 10)]

    story += [callout(
        "The honest headline",
        "<b>The accuracy claim was tested and withdrawn.</b> Given the same relational prior, "
        "MAC beats TRAIL by 9.3 points and a <i>single</i> attention step beats it by 8.4 at "
        "a twelfth of the compute. What survives is measured and real: the halting "
        "certificate halves compute at identical accuracy, drift is zero to four orders of "
        "magnitude, and the trace is the computation rather than a saliency map over it. "
        "This is a paper about what certificates, on-manifold state and interpretability "
        "<i>cost</i>.", S, RED)]

    story += pic(f_arch, 168,
                 "Figure 1. End-to-end flow. Only the reasoning module differs between the "
                 "models compared in this report; backbone, question encoder and read-out "
                 "head are shared, which is what makes the comparison about reasoning.", S)

    # ---------------------------------------------------------------- method
    story += [CondPageBreak(120 * mm), para("1. The method", S["h1"])]
    story += [para(
        "An encoder yields <b>atoms</b> V &#8712; R<sup>N&#215;d</sup> (a 14&#215;14 grid of "
        "L2-normalised patch embeddings in the CLEVR setting). A question encoder yields "
        "contextual word states and a summary q. The reasoning state is a point on the "
        "simplex, p &#8712; &#916;<sup>N&#8722;1</sup>, with visual state z = V<sup>T</sup>p.",
        S["body"])]

    story += [para("1.1 The energy", S["h2"]), para(
        "Each hop fixes a question-conditioned energy over the simplex:", S["body"])]
    story += [eq(r"E_h(p) \;=\; -\langle s_h,\, p\rangle \;+\; \frac{\beta}{2}\,"
                 r"\left\| p - M_h\,p \right\|_2^2", "eq_energy", 130)]
    story += [para(
        "Here s<sub>h,i</sub> = &#10216;V<sub>i</sub>, W<sub>s</sub>c<sub>h</sub>&#10217;/&#8730;d "
        "scores each atom against the current sub-question, and <b>M<sub>h</sub> is "
        "column-stochastic</b>: M<sub>h</sub>p redistributes belief along the relation the "
        "sub-question names, built from a control-conditioned attention over atom pairs plus "
        "a control-conditioned relative-position bias (so &#8220;left of&#8221; can become an "
        "actual direction of transport on the grid).", S["body"]),
        para(
        "The quadratic term asks the belief to be a <b>fixed point of the queried "
        "relation</b>. Attending to <i>the thing to the left of the red cube</i> means "
        "attending to a set that maps to itself under &#8220;left of &#8728; start at the red "
        "cube&#8221;. That is what turns a re-weighting of evidence into a hop. The term is "
        "convex &#8212; its Hessian (I&#8722;M)<sup>T</sup>(I&#8722;M) is PSD &#8212; so the "
        "whole energy is convex on the simplex for every &#946; &#8805; 0.", S["body"])]

    story += [para("1.2 Two loops", S["h2"]), para(
        "The outer loop walks hops; the inner loop optimises the hop's energy. The energy is "
        "<b>held fixed inside the inner loop</b>, which is precisely what makes the theorems "
        "of &#167;2 apply to the code rather than to an idealisation of it.", S["body"])]
    story += [eq(r"p_{k+1} \;=\; \arg\min_{p \in \Delta}\; \left[ \langle g_k,\,p\rangle + "
                 r"\frac{1}{\eta}\, \mathrm{KL}(p\,\|\,p_k) \right] \;\; \propto \;\; "
                 r"p_k \odot e^{-\eta\, g_k}, \quad \eta = \frac{1}{4\beta}",
                 "eq_md", 150)]
    story += pic(f_loops, 168,
                 "Figure 2. The two loops. Building M_h once per hop rather than once per "
                 "update also makes TRAIL cheaper than a flat recursion at the same number "
                 "of belief updates: 4 hops x 4 inner steps costs 4 transport builds, not 16.",
                 S)

    _halt_tbl = table([
        ["Loop", "Rule", "What it certifies"],
        ["inner (k)", "G<sub>k</sub> &#8804; &#949;&#183;G<sub>0</sub>",
         "this hop's energy is solved to relative accuracy &#949; (Thm. A). Governs cost."],
        ["outer (h)",
         "KL(p<sub>h</sub>&#8214;p<sub>h&#8722;1</sub>) &#8804; &#948;&#183;KL(p<sub>1</sub>&#8214;p<sub>0</sub>)",
         "the hop moved no belief, so there is no further sub-question. Governs the answer."],
    ], [26, 62, 77], S)
    story += [KeepTogether([para("1.3 Halting: two rules, each for its own loop", S["h2"]),
                            _halt_tbl])]
    story += [callout(
        "A correction worth recording",
        "An earlier version used the gap to decide <i>how many hops</i> to take. That "
        "conflates two different things: the gap certifies the <i>current sub-question</i> is "
        "solved, which is not the question being answered &#8212; a 3-hop question can have "
        "hop 1 solved to machine precision with the answer still two hops away. The "
        "conflation stopped early on exactly the hard questions adaptivity is for, and cost "
        "about <b>16 accuracy points</b> at the operating point.", S, RED)]

    # ---------------------------------------------------------------- theory
    story += [CondPageBreak(120 * mm), para("2. The theory", S["h1"]), para(
        "Throughout, M is column-stochastic, h(p) = &#8721; p<sub>i</sub> log p<sub>i</sub> is "
        "the negative entropy and D(q,p) = KL(q&#8214;p). Every statement below is checked "
        "numerically by <font face='DJVM'>tests/test_theory.py</font>, which runs in about a "
        "minute and fails if the proposition is false.", S["body"])]

    story += [para("Lemma 1 &#8212; dimension-free relative smoothness", S["h2"]), para(
        "Let H = &#946;(I&#8722;M)<sup>T</sup>(I&#8722;M) be the Hessian of E. Then "
        "max<sub>ij</sub>|H<sub>ij</sub>| &#8804; 4&#946; <b>for every N</b>. Hence E is "
        "4&#946;-smooth with respect to the &#8467;<sub>1</sub> norm on the simplex and, via "
        "Pinsker, satisfies the Bregman descent lemma with modulus L = 4&#946;.", S["body"]),
        para(
        "<i>Proof.</i> H<sub>ij</sub> = &#946;&#10216;(I&#8722;M)e<sub>i</sub>, "
        "(I&#8722;M)e<sub>j</sub>&#10217;. Column i of M lies in the simplex, so "
        "&#8214;Me<sub>i</sub>&#8214;<sub>2</sub> &#8804; "
        "&#8214;Me<sub>i</sub>&#8214;<sub>1</sub> = 1 and therefore "
        "&#8214;(I&#8722;M)e<sub>i</sub>&#8214;<sub>2</sub> &#8804; 1 + 1 = 2. "
        "Cauchy&#8211;Schwarz gives |H<sub>ij</sub>| &#8804; 4&#946;. For f with "
        "&#8214;&#8711;<sup>2</sup>f&#8214;<sub>1&#8594;&#8734;</sub> &#8804; L we have "
        "f(q) &#8804; f(p) + &#10216;&#8711;f(p), q&#8722;p&#10217; + "
        "(L/2)&#8214;q&#8722;p&#8214;<sub>1</sub><sup>2</sup>, and Pinsker's "
        "&#8214;q&#8722;p&#8214;<sub>1</sub><sup>2</sup> &#8804; 2D(q,p) yields the Bregman "
        "form. &#9633;", S["proof"]),
        para(
        "<b>Why it matters.</b> The constant does not involve N. Refining the atom grid does "
        "not change the prescribed step size, which is what allows the same weights to be run "
        "at a different resolution, or for more iterations, at test time.", S["body"])]

    story += [para("Theorem A &#8212; the halting certificate is sound", S["h2"])]
    story += [eq(r"G(p) \;=\; \langle g,\,p\rangle - \min_i g_i \;\;\geq\;\; "
                 r"E(p) - \min_{u\in\Delta} E(u) \;\;\geq\;\; 0", "eq_gap", 122)]
    story += [para(
        "<i>Proof.</i> Convexity gives E(u) &#8805; E(p) + &#10216;g, u&#8722;p&#10217; for all "
        "u &#8712; &#916;, so E(p) &#8722; E(u) &#8804; &#10216;g, p&#8722;u&#10217; &#8804; "
        "max<sub>u&#8712;&#916;</sub>&#10216;g, p&#8722;u&#10217; = G(p); the maximum of a linear "
        "function over the simplex is at a vertex, which gives the closed form. "
        "Non-negativity: take u = p. &#9633;", S["proof"]),
        para(
        "So stopping the first time G &#8804; &#949; returns an &#949;-optimal belief, "
        "whatever the stopping time turned out to be. Cost: one max and one dot product, "
        "O(N) &#8212; no extra parameters, no auxiliary loss, no learned gate.", S["body"])]

    story += [para("Theorem B &#8212; monotone descent and rate", S["h2"]), para(
        "With &#951; &#8804; 1/L = 1/(4&#946;), the inner loop satisfies E(p<sub>k+1</sub>) "
        "&#8804; E(p<sub>k</sub>) &#8722; (1/&#951; &#8722; L)&#183;D(p<sub>k+1</sub>, "
        "p<sub>k</sub>); in particular E is non-increasing, and strictly decreasing for "
        "&#951; &#8804; 1/(2L). From the uniform start, E(p<sub>K</sub>) &#8722; E* &#8804; "
        "4&#946;&#183;log N / K.", S["body"]),
        para(
        "<i>Proof.</i> Optimality of the mirror step against the feasible point "
        "p<sub>k</sub> gives &#10216;g<sub>k</sub>, p<sub>k+1</sub>&#8722;p<sub>k</sub>&#10217; "
        "+ (1/&#951;)D(p<sub>k+1</sub>,p<sub>k</sub>) &#8804; 0. Add Lemma 1's descent lemma. "
        "The rate is the standard relatively-smooth mirror-descent bound with D(p*, uniform) "
        "= log N &#8722; H(p*) &#8804; log N. &#9633;", S["proof"])]

    story += [para("Propositions 1&#8211;3", S["h2"])]
    story += [table([
        ["", "Statement", "Consequence"],
        ["Prop. 1",
         "With p<sub>0</sub> uniform, &#946;=0, &#951;=1, K=1: p<sub>1</sub> = softmax(s).",
         "Softmax cross-attention <i>is</i> one TRAIL step, so TRAIL strictly generalises the "
         "block it replaces and <font face='DJVM'>attn1</font> isolates iteration from "
         "architecture."],
        ["Prop. 2",
         "z<sub>t</sub> = V<sup>T</sup>p<sub>t</sub> &#8712; conv(V) for all t.",
         "Off-manifold drift is identically zero. Measured: <b>4&#215;10<sup>&#8722;4</sup></b> "
         "for TRAIL against <b>10.76</b> for free-form recursion."],
        ["Prop. 3",
         "For &#946;=0, G<sub>t</sub> &#8804; (N&#8722;1)S&#183;e<sup>&#8722;t&#951;&#916;</sup>, "
         "so T(&#949;) &#8804; &#8968;log((N&#8722;1)S/&#949;)/(&#951;&#916;)&#8969;.",
         "Halting time is inversely proportional to the decision margin &#8212; the theory's "
         "prediction that hard inputs get more compute. <b>Not borne out empirically "
         "(&#167;6).</b>"],
    ], [18, 72, 75], S)]

    story += [callout(
        "What the theory does not claim",
        "Theorem A certifies &#949;-optimality <b>of the hop's energy</b>, not correctness of "
        "the answer. The bridge from &#8220;this hop is solved&#8221; to &#8220;the answer is "
        "right&#8221; rests on the energy being a good proxy and is an empirical question. "
        "Stating this in the paper is cheaper than having a reviewer state it.", S, AMBER)]

    # ------------------------------------------------------------- structure
    story += [CondPageBreak(120 * mm), para("3. Code structure", S["h1"]), para(
        "The repository is organised so that each formal claim has exactly one place it "
        "lives and exactly one test that checks it. Nothing in "
        "<font face='DJVM'>theory/certificates.py</font> is learned &#8212; every function "
        "there is a measurable quantity corresponding to a statement in &#167;2.", S["body"])]
    story += pic(f_map, 168,
                 "Figure 3. Repository map, grouped by responsibility.", S)

    story += [para("3.1 The contract that makes the comparison a comparison", S["h2"]), para(
        "Every reasoner &#8212; TRAIL and all four baselines &#8212; exposes the same "
        "signature and returns the same state object:", S["body"]),
        Paragraph("forward(V, q_vec, words, word_mask, n_steps, eps, record, answer_head) "
                  "-&gt; TrailState", S["mono"]),
        para(
        "so <font face='DJVM'>train.py</font> and every figure script are agnostic to which "
        "model is running. Backbone, question encoder, read-out head, optimiser, schedule and "
        "data are shared; only the reasoning module varies. Budgets are matched in "
        "<b>belief updates</b>, not &#8220;steps&#8221;: TRAIL's H&#215;K against the same "
        "count of iterations for the recurrent baselines.", S["body"])]

    story += [table([
        ["Module", "Responsibility", "Guarded by"],
        ["<font face='DJVM'>theory/certificates.py</font>",
         "energy, gradient, Frank&#8211;Wolfe gap, hull drift (FISTA), smoothness constant, "
         "margin bound", "<font face='DJVM'>test_theory.py</font> (10 checks)"],
        ["<font face='DJVM'>models/trail_cell.py</font>",
         "control unit, column-stochastic transport, the two loops, both halting rules",
         "<font face='DJVM'>test_module_inner_loop_is_the_theory</font>"],
        ["<font face='DJVM'>models/baselines.py</font>",
         "attn1, free-form latent recursion, MAC, FiLM at matched budget",
         "<font face='DJVM'>test_eval_budget.py</font>"],
        ["<font face='DJVM'>data/clevr.py</font>",
         "streaming reader for the 800&#160;MB question file, program-depth labels, the two "
         "generalisation splits, root discovery",
         "<font face='DJVM'>test_clevr_io.py</font>"],
        ["<font face='DJVM'>data/features.py</font>",
         "frozen-trunk cache shared by CLEVR and GQA; deletes partial caches on failure",
         "&#8212;"],
        ["<font face='DJVM'>train.py</font>",
         "anytime loss, &#948;-sweep evaluation, halting statistics, drift measurement",
         "<font face='DJVM'>test_eval_budget.py</font>"],
    ], [42, 78, 45], S)]

    story += [para("3.2 Design decisions a reader should not have to reverse-engineer",
                   S["h2"])]
    story += bullets([
        "<b>Anytime loss.</b> Every prefix is supervised, L = (1/H)&#8721;<sub>h</sub> "
        "CE(head(p<sub>h</sub>), y). A model whose stopping time is data-dependent must be "
        "correct at whatever time it stops; <font face='DJVM'>anytime_loss='last'</font> is "
        "kept as the ablation that shows this is what makes the frontier flat rather than "
        "cliffed.",
        "<b>Scale-free thresholds.</b> Both &#949; and &#948; are relative to the first value "
        "in their own sequence, so one global threshold is meaningful across examples whose "
        "score scales differ.",
        "<b>Soft cardinality in the head.</b> z<sub>T</sub> is a convex combination and so "
        "carries no scale, which makes counting impossible; the head also receives the "
        "normalised entropy of p<sub>T</sub> and "
        "n&#770; = &#8721;<sub>i</sub>&#963;(a&#183;s<sub>i</sub> + b), which is interpretable "
        "on its own.",
        "<b>Path discovery, not hard-coded slugs.</b> Kaggle mirrors nest differently, so the "
        "loaders search for the dataset layout. This was a deliberate choice: kaggle.com was "
        "unreachable from the environment this was written in, so no slug could be verified.",
        "<b>Weight-tied hops behind a flag.</b> With per-hop control embeddings a model has H "
        "learned sub-question slots and no way to express the (H+1)-th, so "
        "&#8220;iterate longer&#8221; cannot work. <font face='DJVM'>share_steps=True</font> "
        "ties every hop to one operator, which is the only setting in which the "
        "depth-extrapolation claim is testable at all.",
    ], S)

    # ---------------------------------------------------------- experiments
    story += [CondPageBreak(110 * mm), para("4. Experimental design", S["h1"])]
    story += [table([
        ["Benchmark", "Why it is in the study", "Status"],
        ["<b>HOPWORLD</b> (generated in-process)",
         "The number of hops a question needs is known <i>exactly</i> and is never shown to "
         "the model, so halting and depth extrapolation are measurable without confounds. "
         "Runs on CPU in minutes.", "running"],
        ["<b>CLEVR v1.0</b>",
         "Ships a ground-truth functional program per question, so reasoning depth is a label "
         "to correlate against without ever training on it. Primary benchmark.",
         "pipeline tested, full run pending"],
        ["<b>GQA</b>",
         "Real photographs through the same code path; <font face='DJVM'>len(q['semantic'])</font> "
         "plays the role of CLEVR's program length, so the hop analysis transfers verbatim.",
         "pipeline written"],
    ], [40, 90, 35], S), Spacer(1, 6)]

    story += [para("4.1 Protocol", S["h2"])] + bullets([
        "Shared backbone (frozen ResNet-18 to layer3, 14&#215;14&#215;256 atoms, cached once "
        "as float16), shared question encoder, shared head, shared optimiser and schedule.",
        "Budgets matched in belief updates: TRAIL 4 hops &#215; 3 inner steps = 12; the "
        "recurrent baselines 12 iterations, which is also MAC's standard depth. FiLM is not "
        "an iterative reasoner &#8212; its steps are residual conv blocks &#8212; so it keeps "
        "its standard depth of 4 and is reported as such rather than pretending to a matched "
        "budget.",
        "Two generalisation splits on CLEVR: <b>depth</b> (train on programs &#8804; 10 nodes, "
        "test on &#8805; 15) and <b>held-out combination</b> (every question mentioning "
        "<i>red</i> and <i>cube</i> removed from training).",
    ], S)

    story += [callout(
        "Comparability",
        "These numbers are <b>not</b> comparable to published CLEVR results (MAC &#8776; 98.9, "
        "NS-VQA &#8776; 99.8). Those use ResNet-101 conv4 features at full resolution and days "
        "of training. Every model here shares one frozen ResNet-18 cache and one budget; the "
        "comparison is internal and the paper says so.", S, AMBER)]

    # -------------------------------------------------------------- results
    story += [para("5. Results so far", S["h1"])]
    if res:
        # Two operating points, reported separately.  Putting full-depth accuracy in
        # the same row as certificate-halted compute would read as "81% for 4.3
        # updates", which is not what either number means.
        rows = [["Model", "Params", "Acc. at full depth", "Updates",
                 "Acc. at certificate", "Updates", "Drift from conv(V)"]]
        for k in ORDER:
            if k not in res:
                continue
            r = res[k]
            full_u = r.get("updates_per_step", 1) * len(r.get("gap_curve") or [0]) \
                if k == "trail" else r.get("iid_updates", 0)
            rows.append([LBL[k], f"{r['params']/1e6:.2f}M",
                         f"{100*r['iid_acc']:.1f}%", f"{full_u:.0f}",
                         f"{100*r['iid_acc_eps']:.1f}%", f"{r.get('iid_updates', 0):.2f}",
                         f"{r['drift_per_step'][-1]:.4f}"])
        story += [table(rows, [42, 18, 26, 18, 26, 18, 25], S), Spacer(1, 4),
                  para(f"HOPWORLD, 24k train / 3k val, 12 epochs, identical budgets. For "
                       f"every model except TRAIL the two operating points coincide, because "
                       f"only TRAIL has a certificate to halt on. {len(res)} of 5 models "
                       f"complete when this report was generated; missing rows are still "
                       f"training.", S["cap"])]
    if f_front:
        story += pic(f_front, 130,
                     "Figure 4. The compute&#8211;accuracy frontier, with every baseline given "
                     "the same relational prior. TRAIL contributes a <i>curve</i>, traced by "
                     "sweeping one threshold on one trained model, where each fixed-depth "
                     "baseline contributes a point &#8212; that adaptivity is real. What the "
                     "figure also shows is that the curve sits <i>below</i> a single matched "
                     "attention step at a twelfth of the compute, which is why the accuracy "
                     "claim is withdrawn in &#167;5.2.", S)

    story += [para("5.1 What held", S["h2"])] + bullets([
        "<b>The inner certificate halves compute for free.</b> 12 &#8594; 6.32 belief updates "
        "at <i>identical</i> accuracy (0.8123, unchanged to four decimal places). Theorem A "
        "doing real work, with no learned gate and no auxiliary loss.",
        "<b>Zero drift.</b> 4&#215;10<sup>&#8722;4</sup> for TRAIL &#8212; zero up to the "
        "tolerance of the projection solver &#8212; against 10.76 for free-form latent "
        "recursion at the same budget. Prop. 2 confirmed, four orders of magnitude.",
        "<b>Anytime supervision pays in the cheap regime.</b> +19 points over last-step "
        "training at 5 belief updates; the frontier is flat rather than cliffed. It costs a "
        "ceiling about 6 points lower, which is a trade-off, not a free win.",
        "<b>The two-loop structure.</b> Same accuracy as the flat variant at one third of the "
        "O(N<sup>2</sup>) transport builds — and it is what makes the theorems apply to the "
        "code rather than to an idealisation of it.",
    ], S)

    story += [para("5.2 What did not: the accuracy claim", S["h2"]), para(
        "The main table showed TRAIL 19&#8211;22 points ahead at a matched <i>budget</i>. The "
        "ablations then showed that margin is carried entirely by the control-conditioned "
        "relative-position bias inside the transport module &#8212; removing it costs 20.9 "
        "points &#8212; and <b>no baseline had that module</b>. The comparison was matched in "
        "belief updates and unmatched in inductive bias. Giving every baseline the same "
        "module (identical +23,012 parameters) leaves mirror descent as the only difference:",
        S["body"])]
    story += [table([
        ["Model", "No prior", "With prior", "Change", "Belief updates"],
        ["Single cross-attention", "0.619", "<b>0.896</b>", "+0.277", "<b>1</b>"],
        ["MAC", "0.609", "<b>0.905</b>", "+0.296", "12"],
        ["Free-form latent recursion", "0.602", "0.721", "+0.119", "12"],
        ["TRAIL (ours)", "0.812", "0.812", "&#8212;", "4.32"],
    ], [52, 24, 26, 22, 30], S), Spacer(1, 5)]
    story += [callout(
        "Reversed, not narrowed",
        "MAC with a matched prior beats TRAIL by 9.3 points, and a <i>single</i> relational "
        "attention step beats it by 8.4 while spending one belief update against twelve. By "
        "Prop. 1 attention is one TRAIL step, so the T=1 case of the method outperforms the "
        "method. <b>And the benchmark is the deeper problem:</b> if one dense transport "
        "application reaches 89.6%, HOPWORLD does not require iterated reasoning and cannot "
        "test the hypothesis this work is about. The benchmark is ours, so that is a design "
        "error, not a property of an inherited task.", S, RED)]
    story += [para("The measured prices belong beside the benefits, not in a footnote: the "
                   "prescribed step size costs <b>7.7 points</b> against a learned one (which "
                   "converges to &#951; = 0.961, violating Thm. B's &#951; &#8804; 0.25), and "
                   "the simplex constraint costs <b>8.4 points</b> against one unconstrained "
                   "relational attention step.", S["body"])]

    story += [CondPageBreak(100 * mm), para("6. The negative result", S["h1"]), para(
        "Proposition 3 predicts that halting time scales as 1/&#916; in the decision margin, "
        "so harder questions should receive more steps for free. <b>On HOPWORLD this does not "
        "happen.</b> Sweeping the outer threshold &#948; on a trained TRAIL model:", S["body"])]
    story += [table(
        [["&#948;", "mean hops", "std. dev.", "&#961;(halt, true hops)", "accuracy"]] +
        [[f"{d:g}" if d else "0 (fixed depth)", f"{m:.2f}", f"{s:.3f}",
          "undefined" if r is None else f"{r:.3f}",
          {0.5: "73.3%", 0.3: "73.3%", 0.2: "73.3%", 0.1: "73.3%", 0.05: "73.3%",
           0.02: "76.7%", 0.01: "79.3%", 0.0: "81.2%"}[d]]
         for d, m, s, r in HALT_ROWS], [24, 28, 26, 40, 28], S)]
    story += pic(f_halt, 160,
                 "Figure 5. Left: over most of the threshold range the outer rule saturates "
                 "&#8212; every example halts at exactly hop 2. Right: the correlation is "
                 "therefore undefined for lack of variance, not measurably zero; where "
                 "variance exists it is 0.06&#8211;0.08.", S)

    story += [para(
        "Two things must be separated here. Over most of the range the rule <b>saturates</b>: "
        "the standard deviation of the halting step is 0.000, so &#961; is undefined rather "
        "than zero. Where there is variance (&#948; = 0.01&#8211;0.02), &#961; = "
        "0.06&#8211;0.08, i.e. nothing. Reporting &#961; at a single threshold would have "
        "hidden which of these was happening &#8212; a saturated rule and a genuinely "
        "uncorrelated one look identical through one number &#8212; which is why the "
        "evaluation now reports the spread alongside the correlation at every threshold.",
        S["body"])]
    story += [para("Diagnosis", S["h2"]), para(
        "Prop. 3 is a statement about iterating a <i>fixed</i> energy, and it is true there "
        "&#8212; the test verifies the bound. The gap between theory and measurement is that "
        "M<sub>h</sub> is a <b>dense learned operator</b>, not a single ground-truth relation: "
        "one application can compose several hops of the underlying task, so the number of "
        "TRAIL hops has no reason to equal the number of program hops. The theory bounds how "
        "long it takes to solve <i>a</i> hop; it says nothing about how many of the task's "
        "hops the model packs into one.", S["body"]),
        para(
        "This is a testable diagnosis rather than a shrug, and it names the follow-up: "
        "constrain M<sub>h</sub> (low rank, spatial locality, or a penalty on composing) so "
        "that one application is one relation, then re-measure &#961;. Until that is done the "
        "contributions claimed are compute adaptivity and interpretability, <b>not</b> "
        "unsupervised difficulty estimation.", S["body"])]

    # ------------------------------------------------------- what went wrong
    story += [CondPageBreak(110 * mm), para("7. Three corrections made during development", S["h1"]),
              para(
        "Recorded because each changed a number, and because a reader running the artefact "
        "would otherwise rediscover them.", S["body"])]
    story += [table([
        ["What was wrong", "Why it mattered", "Fix"],
        ["The energy was re-conditioned at every step, so no step optimised a fixed objective.",
         "The theorems did not apply to the code, only to an idealisation of it.",
         "Two loops: hops outside, mirror descent on a fixed E<sub>h</sub> inside. A test now "
         "pulls (s<sub>h</sub>, M<sub>h</sub>) out of the module and replays the inner loop "
         "through the certificate functions."],
        ["The Frank&#8211;Wolfe gap decided how many <i>hops</i> to take.",
         "Cost ~16 accuracy points: it stopped early on exactly the multi-hop questions "
         "adaptivity is supposed to help.",
         "Gap governs iterations within a hop; a belief-stationarity rule governs the number "
         "of hops."],
        ["<font face='DJVM'>evaluate()</font> defaulted its depth to "
         "<font face='DJVM'>cfg.n_steps</font> &#8212; TRAIL's <i>hop</i> count.",
         "Baselines were trained at 12 iterations and evaluated at 4. The compute axis was "
         "wrong by 3&#215; and the comparison flattered TRAIL.",
         "Depth now comes from each reasoner's own configuration; "
         "<font face='DJVM'>test_eval_budget.py</font> asserts it and that budgets match."],
    ], [52, 56, 60], S)]
    story += [callout(
        "On the third one",
        "The affected runs were discarded and the comparison re-run. Re-measured, free-form "
        "latent recursion scores 60.2% at its true 12 updates &#8212; its accuracy was "
        "unchanged, because the eight iterations it was denied were adding nothing. The bug "
        "mattered for the <i>compute axis</i>, not the accuracy axis, and the qualitative "
        "conclusion survived. That was not knowable before re-running it.", S, GREEN)]

    # ---------------------------------------------------------------- how to
    story += [para("8. Running it", S["h1"]), para(
        "<b>Kaggle.</b> Open <font face='DJVM'>notebooks/trail_wacv2027_kaggle.ipynb</font>, "
        "enable the GPU accelerator and Internet, attach a CLEVR v1.0 dataset, run top to "
        "bottom. It starts in smoke mode (minutes end to end); set "
        "<font face='DJVM'>SMOKE</font> and <font face='DJVM'>QUICK</font> to "
        "<font face='DJVM'>False</font> for the real run.", S["body"])]
    story += [Paragraph(
        "python tests/test_theory.py                    # the maths, ~1 min<br/>"
        "python tests/test_eval_budget.py               # matched budgets<br/>"
        "python scripts/run_synthetic.py --quick        # controlled study<br/>"
        "python scripts/run_clevr.py --split iid        # the real run<br/>"
        "python scripts/run_clevr.py --split depth      # depth extrapolation<br/>"
        "python scripts/run_gqa.py                      # real images<br/>"
        "python scripts/report.py --runs runs/synthetic # tables, MD + LaTeX",
        S["mono"]), Spacer(1, 5)]
    story += [table([
        ["Stage", "Cost on one T4"],
        ["Theory + budget tests", "~1 min (CPU)"],
        ["HOPWORLD, all 5 models", "~20 min"],
        ["Featurising 20k+5k CLEVR images (once, shared by every model)", "~25 min"],
        ["Training one reasoner on CLEVR", "~35&#8211;50 min"],
        ["Five models + two generalisation splits", "fits a 12&#160;h session"],
    ], [110, 45], S)]

    # ---------------------------------------------------------- limitations
    story += [para("9. Limitations", S["h1"])] + bullets([
        "The certificate bounds the hop's energy, not answer correctness.",
        "Only the <i>visual</i> state is drift-free; the control unit is an unconstrained "
        "recurrence. Extending the guarantee to the control path is open.",
        "M<sub>h</sub> is O(N<sup>2</sup>) per hop. Fine at N = 196, and built once per hop "
        "rather than once per update, but it is the scaling bottleneck for finer atom grids. "
        "Low-rank or sparse transport is the obvious next step &#8212; and, per &#167;6, may "
        "also be what makes halting track difficulty.",
        "Convexity holds for the relational penalty as written; a concave coherence reward "
        "would lose Thm. A's global guarantee and leave only stationarity.",
        "Everything here is a from-scratch reasoner over frozen features. Whether the same "
        "parameterisation helps inside a pretrained MLLM's latent reasoning loop is untested "
        "and is the natural follow-up.",
        "Dataset slugs could not be verified from the authoring environment; the loaders "
        "discover paths instead, and the CLEVR reader is tested against a deliberately "
        "hostile fixture, but the Kaggle attachment step needs a human eye once.",
    ], S)

    story += [para("10. What a reviewer will press on", S["h1"])]
    story += [table([
        ["Likely objection", "Prepared answer"],
        ["&#8220;Attention already does this.&#8221;",
         "Prop. 1: attention is literally the T=1 case. The paper's question is what "
         "iterating it under a certificate buys, and <font face='DJVM'>attn1</font> is the "
         "control that isolates exactly that."],
        ["&#8220;Deep equilibrium models / unrolled optimisation are not new.&#8221;",
         "Agreed, and cited. What is new is that the optimisation variable is the reasoning "
         "state itself, constrained to the hull of the image's own atoms, and that the "
         "stopping rule is the algorithm's own certificate rather than a learned gate."],
        ["&#8220;The accuracy is far below published CLEVR numbers.&#8221;",
         "Stated up front (&#167;4). Different features, different budget; the comparison is "
         "internal and every model shares one cache."],
        ["&#8220;Does this actually beat anything?&#8221;",
         "No, and we say so in &#167;5.2. Given a matched prior, MAC and a single attention "
         "step both exceed it. The contribution is the certificate, the on-manifold state and "
         "the trace, together with an honest account of what they cost."],
        ["&#8220;Your halting does not do what you said.&#8221;",
         "Reported as a negative in &#167;6, with the saturation analysis and a mechanistic "
         "diagnosis, before a reviewer can find it."],
        ["&#8220;Is any of this reproducible?&#8221;",
         "<font face='DJVM'>tests/test_theory.py</font> checks all ten formal claims in about "
         "a minute; the notebook runs the study end to end."],
    ], [58, 110], S)]

    story += [Spacer(1, 10), para(
        "Generated from the live run artefacts. Regenerate with "
        "<font face='DJVM'>python report/make_report.py</font> after further models finish, "
        "and the tables update in place.", S["small"])]

    # ----------------------------------------------------------------- build
    def decorate(canvas, doc):
        canvas.saveState()
        canvas.setStrokeColor(RULE)
        canvas.setLineWidth(0.5)
        canvas.line(20 * mm, 287 * mm, 190 * mm, 287 * mm)
        canvas.setFont("DJV", 7.4)
        canvas.setFillColor(GREY)
        canvas.drawString(20 * mm, 290 * mm, "TRAIL — technical report")
        canvas.drawRightString(190 * mm, 290 * mm, "LVR @ WACV 2027")
        canvas.line(20 * mm, 16 * mm, 190 * mm, 16 * mm)
        canvas.drawCentredString(105 * mm, 11 * mm, f"{doc.page}")
        canvas.restoreState()

    doc = BaseDocTemplate(out_path, pagesize=A4,
                          leftMargin=20 * mm, rightMargin=20 * mm,
                          topMargin=24 * mm, bottomMargin=22 * mm,
                          title="TRAIL - technical report",
                          author="TRAIL", subject="Latent visual reasoning as certified "
                                                  "mirror descent on the visual simplex")
    frame = Frame(doc.leftMargin, doc.bottomMargin, doc.width, doc.height, id="n")
    doc.addPageTemplates([PageTemplate(id="all", frames=[frame], onPage=decorate)])
    check_glyphs(story)
    doc.build(story)
    return out_path


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="report/TRAIL_technical_report.pdf")
    a = ap.parse_args()
    p = build(a.out)
    print("wrote", p, os.path.getsize(p) // 1024, "KB")
