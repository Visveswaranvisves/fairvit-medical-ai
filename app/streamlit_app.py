from html import escape
import hashlib
import streamlit as st
import torch
from torch import nn
from torchvision.models import vit_b_16
from torchvision import transforms
from huggingface_hub import hf_hub_download
from PIL import Image, ImageOps, ImageDraw


def make_tab_icon():
    """Browser-tab icon: dark tile, lime lungs and an ice-blue medical cross (UI theme colours)."""
    try:
        s = 256
        img = Image.new('RGBA', (s, s), (0, 0, 0, 0))
        d = ImageDraw.Draw(img)
        d.rounded_rectangle((0, 0, s - 1, s - 1), radius=60, fill='#111314')
        d.ellipse((32, 56, 120, 208), fill='#d3ec4c')
        d.ellipse((136, 56, 224, 208), fill='#d3ec4c')
        d.rounded_rectangle((113, 76, 143, 180), radius=9, fill='#8fc7e3')
        d.rounded_rectangle((76, 113, 180, 143), radius=9, fill='#8fc7e3')
        return img
    except Exception:
        return '🫁'


st.set_page_config(page_title='FairViT | Chest X-ray research', page_icon=make_tab_icon(), layout='wide')

# ---------- Healthcare illustrations (inline SVG) ----------
LIME = '#e4f45a'
OLIVE = '#86a80f'
SKY = '#8fc7e3'

CHEST_G = (
    '<path d="M110 96 Q165 76 220 96" stroke="#9fd6ea" stroke-width="3" opacity=".7"/>'
    '<path d="M165 72 V262" stroke="#9fd6ea" stroke-width="3" stroke-dasharray="3 6" opacity=".7"/>'
    '<path d="M158 92 C128 88 104 128 102 176 C100 214 118 236 146 230 C160 227 160 206 160 188 Z" fill="#1c5166" stroke="#7cbcd3" stroke-width="2"/>'
    '<path d="M172 92 C202 88 226 128 228 176 C230 214 212 236 184 230 C170 227 170 206 170 188 Z" fill="#1c5166" stroke="#7cbcd3" stroke-width="2"/>'
    '<g stroke="#9fd6ea" stroke-width="3" opacity=".75" fill="none">'
    '<path d="M108 112 Q135 124 158 116"/><path d="M104 134 Q134 148 158 138"/><path d="M103 156 Q134 170 158 160"/><path d="M104 178 Q134 192 158 182"/><path d="M110 200 Q136 212 158 204"/>'
    '<path d="M222 112 Q195 124 172 116"/><path d="M226 134 Q196 148 172 138"/><path d="M227 156 Q196 170 172 160"/><path d="M226 178 Q196 192 172 182"/><path d="M220 200 Q194 212 172 204"/></g>'
    '<path d="M148 196 C148 170 182 170 190 190 C194 212 170 232 152 218 Z" fill="#2c6f86" opacity=".85"/>'
    f'<circle cx="122" cy="150" r="13" stroke="{LIME}" stroke-width="3.5" fill="none"/>'
    '<text x="92" y="80" font-family="Arial" font-size="13" font-weight="700" fill="#cfeaf5">R</text>'
)

FILM_SVG_SMALL = (
    '<svg viewBox="70 50 190 230" width="100%" height="100%" xmlns="http://www.w3.org/2000/svg" fill="none" stroke-linecap="round" stroke-linejoin="round">'
    '<rect x="76" y="56" width="178" height="218" rx="12" fill="#10303d"/>' + CHEST_G + '</svg>'
)

HERO_ART_SVG = (
    '<svg viewBox="0 0 380 300" width="100%" height="100%" xmlns="http://www.w3.org/2000/svg" fill="none" stroke-linecap="round" stroke-linejoin="round">'
    # hospital silhouette
    '<rect x="196" y="70" width="172" height="230" rx="16" fill="#fff" opacity=".18"/>'
    '<g fill="#fff" opacity=".28"><rect x="214" y="118" width="22" height="18" rx="4"/><rect x="250" y="118" width="22" height="18" rx="4"/><rect x="286" y="118" width="22" height="18" rx="4"/><rect x="322" y="118" width="22" height="18" rx="4"/>'
    '<rect x="214" y="152" width="22" height="18" rx="4"/><rect x="250" y="152" width="22" height="18" rx="4"/><rect x="286" y="152" width="22" height="18" rx="4"/><rect x="322" y="152" width="22" height="18" rx="4"/>'
    '<rect x="214" y="186" width="22" height="18" rx="4"/><rect x="250" y="186" width="22" height="18" rx="4"/><rect x="286" y="186" width="22" height="18" rx="4"/><rect x="322" y="186" width="22" height="18" rx="4"/></g>'
    f'<circle cx="326" cy="46" r="26" fill="{LIME}"/>'
    '<path d="M326 34 V58 M314 46 H338" stroke="#111" stroke-width="6"/>'
    # light box + film
    '<rect x="52" y="38" width="214" height="258" rx="22" fill="#fff" opacity=".93"/>'
    '<g transform="translate(-6 -2)"><rect x="76" y="56" width="178" height="218" rx="12" fill="#10303d"/>' + CHEST_G + '</g>'
    # stethoscope
    '<path d="M286 262 C286 214 330 214 330 170" stroke="#fff" stroke-width="6"/>'
    '<path d="M330 170 C330 146 350 146 350 126 M330 170 C330 146 310 146 310 126" stroke="#fff" stroke-width="6"/>'
    f'<circle cx="286" cy="268" r="17" fill="{LIME}" stroke="#fff" stroke-width="4"/>'
    '<circle cx="286" cy="268" r="6" fill="#111"/>'
    '</svg>'
)

BRAND_SVG = (
    '<svg viewBox="0 0 40 40" width="38" height="38" xmlns="http://www.w3.org/2000/svg" fill="none">'
    f'<g fill="{OLIVE}"><circle cx="20" cy="6" r="3"/><circle cx="20" cy="14" r="3"/><circle cx="12" cy="20" r="3"/><circle cx="20" cy="20" r="3"/><circle cx="28" cy="20" r="3"/><circle cx="20" cy="26" r="3"/><circle cx="20" cy="34" r="3"/></g>'
    '</svg>'
)


def ic(inner, color='#111', size=22, sw=1.9):
    return (f'<svg viewBox="0 0 24 24" width="{size}" height="{size}" xmlns="http://www.w3.org/2000/svg" fill="none" stroke="{color}" '
            f'stroke-width="{sw}" stroke-linecap="round" stroke-linejoin="round">{inner}</svg>')

IC_SEARCH = ic('<circle cx="11" cy="11" r="7"/><path d="M20 20 L16 16"/>', '#8b9599', 18)
IC_SHIELD = ic('<path d="M12 3 L20 6 V12 c0 5 -3.5 8 -8 9 c-4.5 -1 -8 -4 -8 -9 V6 Z"/><path d="M9 12 l2 2 l4 -4"/>', LIME, 20, 2.1)
IC_FILM = ic('<rect x="4" y="3" width="16" height="18" rx="3"/><path d="M12 8 V16 M8 12 H16"/>', '#111', 22)
IC_CHIP = ic('<rect x="7" y="7" width="10" height="10" rx="2"/><path d="M9 3 v4 M15 3 v4 M9 17 v4 M15 17 v4 M3 9 h4 M3 15 h4 M17 9 h4 M17 15 h4"/>', '#111', 22)
IC_GRID = ic('<rect x="4" y="4" width="6" height="6" rx="1.5"/><rect x="14" y="4" width="6" height="6" rx="1.5"/><rect x="4" y="14" width="6" height="6" rx="1.5"/><rect x="14" y="14" width="6" height="6" rx="1.5"/>', '#111', 22)
IC_PULSE = ic('<path d="M3 12 H8 L10 6 L14 18 L16 12 H21"/>', '#111', 22)
IC_CHECK = ic('<path d="M5 12 l4 4 l10 -10"/>', '#111', 14, 2.6)
IC_HOSP = ic('<path d="M5 21 V8 L12 4 L19 8 V21 Z"/><path d="M12 10 V16 M9 13 H15"/><path d="M10 21 V18 H14 V21"/>', '#1d4f3f', 17)
IC_LUNG = ic('<path d="M12 4 V11 M12 11 C9 9 6 11 5.5 15 C5 19 6.5 20 8.5 19.5 C11 19 12 17 12 14 M12 11 C15 9 18 11 18.5 15 C19 19 17.5 20 15.5 19.5 C13 19 12 17 12 14"/>', '#6b2740', 17)
IC_STETH = ic('<path d="M7 4 V10 a4 4 0 0 0 8 0 V4"/><path d="M11 14 V16 a4 4 0 0 0 8 0 V13"/><circle cx="19" cy="11" r="2"/>', '#264a66', 17)
IC_DATA = ic('<ellipse cx="12" cy="6" rx="7" ry="3"/><path d="M5 6 V18 c0 1.7 3.1 3 7 3 s7 -1.3 7 -3 V6"/><path d="M5 12 c0 1.7 3.1 3 7 3 s7 -1.3 7 -3"/>', '#111', 22)
IC_BALANCE = ic('<path d="M12 4 V20 M7 20 H17 M5 8 H19"/><path d="M5 8 L3 14 a3 3 0 0 0 6 0 Z"/><path d="M19 8 L17 14 a3 3 0 0 0 6 0 Z"/>', '#111', 22)
IC_HUMAN = ic('<circle cx="12" cy="8" r="3.5"/><path d="M5 20 c0 -4 3 -6 7 -6 s7 2 7 6"/>', '#111', 22)
IC_PLAY = ic('<path d="M8 5 L19 12 L8 19 Z" fill="#fff" stroke="none"/>', '#fff', 22)

LUNG_PATH = ('M14 3 C6 3 2 14 2 22 C2 28 6 30 10 28 C14 26 14 20 14 16 Z '
             'M18 3 C26 3 30 14 30 22 C30 28 26 30 22 28 C18 26 18 20 18 16 Z')


def lung_grid(score):
    n = round(score * 24) if score is not None else 0
    cells = []
    for i in range(24):
        x, y = (i % 6) * 42 + 5, (i // 6) * 36 + 2
        fill = '#4f86a0' if i < n else '#ffffffd9'
        cells.append(f'<use href="#lg" x="{x}" y="{y}" fill="{fill}"/>')
    return ('<svg viewBox="0 0 252 144" width="100%" xmlns="http://www.w3.org/2000/svg">'
            f'<defs><path id="lg" d="{LUNG_PATH}"/></defs>' + ''.join(cells) + '</svg>')


def avatars():
    def av(bg, svg, left):
        return f'<span class="av" style="background:{bg};left:{left}px">{svg}</span>'
    return '<span class="avs">' + av('#bfeadd', IC_HOSP, 0) + av('#f6cfd9', IC_LUNG, 26) + av('#cfe0f3', IC_STETH, 52) + '</span>'


def hero(title_html, join_label, join_num, join_unit, cta):
    return (
        '<div class="hero"><div class="hero-title">' + title_html + '</div>'
        f'<div class="hero-play"><span class="play">{IC_PLAY}</span><span>Explore now</span></div>'
        f'<div class="hero-join"><small>{escape(join_label)}</small><div class="join-row">{avatars()}'
        f'<div class="join-num"><b>{escape(join_num)}</b><small>{escape(join_unit)}</small></div></div></div>'
        f'<span class="hero-cta">{escape(cta)}<i><svg viewBox="0 0 24 24" width="18" height="18" fill="none" stroke="#111" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round"><path d="M5 12 H19 M13 6 L19 12 L13 18"/></svg></i></span>'
        f'<div class="hero-art">{HERO_ART_SVG}</div></div>'
    )


def res_card(label, note, pill, score, circle, circle_cap, number, unit, scale=None):
    scale_html = f'<div class="res-scale">{scale}</div>' if scale else ''
    return (
        '<div class="res"><div class="res-top"><div class="res-copy">'
        f'<b>{escape(label)}</b><span>{escape(note)}</span>'
        f'<span class="well">{escape(pill)}<i>{IC_CHECK}</i></span></div>'
        f'<div class="res-grid">{lung_grid(score)}</div></div>{scale_html}'
        '<div class="res-bottom">'
        f'<div class="thr"><span class="dot">{escape(circle)}</span><small>{escape(circle_cap)}</small></div>'
        f'<div class="res-num">{number}<small>{escape(unit)}</small></div></div></div>'
    )


def card_head(icon_svg, title, value='', unit=''):
    val = f'<span class="head-val">{value}<small>{unit}</small></span>' if value else ''
    return f'<div class="card-head"><span class="chip-ic">{icon_svg}</span><b>{escape(title)}</b>{val}</div>'


def prep_card():
    bars = ''.join('<span></span>' for _ in range(44))
    return (
        '<div class="wcard">' + card_head(IC_GRID, 'Input preparation', '224 × 224', ' px')
        + '<div class="sub"><span>Resize, then tensor</span><span>Matches training</span></div>'
        '<div class="big">224<small>/px per side</small></div>'
        f'<div class="bars">{bars}</div><div class="bar-scale"><span>0</span><span>224</span></div>'
        '<div class="stats"><div><b>224</b><small>Pixels per side</small></div><div><b>RGB</b><small>3 channels</small></div><div><b>1</b><small>Output score</small></div></div>'
        '<div class="note">Resized to 224 × 224, then converted to a tensor. Matches the current training dataloader.</div></div>'
    )


WAVE_SVG = (
    '<svg viewBox="0 0 300 120" width="100%" xmlns="http://www.w3.org/2000/svg" fill="none" stroke-linecap="round" stroke-linejoin="round">'
    f'<path d="M0 78 C40 18 84 18 124 62 S206 108 246 58 S292 40 300 46" stroke="{SKY}" stroke-width="16" opacity=".4"/>'
    '<path d="M0 62 C40 6 84 6 124 50 S206 96 246 46 S292 28 300 34" stroke="#b8dff0" stroke-width="10" opacity=".7"/>'
    '<path d="M0 60 H300" stroke="#0d0d0d" stroke-width="1.4" stroke-dasharray="4 6" opacity=".35"/>'
    '<circle cx="70" cy="22" r="9" fill="#5ab0d8" stroke="#fff" stroke-width="3.5"/>'
    '</svg>'
)


def thr_card():
    return (
        '<div class="wcard">' + card_head(IC_PULSE, 'Decision threshold', '0.50', ' cutoff')
        + '<div class="sub"><span>Pneumonia-class score</span><span>ViT-B/16 baseline</span></div>'
        f'<div class="wave">{WAVE_SVG}</div>'
        '<div class="thr-bottom"><div class="big">0.50<small>/1.00</small></div>'
        '<div class="thr-note"><small>Not a clinical</small><small>probability</small><b>Research only</b></div></div></div>'
    )


def ctx_card(svg, title, body):
    return (f'<div class="ctx-card"><div class="ctx-icon">{svg}</div>'
            f'<div class="ctx-title">{escape(title)}</div><div class="ctx-body">{escape(body)}</div></div>')


def nav_uri(inner, color):
    svg = (f"<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24' fill='none' stroke='{color}' "
           f"stroke-width='1.9' stroke-linecap='round' stroke-linejoin='round'>{inner}</svg>")
    return 'url("data:image/svg+xml;utf8,' + svg.replace('#', '%23') + '")'

NAV_ICONS = [
    "<rect x='4' y='3' width='16' height='18' rx='3'/><path d='M12 8V16M8 12H16'/>",
    "<rect x='7' y='7' width='10' height='10' rx='2'/><path d='M9 3v4M15 3v4M9 17v4M15 17v4M3 9h4M3 15h4M17 9h4M17 15h4'/>",
]
NAV_CSS = ''.join(
    f'[data-testid="stSidebar"] [role="radiogroup"]>label:nth-of-type({i})::before{{background-image:{nav_uri(icn,"#6b7478")}}}'
    f'[data-testid="stSidebar"] [role="radiogroup"]>label:nth-of-type({i}):has(input:checked)::before{{background-image:{nav_uri(icn,"#ffffff")}}}'
    for i, icn in enumerate(NAV_ICONS, start=1)
)

BASE_CSS = '''
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600&display=swap');
:root{--ink:#0d0d0d;--muted:#5f6c72;--olive:#86a80f;--lime:#e4f45a;--sky:#8fc7e3}
html,body,.stApp,[class*="st-"]{font-family:'Inter','Source Sans Pro',system-ui,sans-serif}
.stApp{background:linear-gradient(160deg,#e1f0f6 0%,#eaf6fa 52%,#dfeef5 100%);background-attachment:fixed;color:var(--ink)}
[data-testid="stHeader"]{background:transparent}
.block-container,[data-testid="stMainBlockContainer"]{max-width:1500px;width:100%;padding:3.2rem 2.2rem 2.4rem}
h1,h2,h3{letter-spacing:-.04em;color:var(--ink)}
[data-testid="stCaptionContainer"]{color:#66737a}
[data-testid="stCaptionContainer"] p{font-size:13px;line-height:1.5}

/* ===== Icon rail (sidebar) ===== */
[data-testid="stSidebar"]{width:96px!important;min-width:96px!important;background:#fff;border-right:0;border-radius:0 34px 34px 0;box-shadow:8px 0 40px #1f4a5a14}
[data-testid="stSidebarHeader"],[data-testid="stSidebarCollapseButton"],[data-testid="stSidebarCollapsedControl"]{display:none!important}
[data-testid="stSidebarUserContent"]{padding:1.8rem .6rem 1rem}
[data-testid="stSidebar"] [data-testid="stRadio"]{display:flex;justify-content:center}
[data-testid="stSidebar"] [role="radiogroup"]{gap:14px;align-items:center}
[data-testid="stSidebar"] [role="radiogroup"]>label{display:flex;align-items:center;justify-content:center;width:52px;height:52px;border-radius:50%;padding:0;margin:0;cursor:pointer;transition:background .15s}
[data-testid="stSidebar"] [role="radiogroup"]>label>div:first-of-type{display:none}
[data-testid="stSidebar"] [role="radiogroup"]>label>div:last-of-type{display:none}
[data-testid="stSidebar"] [role="radiogroup"]>label::before{content:"";width:23px;height:23px;background-size:contain;background-repeat:no-repeat;background-position:center}
[data-testid="stSidebar"] [role="radiogroup"]>label:hover{background:#eef5f8}
[data-testid="stSidebar"] [role="radiogroup"]>label:has(input:checked){background:var(--olive);box-shadow:0 10px 22px #86a80f55}
.rail-brand{display:flex;justify-content:center;margin:0 0 34px}
.rail-div{height:1px;background:#e5ecef;margin:22px 10px}
.rail-bottom{display:flex;flex-direction:column;align-items:center;gap:16px;margin-top:34vh}
.rail-avatar{width:44px;height:44px;border-radius:50%;background:linear-gradient(135deg,#bfeadd,#8fc7e3);display:flex;align-items:center;justify-content:center;font-weight:600;color:#0d3b4a;border:2px solid #fff;box-shadow:0 4px 12px #1f4a5a22}

/* ===== Greeting bar ===== */
.greet{display:flex;align-items:center;justify-content:space-between;gap:20px;flex-wrap:wrap;margin:0 0 22px}
.greet h1{font-size:34px!important;font-weight:500!important;margin:0!important;padding:0!important;line-height:1.1!important}
.greet p{margin:6px 0 0;font-size:13px;color:var(--muted)}
.greet-actions{display:flex;align-items:center;gap:14px;margin-right:130px}
.searchpill{display:flex;align-items:center;gap:10px;background:#fff;border-radius:40px;padding:15px 24px;min-width:260px;font-size:14px;color:#8b9599;pointer-events:none}
.blackpill{display:inline-flex;align-items:center;gap:10px;background:#0d0d0d;color:var(--lime);border-radius:40px;padding:15px 24px;font-size:14px;font-weight:500;white-space:nowrap}

/* ===== Hero ===== */
.hero{position:relative;overflow:hidden;min-height:300px;border-radius:30px;background:linear-gradient(120deg,#adc9cc 0%,#8db3bb 55%,#6e99a5 100%);padding:30px 34px;box-sizing:border-box;width:100%;color:#fff}
.hero-title{font-size:44px;line-height:1.06;font-weight:300;letter-spacing:-.035em;color:#fff;max-width:52%}
.hero-play{position:absolute;left:34px;top:50%;display:flex;align-items:center;gap:12px;font-size:15px;color:#f2fbfc;margin-top:6px}
.hero-play .play{width:38px;height:38px;border-radius:50%;background:#ffffff40;display:inline-flex;align-items:center;justify-content:center}
.hero-join{position:absolute;left:34px;bottom:26px}
.hero-join>small{display:block;font-size:12px;color:#eaf6f8;margin-bottom:8px}
.join-row{display:flex;align-items:center;gap:12px}
.avs{position:relative;display:inline-block;width:86px;height:34px}
.av{position:absolute;top:0;width:34px;height:34px;border-radius:50%;border:2px solid #fff;display:inline-flex;align-items:center;justify-content:center;box-sizing:border-box}
.join-num b{font-size:22px;font-weight:500;letter-spacing:-.03em;display:block;line-height:1}
.join-num small{font-size:12px;color:#eaf6f8}
.hero-cta{position:absolute;right:26px;bottom:24px;display:inline-flex;align-items:center;gap:22px;background:#0d0d0d;color:#fff;border-radius:40px;padding:8px 8px 8px 28px;font-size:15px;font-weight:500}
.hero-cta i{width:42px;height:42px;border-radius:50%;background:#fff;display:inline-flex;align-items:center;justify-content:center}
.hero-art{position:absolute;right:12px;top:14px;width:300px;height:237px;pointer-events:none}

/* ===== Blue result card ===== */
.res{background:#8fc7e3;background-image:radial-gradient(circle at 30% 20%,#a3d3ea 0%,transparent 60%);border-radius:30px;padding:26px 28px;min-height:300px;box-sizing:border-box;color:#fff;display:flex;flex-direction:column;width:100%}
.res-top{display:flex;gap:18px;align-items:flex-start}
.res-copy{flex:1;min-width:0;display:flex;flex-direction:column;gap:6px}
.res-copy b{font-size:13px;font-weight:600;color:#10303d}
.res-copy span{font-size:12px;line-height:1.5;color:#e9f6fb}
.res-copy .well{display:inline-flex;align-items:center;gap:8px;align-self:flex-start;background:var(--lime);color:#111;border-radius:30px;padding:9px 15px;font-size:12px;font-weight:600;margin-top:10px;line-height:1.3}
.res-copy .well i{display:inline-flex;flex:none;width:22px;height:22px;border-radius:50%;background:#fff6;align-items:center;justify-content:center}
.res-grid{flex:none;width:52%;max-width:240px}
.res-scale{font-size:10px;color:#e9f6fb;margin-top:10px;opacity:.9}
.res-bottom{display:flex;align-items:flex-end;justify-content:space-between;gap:12px;margin-top:auto;padding-top:14px}
.thr{display:flex;align-items:center;gap:10px}
.thr .dot{width:50px;height:50px;border-radius:50%;background:#fff;color:#0d0d0d;display:inline-flex;align-items:center;justify-content:center;font-size:13px;font-weight:600}
.thr small{font-size:11px;color:#e9f6fb}
.res-num{font-size:62px;font-weight:300;letter-spacing:-.05em;line-height:1;color:#fff}
.res-num small{font-size:18px;font-weight:400;letter-spacing:0;margin-left:6px;color:#e9f6fb}

/* ===== White cards ===== */
.wcard,[class*="st-key-panel_"]{background:#fff;border-radius:30px;box-sizing:border-box}
.wcard{padding:26px 28px;min-height:500px;display:flex;flex-direction:column;width:100%}
[class*="st-key-panel_"]{padding:26px 28px!important;border:0!important;gap:.8rem;min-width:0;overflow-x:clip}
[class*="st-key-panel_"] [data-testid="stElementContainer"],
[class*="st-key-panel_"] [data-testid="stElementContainer"]>div,
[class*="st-key-panel_"] [data-testid="stFileUploader"],
[class*="st-key-panel_"] [data-testid="stImage"],
[class*="st-key-panel_"] [data-testid="stMarkdownContainer"]{width:100%!important;max-width:100%!important;box-sizing:border-box;min-width:0}
[class*="st-key-panel_"] [data-testid="stImage"] img{max-width:100%;height:auto;border-radius:18px}
.card-head{display:flex;align-items:center;gap:14px}
.card-head b{font-size:20px;font-weight:500;letter-spacing:-.03em;color:var(--ink)}
.chip-ic{width:46px;height:46px;border-radius:15px;background:#f1f4f5;display:inline-flex;align-items:center;justify-content:center;flex:none}
.head-val{margin-left:auto;font-size:20px;font-weight:400;letter-spacing:-.03em}
.head-val small{font-size:12px;color:#7d888d;margin-left:3px}
.sub{display:flex;justify-content:space-between;gap:10px;font-size:11px;color:#6d787d;margin:14px 0 4px}
.big{font-size:60px;font-weight:300;letter-spacing:-.05em;line-height:1.1;margin:22px 0 14px;color:var(--ink)}
.big small{font-size:18px;color:#7d888d;font-weight:400;letter-spacing:0;margin-left:4px}
.bars{display:grid;grid-template-columns:repeat(44,1fr);gap:3px;height:44px}
.bars span{background:#a7c61f;border-radius:2px}
.bar-scale{display:flex;justify-content:space-between;font-size:10px;color:#7d888d;margin:6px 0 16px}
.stats{display:grid;grid-template-columns:repeat(3,1fr);gap:8px;border-top:1px solid #edf1f2;padding-top:16px;margin-top:auto}
.stats b{display:block;font-size:24px;font-weight:400;letter-spacing:-.04em}
.stats small{font-size:11px;color:#7d888d}
.note{font-size:12px;line-height:1.5;color:#7d888d;margin-top:14px}
.wave{margin:18px 0 6px}
.thr-bottom{display:flex;align-items:flex-end;justify-content:space-between;margin-top:auto}
.thr-bottom .big{margin:0}
.thr-note{display:flex;flex-direction:column;align-items:flex-end;text-align:right}
.thr-note small{font-size:10px;color:#7d888d;line-height:1.35}
.thr-note b{font-size:14px;font-weight:600;margin-top:6px}
.viewer{background:#f1f8fb;border:1px dashed #b9d3de;border-radius:22px;min-height:210px;display:flex;align-items:center;justify-content:center;text-align:center;padding:22px;box-sizing:border-box;width:100%}
.viewer .film{width:78px;height:94px;margin:0 auto 12px}
.viewer strong{display:block;font-size:17px;font-weight:500;color:#10303d;margin-bottom:6px}
.viewer .muted{font-size:13px;color:#66737a}
.muted{font-size:13px;color:#66737a}

/* ===== Widgets ===== */
[data-testid="stFileUploaderDropzone"]{border:1px dashed #b9d3de;background:#f3f9fb;border-radius:20px}
[data-testid="stFileUploaderDropzone"] button{border-radius:30px;border:1px solid #cfdde4;background:#fff;color:#0d0d0d}
.stButton>button[kind="primary"]{background:#0d0d0d;color:#fff;border:0;border-radius:40px;min-height:54px;font-weight:500;font-size:15px;width:100%}
.stButton>button[kind="primary"]:hover{background:#26292b;color:var(--lime)}
.stButton>button:disabled{background:#e7eef1!important;color:#93a0a6!important}
[data-testid="stAlert"]{border-radius:18px}
[data-testid="stExpander"]{background:#fff;border:0;border-radius:22px}
[data-testid="stExpander"] details{border:0;border-radius:22px;background:#fff}
[data-testid="stCode"]{border-radius:16px}

/* ===== Context cards, footer ===== */
.ctx-grid{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:1rem;margin:16px 0 4px}
.ctx-card{background:#fff;border-radius:26px;padding:22px;box-sizing:border-box;min-width:0}
.ctx-icon{width:50px;height:50px;border-radius:16px;background:#f1f4f5;display:grid;place-items:center;margin-bottom:14px}
.ctx-card:nth-child(2) .ctx-icon{background:#eef8c2}
.ctx-card:nth-child(3) .ctx-icon{background:#d9edf7}
.ctx-title{font-size:16px;font-weight:500;letter-spacing:-.02em;margin-bottom:6px}
.ctx-body{font-size:13px;line-height:1.55;color:#66737a}
.foot{font-size:12px;color:#66737a;margin-top:22px;padding-bottom:6px}
[data-testid="stHorizontalBlock"]{gap:1rem;min-width:0}
[data-testid="stColumn"]{min-width:0}
[data-testid="stMainBlockContainer"]>div>[data-testid="stVerticalBlock"]{gap:1rem}

@media(max-width:1100px){.ctx-grid{grid-template-columns:repeat(2,minmax(0,1fr))}.hero-title{max-width:60%;font-size:36px}}
@media(max-width:768px){
 [data-testid="stSidebar"]{width:72px!important;min-width:72px!important}
 .block-container,[data-testid="stMainBlockContainer"]{padding:1.6rem 1rem}
 .greet-actions{margin-right:0;flex-wrap:wrap}.searchpill{min-width:0;flex:1}
 .hero{min-height:360px;padding:22px}.hero-title{font-size:30px;max-width:100%}.hero-art{display:none}.hero-play{display:none}
 .hero-cta{left:22px;right:auto}.hero-join{bottom:84px;left:22px}
 .ctx-grid{grid-template-columns:1fr}.res-num{font-size:48px}.big{font-size:48px}
}
'''
st.markdown(BASE_CSS + NAV_CSS + '</style>', unsafe_allow_html=True)


class FairViTMed(nn.Module):
    def __init__(self):
        super().__init__()
        self.vit = vit_b_16(weights=None)
        self.vit.heads.head = nn.Linear(
            self.vit.heads.head.in_features, 1
        )

    def forward(self, x):
        return self.vit(x)

@st.cache_resource
def load_model():
    torch.set_num_threads(1)

    checkpoint_path = hf_hub_download(
        repo_id="Visveswaranvisves/fairvit-medical-ai",
        filename="baseline_model.pth",
        revision="c90fbaa4612fff2db348514eef9f074fcc479636",
    )

    state = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=True,
        mmap=True,
    )

    # Support a plain state dictionary or a wrapped checkpoint.
    if "model_state_dict" in state:
        state = state["model_state_dict"]
    elif "state_dict" in state:
        state = state["state_dict"]

    with torch.device("meta"):
        model = FairViTMed()

    model.load_state_dict(state, strict=True, assign=True)
    model.eval()
    return model
preprocess=transforms.Compose([transforms.Resize((224,224)),transforms.ToTensor()])
with st.sidebar:
 st.markdown(f'<div class="rail-brand">{BRAND_SVG}</div>',unsafe_allow_html=True)
 page=st.radio('Workspace',['Image analysis','Model information'],label_visibility='collapsed')
 st.markdown('<div class="rail-div"></div><div class="rail-bottom"><div class="rail-avatar">V</div></div>',unsafe_allow_html=True)

heading='Hi, Researcher!' if page=='Image analysis' else 'Inside the model'
subline='Let’s look at your chest X-ray analysis overview.' if page=='Image analysis' else 'Architecture, checkpoint and interpretation in one place.'
st.markdown(f'<div class="greet"><div><h1>{heading}</h1><p>{subline}</p></div><div class="greet-actions"><div class="searchpill">{IC_SEARCH}Search imaging metrics</div><span class="blackpill">{IC_SHIELD}Baseline model · Research only</span></div></div>',unsafe_allow_html=True)

if page=='Image analysis':
 top_l,top_r=st.columns([1.7,1],gap='medium')
 with top_l:
  st.markdown(hero('Your X-ray<br>Analysis Starts<br>Here!','Built on NIH ChestX-ray14:','112k+','Radiographs','ViT-B/16 baseline'),unsafe_allow_html=True)
 res_slot=top_r.empty()
 b1,b2,b3=st.columns([1.15,1,1],gap='medium')
 with b1,st.container(border=False,key="panel_1"):
  st.markdown(card_head(IC_FILM,'Image workspace'),unsafe_allow_html=True)
  uploaded=st.file_uploader('Add a chest X-ray sample',type=['png','jpg','jpeg'])
  image=None;digest=None
  if uploaded is not None:
   digest=hashlib.sha256(uploaded.getvalue()).hexdigest()
   try:image=ImageOps.exif_transpose(Image.open(uploaded)).convert('RGB')
   except Exception:
    st.error('This image could not be read. Try a valid PNG or JPEG.')
   if image is not None:
    st.image(image,caption=f'{image.width} × {image.height} px · Uploaded sample',use_container_width=True)
  else:st.markdown(f'<div class="viewer"><div><div class="film">{FILM_SVG_SMALL}</div><strong>A space for your sample</strong><span class="muted">Upload a chest X-ray PNG or JPEG to begin.</span></div></div>',unsafe_allow_html=True)
  run=st.button('Analyse image →',type='primary',use_container_width=True,disabled=image is None)
  st.caption('The app does not verify that an uploaded image is a chest X-ray.')
  if run:
   st.session_state.pop('inference',None)
   try:
    with st.spinner('Loading the research model and analysing the sample…'):
     model=load_model()
     tensor=preprocess(image).unsqueeze(0)
     with torch.inference_mode():score=torch.sigmoid(model(tensor)).item()
    st.session_state.inference={'digest':digest,'score':float(score)}
   except Exception as exc:
    st.error('Model loading or inference failed.')
    with st.expander('Technical details'):st.exception(exc)
 with b2:
  st.markdown(prep_card(),unsafe_allow_html=True)
 with b3:
  st.markdown(thr_card(),unsafe_allow_html=True)
 r=st.session_state.get('inference')
 valid=r is not None and image is not None and r['digest']==digest
 score=r['score'] if valid else None
 number=f'{score:.3f}' if valid else '—'
 status=('Above' if score>.5 else 'Below')+' the pneumonia-class threshold' if valid else 'Your result will appear after analysis.'
 res_slot.markdown(res_card('Model response','Pneumonia-class score · Not a clinical probability',status,score,'0.50','Threshold',number,'/score','0.00 · Threshold 0.50 · 1.00'),unsafe_allow_html=True)
 st.caption('Research demonstration only. Not validated for clinical use; cannot diagnose or rule out disease. A low score does not mean the X-ray is normal.')
else:
 top_l,top_r=st.columns([1.7,1],gap='medium')
 with top_l:
  st.markdown(hero('Built for<br>Fair Medical<br>Diagnostics','Chest radiography research:','14','Thoracic labels','Baseline checkpoint'),unsafe_allow_html=True)
 with top_r:
  st.markdown(res_card('Architecture:','Vision Transformer B/16 · 224 ÷ 16 = 14 × 14 patches','ViT-B/16 · single-output head',1.0,'B/16','Patch size 16','196','/patches'),unsafe_allow_html=True)
 b1,b2,b3=st.columns([1,1,1],gap='medium')
 with b1,st.container(border=False,key="panel_3"):
  st.markdown(card_head(IC_CHIP,'Architecture & checkpoint'),unsafe_allow_html=True)
  st.write('Vision Transformer B/16 with a single-output classification head.')
  st.write('Checkpoint: baseline_model.pth. This interface uses the baseline model, not a fairness-adjusted checkpoint.')
  st.code('224 × 224 RGB → ViT-B/16 → sigmoid score',language=None)
 with b2,st.container(border=False,key="panel_4"):
  st.markdown(card_head(IC_PULSE,'Reading the output'),unsafe_allow_html=True)
  st.write('The model returns a score from 0 to 1. A value above 0.5 is above the configured pneumonia-class threshold.')
  st.write('This is not a calibrated clinical probability. Neither side of the threshold establishes a diagnosis or a normal X-ray.')
 with b3:
  st.markdown(thr_card(),unsafe_allow_html=True)
 st.markdown('<div class="ctx-grid">'
  +ctx_card(IC_DATA,'Research dataset','NIH ChestX-ray14: 112,120 frontal-view radiographs with 14 thoracic pathology labels.')
  +ctx_card(IC_HOSP.replace('#1d4f3f','#111').replace('width="17" height="17"','width="22" height="22"'),'Clinical setting','Explores how AI could assist hospital radiology, while remaining a research demo.')
  +ctx_card(IC_BALANCE,'Fairness focus','Studies performance gaps across patient sex, age and race in medical imaging.')
  +ctx_card(IC_HUMAN,'Human in the loop','Scores are never a diagnosis. Clinician review always comes first.')
  +'</div>',unsafe_allow_html=True)
 with st.expander('Deployment details'):
  st.write('CPU inference; cached checkpoint loading from Hugging Face; evaluation mode and inference mode enabled. The first request may take longer while weights download.')
st.markdown('<div class="foot">FairViT-Med · Baseline model research · Built by Visveswaran V · Portfolio demonstration</div>',unsafe_allow_html=True)
