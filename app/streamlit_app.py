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
    """Browser-tab icon: dark rounded tile, lilac lungs, lime medical cross (theme colours)."""
    try:
        s = 256
        img = Image.new('RGBA', (s, s), (0, 0, 0, 0))
        d = ImageDraw.Draw(img)
        d.rounded_rectangle((0, 0, s - 1, s - 1), radius=60, fill='#232324')
        d.ellipse((34, 58, 118, 208), fill='#bbb0ed')
        d.ellipse((138, 58, 222, 208), fill='#bbb0ed')
        d.rounded_rectangle((113, 74, 143, 182), radius=8, fill='#d8f36a')
        d.rounded_rectangle((74, 113, 182, 143), radius=8, fill='#d8f36a')
        return img
    except Exception:
        return '🫁'


st.set_page_config(page_title='FairViT | Chest X-ray research', page_icon=make_tab_icon(), layout='wide')

# ---------- Healthcare illustrations (inline SVG, theme colours only) ----------
LILAC = '#bdb0ef'
LIME = '#d8f36a'

XRAY_SVG = (
    '<svg viewBox="0 0 120 140" width="100%" height="100%" xmlns="http://www.w3.org/2000/svg" fill="none" '
    'stroke-linecap="round" stroke-linejoin="round">'
    f'<rect x="4" y="4" width="112" height="132" rx="14" stroke="{LILAC}" stroke-width="2" opacity=".7"/>'
    f'<path d="M30 30 Q60 20 90 30" stroke="{LILAC}" stroke-width="2" opacity=".6"/>'
    f'<path d="M60 24 V118" stroke="{LILAC}" stroke-width="2" stroke-dasharray="3 5" opacity=".7"/>'
    f'<path d="M55 38 C38 35 23 54 23 84 C23 105 34 116 51 113 C57 112 57 100 57 90 L57 48 Z" fill="{LILAC}" fill-opacity=".16" stroke="{LILAC}" stroke-width="2"/>'
    f'<path d="M65 38 C82 35 97 54 97 84 C97 105 86 116 69 113 C63 112 63 100 63 90 L63 48 Z" fill="{LILAC}" fill-opacity=".16" stroke="{LILAC}" stroke-width="2"/>'
    f'<g stroke="{LILAC}" stroke-width="1.6" opacity=".55">'
    '<path d="M29 58 Q43 65 55 60"/><path d="M26 72 Q42 80 55 74"/><path d="M26 86 Q42 94 55 88"/>'
    '<path d="M91 58 Q77 65 65 60"/><path d="M94 72 Q78 80 65 74"/><path d="M94 86 Q78 94 65 88"/></g>'
    f'<circle cx="39" cy="80" r="6" stroke="{LIME}" stroke-width="2"/>'
    '</svg>'
)

ECG_SVG = (
    '<svg viewBox="0 0 420 44" preserveAspectRatio="none" width="100%" height="44" xmlns="http://www.w3.org/2000/svg" fill="none">'
    f'<path d="M0 24 H96 L108 24 L118 8 L132 40 L144 14 L152 24 H236 L248 24 L258 6 L272 40 L284 16 L292 24 H420" '
    f'stroke="{LIME}" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round" vector-effect="non-scaling-stroke"/>'
    '</svg>'
)

BRAND_SVG = (
    '<svg viewBox="0 0 40 40" width="34" height="34" xmlns="http://www.w3.org/2000/svg" fill="none">'
    f'<rect x="1" y="1" width="38" height="38" rx="12" fill="{LIME}"/>'
    '<path d="M17 9 H23 V17 H31 V23 H23 V31 H17 V23 H9 V17 H17 Z" fill="#232324"/>'
    '</svg>'
)

HOSPITAL_SVG = (
    '<svg viewBox="0 0 48 48" width="42" height="42" xmlns="http://www.w3.org/2000/svg" fill="none" stroke="#232324" stroke-width="2.4" stroke-linecap="round" stroke-linejoin="round">'
    '<path d="M8 42 V16 L24 8 L40 16 V42 Z"/><path d="M4 42 H44"/>'
    '<path d="M24 18 V30 M18 24 H30"/><path d="M20 42 V35 H28 V42"/>'
    '</svg>'
)

def icon(path_markup, stroke='#77659f'):
    return (f'<svg viewBox="0 0 48 48" width="30" height="30" xmlns="http://www.w3.org/2000/svg" fill="none" '
            f'stroke="{stroke}" stroke-width="2.6" stroke-linecap="round" stroke-linejoin="round">{path_markup}</svg>')

ICON_HOSPITAL = icon('<path d="M8 42 V16 L24 8 L40 16 V42 Z"/><path d="M4 42 H44"/><path d="M24 18 V30 M18 24 H30"/><path d="M20 42 V35 H28 V42"/>')
ICON_STETH = icon('<path d="M12 6 V20 a8 8 0 0 0 16 0 V6"/><path d="M20 28 V32 a8 8 0 0 0 16 0 V28"/><circle cx="36" cy="24" r="4"/><path d="M10 6 H14 M26 6 H30"/>')
ICON_DATA = icon('<ellipse cx="24" cy="12" rx="14" ry="6"/><path d="M10 12 V36 c0 3.3 6.3 6 14 6 s14-2.7 14-6 V12"/><path d="M10 24 c0 3.3 6.3 6 14 6 s14-2.7 14-6"/>')
ICON_BALANCE = icon('<path d="M24 8 V40 M14 40 H34"/><path d="M10 14 H38"/><path d="M10 14 L4 28 a6 6 0 0 0 12 0 Z"/><path d="M38 14 L32 28 a6 6 0 0 0 12 0 Z"/>')


def hero(tag, headline, text, chips):
    chip_html = ''.join(f'<span class="chip">{escape(c)}</span>' for c in chips)
    return (
        '<div class="hero"><div class="hero-copy">'
        f'<span class="hero-tag">{escape(tag)}</span>'
        f'<div class="hero-title">{escape(headline)}</div>'
        f'<div class="hero-text">{escape(text)}</div>'
        f'<div class="ecg">{ECG_SVG}</div>'
        f'<div class="chip-row">{chip_html}</div>'
        '</div>'
        f'<div class="hero-art">{XRAY_SVG}</div>'
        '</div>'
    )


def ctx_card(svg, title, body):
    return (f'<div class="ctx-card"><div class="ctx-icon">{svg}</div>'
            f'<div class="ctx-title">{escape(title)}</div><div class="ctx-body">{escape(body)}</div></div>')


st.markdown('''<style>
.stApp{background:#dedfdd;color:#222223}.block-container{max-width:1440px;padding:2rem 2.1rem 3rem}
[data-testid="stHeader"]{background:#dedfdd}[data-testid="stSidebar"]{background:#232324;color:white;width:235px!important;border-right:0;border-radius:0 26px 26px 0}
[data-testid="stSidebar"] p,[data-testid="stSidebar"] label,[data-testid="stSidebar"] h3{color:#f4f4f0!important}
[data-testid="stSidebar"] [role="radiogroup"] label{padding:12px 15px;border-radius:30px;margin-bottom:8px}
[data-testid="stSidebar"] [role="radiogroup"] label:has(input:checked){background:white;color:#222!important}
[data-testid="stSidebar"] [role="radiogroup"] label:has(input:checked) p{color:#222!important}
h1,h2,h3{letter-spacing:-.045em;color:#202021}h1{font-size:2.5rem!important;font-weight:550!important}h3{font-size:1.05rem!important}
[data-testid="stMetric"]{background:white;border-radius:23px;padding:20px;min-height:115px}
[data-testid="stMetricValue"]{font-size:1.9rem;font-weight:550;letter-spacing:-.05em}
[data-testid="stMetricLabel"]{color:#696969;font-size:12px}
[data-testid="stFileUploaderDropzone"]{border:1px dashed #bdbabf;background:#f5f5f2;border-radius:16px}
.stButton>button[kind="primary"]{background:#d8f36a;color:#202022;border:0;border-radius:28px;min-height:46px;font-weight:650}
.stButton>button:disabled{background:#edf2d7;color:#74796a}
.brand{display:flex;align-items:center;gap:11px;font-size:23px;margin-bottom:37px;color:white;font-weight:550;letter-spacing:-1px}.spark{font-size:36px;color:#d8f36a}.navnote{font-size:11px;color:#b9bab8;line-height:1.7;margin-top:28px}.side-card{background:#d8f36a;border-radius:23px;padding:21px;color:#232324;margin-top:35px}.side-card strong{font-size:17px}.side-card p{font-size:12px;color:#232324!important;margin-top:9px}
.topline{display:flex;justify-content:space-between;align-items:center;margin:5px 0 23px;font-size:12px;color:#60615e}.badge{background:#fff;border-radius:25px;padding:9px 15px;color:#30312e}.muted{color:#71726f;font-size:12px}.viewer{background:#242426;border-radius:18px;min-height:290px;display:flex;align-items:center;justify-content:center;text-align:center;color:#bab9bd;margin:10px 0 18px;padding:25px}.viewer strong{color:#eeedf1;display:block;font-size:19px;font-weight:500;margin:15px 0 8px}.scan-icon{border:1px solid #777380;width:80px;height:95px;margin:auto;border-radius:10px;display:grid;place-items:center;color:#beb2ed;font-size:40px}
.output{background:#252526;border-radius:24px;padding:25px;color:#fff;min-height:315px}.output h3{color:#fff!important;font-size:17px!important;letter-spacing:-.02em}.score{font-size:58px;letter-spacing:-3px;color:#d8f36a;line-height:1.2;margin:20px 0 7px}.output .muted{color:#c0bdc7}.track{height:17px;background:#4a494e;border-radius:12px;position:relative;margin:24px 0 12px}.fill{height:100%;background:#bbb0ed;border-radius:12px}.threshold{position:absolute;left:50%;top:-6px;height:29px;border-left:2px solid #d8f36a}.scale{display:flex;justify-content:space-between;color:#bdbbc4;font-size:10px}.output-status{margin-top:24px;font-size:13px;color:#eeecf3}.bubble-row{display:flex;align-items:center;justify-content:center;gap:12px;padding:10px 0 20px}.bubble{border-radius:50%;width:103px;height:103px;display:flex;flex-direction:column;align-items:center;justify-content:center;background:#bdb0ef;font-size:29px;letter-spacing:-1px}.bubble small{font-size:10px;letter-spacing:0;color:#323139}.bubble:nth-child(2){background:#d8f36a;width:83px;height:83px;font-size:23px}.foot{font-size:11px;color:#696a67;margin-top:25px}
@media(max-width:768px){.block-container{padding:1.4rem 1rem}h1{font-size:2rem!important}.topline{gap:10px}.output{min-height:260px}.viewer{min-height:230px}[data-testid="stMetricValue"]{font-size:1.5rem}}

/* Scope surfaces to named cards, never to Streamlit's structural wrappers. */
[class*="st-key-panel_"]{background:#fff;border:1px solid #e0e2dc!important;border-radius:20px;padding:20px!important;box-sizing:border-box;min-width:0}
[data-testid="stMetric"]{box-sizing:border-box;min-width:0;width:100%}
[data-testid="stSidebarUserContent"]{padding:1.5rem 1rem}
[data-testid="stCaptionContainer"] p{font-size:13px;line-height:1.5}
[data-testid="stMainBlockContainer"]{width:100%;max-width:1600px;padding:2.8rem 2rem 2rem}
[data-testid="stHorizontalBlock"]{min-width:0}
[data-testid="stColumn"]{min-width:0}
@media(max-width:768px){[data-testid="stMainBlockContainer"]{padding:2rem 1rem}[class*="st-key-panel_"]{padding:16px!important}}

[data-testid="stSidebar"] .brand{color:#fff!important}
[data-testid="stMetric"]{padding:18px 20px;min-height:96px;border:1px solid #d2d5cf}
[data-testid="stMetricValue"]{font-size:1.8rem}
[data-testid="stSidebar"] [role="radiogroup"] label p{font-size:14px}
[data-testid="stCaptionContainer"]{color:#565953}
.topline{margin:0 0 4px;font-size:13px;flex-wrap:wrap;gap:8px}
.viewer{min-height:280px;margin:4px 0 12px}.viewer .muted{color:#bdbac5;font-size:14px}
.output{box-sizing:border-box;min-height:300px;padding:24px}
.muted{font-size:13px;color:#555952}.navnote{font-size:12px}.foot{font-size:12px}
.side-card{padding:18px;margin-top:24px}.brand{margin-bottom:26px}
h1{font-size:2.2rem!important}

/* ===== Spacing fixes ===== */
/* Streamlit gives inner elements a fixed pixel width; inside a padded card that pushes them past the right edge. */
[class*="st-key-panel_"] [data-testid="stElementContainer"],
[class*="st-key-panel_"] [data-testid="stElementContainer"]>div,
[class*="st-key-panel_"] [data-testid="stFileUploader"],
[class*="st-key-panel_"] [data-testid="stImage"],
[class*="st-key-panel_"] [data-testid="stMarkdownContainer"]{width:100%!important;max-width:100%!important;box-sizing:border-box;min-width:0}
[class*="st-key-panel_"]{padding:22px 22px 18px!important;gap:.85rem;overflow-x:clip}
[class*="st-key-panel_"] [data-testid="stImage"] img{max-width:100%;height:auto;border-radius:14px}
[class*="st-key-panel_"] .stButton,[class*="st-key-panel_"] .stButton>button{width:100%}
[class*="st-key-panel_"] h3{margin:0;padding:0 0 .15rem}
.viewer{width:100%;box-sizing:border-box;margin:2px 0 6px}
.output{width:100%;box-sizing:border-box}
.bubble-row{padding:6px 0 14px}
h1{padding:.2rem 0 .1rem!important;line-height:1.15!important}
[data-testid="stMainBlockContainer"]{padding-bottom:4rem}
[data-testid="stMainBlockContainer"]>div>[data-testid="stVerticalBlock"]{gap:1rem}
[data-testid="stHorizontalBlock"]{gap:1rem}
[data-testid="stSidebar"] [role="radiogroup"]{gap:6px}
.foot{margin-top:18px;padding-bottom:6px}

/* ===== Healthcare elements ===== */
.brand svg,.hero-art svg{flex:none}
.brand{gap:10px}
.side-card .side-icon{display:flex;align-items:center;justify-content:space-between;margin-bottom:12px}
.side-card .side-icon span{font-size:10px;letter-spacing:.08em;font-weight:650;background:#232324;color:#d8f36a;padding:5px 10px;border-radius:20px}
.hero{display:flex;align-items:stretch;justify-content:space-between;gap:22px;background:#252526;border-radius:24px;padding:24px 28px;margin:4px 0 6px;box-sizing:border-box;width:100%}
.hero-copy{flex:1;min-width:0;display:flex;flex-direction:column}
.hero-tag{font-size:11px;letter-spacing:.12em;color:#bdb0ef;font-weight:600}
.hero-title{font-size:26px;line-height:1.2;letter-spacing:-.04em;color:#fff;font-weight:550;margin:8px 0 6px}
.hero-text{font-size:13px;line-height:1.6;color:#c6c3cd;max-width:560px}
.ecg{margin:12px 0 10px;max-width:560px;opacity:.95}
.chip-row{display:flex;flex-wrap:wrap;gap:8px}
.chip{font-size:12px;color:#232324;background:#d8f36a;border-radius:20px;padding:5px 12px;font-weight:550}
.chip:nth-child(even){background:#bdb0ef}
.hero-art{flex:none;width:120px;height:140px;align-self:center;background:#2e2d31;border-radius:18px;padding:8px;box-sizing:border-box}
.ctx-grid{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:1rem;margin:4px 0 4px}
.ctx-card{background:#fff;border:1px solid #e0e2dc;border-radius:20px;padding:20px;box-sizing:border-box;min-width:0}
.ctx-icon{width:50px;height:50px;border-radius:16px;background:#ece8f7;display:grid;place-items:center;margin-bottom:14px}
.ctx-card:nth-child(even) .ctx-icon{background:#eef8c9}
.ctx-title{font-size:15px;font-weight:600;color:#202021;letter-spacing:-.02em;margin-bottom:6px}
.ctx-body{font-size:13px;line-height:1.55;color:#555952}
@media(max-width:1100px){.ctx-grid{grid-template-columns:repeat(2,minmax(0,1fr))}}
@media(max-width:768px){.hero{flex-direction:column;padding:20px}.hero-art{display:none}.ctx-grid{grid-template-columns:1fr}.hero-title{font-size:21px}}

</style>''',unsafe_allow_html=True)
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
 st.markdown(f'<div class="brand">{BRAND_SVG}fairvit<span style="color:#a9a9a5;font-size:12px"> / med</span></div>',unsafe_allow_html=True)
 page=st.radio('Workspace',['Image analysis','Model information'],label_visibility='collapsed')
 st.markdown(f'<div class="navnote">VISION RESEARCH WORKSPACE<br>Baseline checkpoint<br>ViT-B/16 architecture</div><div class="side-card"><div class="side-icon">{HOSPITAL_SVG}<span>CHEST X-RAY</span></div><strong>Research, made visible.</strong><p>Inspect a sample. Explore a score. Understand the limitations.</p></div><div class="navnote">Built by Visveswaran V<br>Portfolio demonstration</div>',unsafe_allow_html=True)
st.markdown('<div class="topline"><span>FairViT-Med / Research workspace</span><span class="badge">Baseline model · Research only</span></div>',unsafe_allow_html=True)
st.title('Image overview' if page=='Image analysis' else 'Inside the model')
st.caption('Explore chest X-ray classification with a Vision Transformer.' if page=='Image analysis' else 'Architecture, checkpoint and interpretation in one place.')
if page=='Image analysis':
 st.markdown(hero('CHEST RADIOGRAPHY · RESPONSIBLE AI','Fairness-aware chest X-ray research','A Vision Transformer explored for equitable medical image analysis across patient groups.',['NIH ChestX-ray14','ViT-B/16','Healthcare AI']),unsafe_allow_html=True)
 a,b,c=st.columns(3)
 a.metric('Architecture','ViT-B/16')
 b.metric('Model input','224 × 224')
 c.metric('Decision threshold','0.50')
 left,right=st.columns([1.05,1],gap='medium')
 with left,st.container(border=False,key="panel_1"):
  st.subheader('Image workspace')
  uploaded=st.file_uploader('Add a chest X-ray sample',type=['png','jpg','jpeg'])
  image=None;digest=None
  if uploaded is not None:
   digest=hashlib.sha256(uploaded.getvalue()).hexdigest()
   try:image=ImageOps.exif_transpose(Image.open(uploaded)).convert('RGB')
   except Exception:
    st.error('This image could not be read. Try a valid PNG or JPEG.')
   if image is not None:
    st.image(image,caption=f'{image.width} × {image.height} px · Uploaded sample',use_container_width=True)
  else:st.markdown(f'<div class="viewer"><div><div style="width:92px;height:108px;margin:auto">{XRAY_SVG}</div><strong>A space for your sample</strong><span class="muted">Upload a chest X-ray PNG or JPEG to begin.</span></div></div>',unsafe_allow_html=True)
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
 with right:
  with st.container(border=False,key="panel_2"):
   st.subheader('Input preparation')
   st.markdown('<div class="bubble-row"><div class="bubble">224<small>PIXELS PER SIDE</small></div><div class="bubble">RGB<small>3 CHANNELS</small></div></div>',unsafe_allow_html=True)
   st.caption('Resized to 224 × 224, then converted to a tensor. Matches the current training dataloader.')
  r=st.session_state.get('inference')
  valid=r is not None and image is not None and r['digest']==digest
  score=r['score'] if valid else None
  number=f'{score:.3f}' if valid else '—'
  status=('Above' if score>.5 else 'Below')+' the pneumonia-class threshold' if valid else 'Your result will appear after analysis.'
  fill=f'{score*100:.2f}' if valid else '0'
  st.markdown(f'<div class="output"><h3>Model response</h3><div class="muted">Pneumonia-class score</div><div class="score">{number}</div><div class="muted">Model score · Not a clinical probability</div><div class="track"><div class="fill" style="width:{fill}%"></div><div class="threshold"></div></div><div class="scale"><span>0.00</span><span>Threshold 0.50</span><span>1.00</span></div><div class="output-status">{status}</div></div>',unsafe_allow_html=True)
 st.caption('Research demonstration only. Not validated for clinical use; cannot diagnose or rule out disease. A low score does not mean the X-ray is normal.')
else:
 st.markdown(hero('MODEL · CHECKPOINT · INTERPRETATION','Built for equitable diagnostics','From chest radiograph to a single score, with the limits of that score kept in plain sight.',['Chest radiography','Baseline checkpoint','Research only']),unsafe_allow_html=True)
 a,b=st.columns(2)
 with a,st.container(border=False,key="panel_3"):
  st.subheader('Architecture & checkpoint')
  st.write('Vision Transformer B/16 with a single-output classification head.')
  st.write('Checkpoint: baseline_model.pth. This interface uses the baseline model, not a fairness-adjusted checkpoint.')
  st.code('224 × 224 RGB → ViT-B/16 → sigmoid score',language=None)
 with b,st.container(border=False,key="panel_4"):
  st.subheader('Reading the output')
  st.write('The model returns a score from 0 to 1. A value above 0.5 is above the configured pneumonia-class threshold.')
  st.write('This is not a calibrated clinical probability. Neither side of the threshold establishes a diagnosis or a normal X-ray.')
 st.markdown('<div class="ctx-grid">'
  +ctx_card(ICON_DATA,'Research dataset','NIH ChestX-ray14: 112,120 frontal-view radiographs with 14 thoracic pathology labels.')
  +ctx_card(ICON_HOSPITAL,'Clinical setting','Explores how AI could assist hospital radiology, while remaining a research demo.')
  +ctx_card(ICON_BALANCE,'Fairness focus','Studies performance gaps across patient sex, age and race in medical imaging.')
  +ctx_card(ICON_STETH,'Human in the loop','Scores are never a diagnosis. Clinician review always comes first.')
  +'</div>',unsafe_allow_html=True)
 with st.expander('Deployment details'):
  st.write('CPU inference; cached checkpoint loading from Hugging Face; evaluation mode and inference mode enabled. The first request may take longer while weights download.')
st.markdown('<div class="foot">FairViT-Med · Baseline model research · Visveswaran V</div>',unsafe_allow_html=True)
