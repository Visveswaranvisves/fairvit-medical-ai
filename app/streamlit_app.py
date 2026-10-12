from html import escape
import hashlib
import streamlit as st
import torch
from torch import nn
from torchvision.models import vit_b_16
from torchvision import transforms
from huggingface_hub import hf_hub_download
from PIL import Image, ImageOps

st.set_page_config(page_title='FairViT | Image research',page_icon='✦',layout='wide')
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
[data-testid="stVerticalBlockBorderWrapper"]>div{background:white;border:0!important;border-radius:24px;padding:16px}
[data-testid="stFileUploaderDropzone"]{border:1px dashed #bdbabf;background:#f5f5f2;border-radius:16px}
.stButton>button[kind="primary"]{background:#d8f36a;color:#202022;border:0;border-radius:28px;min-height:46px;font-weight:650}
.stButton>button:disabled{background:#edf2d7;color:#74796a}
.brand{display:flex;align-items:center;gap:11px;font-size:23px;margin-bottom:37px;color:white;font-weight:550;letter-spacing:-1px}.spark{font-size:36px;color:#d8f36a}.navnote{font-size:11px;color:#b9bab8;line-height:1.7;margin-top:28px}.side-card{background:#d8f36a;border-radius:23px;padding:21px;color:#232324;margin-top:35px}.side-card strong{font-size:17px}.side-card p{font-size:12px;color:#232324!important;margin-top:9px}
.topline{display:flex;justify-content:space-between;align-items:center;margin:5px 0 23px;font-size:12px;color:#60615e}.badge{background:#fff;border-radius:25px;padding:9px 15px;color:#30312e}.muted{color:#71726f;font-size:12px}.viewer{background:#242426;border-radius:18px;min-height:290px;display:flex;align-items:center;justify-content:center;text-align:center;color:#bab9bd;margin:10px 0 18px;padding:25px}.viewer strong{color:#eeedf1;display:block;font-size:19px;font-weight:500;margin:15px 0 8px}.scan-icon{border:1px solid #777380;width:80px;height:95px;margin:auto;border-radius:10px;display:grid;place-items:center;color:#beb2ed;font-size:40px}
.output{background:#252526;border-radius:24px;padding:25px;color:#fff;min-height:315px}.output h3{color:#fff!important;font-size:17px!important;letter-spacing:-.02em}.score{font-size:58px;letter-spacing:-3px;color:#d8f36a;line-height:1.2;margin:20px 0 7px}.output .muted{color:#c0bdc7}.track{height:17px;background:#4a494e;border-radius:12px;position:relative;margin:24px 0 12px}.fill{height:100%;background:#bbb0ed;border-radius:12px}.threshold{position:absolute;left:50%;top:-6px;height:29px;border-left:2px solid #d8f36a}.scale{display:flex;justify-content:space-between;color:#bdbbc4;font-size:10px}.output-status{margin-top:24px;font-size:13px;color:#eeecf3}.bubble-row{display:flex;align-items:center;justify-content:center;gap:12px;padding:10px 0 20px}.bubble{border-radius:50%;width:103px;height:103px;display:flex;flex-direction:column;align-items:center;justify-content:center;background:#bdb0ef;font-size:29px;letter-spacing:-1px}.bubble small{font-size:10px;letter-spacing:0;color:#323139}.bubble:nth-child(2){background:#d8f36a;width:83px;height:83px;font-size:23px}.foot{font-size:11px;color:#696a67;margin-top:25px}
@media(max-width:768px){.block-container{padding:1.4rem 1rem}h1{font-size:2rem!important}.topline{gap:10px}.output{min-height:260px}.viewer{min-height:230px}[data-testid="stMetricValue"]{font-size:1.5rem}}
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
 st.markdown('<div class="brand"><span class="spark">✦</span>fairvit<span style="color:#a9a9a5;font-size:12px"> / med</span></div>',unsafe_allow_html=True)
 page=st.radio('Workspace',['Image analysis','Model information'],label_visibility='collapsed')
 st.markdown('<div class="navnote">VISION RESEARCH WORKSPACE<br>Baseline checkpoint<br>ViT-B/16 architecture</div><div class="side-card"><strong>Research, made visible.</strong><p>Inspect a sample. Explore a score. Understand the limitations.</p></div><div class="navnote">Built by Visveswaran V<br>Portfolio demonstration</div>',unsafe_allow_html=True)
st.markdown('<div class="topline"><span>FairViT-Med / Research workspace</span><span class="badge">Baseline model · Research only</span></div>',unsafe_allow_html=True)
st.title('Image overview' if page=='Image analysis' else 'Inside the model')
st.caption('Explore chest X-ray classification with a Vision Transformer.' if page=='Image analysis' else 'Architecture, checkpoint and interpretation in one place.')
if page=='Image analysis':
 a,b,c=st.columns(3)
 a.metric('Architecture','ViT-B/16')
 b.metric('Model input','224 × 224')
 c.metric('Decision threshold','0.50')
 st.write('')
 left,right=st.columns([1.05,1],gap='medium')
 with left,st.container(border=True):
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
  else:st.markdown('<div class="viewer"><div><div class="scan-icon">⊞</div><strong>A space for your sample</strong><span class="muted">Upload a PNG or JPEG to begin.</span></div></div>',unsafe_allow_html=True)
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
  with st.container(border=True):
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
 a,b=st.columns(2)
 with a,st.container(border=True):
  st.subheader('Architecture & checkpoint')
  st.write('Vision Transformer B/16 with a single-output classification head.')
  st.write('Checkpoint: baseline_model.pth. This interface uses the baseline model, not a fairness-adjusted checkpoint.')
  st.code('224 × 224 RGB → ViT-B/16 → sigmoid score',language=None)
 with b,st.container(border=True):
  st.subheader('Reading the output')
  st.write('The model returns a score from 0 to 1. A value above 0.5 is above the configured pneumonia-class threshold.')
  st.write('This is not a calibrated clinical probability. Neither side of the threshold establishes a diagnosis or a normal X-ray.')
 with st.expander('Deployment details'):
  st.write('CPU inference; cached checkpoint loading from Hugging Face; evaluation mode and inference mode enabled. The first request may take longer while weights download.')
st.markdown('<div class="foot">FairViT-Med · Baseline model research · Visveswaran V</div>',unsafe_allow_html=True)
