import streamlit as st
import torch
from torch import nn
from torchvision.models import vit_b_16
from torchvision import transforms
from huggingface_hub import hf_hub_download
from PIL import Image, ImageOps

st.set_page_config(
    page_title="FairViT-Med",
    page_icon="🔬",
    layout="centered",
)

st.title("FairViT-Med")
st.caption("Chest X-ray classification · Baseline model")
st.info(
    "Research demonstration only. This model is not validated "
    "for clinical use and cannot diagnose or rule out disease."
)


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


# Matches the current training dataloader in your repository.
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

uploaded = st.file_uploader(
    "Upload a sample chest X-ray",
    type=["png", "jpg", "jpeg"],
)

if uploaded is not None:
    try:
        image = ImageOps.exif_transpose(
            Image.open(uploaded)
        ).convert("RGB")
    except Exception:
        st.error("Could not read this image. Try another PNG or JPEG.")
        st.stop()

    st.image(image, caption="Uploaded image", use_container_width=True)

    if st.button("Run model", type="primary"):
        try:
            with st.spinner(
                "Loading model and analysing image. "
                "The first run downloads approximately 327 MB..."
            ):
                model = load_model()
                tensor = preprocess(image).unsqueeze(0)

                with torch.inference_mode():
                    score = torch.sigmoid(model(tensor)).item()

            st.metric("Pneumonia-class model score", f"{score:.3f}")
            st.write(
                "**Model output:** "
                + (
                    "Above the pneumonia-class threshold"
                    if score > 0.5
                    else "Below the pneumonia-class threshold"
                )
            )
            st.caption(
                "Threshold: 0.5. This score is not a calibrated "
                "clinical probability. A low score does not mean "
                "the X-ray is normal."
            )
        except Exception as exc:
            st.error("Model loading or inference failed.")
            st.exception(exc)
else:
    st.write("Upload an image, then click Run model.")
