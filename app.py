import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import pickle
import os


# ============================================
# PATHS & DOWNLOAD
# ============================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "best_model.pth")
WORD2IDX_PATH = os.path.join(BASE_DIR, "word2idx.pkl")
IDX2WORD_PATH = os.path.join(BASE_DIR, "idx2word.pkl")

if os.path.isdir(MODEL_PATH):
    st.error(
        "'best_model.pth' is a FOLDER, but it must be a single .pth file.\n\n"
        "Fix:\n"
        "- Delete/rename the 'best_model.pth' folder.\n"
        "- Put the real 'best_model.pth' file next to app.py.\n"
        "- If you only have the folder contents, re-zip the *contents* so the zip root contains "
        "'version', 'data.pkl', and '.data/' (no extra parent folder), then rename .zip -> .pth."
    )
    st.stop()

if not os.path.exists(MODEL_PATH):
    st.error(
        "Model file 'best_model.pth' not found.\n\n"
        "Put 'best_model.pth' in the same folder as 'app.py', then rerun the app."
    )
    st.stop()


st.set_page_config(page_title="Neural Storyteller", page_icon="🖼️", layout="wide")

# ============================================
# MODEL ARCHITECTURE
# ============================================
class Encoder(nn.Module):
    def __init__(self, embed_size):
        super().__init__()
        self.fc = nn.Linear(2048, embed_size)
    def forward(self, x):
        return self.fc(x)

class Decoder(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers=2, dropout=0.5):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers=num_layers, 
                           batch_first=True, dropout=dropout if num_layers > 1 else 0)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, vocab_size)
    
    def forward(self, captions, hidden):
        emb = self.embedding(captions)
        emb = self.dropout(emb)
        out, _ = self.lstm(emb, hidden)
        out = self.dropout(out)
        return self.fc(out)

class ImageCaptionModel(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers=2):
        super().__init__()
        self.encoder = Encoder(embed_size)
        self.decoder = Decoder(embed_size, hidden_size, vocab_size, num_layers=num_layers)
        self.hidden_size = hidden_size
        self.embed_size = embed_size
        self.num_layers = num_layers
        self.hidden_projection = nn.Linear(embed_size, hidden_size)

# ============================================
# LOAD MODEL
# ============================================
if hasattr(st, "cache_resource"):
    cache_model = st.cache_resource
else:
    # Fallback for older Streamlit versions
    cache_model = st.cache


@cache_model
def load_model():
    # Load vocabulary
    try:
        with open(WORD2IDX_PATH, "rb") as f:
            word2idx = pickle.load(f)
        with open(IDX2WORD_PATH, "rb") as f:
            idx2word = pickle.load(f)
    except FileNotFoundError as e:
        st.error(f"Vocabulary file not found: {e}")
        st.stop()

    vocab_size = len(word2idx)
    model = ImageCaptionModel(512, 1024, vocab_size, num_layers=2)

    # Load model weights (handle older PyTorch versions without weights_only)
    try:
        checkpoint = torch.load(MODEL_PATH, map_location="cpu", weights_only=False)
    except TypeError:
        checkpoint = torch.load(MODEL_PATH, map_location="cpu")
    except RuntimeError as e:
        st.error(f"Failed to load 'best_model.pth': {e}")
        st.info(
            "This usually means the file is corrupted OR it isn't a real PyTorch checkpoint file.\n\n"
            "If you created 'best_model.pth' by zipping a folder, open it with 7-Zip and confirm the TOP LEVEL contains:\n"
            "- version\n"
            "- data.pkl\n"
            "- .data/\n"
            "If you see an extra folder first, re-create the zip from inside the folder so those files are at the root."
        )
        st.stop()
    except FileNotFoundError as e:
        st.error(f"Model file not found: {e}")
        st.stop()
    except PermissionError as e:
        st.error(f"Cannot read model file (permission error): {e}")
        st.info(
            "Close any program using 'best_model.pth', ensure the file is not read-only, "
            "and consider moving this project out of OneDrive (for example to 'C:\\Users\\abdul\\Documents') "
            "to avoid sync locks."
        )
        st.stop()

    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    else:
        state_dict = checkpoint

    model.load_state_dict(state_dict)
    model.eval()

    # Load ResNet backbone with compatibility for different torchvision versions
    try:
        resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
    except AttributeError:
        # Older torchvision: use pretrained=True
        resnet = models.resnet50(pretrained=True)

    resnet = nn.Sequential(*list(resnet.children())[:-1])
    resnet.eval()

    transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
            ),
        ]
    )

    return model, resnet, transform, word2idx, idx2word

model, resnet, transform, word2idx, idx2word = load_model()

# ============================================
# CAPTION GENERATION
# ============================================
def generate_caption(feature, beam_width=5, max_len=30):
    model.eval()
    with torch.no_grad():
        encoded = model.encoder(feature)
        projected = model.hidden_projection(encoded)
        h = projected.unsqueeze(0).repeat(model.num_layers, 1, 1)
        c = torch.zeros_like(h)
        
        beams = [([word2idx["<start>"]], 0.0, h, c)]
        completed = []
        
        for _ in range(max_len):
            candidates = []
            for seq, score, h_state, c_state in beams:
                if seq[-1] == word2idx["<end>"]:
                    completed.append((seq, score))
                    continue
                
                word = torch.tensor([[seq[-1]]])
                emb = model.decoder.embedding(word)
                out, (new_h, new_c) = model.decoder.lstm(emb, (h_state, c_state))
                logits = model.decoder.fc(out.squeeze(1))
                log_probs = torch.log_softmax(logits, dim=-1)
                
                top_probs, top_indices = log_probs[0].topk(beam_width)
                for prob, idx in zip(top_probs, top_indices):
                    candidates.append((seq + [idx.item()], score + prob.item(), new_h, new_c))
            
            if not candidates:
                break
            beams = sorted(candidates, key=lambda x: x[1], reverse=True)[:beam_width]
        
        completed.extend(beams)
        best_seq, _ = max(completed, key=lambda x: x[1]) if completed else ([word2idx["<start>"], word2idx["<end>"]], 0)
        
        caption = [idx2word[idx] for idx in best_seq 
                  if idx not in [word2idx["<start>"], word2idx["<end>"], word2idx["<pad>"]]]
        return " ".join(caption) if caption else "No caption generated"

# ============================================
# STREAMLIT UI
# ============================================
st.title("🖼️ Neural Storyteller - Image Captioning")
st.markdown("### Upload an image and get AI-generated captions!")

with st.sidebar:
    st.header("⚙️ Model Info")
    st.info("""
    **Architecture:**
    - Encoder: ResNet50
    - Decoder: 2-layer LSTM
    - Vocabulary: 7,662 words
    
    **Training:**
    - Dataset: Flickr30k
    - Epochs: 44
    - Val Loss: 2.75
    - BLEU-4: 0.18-0.22
    """)

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    col1, col2 = st.columns([1, 1])
    
    with col1:
        image = Image.open(uploaded_file).convert('RGB')
        st.image(image, caption='Uploaded Image', use_column_width=True)
    
    with col2:
        with st.spinner('🤖 Generating caption...'):
            img_tensor = transform(image).unsqueeze(0)
            
            with torch.no_grad():
                feature = resnet(img_tensor).squeeze()
                if len(feature.shape) == 1:
                    feature = feature.unsqueeze(0)
            
            caption = generate_caption(feature, beam_width=5)
            
            st.success("✨ Generated Caption")
            st.markdown(f"### *{caption}*")
            st.markdown(f"**Words:** {len(caption.split())}")
else:
    st.info("👆 Upload an image to get started!")

st.markdown("---")
st.markdown("*Built with PyTorch, ResNet50, and Streamlit | Trained on dual T4 GPUs*")