import torch
import torch.nn as nn
from transformers import BertModel, BertTokenizer, HubertModel, Wav2Vec2FeatureExtractor
import torchaudio
import os
import shutil
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# ValenceRegressor
class ValenceRegressor(nn.Module):
    def __init__(self, audio_dim=768, text_dim=768, hidden_dim=192, num_heads=6, num_layers=2, dropout=0.5):
        super().__init__()
        self.audio_transformer = nn.ModuleList([
            nn.TransformerEncoderLayer(d_model=audio_dim, nhead=num_heads, dim_feedforward=hidden_dim*4, dropout=dropout, batch_first=True)
            for _ in range(num_layers)
        ])
        self.audio_layer_norm = nn.LayerNorm(audio_dim)
        self.audio_attention_pool = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim*2), nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden_dim*2, 1)
        )
        self.text_encoder = BertModel.from_pretrained("bert-base-uncased")
        for param in self.text_encoder.parameters():
            param.requires_grad = False
        for param in list(self.text_encoder.parameters())[-2:]:
            param.requires_grad = True
        self.audio_projection = nn.Linear(audio_dim, hidden_dim)
        self.text_projection = nn.Linear(text_dim, hidden_dim)
        self.audio_to_text_attention = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=num_heads//2, dropout=dropout, batch_first=True)
        self.text_to_audio_attention = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=num_heads//2, dropout=dropout, batch_first=True)
        self.audio_gate = nn.Sequential(nn.Linear(hidden_dim*2, hidden_dim), nn.Sigmoid())
        self.text_gate = nn.Sequential(nn.Linear(hidden_dim*2, hidden_dim), nn.Sigmoid())
        self.fusion_layer = nn.Sequential(
            nn.Linear(hidden_dim*2, hidden_dim*2), nn.LayerNorm(hidden_dim*2), nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden_dim*2, hidden_dim)
        )
        self.shared_fc = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.GELU(), nn.Dropout(dropout))
        self.output_branch = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim//2), nn.LayerNorm(hidden_dim//2), nn.GELU(), nn.Dropout(dropout*0.5), nn.Linear(hidden_dim//2, 1)
        )

    def audio_attention_pooling(self, x, audio_mask=None):
        weights = self.audio_attention_pool(x)
        if audio_mask is not None:
            weights = weights.masked_fill(~audio_mask.bool().unsqueeze(-1), float('-inf'))
        weights = torch.softmax(weights, dim=1)
        output = torch.bmm(weights.transpose(1, 2), x)
        return output.squeeze(1)

    def forward(self, audio_features, input_ids, attention_mask):
        audio_mask = (audio_features.abs().sum(dim=-1) > 1e-6)
        audio_repr = audio_features
        for layer in self.audio_transformer:
            audio_key_padding_mask = (~audio_mask).float()
            audio_repr = layer(audio_repr, src_key_padding_mask=audio_key_padding_mask)
        audio_repr = self.audio_layer_norm(audio_repr)
        text_outputs = self.text_encoder(input_ids=input_ids, attention_mask=attention_mask)
        text_repr = text_outputs.last_hidden_state
        audio_proj = self.audio_projection(audio_repr)
        text_proj = self.text_projection(text_repr)
        audio_attended_text, _ = self.audio_to_text_attention(
            query=audio_proj, key=text_proj, value=text_proj, key_padding_mask=(1 - attention_mask).bool()
        )
        text_attended_audio, _ = self.text_to_audio_attention(
            query=text_proj, key=audio_proj, value=audio_proj, key_padding_mask=(~audio_mask).bool()
        )
        audio_concat = torch.cat([audio_proj, audio_attended_text], dim=-1)
        text_concat = torch.cat([text_proj, text_attended_audio], dim=-1)
        audio_gate_value = self.audio_gate(audio_concat)
        text_gate_value = self.text_gate(text_concat)
        gated_audio = audio_proj * audio_gate_value
        gated_text = text_proj * text_gate_value
        pooled_audio = self.audio_attention_pooling(gated_audio, audio_mask)
        text_sum = torch.sum(gated_text * attention_mask.unsqueeze(-1), dim=1)
        text_count = torch.sum(attention_mask, dim=1, keepdim=True).clamp(min=1)
        pooled_text = text_sum / text_count
        fused = torch.cat([pooled_audio, pooled_text], dim=1)
        joint_repr = self.fusion_layer(fused)
        shared = self.shared_fc(joint_repr)
        output = self.output_branch(shared)
        scaled_output = 1.0 + 4.0 * torch.sigmoid(output)
        return scaled_output

# MultimodalArousalDominanceModel
class MultimodalArousalDominanceModel(nn.Module):
    def __init__(self, audio_dim=768, text_dim=768, hidden_dim=192, num_heads=6, num_layers=2, dropout=0.5):
        super().__init__()
        self.audio_transformer = nn.ModuleList([
            nn.TransformerEncoderLayer(d_model=audio_dim, nhead=num_heads, dim_feedforward=hidden_dim*4, dropout=dropout, batch_first=True)
            for _ in range(num_layers)
        ])
        self.audio_layer_norm = nn.LayerNorm(audio_dim)
        self.audio_attention_pool = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim*2), nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden_dim*2, 1)
        )
        self.text_encoder = BertModel.from_pretrained("bert-base-uncased")
        for param in self.text_encoder.parameters():
            param.requires_grad = False
        for param in list(self.text_encoder.parameters())[-2:]:
            param.requires_grad = True
        self.audio_projection = nn.Linear(audio_dim, hidden_dim)
        self.text_projection = nn.Linear(text_dim, hidden_dim)
        self.audio_to_text_attention = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=num_heads//2, dropout=dropout, batch_first=True)
        self.text_to_audio_attention = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=num_heads//2, dropout=dropout, batch_first=True)
        self.audio_gate = nn.Sequential(nn.Linear(hidden_dim*2, hidden_dim), nn.Sigmoid())
        self.text_gate = nn.Sequential(nn.Linear(hidden_dim*2, hidden_dim), nn.Sigmoid())
        self.fusion_layer_arousal = nn.Sequential(
            nn.Linear(hidden_dim*2, hidden_dim*2), nn.LayerNorm(hidden_dim*2), nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden_dim*2, hidden_dim)
        )
        self.fusion_layer_dominance = nn.Sequential(
            nn.Linear(hidden_dim*2, hidden_dim*2), nn.LayerNorm(hidden_dim*2), nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden_dim*2, hidden_dim)
        )
        self.shared_fc_arousal = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.GELU(), nn.Dropout(dropout)
        )
        self.shared_fc_dominance = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.GELU(), nn.Dropout(dropout)
        )
        self.output_branch_arousal = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim//2), nn.LayerNorm(hidden_dim//2), nn.GELU(), nn.Dropout(dropout*0.5), nn.Linear(hidden_dim//2, 1)
        )
        self.output_branch_dominance = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim//2), nn.LayerNorm(hidden_dim//2), nn.GELU(), nn.Dropout(dropout*0.5), nn.Linear(hidden_dim//2, 1)
        )

    def audio_attention_pooling(self, x, audio_mask=None):
        weights = self.audio_attention_pool(x)
        if audio_mask is not None:
            weights = weights.masked_fill(~audio_mask.bool().unsqueeze(-1), float('-inf'))
        weights = torch.softmax(weights, dim=1)
        output = torch.bmm(weights.transpose(1, 2), x)
        return output.squeeze(1)

    def forward(self, audio_features, input_ids, attention_mask):
        audio_mask = (audio_features.abs().sum(dim=-1) > 1e-6)
        audio_repr = audio_features
        for layer in self.audio_transformer:
            audio_key_padding_mask = (~audio_mask).float()
            audio_repr = layer(audio_repr, src_key_padding_mask=audio_key_padding_mask)
        audio_repr = self.audio_layer_norm(audio_repr)
        text_outputs = self.text_encoder(input_ids=input_ids, attention_mask=attention_mask)
        text_repr = text_outputs.last_hidden_state
        audio_proj = self.audio_projection(audio_repr)
        text_proj = self.text_projection(text_repr)
        audio_attended_text, _ = self.audio_to_text_attention(
            query=audio_proj, key=text_proj, value=text_proj, key_padding_mask=(1 - attention_mask).bool()
        )
        text_attended_audio, _ = self.text_to_audio_attention(
            query=text_proj, key=audio_proj, value=audio_proj, key_padding_mask=(~audio_mask).bool()
        )
        audio_concat = torch.cat([audio_proj, audio_attended_text], dim=-1)
        text_concat = torch.cat([text_proj, text_attended_audio], dim=-1)
        audio_gate_value = self.audio_gate(audio_concat)
        text_gate_value = self.text_gate(text_concat)
        gated_audio = audio_proj * audio_gate_value
        gated_text = text_proj * text_gate_value
        pooled_audio = self.audio_attention_pooling(gated_audio, audio_mask)
        text_sum = torch.sum(gated_text * attention_mask.unsqueeze(-1), dim=1)
        text_count = torch.sum(attention_mask, dim=1, keepdim=True).clamp(min=1)
        pooled_text = text_sum / text_count
        fused = torch.cat([pooled_audio, pooled_text], dim=1)
        joint_repr_arousal = self.fusion_layer_arousal(fused)
        joint_repr_dominance = self.fusion_layer_dominance(fused)
        shared_arousal = self.shared_fc_arousal(joint_repr_arousal)
        shared_dominance = self.shared_fc_dominance(joint_repr_dominance)
        output_arousal = self.output_branch_arousal(shared_arousal)
        output_dominance = self.output_branch_dominance(shared_dominance)
        scaled_arousal = 1.0 + 4.0 * torch.sigmoid(output_arousal)
        scaled_dominance = 1.0 + 4.0 * torch.sigmoid(output_dominance)
        return scaled_arousal, scaled_dominance

# Function to clear Hugging Face cache
def clear_huggingface_cache():
    cache_dir = Path.home() / ".cache" / "huggingface" / "transformers"
    if cache_dir.exists():
        shutil.rmtree(cache_dir)
        print(f"Cleared cache at {cache_dir}")

# Function to load feature extractor with retry
def load_feature_extractor(model_name, max_retries=3):
    for attempt in range(max_retries):
        try:
            return Wav2Vec2FeatureExtractor.from_pretrained(model_name)
        except (OSError, ValueError) as e:
            print(f"Attempt {attempt+1}/{max_retries} failed: {e}")
            if attempt < max_retries - 1:
                clear_huggingface_cache()
    raise RuntimeError(f"Failed to load Wav2Vec2FeatureExtractor after {max_retries} attempts")

# Function to extract audio features using HuBERT
def extract_hubert_features(audio_path, processor, hubert_model, device, sampling_rate=16000, max_audio_samples=128000):
    audio, sr = torchaudio.load(audio_path)
    if sr != sampling_rate:
        audio = torchaudio.transforms.Resample(sr, sampling_rate)(audio)
    audio = audio.squeeze(0)
    if audio.dim() > 1:
        audio = audio[0]
    if audio.size(0) > max_audio_samples:
        audio = audio[:max_audio_samples]
    elif audio.size(0) < max_audio_samples:
        audio = torch.nn.functional.pad(audio, (0, max_audio_samples - audio.size(0)))
    audio = audio.cpu().numpy()
    inputs = processor(audio, sampling_rate=sampling_rate, return_tensors="pt", padding=False, truncation=False)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = hubert_model(**inputs)
    return outputs.last_hidden_state

# Function to preprocess inputs with empty transcription
def preprocess_inputs(audio_path, tokenizer, processor, hubert_model, device, max_length=512):
    # Use an empty text since we're not using transcriptions
    empty_text = " "
    encoding = tokenizer(empty_text, max_length=max_length, padding='max_length', truncation=True, return_tensors='pt')
    input_ids = encoding['input_ids'].to(device)
    attention_mask = encoding['attention_mask'].to(device)
    audio_features = extract_hubert_features(audio_path, processor, hubert_model, device)
    return input_ids, attention_mask, audio_features

# Function to load state dict flexibly
def load_state_dict(model, checkpoint_path, device):
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    if isinstance(checkpoint, dict) and not all(k.startswith(('audio_', 'text_', 'fusion_', 'shared_', 'output_')) for k in checkpoint.keys()):
        # Try common keys if not direct state dict
        for key in ['model_state_dict', 'state_dict', 'model']:
            if key in checkpoint:
                state_dict = checkpoint[key]
                break
        else:
            state_dict = checkpoint  # Assume direct state dict if no common keys
    else:
        state_dict = checkpoint  # Direct state dict
    # Handle DataParallel prefix
    state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict)
    return model

# Function to load models
def load_models(valence_checkpoint_path, ad_checkpoint_path, device):
    valence_model = ValenceRegressor(audio_dim=768, text_dim=768, hidden_dim=192, num_heads=6, num_layers=2, dropout=0.5)
    valence_model = load_state_dict(valence_model, valence_checkpoint_path, device)
    if torch.cuda.device_count() > 1:
        valence_model = nn.DataParallel(valence_model)
    valence_model.to(device)
    valence_model.eval()

    ad_model = MultimodalArousalDominanceModel(audio_dim=768, text_dim=768, hidden_dim=192, num_heads=6, num_layers=2, dropout=0.5)
    ad_model = load_state_dict(ad_model, ad_checkpoint_path, device)
    if torch.cuda.device_count() > 1:
        ad_model = nn.DataParallel(ad_model)
    ad_model.to(device)
    ad_model.eval()

    return valence_model, ad_model

# Main function for prediction - modified to not require transcription
def predict_emotions(audio_path, valence_checkpoint_path, ad_checkpoint_path):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    processor = load_feature_extractor('facebook/hubert-base-ls960')
    hubert_model = HubertModel.from_pretrained('facebook/hubert-base-ls960').to(device)
    hubert_model.eval()
    
    valence_model, ad_model = load_models(valence_checkpoint_path, ad_checkpoint_path, device)
    
    input_ids, attention_mask, audio_features = preprocess_inputs(
        audio_path=audio_path,
        tokenizer=tokenizer,
        processor=processor,
        hubert_model=hubert_model,
        device=device,
        max_length=512
    )
    
    with torch.no_grad():
        valence_pred = valence_model(audio_features, input_ids, attention_mask)
        arousal_pred, dominance_pred = ad_model(audio_features, input_ids, attention_mask)
    
    return {
        'valence': valence_pred.item(),
        'arousal': arousal_pred.item(),
        'dominance': dominance_pred.item()
    }

# Example usage
if __name__ == "__main__":
    valence_checkpoint_path = os.getenv("VALENCE_CHECKPOINT_PATH", '/Users/suchirmvelpanur/Desktop/Generative AI and its Applications/Project/VAD_Models/V.pth')
    ad_checkpoint_path = os.getenv("AD_CHECKPOINT_PATH", '/Users/suchirmvelpanur/Desktop/Generative AI and its Applications/Project/VAD_Models/AD.pth')
    audio_path = os.getenv("TEST_AUDIO_PATH", '/Users/suchirmvelpanur/Desktop/Generative AI and its Applications/Project/VAD_Models/Ses01F_impro01_F000.wav')
    
    predictions = predict_emotions(audio_path, valence_checkpoint_path, ad_checkpoint_path)
    
    print(f"Predicted Valence: {predictions['valence']:.4f}")
    print(f"Predicted Arousal: {predictions['arousal']:.4f}")
    print(f"Predicted Dominance: {predictions['dominance']:.4f}")

