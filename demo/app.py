"""
Gradio demo app for smell-to-molecule translation
"""
import gradio as gr
import torch
from transformers import AutoTokenizer
import sys
sys.path.append('..')

from src.models.neobert_model import SmellToMoleculeModel
import json

# Load model and tokenizer
model = SmellToMoleculeModel()
model.load_state_dict(torch.load('../models/checkpoints/best_model.pt'))
model.eval()

tokenizer = AutoTokenizer.from_pretrained('neobert-base')

# Load chemical mapping
with open('../data/processed/chemicals_mapping.json', 'r') as f:
    idx_to_chemical = json.load(f)

def predict_molecules(description, top_k=10):
    """Predict molecules from smell description"""
    # Tokenize
    inputs = tokenizer(
        description,
        return_tensors='pt',
        max_length=128,
        padding=True,
        truncation=True
    )
    
    # Predict
    with torch.no_grad():
        outputs, _ = model(**inputs)
    
    # Get top-k predictions
    probs = outputs[0].numpy()
    top_indices = probs.argsort()[-top_k:][::-1]
    
    # Format results
    results = []
    for idx in top_indices:
        chem = idx_to_chemical[str(idx)]
        results.append({
            'Chemical': chem['name'],
            'Formula': chem['formula'],
            'Confidence': f"{probs[idx]:.1%}"
        })
    
    return results

# Gradio interface
with gr.Blocks(title="Smell-to-Molecule Translator") as demo:
    gr.Markdown("# 🧪 Smell-to-Molecule Translator")
    gr.Markdown("Powered by NeoBERT - Describe a smell and get chemical predictions!")
    
    with gr.Row():
        with gr.Column():
            description_input = gr.Textbox(
                label="Smell Description",
                placeholder="e.g., Fresh citrus with floral notes and a woody base",
                lines=3
            )
            top_k_slider = gr.Slider(
                minimum=1,
                maximum=20,
                value=10,
                step=1,
                label="Number of predictions"
            )
            submit_btn = gr.Button("Translate to Molecules", variant="primary")
        
        with gr.Column():
            output = gr.Dataframe(
                headers=["Chemical", "Formula", "Confidence"],
                label="Predicted Molecules"
            )
    
    # Examples
    gr.Examples(
        examples=[
            ["Fresh citrus with bergamot and lemon notes"],
            ["Warm, woody, slightly sweet with sandalwood"],
            ["Sweet vanilla with hints of caramel and toffee"],
            ["Floral and powdery with violet and iris"],
            ["Green, fresh, herbaceous with basil and mint"]
        ],
        inputs=description_input
    )
    
    submit_btn.click(
        fn=predict_molecules,
        inputs=[description_input, top_k_slider],
        outputs=output
    )

if __name__ == '__main__':
    demo.launch(share=True)
