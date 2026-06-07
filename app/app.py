import gradio as gr
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch

MODEL_PATH = "AbhijithSK/medical-summarizer"

print("Loading model...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model     = AutoModelForSeq2SeqLM.from_pretrained(MODEL_PATH)
model.eval()
print("Model ready.")

def summarize(text):
    if not text.strip():
        return "Please enter some medical text."
    inputs = tokenizer(
        text,
        return_tensors="pt",
        max_length=512,
        truncation=True
    )
    with torch.no_grad():
        summary_ids = model.generate(
            inputs["input_ids"],
            max_length=128,
            min_length=30,
            num_beams=4,
            length_penalty=2.0,
            early_stopping=True
        )
    return tokenizer.decode(summary_ids[0], skip_special_tokens=True)

examples = [
    ["Patients with type 2 diabetes mellitus often present with hyperglycemia, insulin resistance, and progressive beta cell dysfunction. The condition is associated with increased risk of cardiovascular disease, nephropathy, retinopathy, and neuropathy. Treatment includes lifestyle modifications, metformin as first-line therapy, and additional agents such as GLP-1 receptor agonists and SGLT2 inhibitors when glycemic targets are not met."],
    ["A 45-year-old male presented with chest pain radiating to the left arm, diaphoresis, and shortness of breath. ECG showed ST elevation in leads II, III, and aVF consistent with inferior myocardial infarction. The patient was immediately taken for primary percutaneous coronary intervention. Coronary angiography revealed 100% occlusion of the right coronary artery which was successfully stented."],
]

demo = gr.Interface(
    fn=summarize,
    inputs=gr.Textbox(
        lines=10,
        placeholder="Paste a medical article or clinical note here...",
        label="Medical Text"
    ),
    outputs=gr.Textbox(
        lines=5,
        label="Summary"
    ),
    title="Medical Text Summarizer",
    description="Summarizes medical articles and clinical notes using a fine-tuned BART model trained on PubMed data.",
    examples=examples,
    theme=gr.themes.Soft()
)

if __name__ == "__main__":
    demo.launch()