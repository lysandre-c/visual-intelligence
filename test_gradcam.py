import torch
import json
import numpy as np
from PIL import Image
from transformers import AutoProcessor, LlavaForConditionalGeneration
from peft import PeftModel
from src.analysis.vlm_saliency import _resolve_vision_tower, _resolve_device

def test():
    hf_model_id = "llava-hf/llava-1.5-7b-hf"
    adapter_path = "results/rl_alignment/checkpoint-900"
    
    print("Loading processor...")
    processor = AutoProcessor.from_pretrained(hf_model_id)
    
    print("Loading base model...")
    base_model = LlavaForConditionalGeneration.from_pretrained(
        hf_model_id,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    
    print("Loading PEFT model...")
    dpo_model = PeftModel.from_pretrained(base_model, adapter_path)
    dpo_model.eval()
    
    # Create fake image and prompt
    image = Image.new('RGB', (336, 336), color='white')
    prompt = "USER: <image>\nWhich line is longer?\nOptions:\nA) Top\nB) Bottom\nASSISTANT:"
    
    inputs = processor(text=prompt, images=image, return_tensors="pt").to(dpo_model.device)
    if "pixel_values" in inputs:
        inputs["pixel_values"].requires_grad_(True)
        print("Set pixel_values.requires_grad = True")
        
    vision_tower = _resolve_vision_tower(dpo_model)
    if hasattr(vision_tower, "vision_model"):
        vision_model = vision_tower.vision_model
    else:
        vision_model = vision_tower
    target_layer = vision_model.encoder.layers[-2]
    
    activations = []
    gradients = []
    
    def forward_hook(module, input, output):
        print(f"Forward hook called! Output shape: {output[0].shape}")
        activations.append(output[0])
        
    def backward_hook(module, grad_input, grad_output):
        print(f"Backward hook called! Grad output shape: {grad_output[0].shape}")
        gradients.append(grad_output[0])
        
    h1 = target_layer.register_forward_hook(forward_hook)
    h2 = target_layer.register_full_backward_hook(backward_hook)
    
    print("Running forward...")
    outputs = dpo_model(**inputs, output_hidden_states=True)
    logits = outputs.logits
    next_token_logits = logits[0, -1, :]
    target_token_id = processor.tokenizer.encode(" A", add_special_tokens=False)[-1]
    score = next_token_logits[target_token_id]
    
    print("Running backward...")
    dpo_model.zero_grad()
    score.backward(retain_graph=True)
    
    h1.remove()
    h2.remove()
    
    print(f"Activations captured: {len(activations)}")
    print(f"Gradients captured: {len(gradients)}")

test()
