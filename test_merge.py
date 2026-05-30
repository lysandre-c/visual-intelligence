import torch
from transformers import pipeline, AutoProcessor
from peft import PeftModel
from pathlib import Path

def test():
    hf_model_id = "llava-hf/llava-1.5-7b-hf"
    adapter_path = "results/rl_alignment/final"
    
    # Base model via pipeline
    pipe_base = pipeline("image-text-to-text", model=hf_model_id, device_map="auto", model_kwargs={"torch_dtype": torch.float16})
    
    # Adapter model via PeftModel (as in vlm.py)
    pipe_dpo = pipeline("image-text-to-text", model=hf_model_id, device_map="auto", model_kwargs={"torch_dtype": torch.float16})
    
    projector_path = Path(adapter_path) / "multi_modal_projector.pt"
    if projector_path.exists():
        print("Loading projector...")
        projector_state = torch.load(str(projector_path), map_location="cpu", weights_only=True)
        pipe_dpo.model.model.multi_modal_projector.load_state_dict(projector_state)
    
    pipe_dpo.model = PeftModel.from_pretrained(pipe_dpo.model, adapter_path)
    
    # Test generation
    from PIL import Image
    # create a dummy image
    img = Image.new("RGB", (224, 224), color="white")
    prompt = "USER: <image>\nWhich line is longer?\nASSISTANT:"
    
    out_base = pipe_base(img, text=prompt, generate_kwargs={"max_new_tokens": 20})
    out_dpo = pipe_dpo(img, text=prompt, generate_kwargs={"max_new_tokens": 20})
    
    print("Base:", out_base)
    print("DPO:", out_dpo)

    # Now with merge and unload
    pipe_merged = pipeline("image-text-to-text", model=hf_model_id, device_map="auto", model_kwargs={"torch_dtype": torch.float16})
    if projector_path.exists():
        projector_state = torch.load(str(projector_path), map_location="cpu", weights_only=True)
        pipe_merged.model.model.multi_modal_projector.load_state_dict(projector_state)
    pipe_merged.model = PeftModel.from_pretrained(pipe_merged.model, adapter_path).merge_and_unload()
    out_merged = pipe_merged(img, text=prompt, generate_kwargs={"max_new_tokens": 20})
    
    print("Merged:", out_merged)

if __name__ == "__main__":
    test()
