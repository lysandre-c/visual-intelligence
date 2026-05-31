#!/usr/bin/env python3
import json
from pathlib import Path
from PIL import Image

# Import the protocol and model
from src.probing.vlm_protocol import VLMProtocol
from src.models.vlm import LLaVAProber
from src.stimuli.base import StimulusPair

def load_cached_results(results_path: Path):
    with open(results_path) as f:
        return {r["stimulus_id"]: r for r in json.load(f)}

def test():
    adapter_path = "results/rl_alignment/checkpoint-800"
    manifest_muller = "data/stimuli/geometric/muller_lyer/manifest.json"
    manifest_color = "data/stimuli/color/simultaneous_contrast/manifest.json"
    
    # Load 5 items from each
    with open(manifest_muller) as f:
        muller_items = json.load(f)[:5]
    with open(manifest_color) as f:
        color_items = json.load(f)[:5]
        
    all_items = muller_items + color_items
    
    # Initialize Prober and Protocol
    print("Initializing LLaVAProber with symDPO...")
    prober = LLaVAProber(
        hf_model_id="llava-hf/llava-1.5-7b-hf",
        adapter_path=adapter_path,
        model_name="llava_symDPO",
        device="cuda"
    )
    protocol = VLMProtocol(prober)
    
    # Load cached llava_1.5 results
    print("Loading baseline cached results...")
    muller_baseline = load_cached_results(Path("results/full/llava_1.5_muller_lyer_results.json"))
    color_baseline = load_cached_results(Path("results/full/llava_1.5_simultaneous_contrast_results.json"))
    baselines = {**muller_baseline, **color_baseline}
    
    print("\n--- Running evaluation ---")
    differences = 0
    for item in all_items:
        sid = item["stimulus_id"]
        illusion_img = Image.open(item["illusion_path"]).convert("RGB")
        control_img = Image.open(item["control_path"]).convert("RGB")
        
        # probe
        dist = protocol.probe_stimulus(
            illusion=illusion_img,
            control=control_img,
            category=item["category"],
            illusion_type=item["illusion_type"],
            correct_answer=item["correct_answer"],
            illusory_answer=item["illusory_answer"],
            stimulus_id=sid,
        )
        
        predicted = dist.to_label()
        baseline_record = baselines.get(sid)
        baseline_pred = baseline_record["predicted_label"] if baseline_record else "UNKNOWN"
        
        print(f"[{sid}] Base: {baseline_pred:10s} | symDPO: {predicted:10s} | {'SAME' if predicted == baseline_pred else 'DIFF'}")
        if predicted != baseline_pred:
            differences += 1
            
    print(f"\nTotal differences found: {differences} / {len(all_items)}")

if __name__ == "__main__":
    test()
