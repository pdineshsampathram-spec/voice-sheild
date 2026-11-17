#!/usr/bin/env python3
"""
export_ecapa_onnx.py — Export SpeechBrain ECAPA-TDNN to ONNX for Android.

Source model: speechbrain/spkrec-ecapa-voxceleb
License: Apache-2.0
Architecture: ECAPA-TDNN (Emphasized Channel Attention, Propagation and Aggregation)
Training data: VoxCeleb1 + VoxCeleb2
Embedding dimension: 192

This script:
1. Downloads the pretrained model from HuggingFace
2. Extracts the embedding_model component
3. Inspects the actual preprocessing pipeline to derive exact parameters
4. Exports to ONNX with dynamic axes
5. Verifies numerical parity (PyTorch vs ONNX)
6. Reports the exact input/output contract for Android implementation

Usage:
    pip install speechbrain torch torchaudio onnx onnxruntime numpy
    python3 export_ecapa_onnx.py

Output:
    ecapa_tdnn.onnx — the exported model
    ecapa_contract.json — exact I/O contract for Android
"""

import json
import os
import sys
import time
import traceback

def check_dependencies():
    """Verify all required packages are available."""
    missing = []
    for pkg in ['torch', 'torchaudio', 'speechbrain', 'onnx', 'onnxruntime', 'numpy']:
        try:
            __import__(pkg)
        except ImportError:
            missing.append(pkg)
    
    if missing:
        print(f"Missing packages: {', '.join(missing)}")
        print(f"Install with: pip install {' '.join(missing)}")
        sys.exit(1)
    
    import torch
    import onnxruntime
    print(f"PyTorch: {torch.__version__}")
    print(f"ONNX Runtime: {onnxruntime.__version__}")

def load_speechbrain_model():
    """Load the pretrained ECAPA-TDNN from SpeechBrain."""
    from speechbrain.inference.speaker import EncoderClassifier
    
    print("\n=== Loading SpeechBrain ECAPA-TDNN ===")
    print("Source: speechbrain/spkrec-ecapa-voxceleb")
    
    classifier = EncoderClassifier.from_hparams(
        source="speechbrain/spkrec-ecapa-voxceleb",
        savedir="pretrained_models/spkrec-ecapa-voxceleb"
    )
    
    return classifier

def inspect_preprocessing(classifier):
    """
    CRITICAL: Derive exact preprocessing parameters from the actual model.
    
    DO NOT hardcode parameters. Extract them from the SpeechBrain hyperparams
    to ensure the Android implementation matches exactly.
    """
    print("\n=== Inspecting Preprocessing Pipeline ===")
    
    contract = {
        "model_name": "ECAPA-TDNN",
        "model_source": "speechbrain/spkrec-ecapa-voxceleb",
        "license": "Apache-2.0",
        "embedding_dim": 192,
        "sample_rate": 16000,
    }
    
    # Extract feature extraction params from the model's modules/hparams
    compute_feat = getattr(classifier.mods, 'compute_features', None) or getattr(classifier.hparams, 'compute_features', None)
    if compute_feat is not None:
        print(f"Feature extractor type: {type(compute_feat).__name__}")
        stft = getattr(compute_feat, 'compute_STFT', None)
        fbanks = getattr(compute_feat, 'compute_fbanks', None)
        
        if stft is not None:
            if hasattr(stft, 'sample_rate'):
                contract["sample_rate"] = int(stft.sample_rate)
            if hasattr(stft, 'win_length'):
                contract["win_length_samples"] = int(stft.win_length)
                contract["win_length_ms"] = float(stft.win_length) / contract["sample_rate"] * 1000
            if hasattr(stft, 'hop_length'):
                contract["hop_length_samples"] = int(stft.hop_length)
                contract["hop_length_ms"] = float(stft.hop_length) / contract["sample_rate"] * 1000
            if hasattr(stft, 'n_fft'):
                contract["n_fft"] = int(stft.n_fft)
        
        if fbanks is not None:
            if hasattr(fbanks, 'n_mels'):
                contract["n_mels"] = int(fbanks.n_mels)
            if hasattr(fbanks, 'f_min'):
                contract["f_min"] = float(fbanks.f_min)
            if hasattr(fbanks, 'f_max'):
                contract["f_max"] = float(fbanks.f_max)
                
        print(f"  sample_rate: {contract.get('sample_rate')}")
        print(f"  n_mels: {contract.get('n_mels')}")
        print(f"  win_length_samples: {contract.get('win_length_samples')} ({contract.get('win_length_ms')}ms)")
        print(f"  hop_length_samples: {contract.get('hop_length_samples')} ({contract.get('hop_length_ms')}ms)")
        print(f"  n_fft: {contract.get('n_fft')}")
        print(f"  f_min: {contract.get('f_min')}, f_max: {contract.get('f_max')}")
    else:
        print("WARNING: compute_features not found")
    
    # Check for mean_var_norm (CMVN)
    mvn = getattr(classifier.mods, 'mean_var_norm', None) or getattr(classifier.hparams, 'mean_var_norm', None)
    if mvn is not None:
        print(f"Normalization: {type(mvn).__name__}")
        contract["normalization"] = type(mvn).__name__
        if hasattr(mvn, 'norm_type'):
            contract["norm_type"] = str(mvn.norm_type)
        if hasattr(mvn, 'std_norm'):
            contract["std_norm"] = bool(mvn.std_norm)
        print(f"  norm_type: {contract.get('norm_type')}, std_norm: {contract.get('std_norm')}")
    else:
        contract["normalization"] = "none_found"
        print("WARNING: mean_var_norm not found")
    
    return contract

def export_to_onnx(classifier, contract, output_path="ecapa_tdnn.onnx"):
    """Export the embedding model to ONNX."""
    import torch
    import numpy as np
    
    print("\n=== Exporting to ONNX ===")
    
    embedding_model = classifier.mods.embedding_model
    embedding_model.eval()
    
    # Count parameters
    total_params = sum(p.numel() for p in embedding_model.parameters())
    trainable_params = sum(p.numel() for p in embedding_model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    contract["total_parameters"] = total_params
    
    # Determine input shape
    # The embedding model takes mel features: [batch, time, n_mels]
    n_mels = contract.get("n_mels", 80)
    time_steps = 300  # ~3 seconds of features at 10ms hop
    
    # Create dummy input
    dummy_input = torch.randn(1, time_steps, n_mels)
    print(f"Dummy input shape: {dummy_input.shape} (batch=1, time={time_steps}, mels={n_mels})")
    
    # Test forward pass first
    with torch.no_grad():
        try:
            test_output = embedding_model(dummy_input)
            print(f"PyTorch output shape: {test_output.shape}")
            embedding_dim = test_output.shape[-1]
            contract["embedding_dim"] = int(embedding_dim)
            print(f"Embedding dimension: {embedding_dim}")
        except Exception as e:
            print(f"Forward pass failed: {e}")
            print("Trying alternative input format...")
            # Some versions expect [batch, n_mels, time]
            dummy_input = torch.randn(1, n_mels, time_steps)
            test_output = embedding_model(dummy_input)
            print(f"PyTorch output shape (transposed input): {test_output.shape}")
            embedding_dim = test_output.shape[-1]
            contract["embedding_dim"] = int(embedding_dim)
            contract["input_format"] = "batch_mels_time"
    
    # Record reference output for parity check
    reference_output = test_output.numpy().flatten()
    
    # Export
    print(f"\nExporting to {output_path}...")
    
    input_names = ["mel_features"]
    output_names = ["embedding"]
    
    # Determine dynamic axes based on input format
    if contract.get("input_format") == "batch_mels_time":
        dynamic_axes = {
            "mel_features": {0: "batch", 2: "time"},
            "embedding": {0: "batch"}
        }
    else:
        dynamic_axes = {
            "mel_features": {0: "batch", 1: "time"},
            "embedding": {0: "batch"}
        }
    
    torch.onnx.export(
        embedding_model,
        dummy_input,
        output_path,
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=dynamic_axes,
        opset_version=17,
        do_constant_folding=True,
        export_params=True,
        verbose=False
    )
    
    # Check file size
    file_size = os.path.getsize(output_path)
    contract["onnx_file_size_bytes"] = file_size
    contract["onnx_file_size_mb"] = round(file_size / (1024 * 1024), 2)
    print(f"ONNX file size: {contract['onnx_file_size_mb']} MB")
    
    return reference_output, dummy_input

def verify_onnx_parity(onnx_path, reference_output, dummy_input, contract):
    """Verify numerical parity between PyTorch and ONNX outputs."""
    import onnx
    import onnxruntime as ort
    import numpy as np
    
    print("\n=== Verifying ONNX Parity ===")
    
    # Validate ONNX model structure
    model = onnx.load(onnx_path)
    onnx.checker.check_model(model)
    print("ONNX model validation: PASSED")
    
    # Log graph I/O
    print("\nONNX Graph Inputs:")
    for inp in model.graph.input:
        shape = [d.dim_value if d.dim_value > 0 else d.dim_param for d in inp.type.tensor_type.shape.dim]
        print(f"  {inp.name}: shape={shape}, dtype={inp.type.tensor_type.elem_type}")
        contract["onnx_input_name"] = inp.name
        contract["onnx_input_shape"] = str(shape)
    
    print("\nONNX Graph Outputs:")
    for out in model.graph.output:
        shape = [d.dim_value if d.dim_value > 0 else d.dim_param for d in out.type.tensor_type.shape.dim]
        print(f"  {out.name}: shape={shape}, dtype={out.type.tensor_type.elem_type}")
        contract["onnx_output_name"] = out.name
        contract["onnx_output_shape"] = str(shape)
    
    # Run ONNX inference
    session = ort.InferenceSession(onnx_path)
    input_name = session.get_inputs()[0].name
    
    onnx_input = dummy_input.numpy()
    onnx_output = session.run(None, {input_name: onnx_input})[0].flatten()
    
    # Compare
    max_diff = np.max(np.abs(reference_output - onnx_output))
    mean_diff = np.mean(np.abs(reference_output - onnx_output))
    
    print(f"\nNumerical Parity:")
    print(f"  Max absolute difference: {max_diff:.8f}")
    print(f"  Mean absolute difference: {mean_diff:.8f}")
    print(f"  Parity threshold: 1e-4")
    
    contract["parity_max_diff"] = float(max_diff)
    contract["parity_mean_diff"] = float(mean_diff)
    
    if max_diff < 1e-4:
        print("  Result: ✅ PASSED — PyTorch and ONNX outputs match")
        contract["parity_verified"] = True
    elif max_diff < 1e-3:
        print("  Result: ⚠️ MARGINAL — Difference is small but above threshold")
        contract["parity_verified"] = "marginal"
    else:
        print("  Result: ❌ FAILED — Significant numerical divergence")
        contract["parity_verified"] = False
    
    # Benchmark ONNX inference time
    print("\nBenchmarking ONNX inference...")
    times = []
    for _ in range(10):
        start = time.perf_counter()
        session.run(None, {input_name: onnx_input})
        times.append((time.perf_counter() - start) * 1000)
    
    avg_ms = sum(times) / len(times)
    contract["onnx_inference_avg_ms"] = round(avg_ms, 2)
    print(f"  Average inference time (CPU, {len(times)} runs): {avg_ms:.2f} ms")
    
    return contract

def save_contract(contract, output_path="ecapa_contract.json"):
    """Save the model contract for Android implementation reference."""
    print(f"\n=== Saving Contract to {output_path} ===")
    
    with open(output_path, 'w') as f:
        json.dump(contract, f, indent=2, default=str)
    
    print(json.dumps(contract, indent=2, default=str))
    
    print(f"\nContract saved to {output_path}")
    print("\n" + "="*60)
    print("IMPORTANT: Use ecapa_contract.json as the SOLE SOURCE OF TRUTH")
    print("for Android MelFeatureExtractor and EcapaTdnnEngine parameters.")
    print("DO NOT hardcode values — read them from this contract.")
    print("="*60)

def main():
    print("="*60)
    print("ECAPA-TDNN ONNX Export for VoiceShield")
    print("="*60)
    
    check_dependencies()
    
    try:
        # Step 1: Load model
        classifier = load_speechbrain_model()
        
        # Step 2: Inspect preprocessing (derive params, don't assume)
        contract = inspect_preprocessing(classifier)
        
        # Step 3: Export to ONNX
        output_dir = os.path.dirname(os.path.abspath(__file__))
        onnx_path = os.path.join(output_dir, "ecapa_tdnn.onnx")
        contract_path = os.path.join(output_dir, "ecapa_contract.json")
        
        reference_output, dummy_input = export_to_onnx(
            classifier, contract, onnx_path
        )
        
        # Step 4: Verify parity
        contract = verify_onnx_parity(
            onnx_path, reference_output, dummy_input, contract
        )
        
        # Step 5: Save contract
        save_contract(contract, contract_path)
        
        # Summary
        print("\n" + "="*60)
        print("EXPORT COMPLETE")
        print(f"  ONNX model: {onnx_path}")
        print(f"  Contract:   {contract_path}")
        print(f"  Size:       {contract.get('onnx_file_size_mb', '?')} MB")
        print(f"  Embedding:  {contract.get('embedding_dim', '?')} dimensions")
        print(f"  Parity:     {'VERIFIED' if contract.get('parity_verified') == True else 'CHECK NEEDED'}")
        print("="*60)
        
        # Copy instructions
        print(f"\nNext step: Copy ONNX to Android assets:")
        print(f"  cp {onnx_path} mobile/app/src/main/assets/models/ecapa_tdnn.onnx")
        
    except Exception as e:
        print(f"\n❌ EXPORT FAILED: {e}")
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
