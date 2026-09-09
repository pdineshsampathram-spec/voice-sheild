#!/usr/bin/env python3
"""
Export AASIST-L (Anti-Spoofing) PyTorch model to ONNX for Android on-device inference.

Model Provenance:
- Source: SpeechAntiSpoofingBenchmarks/AASIST-L (Kirill Borodin)
- Base Paper: Jung et al., ICASSP 2022 (AASIST: Audio Anti-Spoofing using Integrated Spectro-Temporal Graph Attention Networks)
- Parameters: 85,306 (~0.085M)
- Architecture: RawNet2 residual sinc-convolution + heterogeneous spectro-temporal graph attention
- Checkpoint: ASVspoof2019 LA pretrained weights
- License: MIT License

Input Tensor Contract:
- Shape: [batch_size, 64600]
- Type: float32
- Sample Rate: 16,000 Hz mono PCM
- Duration: 4.0375 seconds (64,600 samples)

Output Tensor Contract:
- Shape: [batch_size, 2]
- Type: float32
- Semantics: [bona_fide_logit, spoof_logit]
"""

import os
import sys
import numpy as np
import torch

# Ensure ml/models is in path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "models"))
from _net import Model as AASISTNet

_D_ARGS = {
    "architecture": "AASIST",
    "nb_samp": 64600,
    "first_conv": 128,
    "filts": [70, [1, 32], [32, 32], [32, 24], [24, 24]],
    "gat_dims": [24, 32],
    "pool_ratios": [0.4, 0.5, 0.7, 0.5],
    "temperatures": [2.0, 2.0, 100.0, 100.0],
}


def export_onnx(output_path="ml/models/aasist_l.onnx", ckpt_path="ml/models/AASIST-L.pth"):
    print(f"Loading AASIST-L checkpoint from: {ckpt_path}")
    net = AASISTNet(_D_ARGS)
    sd = torch.load(ckpt_path, map_location="cpu")
    if isinstance(sd, dict) and "state_dict" in sd:
        sd = sd["state_dict"]
    net.load_state_dict(sd, strict=True)
    net.eval()
    print("PyTorch model loaded successfully. Total parameters:", sum(p.numel() for p in net.parameters()))

    # Dummy input: 1 audio sample of 64,600 samples
    dummy_input = torch.randn(1, 64600, dtype=torch.float32)

    with torch.no_grad():
        pt_out = net(dummy_input)
        # Note: AASIST net returns (last_hidden, output_logits) or output_logits
        if isinstance(pt_out, tuple):
            print(f"Model returned tuple of {len(pt_out)} items. Selecting logits output.")
            # Wrapper module to return only logits
            class LogitsWrapper(torch.nn.Module):
                def __init__(self, model):
                    super().__init__()
                    self.model = model

                def forward(self, x):
                    res = self.model(x)
                    return res[1] if isinstance(res, tuple) else res

            export_model = LogitsWrapper(net)
            pt_logits = export_model(dummy_input)
        else:
            export_model = net
            pt_logits = pt_out

    print(f"PyTorch logits shape: {pt_logits.shape}, sample: {pt_logits.numpy()}")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    print(f"Exporting to ONNX at: {output_path}")

    torch.onnx.export(
        export_model,
        dummy_input,
        output_path,
        export_params=True,
        opset_version=17,
        do_constant_folding=True,
        input_names=["audio"],
        output_names=["logits"],
        dynamo=False,
        dynamic_axes={
            "audio": {0: "batch_size"},
            "logits": {0: "batch_size"},
        },
    )

    print(f"ONNX export completed: {output_path} ({os.path.getsize(output_path) / 1024:.1f} KB)")

    # Verify with onnxruntime
    try:
        import onnxruntime as ort
        session = ort.InferenceSession(output_path, providers=["CPUExecutionProvider"])
        ort_inputs = {"audio": dummy_input.numpy()}
        ort_out = session.run(None, ort_inputs)[0]
        diff = np.max(np.abs(pt_logits.numpy() - ort_out))
        print(f"ONNX Runtime verification passed! Max difference vs PyTorch: {diff:.6f}")
    except ImportError:
        print("onnxruntime not installed yet, skipping ONNX Runtime verification.")


if __name__ == "__main__":
    export_onnx()
