"""Exact code-digest migrations that preserve compatible interrupted runs."""

from __future__ import annotations

from typing import Final

CHECKPOINT_RETENTION_MIGRATIONS: Final = (
    {
        # Stop requests are handled only after a durable checkpoint is saved.
        # Model math, optimizer, sampler, RNG and persisted state are unchanged.
        "id": "saved-boundary-runtime-stop-v1",
        "from_files": {
            "training.py": (
                "c7d81ff32225908347a3b0450023b1cdbcc90b8aa7b1e589569b9f56cb98f448"
            ),
        },
        "to_files": {
            "training.py": (
                "65cf7aade6258e2e4ed58842bffc3010696c9f4551c8907efc5dbb91f555e06f"
            ),
        },
    },
    {
        # Prevalidated worker metadata and device constants preserve values, gradients,
        # and device RNG order. Nonpersistent buffers add no checkpoint state keys.
        "id": "asynchronous-model-hotpath-v1",
        "from_files": {
            "training.py": (
                "800e2994473534506ce0f234b613f2c7a4a31614ddb446a739d7fe8da17404d2"
            ),
            "models/forecast.py": (
                "fdcd685524e1fe553bff3c6ab5a528f0a19519b92f701723e1e38e170b5dfa6c"
            ),
            "models/quant.py": (
                "63cbccf1900a5cbf54eadaa480aa5fa2c8e87670947863997027f9876156c558"
            ),
            "models/ranking.py": (
                "4bd617205bf73bb342cbeb3a1d065a3ef18ad1b22d1a061b99baee493c430cc1"
            ),
            "models/scale_features.py": (
                "4332f347af42fe571a505319588a7037fc694ea978adb954bc4ab8f854eebc89"
            ),
        },
        "to_files": {
            "training.py": (
                "c7d81ff32225908347a3b0450023b1cdbcc90b8aa7b1e589569b9f56cb98f448"
            ),
            "models/forecast.py": (
                "7054dfcc0024757aab452bf91132f6bfb814845d14aceec5c85ec3c153a13e3a"
            ),
            "models/quant.py": (
                "5094d0cf9d0fe2c6e9f495a0a052f8a792ce45ffdcc75ed5ea4006da4f23c274"
            ),
            "models/ranking.py": (
                "b20e6c04d6c0f4ae5c8a597505350e92c7f1bc485bf2ee9526a16bf41d1d0ee0"
            ),
            "models/scale_features.py": (
                "ac70243985305e2b522f5a21f1c68fbf46aa8fb49f90a7e568d0c4e740f894c5"
            ),
        },
    },
    {
        # Fresh initialization ignores calibration-cache warmth. Resumed parameters,
        # optimizer state and RNG are still restored from the unchanged checkpoint.
        "id": "cache-independent-model-initialization-v1",
        "from_files": {
            "training.py": (
                "18c947d1754969a14c0a4f4ce0a4b5fac1b216c590d2b89f96f81a4b37c7cb76"
            ),
            "models/forecast.py": (
                "fdcd685524e1fe553bff3c6ab5a528f0a19519b92f701723e1e38e170b5dfa6c"
            ),
            "models/quant.py": (
                "63cbccf1900a5cbf54eadaa480aa5fa2c8e87670947863997027f9876156c558"
            ),
            "models/ranking.py": (
                "4bd617205bf73bb342cbeb3a1d065a3ef18ad1b22d1a061b99baee493c430cc1"
            ),
            "models/scale_features.py": (
                "4332f347af42fe571a505319588a7037fc694ea978adb954bc4ab8f854eebc89"
            ),
        },
        "to_files": {
            "training.py": (
                "c7d81ff32225908347a3b0450023b1cdbcc90b8aa7b1e589569b9f56cb98f448"
            ),
            "models/forecast.py": (
                "7054dfcc0024757aab452bf91132f6bfb814845d14aceec5c85ec3c153a13e3a"
            ),
            "models/quant.py": (
                "5094d0cf9d0fe2c6e9f495a0a052f8a792ce45ffdcc75ed5ea4006da4f23c274"
            ),
            "models/ranking.py": (
                "b20e6c04d6c0f4ae5c8a597505350e92c7f1bc485bf2ee9526a16bf41d1d0ee0"
            ),
            "models/scale_features.py": (
                "ac70243985305e2b522f5a21f1c68fbf46aa8fb49f90a7e568d0c4e740f894c5"
            ),
        },
    },
    {
        # Execution-only: identical windows, loss, optimizer and model parameters.
        # Probe updates are rolled back; existing checkpoint progress/RNG is loaded
        # afterwards. All other source/config/data fingerprints must still match.
        "id": "end-to-end-pipeline-probe-v2",
        "from_files": {
            "training.py": (
                "b34e527e81fc8a09fcc668b667a1a123a8501936603266c23d5163a65b387b56"
            ),
            "models/forecast.py": (
                "fdcd685524e1fe553bff3c6ab5a528f0a19519b92f701723e1e38e170b5dfa6c"
            ),
            "models/quant.py": (
                "63cbccf1900a5cbf54eadaa480aa5fa2c8e87670947863997027f9876156c558"
            ),
            "models/ranking.py": (
                "4bd617205bf73bb342cbeb3a1d065a3ef18ad1b22d1a061b99baee493c430cc1"
            ),
            "models/scale_features.py": (
                "4332f347af42fe571a505319588a7037fc694ea978adb954bc4ab8f854eebc89"
            ),
        },
        "to_files": {
            "training.py": (
                "c7d81ff32225908347a3b0450023b1cdbcc90b8aa7b1e589569b9f56cb98f448"
            ),
            "models/forecast.py": (
                "7054dfcc0024757aab452bf91132f6bfb814845d14aceec5c85ec3c153a13e3a"
            ),
            "models/quant.py": (
                "5094d0cf9d0fe2c6e9f495a0a052f8a792ce45ffdcc75ed5ea4006da4f23c274"
            ),
            "models/ranking.py": (
                "b20e6c04d6c0f4ae5c8a597505350e92c7f1bc485bf2ee9526a16bf41d1d0ee0"
            ),
            "models/scale_features.py": (
                "ac70243985305e2b522f5a21f1c68fbf46aa8fb49f90a7e568d0c4e740f894c5"
            ),
        },
    },
    {
        # Only provenance lookup changes; all data/model/optimizer semantics must still match.
        "id": "dataset-scoped-provenance-v1",
        "from_files": {
            "tracking.py": (
                "baee48c80779374c8e1cdc215540b9d3f06087ceaeef8a626d01ce799d2d3dea"
            ),
        },
        "to_files": {
            "tracking.py": (
                "29fd76b6818c6b0d11efb13298e1cde25feaca4cbe6fc5bfc9e811345e9d129e"
            ),
        },
    },
    {
        "id": "adaptive-gpu-resource-planning-v1",
        "from_files": {
            "training.py": (
                "45d495828451462edb0e059c8e0d67700f104cbe0c6b421e4afaa92b5533017a"
            ),
        },
        "to_files": {
            "training.py": (
                "8c733eacca16cef30ada2ee11dfe8a47500a9a67c8075e1844ae0406f9dd3009"
            ),
        },
    },
    {
        "id": "best-five-plus-adaptive-gpu-resource-planning-v1",
        "from_files": {
            "checkpointing.py": (
                "5e73422fe8686366220dc7d07159c39b54228b3913f86ee759b0761d554c6b46"
            ),
            "run_contract.py": (
                "d54028f40a8e1eb5e8a86cd02c3855704c3301fd10c118d2d3e7645e53870ab8"
            ),
            "training.py": (
                "45d495828451462edb0e059c8e0d67700f104cbe0c6b421e4afaa92b5533017a"
            ),
        },
        "to_files": {
            "checkpointing.py": (
                "5c58845e417dcf3aed6381f9136048ee37535586c9cd63617b8b9d0ec45a2194"
            ),
            "run_contract.py": (
                "2b41ead1d3a74040419d393374755aee137d5fbe68a49da88eb7fbe9a1773044"
            ),
            "training.py": (
                "8c733eacca16cef30ada2ee11dfe8a47500a9a67c8075e1844ae0406f9dd3009"
            ),
        },
    },
    {
        "id": "hardware-portable-runtime-plan-v2",
        "from_files": {
            "training.py": (
                "d54aaa84c5adda60051cb3574baa54e4009211629198f540c99ae11b83173437"
            ),
        },
        "to_files": {
            "training.py": (
                "8c733eacca16cef30ada2ee11dfe8a47500a9a67c8075e1844ae0406f9dd3009"
            ),
        },
    },
    {
        "id": "adaptive-gpu-batch-probe-binding-v1",
        "from_files": {
            "training.py": (
                "a2521d3b5389c345e2e956e441f7806840e5b721c0c73be3ed1550a6bba4c1ff"
            ),
        },
        "to_files": {
            "training.py": (
                "8c733eacca16cef30ada2ee11dfe8a47500a9a67c8075e1844ae0406f9dd3009"
            ),
        },
    },
    {
        "id": "adaptive-gpu-import-order-v1",
        "from_files": {
            "training.py": (
                "75277a0a7271c8557daae4544b55f65d12c5700b84039634fa38b283054318d1"
            ),
        },
        "to_files": {
            "training.py": (
                "8c733eacca16cef30ada2ee11dfe8a47500a9a67c8075e1844ae0406f9dd3009"
            ),
        },
    },
)
