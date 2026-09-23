import json

from toy_e2e.scripts.summarize_attention_shapes import summarize


def test_summarize_groups_exact_attention_shapes_per_forward(tmp_path):
    shape_dir = tmp_path / "c16" / "decode"
    shape_dir.mkdir(parents=True)
    shapes = [
        {
            "family": "attention",
            "mode": "mla_decode_projected_value",
            "kernel_name": "composed_decode_project_value_fallback",
            "dtype": "torch.float8_e4m3fn",
            "shape_params": {
                "batch_size": 16,
                "num_q_heads": 12,
                "kv_lora_rank": 512,
                "value_head_dim": 128,
                "gate_kind": "sigmoid",
            },
            "timestamp_ns": 1,
        },
        {
            "family": "attention",
            "mode": "mla_project_value",
            "kernel_name": "torch_bmm_sigmoid_mul_fallback",
            "dtype": "torch.bfloat16",
            "shape_params": {
                "batch_size": 16,
                "num_heads": 12,
                "latent_dim": 512,
                "value_dim": 128,
                "gate_kind": "sigmoid",
            },
            "timestamp_ns": 2,
        },
        {
            "family": "attn_res",
            "mode": "fwd",
            "kernel_name": "gluon_attn_res_fwd_gfx1250",
            "dtype": "torch.bfloat16",
            "shape_params": {
                "tokens": 16,
                "hidden_size": 7168,
                "valid_blocks": 6,
            },
            "timestamp_ns": 3,
        },
        {
            "family": "gemm",
            "mode": "mm",
            "kernel_name": "unrelated",
            "dtype": "torch.bfloat16",
            "shape_params": {},
            "timestamp_ns": 4,
        },
    ]
    shapes_path = shape_dir / "decode.shapes.json"
    shapes_path.write_text(json.dumps(shapes * 2), encoding="utf-8")
    manifest = {
        "runs": [
            {
                "concurrency": 16,
                "prefill": {
                    "forward_count": 1,
                    "shapes": str(tmp_path / "c16" / "prefill" / "prefill.shapes.json"),
                },
                "decode": {
                    "forward_count": 2,
                    "shapes": "/relocated/decode.shapes.json",
                },
            }
        ]
    }
    prefill_dir = tmp_path / "c16" / "prefill"
    prefill_dir.mkdir()
    (prefill_dir / "prefill.shapes.json").write_text("[]", encoding="utf-8")
    manifest_path = tmp_path / "profile_manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    result = summarize(manifest_path)

    decode = result["profiles"][1]
    assert decode["name"] == "c16_decode"
    assert decode["components"]["mla.decode_projected_value"]["calls_per_forward"] == 1
    assert decode["components"]["mla.value_projection_gate"]["calls_per_forward"] == 1
    assert decode["components"]["attn_res"]["calls_per_forward"] == 1
    assert len(decode["fallbacks"]) == 2
    assert {call["kernel_name"] for call in decode["calls"]} == {
        "composed_decode_project_value_fallback",
        "torch_bmm_sigmoid_mul_fallback",
        "gluon_attn_res_fwd_gfx1250",
    }


def test_summarize_skips_disabled_stage(tmp_path):
    shape_dir = tmp_path / "c16" / "prefill"
    shape_dir.mkdir(parents=True)
    shapes_path = shape_dir / "prefill.shapes.json"
    shapes_path.write_text("[]", encoding="utf-8")
    manifest_path = tmp_path / "profile_manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "runs": [
                    {
                        "concurrency": 16,
                        "prefill": {
                            "forward_count": 8,
                            "shapes": str(shapes_path),
                        },
                        "decode": {
                            "forward_count": 0,
                            "shapes": None,
                        },
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    result = summarize(manifest_path)

    assert [profile["name"] for profile in result["profiles"]] == ["c16_prefill"]
