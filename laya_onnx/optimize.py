"""Offline ONNX graph optimization for a PC backend (Intel CPU / 16 GB).

Order of work:
1. Shape inference
2. ONNX Runtime graph fusions (ORT_ENABLE_ALL) written back to disk
3. Optional dynamic INT8 on MatMul/Gemm (best CPU win)
4. Optional FP16 weights (useful on CUDA, usually slower on CPU)
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path


def _find_graph(bundle: Path) -> Path:
    for name in ("laya.onnx", "model.onnx"):
        path = bundle / name if bundle.is_dir() else bundle
        if bundle.is_file() and bundle.suffix == ".onnx":
            return bundle
        if path.is_file():
            return path
    raise FileNotFoundError(f"No ONNX graph under {bundle}")


def infer_shapes(graph: Path) -> Path:
    import onnx

    model = onnx.load(str(graph), load_external_data=True)
    try:
        model = onnx.shape_inference.infer_shapes(model)
        onnx.save(model, str(graph))
    except Exception:
        pass
    return graph


def fuse_graph(graph: Path, *, threads: int | None = None) -> Path:
    import onnxruntime as ort

    tmp = graph.with_suffix(".opt.onnx")
    opts = ort.SessionOptions()
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    opts.optimized_model_filepath = str(tmp)
    if threads:
        opts.intra_op_num_threads = int(threads)
        opts.inter_op_num_threads = 1
    sess = ort.InferenceSession(str(graph), sess_options=opts, providers=["CPUExecutionProvider"])
    del sess
    if tmp.is_file() and tmp.stat().st_size > 0:
        tmp.replace(graph)
        extra = Path(str(tmp) + ".data")
        dest_extra = Path(str(graph) + ".data")
        if extra.is_file():
            extra.replace(dest_extra)
    return graph


def quantize_dynamic_int8(graph: Path) -> Path:
    from onnxruntime.quantization import QuantType, quantize_dynamic

    out = graph.with_suffix(".int8.onnx")
    quantize_dynamic(
        str(graph),
        str(out),
        weight_type=QuantType.QInt8,
        extra_options={"WeightSymmetric": True, "EnableSubgraph": True},
    )
    out.replace(graph)
    return graph


def convert_weights_fp16(graph: Path) -> Path:
    import numpy as np
    import onnx
    from onnx import TensorProto, numpy_helper

    model = onnx.load(str(graph), load_external_data=True)
    for init in model.graph.initializer:
        if init.data_type != TensorProto.FLOAT:
            continue
        array = numpy_helper.to_array(init).astype(np.float16)
        new = numpy_helper.from_array(array, init.name)
        init.CopyFrom(new)
    onnx.save(model, str(graph))
    return graph


def optimize_bundle(source, output=None, *, precision="int8", threads=None, fuse=True):
    source = Path(source).expanduser()
    src_graph = _find_graph(source)
    if output is None:
        dest_graph = src_graph
        dest_root = src_graph.parent
    else:
        output = Path(output).expanduser()
        if output.suffix == ".onnx":
            dest_graph = output
            dest_root = output.parent
            dest_root.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src_graph, dest_graph)
        else:
            if output.exists() and output.resolve() != source.resolve():
                raise FileExistsError(f"Output already exists: {output}")
            output.mkdir(parents=True, exist_ok=True)
            if source.is_dir() and output.resolve() != source.resolve():
                for item in source.iterdir():
                    target = output / item.name
                    if item.is_dir():
                        shutil.copytree(item, target, dirs_exist_ok=True)
                    else:
                        shutil.copy2(item, target)
            dest_graph = output / src_graph.name
            if not dest_graph.is_file():
                shutil.copy2(src_graph, dest_graph)
            dest_root = output

    infer_shapes(dest_graph)
    if fuse:
        fuse_graph(dest_graph, threads=threads)
    if precision == "int8":
        quantize_dynamic_int8(dest_graph)
        infer_shapes(dest_graph)
        if fuse:
            fuse_graph(dest_graph, threads=threads)
    elif precision == "fp16":
        convert_weights_fp16(dest_graph)

    manifest = dest_root / "onnx_config.json"
    payload = {}
    if manifest.is_file():
        try:
            payload = json.loads(manifest.read_text())
        except json.JSONDecodeError:
            payload = {}
    payload.update({"format": payload.get("format", "laya-onnx"), "optimized": True, "precision": precision, "graph": dest_graph.name})
    manifest.write_text(json.dumps(payload, indent=2) + "\n")
    return dest_graph
