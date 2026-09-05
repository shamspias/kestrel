"""Batch-1 TensorRT FP16 latency for KESTREL checkpoints and ultralytics baselines on the local GPU.

KESTREL: fold the re-parameterisable convs, wrap in ExportWrapper (fixed depth, closed vocabulary), export ONNX,
build a TensorRT engine (FP16) and time it. One engine per decoder depth 1..L gives the static-depth latency ladder
that the anytime curve is plotted against. The measured graph contains no NMS.
YOLO: ultralytics ONNX export, then the same TensorRT builder and runner. For NMS-based YOLOs the graph excludes NMS
(ultralytics convention); for YOLO26 / YOLOv10 end-to-end graphs nothing is excluded.

Usage:
  python scripts/trt_latency.py kestrel --ckpt runs/kestrel_n/best.pt --size 512 --out runs/kestrel_n/trt.json
  python scripts/trt_latency.py yolo --ckpt runs/baselines/yolo26n_scratch/weights/best.pt --size 512 --out runs/baselines/yolo26n_scratch.trt.json
  python scripts/trt_latency.py kestrel --preset N --size 512      # random-init pipeline check
Every measurement must be taken on an otherwise idle GPU."""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("YOLO_AUTOINSTALL", "False")          # ultralytics must not pip-install into the venv (it once replaced torch)

TRT_LOGGER = None


def to_fp16_onnx(onnx_path: str, block_ops=()) -> str:
    """TensorRT 11 networks are strongly typed (no FP16 builder flag): precision is whatever the ONNX graph says.
    Convert weights/activations to fp16 (inputs and outputs stay fp32; a few numerically delicate ops stay fp32)."""
    import onnx
    from onnxconverter_common import float16
    m = onnx.load(onnx_path)
    block = list(float16.DEFAULT_OP_BLOCK_LIST) + ["RoiAlign", "TopK", "Softmax", "Pow", "ReduceMean", "ReduceL2", "Sqrt", "Exp", "Softplus", "Erf", "Sin", "Cos", "Range"] + list(block_ops)
    m16 = float16.convert_float_to_float16(m, keep_io_types=True, op_block_list=block, disable_shape_infer=True)
    out = onnx_path.replace(".onnx", "_fp16.onnx")
    onnx.save(m16, out)
    return out


def build_engine(onnx_path: str, engine_path: str, fp16: bool = True, workspace_gb: float = 1.0) -> str:
    import tensorrt as trt
    global TRT_LOGGER
    TRT_LOGGER = TRT_LOGGER or trt.Logger(trt.Logger.WARNING)
    trt.init_libnvinfer_plugins(TRT_LOGGER, "")                  # RoiAlign is imported through the ROIAlign_TRT plugin
    if fp16 == "convert":                                       # optional post-hoc conversion (fragile with strongly typed networks)
        onnx_path = to_fp16_onnx(onnx_path)
    builder = trt.Builder(TRT_LOGGER)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED))
    parser = trt.OnnxParser(network, TRT_LOGGER)
    with open(onnx_path, "rb") as f:
        if not parser.parse(f.read()):
            raise RuntimeError("\n".join(str(parser.get_error(i)) for i in range(parser.num_errors)))
    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, int(workspace_gb * (1 << 30)))
    t0 = time.time()
    plan = builder.build_serialized_network(network, config)
    if plan is None:
        raise RuntimeError("TensorRT build failed")
    with open(engine_path, "wb") as f:
        f.write(plan)
    print(f"  engine built in {time.time() - t0:.0f}s → {engine_path} ({os.path.getsize(engine_path) / 1e6:.1f} MB)")
    return engine_path


class Engine:
    """Minimal TensorRT runner on torch CUDA tensors (handles ultralytics' metadata-prefixed .engine files)."""

    def __init__(self, path: str):
        import tensorrt as trt
        global TRT_LOGGER
        TRT_LOGGER = TRT_LOGGER or trt.Logger(trt.Logger.WARNING)
        trt.init_libnvinfer_plugins(TRT_LOGGER, "")
        with open(path, "rb") as f:
            blob = f.read()
        runtime = trt.Runtime(TRT_LOGGER)
        self.engine = runtime.deserialize_cuda_engine(blob)
        if self.engine is None:                                   # ultralytics: 4-byte little-endian metadata length + JSON + engine
            n = int.from_bytes(blob[:4], "little")
            self.meta = json.loads(blob[4:4 + n].decode())
            self.engine = runtime.deserialize_cuda_engine(blob[4 + n:])
        self.ctx = self.engine.create_execution_context()
        self.io = {}
        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            shape = tuple(self.engine.get_tensor_shape(name))
            if -1 in shape:                                      # dynamic batch: fix to 1
                shape = tuple(1 if s == -1 else s for s in shape)
                if self.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
                    self.ctx.set_input_shape(name, shape)
            dtype = {trt.float32: torch.float32, trt.float16: torch.float16, trt.int32: torch.int32, trt.int64: torch.int64,
                     trt.bool: torch.bool}[self.engine.get_tensor_dtype(name)]
            self.io[name] = torch.empty(shape, dtype=dtype, device="cuda")
            self.ctx.set_tensor_address(name, self.io[name].data_ptr())
        self.inputs = [self.engine.get_tensor_name(i) for i in range(self.engine.num_io_tensors)
                       if self.engine.get_tensor_mode(self.engine.get_tensor_name(i)) == trt.TensorIOMode.INPUT]
        self.stream = torch.cuda.Stream()

    def __call__(self, x: torch.Tensor | None = None):
        if x is not None:
            self.io[self.inputs[0]].copy_(x.to(self.io[self.inputs[0]].dtype))
        self.ctx.execute_async_v3(self.stream.cuda_stream)
        self.stream.synchronize()
        return {k: v for k, v in self.io.items() if k not in self.inputs}

    def capture_graph(self):
        """Capture one engine execution into a CUDA graph (the standard trtexec --useCudaGraph deployment mode);
        replay removes per-kernel launch overhead, which dominates for small graphs with many nodes."""
        self.ctx.execute_async_v3(self.stream.cuda_stream); self.stream.synchronize()   # warm-up on the capture stream
        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph, stream=self.stream):
            self.ctx.execute_async_v3(self.stream.cuda_stream)
        return self

    def replay(self):
        self.graph.replay()
        torch.cuda.synchronize()


def time_fn(fn, n: int = 200, warm: int = 30) -> dict:
    for _ in range(warm):
        fn()
    torch.cuda.synchronize()
    ts = []
    for _ in range(n):
        t0 = time.perf_counter(); fn(); torch.cuda.synchronize(); ts.append(1000 * (time.perf_counter() - t0))
    ts = np.array(ts)
    return dict(ms_mean=float(ts.mean()), ms_median=float(np.median(ts)), ms_p90=float(np.percentile(ts, 90)), n=n)


def kestrel(a):
    from evaluate import build_model, load_checkpoint
    from kestrel import ExportWrapper, count_params
    dev = torch.device("cuda")
    if a.ckpt:
        model, ck = load_checkpoint(a.ckpt, dev)
        tag = os.path.dirname(a.ckpt)
    else:
        model, ck = build_model(a.preset, 20).to(dev).eval(), {}
        tag = a.out_dir or "runs/trt_check"
    os.makedirs(tag, exist_ok=True)
    model.reparameterize()
    L = model.cfg.dec_layers
    out = dict(ckpt=a.ckpt, size=a.size, params_M=count_params(model) / 1e6, gpu=torch.cuda.get_device_name(0), trt={}, torch_fp16={}, torch_fp32={})
    x = torch.randn(1, 3, a.size, a.size, device=dev)
    depths = list(range(1, L + 1)) if a.all_depths else [L]
    for l in depths:
        w = ExportWrapper(model, max_layers=l).eval()
        onnx_path, eng_path = f"{tag}/kestrel_L{l}_{a.size}.onnx", f"{tag}/kestrel_L{l}_{a.size}.engine"
        with torch.no_grad():
            rb, rs = w(x)
            # Trace under autocast so the ONNX graph carries PyTorch's own mixed-precision casts (fp16 matmul/conv,
            # fp32 norms/softmax/reductions): TensorRT 11 networks are strongly typed and take precision from the graph.
            # Constant folding is off because the TorchScript exporter's folder chokes on mixed cpu/cuda constants.
            with torch.autocast("cuda", dtype=torch.float16, enabled=not a.fp32):
                torch.onnx.export(w, x, onnx_path, opset_version=17, input_names=["images"], output_names=["boxes", "scores"], dynamo=False, do_constant_folding=False)
        if not os.path.exists(eng_path) or a.rebuild:
            build_engine(onnx_path, eng_path, fp16=not a.fp32, workspace_gb=a.workspace)
        eng = Engine(eng_path)
        o = eng(x)
        # permutation-invariant parity: the top-K query selection can reorder under fp16, so match each eager query to its
        # nearest engine query (L1 over box+scores) and report the worst match
        ref = torch.cat([rb[0], rs[0] * a.size], 1); got = torch.cat([o["boxes"][0].float(), o["scores"][0].float() * a.size], 1)
        j = torch.cdist(ref, got, p=1).argmin(1)
        db = (rb[0] - o["boxes"][0, j].float()).abs().max().item(); ds = (rs[0] - o["scores"][0, j].float()).abs().max().item()
        r = time_fn(lambda: eng(None), a.n)
        r.update(max_abs_diff_boxes=db, max_abs_diff_scores=ds, fp16=not a.fp32)
        try:
            eng.capture_graph(); rg = time_fn(eng.replay, a.n)
            og = eng.replay() or {k: v for k, v in eng.io.items() if k not in eng.inputs}
            rg["graph_matches_engine"] = bool((og["boxes"] - o["boxes"]).abs().max().item() < 1e-3)
            r["cuda_graph"] = rg
        except Exception as e:
            r["cuda_graph"] = dict(error=str(e)[:200])
        out["trt"][f"L{l}"] = r
        cg = r["cuda_graph"].get("ms_median")
        print(f"  L={l}: TRT {'FP16' if not a.fp32 else 'FP32'} {r['ms_median']:.2f} ms (median), p90 {r['ms_p90']:.2f}; +CUDA graph {cg if cg is None else round(cg, 2)} ms; matched-query |Δbox| {db:.3f} px |Δscore| {ds:.4f}")
        del eng; torch.cuda.empty_cache()
        with torch.no_grad():
            out["torch_fp32"][f"L{l}"] = time_fn(lambda: w(x), a.n)
            with torch.autocast("cuda", dtype=torch.float16):
                out["torch_fp16"][f"L{l}"] = time_fn(lambda: w(x), a.n)
        print(f"        torch eager fp32 {out['torch_fp32'][f'L{l}']['ms_median']:.2f} ms, fp16 autocast {out['torch_fp16'][f'L{l}']['ms_median']:.2f} ms")
    if a.out:
        json.dump(out, open(a.out, "w"), indent=1)
    return out


def yolo(a):
    """ultralytics ONNX export (raw graph: no NMS for NMS-based YOLOs, full end-to-end graph for YOLO26 / YOLOv10),
    then the same TensorRT builder and runner as KESTREL, so the two families are timed identically."""
    from ultralytics import YOLO
    dev = torch.device("cuda")
    m = YOLO(a.ckpt)
    onnx_path, eng_path = a.ckpt.replace(".pt", f"_{a.size}.onnx"), a.ckpt.replace(".pt", f"_{a.size}.engine")
    if not os.path.exists(onnx_path) or a.rebuild:
        # half=True traces the model in fp16 on the GPU (ultralytics' own TensorRT recipe), so the graph is fp16 end to end
        m.export(format="onnx", imgsz=a.size, half=not a.fp32, dynamic=False, simplify=True, opset=17, batch=1, device=0 if not a.fp32 else "cpu", verbose=False)
        os.replace(a.ckpt.replace(".pt", ".onnx"), onnx_path)
    if not os.path.exists(eng_path) or a.rebuild:
        build_engine(onnx_path, eng_path, fp16=not a.fp32, workspace_gb=a.workspace)
    eng = Engine(eng_path)
    x = torch.randn(1, 3, a.size, a.size, device=dev)
    o = eng(x)
    r = time_fn(lambda: eng(None), a.n)
    try:
        eng.capture_graph(); rg = time_fn(eng.replay, a.n); eng.replay()
        rg["graph_matches_engine"] = all(bool((eng.io[k].float() - o[k].float()).abs().max().item() < 1e-3) for k in o)
        r["cuda_graph"] = rg
    except Exception as e:
        r["cuda_graph"] = dict(error=str(e)[:200])
    net = m.model.to(dev).eval().float()
    with torch.no_grad():
        fp32 = time_fn(lambda: net(x), a.n)
        with torch.autocast("cuda", dtype=torch.float16):
            fp16 = time_fn(lambda: net(x), a.n)
    out = dict(ckpt=a.ckpt, size=a.size, gpu=torch.cuda.get_device_name(0), params_M=sum(p.numel() for p in net.parameters()) / 1e6,
               end2end=bool(getattr(net, "end2end", False)), trt=r, torch_fp32=fp32, torch_fp16=fp16, fp16=not a.fp32,
               outputs={k: list(v.shape) for k, v in eng(None).items()})
    print(f"  {a.ckpt}: TRT {'FP16' if not a.fp32 else 'FP32'} {r['ms_median']:.2f} ms (median), +CUDA graph {r['cuda_graph'].get('ms_median')}; torch fp32 {fp32['ms_median']:.2f}, fp16 {fp16['ms_median']:.2f}; end2end={out['end2end']} outputs={out['outputs']}")
    if a.out:
        json.dump(out, open(a.out, "w"), indent=1)
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("kind", choices=["kestrel", "yolo"])
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--preset", default="N")
    ap.add_argument("--size", type=int, default=512)
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--all-depths", action="store_true", help="kestrel: one engine per decoder depth 1..L")
    ap.add_argument("--fp32", action="store_true")
    ap.add_argument("--rebuild", action="store_true")
    ap.add_argument("--workspace", type=float, default=1.0, help="TensorRT workspace in GB")
    ap.add_argument("--out", default=None)
    ap.add_argument("--out-dir", default=None)
    a = ap.parse_args()
    (kestrel if a.kind == "kestrel" else yolo)(a)
