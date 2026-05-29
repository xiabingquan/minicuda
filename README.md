# minicuda

[中文版](README_zh.md)

An experimental CUDA learning project that explores a new learning paradigm in the age of LLMs. Instead of following traditional textbooks or video courses alone, this repository attempts to learn CUDA programming **with the assistance of LLM all along** — from drafting the study plan, to discussing concepts, and recording notes.

## Roadmap

![CUDA Learning Roadmap](assets/roadmap.excalidraw.svg)

The curriculum consists of 10 courses, progressing from basic environment setup to reading and reproducing production-level CUDA codebases.

| Course | Topic | Focus |
|--------|-------|-------|
| 1 | Environment & Hello CUDA | CMake, nvcc, pybind11, PyTorch C++ extension |
| 2 | Memory Hierarchy & Matrix Ops | Shared memory, coalescing, bank conflict, tiled GEMM |
| 3 | Profiling & Performance Analysis | nsys, ncu, roofline model, CUDA streams |
| 4 | Reduction & ML Kernels | Warp primitives, softmax, RMSNorm, LayerNorm, activations |
| 5 | Advanced Kernels & Fusion | Online algorithms, prefix sum, TopK, RoPE, kernel fusion |
| 6 | Tensor Core & High-Performance GEMM | WMMA, double buffering, CuTe, CUTLASS |
| 7 | Hopper Features & Async Execution | TMA, pipeline, warpgroup, thread block cluster |
| 8 | Flash Attention | Online softmax + tiling, forward/backward, causal masking |
| 9 | DeepEP | MoE expert parallelism, all-to-all communication |
| 10 | DeepGEMM | FP8 GEMM, JIT compilation, persistent kernel |

See [plans/overview.md](plans/overview.md) for the full study plan with detailed learning objectives, knowledge points, practice projects, and acceptance criteria for each course.

## Repository Structure

```
csrc/             CUDA kernel implementations (.cu)
binding/          pybind11 / PyTorch extension bindings (.cpp)
tests/            Python correctness & performance tests
include/          Shared C++ header files
3rdparty/         Third-party dependencies (e.g. CUTLASS)
plans/            Study plans and course outlines
  overview.md     Full curriculum overview
  course-XX.md    Detailed plan for each course
notes/            Learning notes organized by course
  course-XX/      Notes, summaries, and takeaways
assets/           Static assets (diagrams, images)
```

## Learning Approach

Each kernel follows a consistent workflow:

1. **Implement** — Write the CUDA kernel in `csrc/`
2. **Verify** — Test correctness against PyTorch reference with `torch.allclose`
3. **Profile** — Analyze with `nsys` (timeline) and `ncu` (kernel-level metrics)
4. **Optimize** — Iterate based on profiling results, keep multiple versions for comparison

## License

[MIT](LICENSE)
