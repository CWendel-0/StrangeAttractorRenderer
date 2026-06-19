# Strange Attractor Renderer

A real-time GPU renderer and explorer for chaotic dynamical systems —
strange attractors. It integrates millions of trajectory points per
second on the GPU, accumulates them into a density histogram, and
composites the result into a smooth, anti-aliased image with
density-estimation blur, color gradients, and live Lyapunov-exponent
and fractal dimension analysis.

![platform](https://img.shields.io/badge/platform-Windows%20%7C%20macOS%20%7C%20Linux-informational)
![license](https://img.shields.io/badge/license-AGPL--3.0-blue)

## What it does

Strange attractors are the trajectories traced out by certain chaotic
systems — sets of points that never repeat, never settle down, yet stay
confined to a bounded, often beautifully structured region of space.
This app simulates **8,192 trajectories in parallel**, each stepped
**512 times per GPU dispatch**, and splats every point into a
high-resolution histogram texture. Frame after frame, the histogram
accumulates, building up an increasingly detailed picture of the
attractor's structure — entirely on the GPU, with no CPU-side
trajectory storage. This is a way of generating and rendering
strange attractors **fast**, with optimizations for speed without
sacrificing fidelity or quality. Other renderers for this type of
data can take hours to produce a result that this program achieves
in seconds, in real-time, visible as you watch it build on screen.

On top of the raw density data, the renderer applies:

- **Density-estimation (DE) blur** — a multi-pass, density-relative
  Gaussian blur (à la Hamilton/Apophysis-style fractal flame rendering)
  that smooths sparse regions while keeping dense regions sharp.
- **Supersampling** (1×/2×/4×) for clean edges.
- **Two independent color gradients** (RGB/Oklab/OKLCh interpolated) — one
  mapped to point density, one to local trajectory speed — combined
  with a choice of 14 blend modes (Add, Multiply, Screen, Overlay,
  Color Dodge, etc.), or a classic monochrome log-density mode.
- **Live Lyapunov exponent (λ₁) and Kaplan-Yorke dimension (D_KY)**
  estimation via shadow-orbit tracking, computed continuously in a
  background thread.
- An **arcball camera** (rotate/pan/zoom) and a built-in **chaotic
  parameter search** that randomizes and auto-tunes attractor
  coefficients until it finds one that's actually chaotic.

Because the entire simulate → splat → blur → composite pipeline runs on
the GPU via `wgpu`, the image keeps refining in real time — you can
rotate the camera, tweak parameters, or change the color gradient and
watch the attractor reshape and re-render at interactive frame rates,
even while millions of new points are being accumulated every second.

## Sample images produced
These sample images were made entirely within the program, and rendered
in a matter of a few (less than 5) seconds.

<img width="1108" height="768" alt="ColorfulPickover" src="https://github.com/user-attachments/assets/a66bc3c7-0962-4aa0-9823-900d31511280" />
<img width="2214" height="1258" alt="ZoomedInLorenz" src="https://github.com/user-attachments/assets/1a4a188a-0665-4ded-bdcd-2d6a12c1f294" />
<img width="692" height="763" alt="Polysin" src="https://github.com/user-attachments/assets/a046058a-3304-4768-996a-6fb1e12443bd" />
<img width="1025" height="1011" alt="Icon" src="https://github.com/user-attachments/assets/edfe938b-3913-41ce-80ea-7c45115423ba" />
<img width="631" height="1046" alt="GreenPolySin" src="https://github.com/user-attachments/assets/eef0e728-7be3-4fda-8d3d-00b89ff5f25b" />

## Supported attractors

| Type | Params | Notes |
|---|---|---|
| Lorenz | 4 | The classic butterfly attractor |
| Lorenz 84 | 5 | Atmospheric circulation model |
| Rössler | 4 | Single-scroll chaotic flow |
| Thomas | 2 | Cyclically symmetric, highly damped |
| Chaotic Flow | 22 | General polynomial flow (12 monomial weights) |
| Pickover | 4 | Classic 4-parameter discrete map |
| Clifford | 4 | Trigonometric discrete map |
| Icon | 6 | Symmetric icon map |
| Icon B | 6 | Variant icon map |
| Polynomial A / B / C | 3 / 6 / 18 | Generalized polynomial maps |
| Polynomial Abs | 21 | Adds `\|x\|, \|y\|, \|z\|` terms |
| Polynomial Power | 24 | Adds an asymmetric power term |
| Polynomial Sin | 39 | Sprott sinusoidal map |
| Polynomial Sprott | up to 169 | Full Sprott polynomial map, order 2–5 |

ODE-based attractors (Lorenz, Lorenz 84, Rössler, Thomas) are integrated
with Euler steps; the rest are discrete iterated maps.

## Controls

- **Left-drag** on the canvas — rotate the camera (arcball)
- **Middle-drag** — pan
- **Scroll wheel** — zoom
- **Randomize** button — runs a parallel background search that mutates
  the current attractor's parameters until it lands on a chaotic
  configuration, then snaps the camera to fit it

## Interface overview

- **Attractor window** — pick the attractor type, edit its parameters
  (with per-parameter freeze checkboxes so the randomizer leaves them
  alone), and trigger the search.
- **Left panel** — rendering mode (Monochrome / Colorful), canvas size
  and expand/collapse, brightness/gamma/background color, anti-aliasing
  level, DE blur tuning (min/max σ, alpha power, noise), blend mode, and
  the two gradient editors.
- **Canvas** — the live render. Can be expanded to fill the window or
  resized/floated independently.
- **Metrics overlay** — bottom-left HUD showing λ₁, D_KY, and a running
  iteration counter.

## Building from source

Requires a recent [Rust toolchain](https://rustup.rs/) and a GPU with
Vulkan, Metal, or DirectX 12 support (via `wgpu`).

```sh
cargo run --release
```

The binary is `strange-attractor` (workspace crate `app`).

## Project layout


- `crates/sim` — attractor definitions (CPU-side parameter
  descriptors, defaults, bounds estimation, Lyapunov/search workers)
- `crates/gpu` — `wgpu` pipelines: per-attractor-type simulation
  compute shaders, histogram accumulation, DE blur, and compositing
- `crates/app` — windowing, `egui` UI, camera, and the main render loop

## Releases

Tagged pushes (`v*`) build and package binaries for Windows, macOS
(Apple Silicon and Intel), and Linux via GitHub Actions.

## License

[GNU AGPL v3](LICENSE)
