# Installation

## From PyPI

```bash
pip install mime-engine
```

The `mime-engine` distribution exposes the `mime` Python package and
pulls in [MADDENING](https://microrobotica.org/maddening/) as a
dependency, so a single command gets you both layers.

> **Why `mime-engine`?** The bare `mime` distribution name on PyPI is
> taken by the long-standing email-encoding library. The import path
> is unaffected — you still write `import mime`.

### GPU acceleration

MIME inherits its JAX hardware story from MADDENING. To add CUDA 12:

```bash
pip install "mime-engine" "jax[cuda12]"
```

Or TPU:

```bash
pip install "mime-engine" "jax[tpu]"
```

You need matching system drivers (`nvidia-smi` for CUDA).

### Optional extras

| Extra | What it adds | Install command |
|-------|---|---|
| `dev` | pytest + pytest-xdist | `pip install "mime-engine[dev]"` |
| `docs` | Sphinx + MyST + bibtex (for building the docs locally) | `pip install "mime-engine[docs]"` |
| `viz` | PyVista 3D rendering | `pip install "mime-engine[viz]"` |
| `runner` | pyzmq + pyyaml + matplotlib (used by the MICROROBOTICA runner) | `pip install "mime-engine[runner]"` |
| `ci` | Minimal CI bundle (pytest + pyyaml) | `pip install "mime-engine[ci]"` |

## From source

```bash
git clone https://github.com/Microrobotics-Simulation-Framework/MIME
cd MIME
pip install -e ".[dev]"
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest tests/ -v
```

## Inside the MICROROBOTICA IDE

If you're using the
[MICROROBOTICA](https://microrobotica.org/) IDE, install MIME into
the same Python environment the IDE's embedded interpreter uses. The
IDE locates `mime` through `PYTHONPATH`; once `pip install mime-engine`
finishes, **File → Open Experiment** will work end-to-end without any
further configuration.

## Verifying the install

```python
import jax
import mime

print(f"JAX backend: {jax.devices()[0].platform}")  # 'cpu' / 'gpu' / 'tpu'
print(f"MIME nodes:  {dir(mime.nodes)}")
```
