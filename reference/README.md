# Reference Python environment

This directory defines a **separate** virtual environment from the main MLX
package: TensorFlow, JAX, and editable `magenta_rt` from `../vendor/magenta-realtime`.

**Git checkout only:** the upstream tree at `vendor/magenta-realtime/` is
vendored directly into this repository (a normal `git clone` already brings it
down). A PyPI **sdist** tarball does not ship that tree, so use a git clone for
`reference/setup_reference_venv.sh`.

From the **repository root**:

```bash
bash reference/setup_reference_venv.sh
```

Then point `MAGENTA_RT_REF_PYTHON` at `reference/.venv/bin/python` (or pass
`--ref-python` to `scripts/run_reference_stream.py`).

The setup script installs PyPI dependencies with `uv sync` here, then runs
`reference/install_reference_t5x.sh` to build the patched **t5x** stack required
for Depthformer in `reference/runner/generate_ref_runner.py`.

### Optional t5x import check

After install, verify the stack in your shell (not run automatically — some
macOS builds abort during `import t5x` in non-interactive contexts even when the
install is fine):

```bash
reference/.venv/bin/python -c "import t5x.models, t5x.decoding; print('ok')"
```

To force the same check from the install script:

```bash
MAGENTA_RT_REFERENCE_T5X_SMOKE=1 bash reference/install_reference_t5x.sh reference/.venv
```
