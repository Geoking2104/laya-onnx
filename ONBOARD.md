# Onboard your AI agent to run laya-onnx

Paste this file (or the one-liner in the README) into Claude Code, Codex, Cursor, Copilot, or any coding agent. The agent must follow every step. Safe to re-run.

## Goal

Install `laya-onnx` in the current workspace so the human can call `laya-onnx predict` and `laya-onnx-snake` on CPU (Intel i7 / 16 GB class). Do not download multi-GB weights unless the human asks.

## Steps

1. Confirm Python ≥ 3.11 (`python3 --version`). If missing, tell the human how to install it and stop.
2. If this directory is not already the repo:
   ```bash
   git clone https://github.com/Geoking2104/laya-onnx.git
   cd laya-onnx
   ```
3. Create a venv and install extras:
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   pip install -U pip
   pip install -e ".[demo,dev]"
   ```
   On Intel CPU also: `pip install -e ".[openvino]"` (ignore failure if wheels are missing).
   Only if the human asked to export graphs: `pip install -e ".[export]"`.
4. Verify:
   ```bash
   laya-onnx --help
   PYTHONPATH=. python -c "from laya_onnx import load; print('import ok')"
   ```
5. Write `AGENTS.md` at the repo root if it does not exist. Point at README, `examples/`, `laya-onnx predict`, `optimize --precision int8`, and `examples/snake.html`.
6. Optional MCP: this repo does **not** ship a hosted MCP server yet. Do not invent RunPod keys. If the human wants GPU later: `npx skills add runpod/runpod-plugins-official`.
7. Report back Python version, install path, working commands, and that weights live on Hugging Face (`receptron/laya-onnx`).

Do not commit `.venv`, ONNX weights, or API keys.
