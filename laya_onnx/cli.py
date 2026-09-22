"""Command-line prediction and checkpoint conversion."""

import argparse
import json
from pathlib import Path

from . import __version__
from .agent import Agent


def main(argv=None):
    parser = argparse.ArgumentParser(prog="laya-onnx", description=__doc__)
    parser.add_argument("--version", action="version", version=__version__)
    commands = parser.add_subparsers(dest="command", required=True)

    predict = commands.add_parser("predict")
    predict.add_argument("--model", default="receptron/laya-onnx")
    predict.add_argument("--subfolder")
    predict.add_argument("--revision")
    predict.add_argument("--providers", choices=("cpu", "openvino", "cuda"), default="cpu")
    predict.add_argument("--threads", type=int)
    source = predict.add_mutually_exclusive_group(required=True)
    source.add_argument("--state", help="Plain text input")
    source.add_argument("--state-file", type=Path, help="JSON state file")
    predict.add_argument("--questions", required=True, type=Path)
    predict.add_argument("--batch-size", type=int, default=16)

    convert = commands.add_parser("convert")
    convert.add_argument("--model", default="convaiinnovations/laya")
    convert.add_argument("--subfolder")
    convert.add_argument("--revision")
    convert.add_argument("--output", type=Path, required=True)
    convert.add_argument("--precision", choices=("fp32", "int8"), default="fp32")
    convert.add_argument("--opset", type=int, default=17)

    args = parser.parse_args(argv)
    if args.command == "convert":
        from .convert import convert as do_convert

        result = do_convert(
            args.model,
            args.output,
            precision=args.precision,
            revision=args.revision,
            subfolder=args.subfolder,
            opset=args.opset,
        )
        print(json.dumps({"output": str(result), "precision": args.precision}))
        return
    state = args.state if args.state is not None else json.loads(args.state_file.read_text())
    questions = json.loads(args.questions.read_text())
    agent = Agent(
        args.model,
        providers=args.providers,
        revision=args.revision,
        subfolder=args.subfolder,
        batch_size=args.batch_size,
        threads=args.threads,
    )
    print(json.dumps(agent.predict(state, questions), ensure_ascii=False, indent=2))
