"""Replay or export a recorded Snake run (the JSONL written by ``--record``)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from rich.console import Console
from rich.table import Table

_TEMPLATE = """<!DOCTYPE html>
<html lang="en"><head><meta charset="UTF-8"/><meta name="viewport" content="width=device-width,initial-scale=1"/>
<title>laya-onnx snake replay</title>
<style>
html,body{margin:0;background:#070b12;color:#e2e8f0;font-family:system-ui,sans-serif}
.wrap{max-width:680px;margin:0 auto;padding:20px}
canvas{display:block;width:100%;max-width:640px;background:#020617;border:2px solid #34d399;border-radius:10px}
.row{display:flex;gap:10px;align-items:center;margin-top:10px}
button{padding:8px 12px;border:0;border-radius:8px;background:#334155;color:#fff}
#play{background:#34d399;color:#052e16;font-weight:700}
#st{font-size:13px;color:#94a3b8;margin-top:8px}
</style></head><body><div class="wrap">
<h1>Snake replay</h1><p id="model"></p>
<canvas id="c" width="640" height="420"></canvas>
<div class="row"><button id="play">Pause</button><button id="reset">Reset</button>
<input id="seek" type="range" min="0" value="0" style="flex:1"/></div>
<div id="st">-</div>
<script>
var FRAMES=__FRAMES__, DECISIONS=__DECISIONS__, MODEL=__MODEL__;
var i=0, playing=true, timer=null, c=document.getElementById("c"), ctx=c.getContext("2d");
document.getElementById("model").textContent="model: "+(MODEL.model||"?")+"   frames: "+FRAMES.length;
function draw(){
  var g=FRAMES[i]||{}, W=g.width||1, H=g.height||1, cw=c.width/W, ch=c.height/H;
  ctx.fillStyle="#020617"; ctx.fillRect(0,0,c.width,c.height);
  ctx.strokeStyle="#1e293b";
  for(var x=0;x<=W;x++){ctx.beginPath();ctx.moveTo(x*cw,0);ctx.lineTo(x*cw,c.height);ctx.stroke();}
  for(var y=0;y<=H;y++){ctx.beginPath();ctx.moveTo(0,y*ch);ctx.lineTo(c.width,y*ch);ctx.stroke();}
  var f=g.food; if(f){ctx.fillStyle="#fbbf24";ctx.fillRect(f[0]*cw+2,f[1]*ch+2,cw-4,ch-4);}
  var b=g.body||[];
  for(var k=b.length-1;k>=0;k--){ctx.fillStyle=k===0?"#6ee7b7":"#059669";ctx.fillRect(b[k][0]*cw+2,b[k][1]*ch+2,cw-4,ch-4);}
  var d=DECISIONS[i]||{};
  document.getElementById("st").textContent="frame "+(i+1)+"/"+FRAMES.length+"  move "+(d.executed||"-")+(d.intervened?" (guarded)":"")+"  score "+(g.score||0)+"  len "+(g.length||0);
  document.getElementById("seek").value=i;
}
function tick(){if(!playing)return; i=(i+1)%Math.max(FRAMES.length,1); draw();}
function start(){if(timer)return; playing=true; document.getElementById("play").textContent="Pause"; timer=setInterval(tick,120);}
function stop(){playing=false; document.getElementById("play").textContent="Play"; clearInterval(timer); timer=null;}
document.getElementById("play").onclick=function(){playing?stop():start();};
document.getElementById("reset").onclick=function(){i=0;draw();};
document.getElementById("seek").oninput=function(e){i=Number(e.target.value);draw();};
var s=document.getElementById("seek"); s.max=Math.max(FRAMES.length-1,0);
draw(); start();
</script></div></body></html>
"""


def _read_record(path: Path):
    meta, frames, end = None, [], None
    with path.open(encoding="utf-8") as handle:
        for lineno, line in enumerate(handle, 1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise SystemExit(f"{path}: line {lineno} is not valid JSON: {exc}")
            kind = record.get("type")
            if kind == "metadata":
                meta = record
            elif kind == "frame":
                frames.append(record)
            elif kind == "end":
                end = record
    if meta is None:
        raise SystemExit(f"{path}: missing metadata header (is this a --record file?)")
    return meta, frames, end


def _render_html(meta: dict, frames: list) -> str:
    games = [frame.get("game") for frame in frames]
    decisions = [frame.get("decision") for frame in frames]
    return (
        _TEMPLATE.replace("__FRAMES__", json.dumps(games))
        .replace("__DECISIONS__", json.dumps(decisions))
        .replace("__MODEL__", json.dumps(meta.get("model") or {}))
    )


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="laya-onnx-snake export", description=__doc__)
    parser.add_argument("record", type=Path, help="JSONL file written by --record")
    parser.add_argument("--limit", type=int, help="Show only the first N frames in the table")
    parser.add_argument("--html", type=Path, help="Write a self-contained animated HTML replay")
    parser.add_argument("--json", action="store_true", help="Print the summary JSON only")
    args = parser.parse_args(argv)
    if not args.record.is_file():
        parser.error(f"no such record: {args.record}")

    meta, frames, end = _read_record(args.record)
    summary = (end or {}).get("summary") or {}
    if args.json:
        print(json.dumps(summary, indent=2))
        return 0

    console = Console()
    model = (meta.get("model") or {}).get("model", "?")
    console.print(
        f"[bold]{args.record}[/]  [dim]{meta.get('format', '?')}[/]  "
        f"model={model}  frames={len(frames)}  created={meta.get('created_utc', '?')}"
    )
    if summary:
        console.print(json.dumps(summary, indent=2))
    shown = frames if args.limit is None else frames[: args.limit]
    if shown:
        table = Table(title="frames")
        for column in ("at", "executed", "chosen", "guarded", "score", "length"):
            table.add_column(column)
        for frame in shown:
            decision = frame.get("decision") or {}
            game = frame.get("game") or {}
            table.add_row(
                f"{float(frame.get('at', 0.0)):.2f}",
                str(decision.get("executed", "")),
                str(decision.get("chosen", "")),
                "yes" if decision.get("intervened") else "",
                str(game.get("score", "")),
                str(game.get("length", "")),
            )
        console.print(table)
    if args.html:
        args.html.parent.mkdir(parents=True, exist_ok=True)
        args.html.write_text(_render_html(meta, frames), encoding="utf-8")
        console.print(f"[green]wrote[/] {args.html}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
