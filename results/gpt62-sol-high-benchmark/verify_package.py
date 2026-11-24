"""Read-only offline verification of one complete ORBIT-Q review package."""
import hashlib
import json
import math
import re
import struct
from pathlib import Path
from xml.etree import ElementTree

def verify(root):
    read=lambda p:json.loads((root/p).read_text())
    sha=lambda path:hashlib.sha256(path.read_bytes()).hexdigest()
    files=read("manifest.json")
    assert set(files)=={str(f.relative_to(root)) for f in root.rglob("*") if f.is_file() and "__pycache__" not in f.parts and f.name!="manifest.json"}
    assert all(sha(root/k)==v for k,v in files.items())
    r=read("trials.json");p=read("paired-runtimes.json");e=read("timing-evidence.json")
    protocol=read("execution-protocol.json");v=protocol["solver"].split("-")[-1]
    assert protocol["solver_effort"]==protocol["audit_effort"]=="high" and protocol["verifier"]=="gpt-6-astra"
    assert len(r)==len(p)==12 and list(e)==[str(i) for i in range(1,13)]
    passes=[];rr=[];img=protocol["image_id"]
    for i,(row,pair) in enumerate(zip(r,p),1):
        d=root/f"challenge-{i:02d}";reward=json.loads((d/"reward.json").read_text());trace=e[str(i)]
        assert row["task"]==pair["task"]==i and row["round"]==(2 if v=="terra" or v=="astra" and i==7 else 1)
        assert row["reward"]==reward["reward"]==reward["functional_score"]*reward["static_policy_score"]*reward["llm_audit_score"]
        assert math.isfinite(row["cost_usd"]) and row["cost_usd"]>=0 and row["solver_wall_sec"]>0
        assert 0<=row["n_cache_tokens"]<=row["n_input_tokens"]
        if row["reward"]==1:
            passes.append(i);assert sha(d/f"solution_{i}.py")==row["candidate_sha256"]
        if pair["expert_sec"] and pair["model_sec"]:
            assert trace["expert"]["config_hash"]==trace["candidate"]["config_hash"]
            rr.append(pair["model_sec"]/pair["expert_sec"])
        for role,t in trace.items():
            assert t["stamp"]["image"]==img and t["stamp"]["cpu"]==6 and t["stamp"]["memory_mib"]==10240 and t["stamp"]["network"]=="none"
            assert t["sec"]==pair["expert_sec" if role=="expert" else "model_sec"]
            assert t["sec"] is None or t["passed"]
    lab=protocol["solver"].split("-")
    run,=[a for a in read("resources-comparison.json") if a["label"]==f"{lab[1]} {lab[2].title()} high"]
    ax,=[a for a in read("runtime-comparison.json")["new"] if a["variant"]==v]
    assert run["passes"]==passes and math.isclose(run["total_cost_usd"],sum(t["cost_usd"] for t in r))
    assert math.isclose(run["agent_wall_sec"],sum(t["solver_wall_sec"] for t in r))
    published=read("published-runtimes.json");adjusted=[]
    for raw,display in zip(p,published,strict=True):
        assert display["task"]==raw["task"] and display["model_sec"]==raw["model_sec"]
        if display["expert_reference_basis"]=="estimated":
            assert display["task"]==8 and raw["expert_oom"] and display["expert_reference_sec"]==50 and raw["expert_sec"] is None
        else:assert display["expert_reference_sec"]==raw["expert_sec"]
        if display["model_over_expert"] is not None:
            assert math.isclose(display["model_over_expert"],display["model_sec"]/display["expert_reference_sec"])
            adjusted.append(display["model_over_expert"])
    assert ax["pairs"]==len(adjusted) and math.isclose(ax["ratio"],math.exp(sum(map(math.log,adjusted))/len(adjusted)))
    assert ax["measured_pairs"]==len(rr) and math.isclose(ax["measured_ratio"],math.exp(sum(map(math.log,rr))/len(rr)))
    assert math.isclose(ax["failure_pct"],100*(12-len(passes))/12)
    links=re.findall(r"!\[.*?\]\((.*?)\)",(root/"README.md").read_text())
    assert len(set(links))==5
    for link in links:
        png=root/link; dims=struct.unpack(">II",png.read_bytes()[16:24])
        assert dims[0]>900 and dims[1]>700
        if "square" in link:assert dims[0]==dims[1]
        assert png.with_suffix(".pdf").read_bytes()[:5]==b"%PDF-"
        svg=ElementTree.parse(png.with_suffix(".svg"))
        texts=["".join(e.itertext()).strip() for e in svg.iter("{http://www.w3.org/2000/svg}text")]
        removed={"5.6 Sol high","5.6 Sol ultra","5.6 Terra high","5.6 Luna high","Grok 4.5 high"}
        if png.stem in {"task-outcomes","agent-resources","resource-efficiency-square"}:
            for r0 in read("resources-comparison.json"):
                assert any(r0["label"] in t for t in texts)==(r0["label"] not in removed), (png.stem,r0["label"])
        if png.stem=="artifact-efficiency":
            for name,_,_ in read("runtime-comparison.json")["historical"]:
                assert (("6 Astra" if name=="Astra" else name) in texts)==(name not in {"Sol","Sol ultra","Terra","Luna","Grok 4.5"}),name
    print(f"{protocol['solver']}: {len(passes)}/12 valid, {len(rr)} measured pairs, 5 figure sets verified.")

if __name__=="__main__": verify(Path(__file__).resolve().parent)
