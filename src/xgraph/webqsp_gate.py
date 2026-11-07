"""Outcome-blind finalization of the amendment-8a development gate."""

import argparse
import ast
import copy
import json
import math
import os
import subprocess
from datetime import UTC, datetime
from pathlib import Path

from xgraph import webqsp_native as native
from xgraph import webqsp_test as study
from xgraph.provenance import sha256_file

ROOT = Path(__file__).resolve().parents[2]
DECISION = ROOT / "eval/webqsp-8a-gate-2026-10-30/decision.json"
NATIVE_OWNER = ROOT.parent / "codex-webqsp-gate"
KEYS = {
    "registration",
    "decision",
    "pilot_receipt",
    "baseline_timings",
    "native_gcr",
    "native_engine",
}
CONTRACTS = {
    "apply_registered_gate",
    "gate_decision",
    "require_native_checks",
    "matched_prices",
    "read_baseline_prices",
    "baseline_price_contract",
    "size_protocol",
    "two_tier_sizing",
    "verify_pilot",
    "verify_occupancy_prices",
    "native_pricing_charge",
    "build_jobs",
    "apply_budget_admission",
    "cluster_subpanel",
    "select_ids",
    "positive",
    "power_report",
    "descriptive_power",
    "cluster_summary",
    "read_sealed",
    "identity",
    "verify_identity",
    "resolved_config",
}


def verify_contract(commit):
    """Refuse to apply new admission semantics to an immutable old registration."""
    old = subprocess.check_output(["git", "show", f"{commit}:src/xgraph/webqsp_test.py"], cwd=ROOT)

    def functions(source):
        tree = ast.parse(source)
        result = {
            n.name: ast.dump(n)
            for n in tree.body
            if isinstance(n, ast.FunctionDef) and n.name in CONTRACTS
        }
        for node in tree.body:
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name) and target.id in {
                        "FAMILIES",
                        "REFERENCES",
                        "SEED",
                        "CORE_ARMS",
                    }:
                        result[target.id] = ast.dump(node.value)
        return result

    if functions(old) != functions(Path(study.__file__).read_text()):
        raise ValueError("finalizer admission contracts differ from the registered source")
    for name in ("webqsp_gcr_pricing.py", "webqsp_gcr_decision.py"):
        source = subprocess.check_output(["git", "show", f"{commit}:src/xgraph/{name}"], cwd=ROOT)
        if source != (ROOT / "src/xgraph" / name).read_bytes():
            raise ValueError(f"finalizer native price contract differs: {name}")


def checked_json(item):
    study.verify_identity(item)
    return json.loads(Path(item["path"]).read_text())


def check_gcr(item, registration):
    qualified = checked_json(item)
    if (
        qualified.get("artifact_kind") != "xgraph.gcr_native_qualification"
        or qualified.get("status") != "QUALIFIED"
        or qualified.get("scope") != "validation"
        or qualified.get("families") != dict.fromkeys(study.FAMILIES, True)
        or qualified.get("producer_commit") != registration["config"]["gcr_source_commit"]
        or qualified.get("recipe_sha256") != study.baseline_price_contract(registration, "gcr")
        or not qualified.get("evidence")
    ):
        raise ValueError("GCR native qualification is incomplete or uses another frozen recipe")
    for ref in qualified["evidence"]:
        study.verify_identity(ref)
    harness = study.read_sealed(qualified["execution_harness"]["path"])
    study.verify_identity(qualified["execution_harness"])
    ledger_ref = next(
        (r for r in qualified["evidence"] if r["path"] == harness["ledger_path"]), None
    )
    if ledger_ref is None:
        raise ValueError("GCR native qualification has no hash-bound lifetime ledger")
    ledger = checked_json(ledger_ref)
    spent = study.positive(
        qualified["protocol_qualification_gpu_seconds"], "GCR qualification charge"
    )
    if (
        ledger["active"] is not None
        or spent != ledger["spent_seconds"]
        or not math.isclose(
            sum(a["elapsed_seconds"] for a in ledger["attempts"]), spent, rel_tol=0, abs_tol=1e-6
        )
    ):
        raise ValueError("GCR native qualification has unsettled or changed lifetime accounting")
    return qualified


def check_native_engine(item, registration):
    """Verify E in its producer environment while leaving its lifetime plan unchanged."""
    path = Path(item["path"]).resolve()
    owner = NATIVE_OWNER.resolve()
    if owner == ROOT.resolve() and path.parent == native.OUT.resolve():
        return native.verify_qualification(item, registration)
    if path != owner / "target/webqsp-native-q/qualification.json":
        raise ValueError("native engine receipt must use the recorded owner lifetime destination")
    # The same verifier runs with the producer's paths and interpreter. Its original
    # environment, stage argv, source hashes, receipts and settled ledger all still apply.
    if (owner / "src/xgraph/webqsp_native.py").read_bytes() != Path(native.__file__).read_bytes():
        raise ValueError("native engine owner verifier source changed")
    script = (
        "import json, sys; from pathlib import Path; "
        "sys.path.insert(0, sys.argv[1]); "
        "from xgraph import webqsp_native as native; "
        "native.ROOT = Path(sys.argv[2]); "
        "native.OUT = native.ROOT / 'target/webqsp-native-q'; "
        "data = json.load(sys.stdin); "
        "print(json.dumps(native.verify_qualification(data['item'], data['registration'])))"
    )
    result = subprocess.run(
        [str(owner / ".venv/bin/python"), "-I", "-B", "-c", script, str(ROOT / "src"), str(owner)],
        input=json.dumps({"item": item, "registration": registration}),
        capture_output=True,
        text=True,
        cwd=ROOT,
        env={
            **os.environ,
            "CUDA_VISIBLE_DEVICES": "",
            "PYTHONDONTWRITEBYTECODE": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "OMP_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
        },
        timeout=60,
    )
    if result.returncode:
        raise ValueError(f"native engine receipt verification failed: {result.stderr.strip()}")
    return json.loads(result.stdout)


def finalize(inputs, inputs_sha256, *, dry_run):
    """Verify the owner-bound metadata and exclusively create the registered gate."""
    if sha256_file(inputs) != inputs_sha256:
        raise ValueError("gate input manifest hash changed")
    manifest = json.loads(Path(inputs).read_text())
    if not Path(manifest["pilot_receipt"]["path"]).is_file():
        raise ValueError("pilot completion receipt is missing; do not open active timings")
    if set(manifest) != KEYS:
        raise ValueError("gate manifest needs exactly the registered six input identities")
    registration = study.read_sealed(manifest["registration"]["path"])
    study.verify_identity(manifest["registration"])
    decision = checked_json(manifest["decision"])
    subgraphrag = decision.get("subgraphrag")
    if (
        decision["registration"] != manifest["registration"]
        or decision.get("development_only") is not True
        or decision.get("webqsp_test_accessed") is not False
        or decision.get("gcr") != "registered_native_price_only"
        or decision.get("native_gcr") != manifest["native_gcr"]
        or decision.get("dog", {}).get("admit") is not False
        or not decision["dog"].get("reason")
        or (
            subgraphrag is not None
            and (
                not isinstance(subgraphrag, dict)
                or subgraphrag.get("admit") is not False
                or not subgraphrag.get("reason")
            )
        )
    ):
        raise ValueError(
            "gate decision changed its registered outcome-blind exclusions or evidence"
        )
    config = registration["config"]
    destination = Path(config["gate"])
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(f"development gate already exists: {destination}")
    if (
        str(destination) != registration["conditional_rule"]["gate_file"]
        or registration["conditional_rule"].get("missing_gate_blocks") is not True
        or registration.get("amendment", {}).get("amendment") != 8
        or registration["amendment"].get("addendum") != "8a"
        or registration["pricing"].get("cohort_rule") != "two_tier"
    ):
        raise ValueError("finalizer requires the registered amendment-8a two-tier gate destination")
    verify_contract(registration["commit"])
    receipt = checked_json(manifest["pilot_receipt"])
    expected = registration["pilot"]["occupancy_receipt"]
    if manifest["pilot_receipt"]["path"] != expected["path"] or any(
        receipt.get(k) != expected[k] for k in ("seal_sha256", "command")
    ):
        raise ValueError("pilot completion receipt changed its registered producer seal or command")
    if manifest["baseline_timings"]["path"] != config["baseline_timings"]:
        raise ValueError("baseline timing identity must name the registered file")
    # Only the metadata that verify_pilot consumes is opened, never receipt.files wholesale.
    timings = study.verify_pilot(registration, config["timings"])
    study.verify_identity(manifest["baseline_timings"])
    rates = study.matched_prices(registration, config["baseline_timings"], config["timings"])
    gcr = check_gcr(manifest["native_gcr"], registration)
    engine = check_native_engine(manifest["native_engine"], registration)
    now = datetime.now(UTC)
    gate = {
        "decided_at": now.isoformat(),
        "development_only": True,
        "webqsp_test_accessed": False,
        "wt_pc": copy.deepcopy(registration["conditional_rule"]["wt_pc_resolution"]),
        "dog": {**decision["dog"], "evidence": [manifest["decision"]]},
        "native_qualification": {
            "families": gcr["families"],
            "evidence": [manifest["native_gcr"], manifest["native_engine"]],
            "scope": "Frozen GCR planner/readers in both families; exact registered engine/adapter "
            "and native tensor agreement. No accuracy or baseline-price inference.",
        },
        "protocol_qualification_gpu_seconds": gcr["protocol_qualification_gpu_seconds"]
        + engine["protocol_qualification_gpu_seconds"],
        "inputs": copy.deepcopy(manifest),
        "input_manifest": study.identity(inputs),
        "implementation": study.identity(__file__),
    }
    if subgraphrag is not None:
        gate["subgraphrag"] = {**subgraphrag, "evidence": [manifest["decision"]]}
    gate = study.apply_registered_gate(registration, gate)
    study.gate_decision(gate)
    study.require_native_checks(gate)
    if now > datetime.fromisoformat(registration["freeze_deadline"]):
        raise ValueError("execution freeze deadline passed")
    if any(
        s.get("status") == "PENDING" and s.get("blocks_execution", True)
        for s in registration.get("amendment_slots", [])
    ):
        raise ValueError("pending blocking amendment slot")
    for item in gate["wt_pc"].get("evidence", []):
        study.verify_identity(item)
    pricing_charge = study.native_pricing_charge(registration, config["baseline_timings"])
    initial = gate["protocol_qualification_gpu_seconds"] + pricing_charge
    if initial > registration["pricing"]["reserve_seconds"]:
        raise ValueError("native qualification plus pricing exceeds the existing core reserve")
    sizing = study.size_protocol(registration, timings, gate, baseline_rates=rates)
    if not sizing.get("execution_ready"):
        pending = {
            k: v["status"]
            for k, v in sizing["baseline_admission"].items()
            if v["status"].startswith("PENDING")
        }
        raise ValueError(f"matched prices/qualification or core power still pending: {pending}")
    # Check the metadata again immediately before writing; the manifest is also bound in the gate.
    for ref in manifest.values():
        study.verify_identity(ref)
    if sha256_file(inputs) != inputs_sha256:
        raise ValueError("gate input manifest changed during verification")
    result = {
        "gate": gate,
        "sizing": sizing,
        "initial_core_gpu_seconds": initial,
        "bound": list(manifest.values()),
        "destination": str(destination),
        "dry_run": dry_run,
    }
    if not dry_run:
        study.write_json(destination, gate)
        result["written"] = study.identity(destination)
    return result


def inventory(decision_path, engine_path):
    """Print root metadata identities only after the registered pilot has completed."""
    decision = json.loads(Path(decision_path).read_text())
    registration = study.read_sealed(decision["registration"]["path"])
    study.verify_identity(decision["registration"])
    receipt = Path(registration["pilot"]["occupancy_receipt"]["path"])
    if not receipt.is_file():
        raise ValueError("pilot completion receipt is missing; no timing hashes may be bound yet")
    return {
        "registration": decision["registration"],
        "decision": study.identity(decision_path),
        "pilot_receipt": study.identity(receipt),
        "baseline_timings": study.identity(registration["config"]["baseline_timings"]),
        "native_gcr": decision["native_gcr"],
        "native_engine": study.identity(engine_path),
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--decision", type=Path, default=DECISION)
    parser.add_argument(
        "--native-engine",
        type=Path,
        default=NATIVE_OWNER / "target/webqsp-native-q/qualification.json",
    )
    parser.add_argument("--inputs", type=Path)
    parser.add_argument("--inputs-sha256")
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--inventory", action="store_true")
    modes.add_argument("--dry-run", action="store_true")
    modes.add_argument("--run", action="store_true")
    args = parser.parse_args(argv)
    if args.inventory:
        result = inventory(args.decision, args.native_engine)
    elif args.inputs and args.inputs_sha256:
        result = finalize(args.inputs, args.inputs_sha256, dry_run=not args.run)
    elif args.run or args.inputs or args.inputs_sha256:
        parser.error("finalization needs --inputs and --inputs-sha256")
    else:
        decision = json.loads(args.decision.read_text())
        registration = checked_json(decision["registration"])
        paths = [
            registration["config"]["gate"],
            registration["config"]["baseline_timings"],
            registration["pilot"]["occupancy_receipt"]["path"],
            str(args.native_engine),
        ]
        result = {
            "status": "PREVIEW_ONLY",
            "stats": {p: Path(p).is_file() for p in paths},
            "registration": decision["registration"],
            "proposed_decisions": {
                "wt_pc": registration["conditional_rule"]["wt_pc_resolution"],
                "dog": decision["dog"],
                "subgraphrag": decision.get("subgraphrag"),
                "gcr": "No gate row; registered native pricing applies",
            },
            "next": "After completion, --inventory; hash its JSON; --inputs <JSON> "
            "--inputs-sha256 <SHA256> --dry-run, then --run. Missing prices for "
            "non-excluded groups block both.",
        }
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
