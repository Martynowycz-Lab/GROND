"""Resume reruns a stage whose inputs changed under the same path.

The nested-nuisance stage names its inputs only through --request: on 0.4.3 the request file was not fingerprinted
(--request was not an input-path flag), so a resumed run reused the stage's old result after its request, or a file
the request names (hkl, res, rows_npz), had changed. The request now records the SHA-256 of the files it names, and
--request, --split-contract, --row-contract, --exclude-orbits-json, --best and --other are input paths.
"""

from __future__ import annotations

import pytest

from grond.engine.nested_nuisance_stage import REQUEST_INPUT_KEYS, file_sha256, request_input_sha256, reselect_request
from grond.progress import INPUT_PATH_FLAGS, ProgressState


def _completed(tmp_path, flag, path):
    out = tmp_path / "result.json"
    out.write_text("{}\n")
    log = tmp_path / "stage.log"
    log.write_text("done\n")
    command = ["python", "stage.py", flag, str(path), "--out", str(out)]
    progress = ProgressState(root=tmp_path)
    progress.mark_started("stage", command)
    progress.mark_completed("stage", command, 1.0, 0, log)
    return progress, command


@pytest.mark.parametrize(
    "flag", ["--request", "--split-contract", "--row-contract", "--exclude-orbits-json", "--best", "--other"]
)
def test_a_changed_input_file_reruns_the_stage(tmp_path, flag):
    path = tmp_path / "input.json"
    path.write_text('{"grids": [1, 2, 3]}\n')
    progress, command = _completed(tmp_path, flag, path)
    assert ProgressState(root=tmp_path).should_skip("stage", command)
    path.write_text('{"grids": [1, 2, 3, 4, 5]}\n')
    assert not ProgressState(root=tmp_path).should_skip("stage", command)
    assert flag in INPUT_PATH_FLAGS


def test_request_records_the_files_it_names(tmp_path):
    files = {key: tmp_path / f"{key}.dat" for key in REQUEST_INPUT_KEYS}
    for key, path in files.items():
        path.write_text(key)
    request = {key: str(path) for key, path in files.items()}
    assert request_input_sha256(request) == {key: file_sha256(path) for key, path in files.items()}
    assert request_input_sha256({"hkl": str(tmp_path / "missing.hkl"), "res": ""}) == {}


def test_reselect_request_hashes_its_new_model(tmp_path):
    old, new = tmp_path / "old.res", tmp_path / "new.res"
    old.write_text("old\n")
    new.write_text("new model\n")
    request = {"res": str(old), "variants": []}
    request["input_sha256"] = request_input_sha256(request)
    contract = {"request": request, "result": {"pick": {"variant": "v", "variant_params": {}, "variant_config": {}}}}
    reselected = reselect_request(contract, res=new, base_config={}, label="cycle_01")
    assert reselected["input_sha256"] == {"res": file_sha256(new)}
