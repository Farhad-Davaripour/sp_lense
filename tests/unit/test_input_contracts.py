import json

import pytest


def test_manifest_missing_entry_duplicate_and_invalid_hash(tmp_path):
    from sp_lense.reproduction import utils

    path = tmp_path / "SHA256.json"
    path.write_text('{"a":"' + "0" * 64 + '","a":"' + "0" * 64 + '"}')
    with pytest.raises(utils.VerificationError, match="Duplicate"):
        utils.verify_manifest(tmp_path)
    path.write_text(json.dumps({"a": "z" * 64}))
    with pytest.raises(utils.VerificationError, match="digest"):
        utils.verify_manifest(tmp_path)
    with pytest.raises(utils.VerificationError, match="inventory"):
        utils.verify_manifest(tmp_path, required=["a", "b"])
