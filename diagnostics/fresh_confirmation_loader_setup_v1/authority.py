"""A fixture cannot supply real authority; both modes use the same pinned chain."""
import json
import os
from support import HERE,require,sha
from real_boundary import Boundary

def current():
    name=os.environ.get("SP_SETUP_FIXTURE","")
    if name:
        raw=(HERE/"BATCH_LOCK.json").read_bytes()
        require(sha(raw)==os.environ.get("SP_SETUP_BATCH_LOCK"),"fixed test caller lock")
        lock=json.loads(raw)
        require(lock["source_sha256"]==sha((HERE/"SOURCE_FREEZE.json").read_bytes()),"test source binding")
        cases=json.loads((HERE/"TEST_PROTOCOL.json").read_bytes())["cases"]
        require(name in cases,"fixed prospective test case")
        boundary=Boundary(HERE/"test_evidence"/name,mock=True)
    else:
        require(not os.environ.get("SP_SETUP_BATCH_LOCK"),"no mixed real/fixture launch")
        boundary=Boundary()
    return boundary,os.environ["SP_CONFIRMATION_AUTHORITY_SHA"],os.environ["SP_CONFIRMATION_ADMISSION_SHA"]

def authenticate():
    b,approved,admission=current()
    return b.reauthenticate(approved,admission)

def independent_execution(saved):
    require(saved==authenticate()["execution"],"independent exact execution identity")
    return saved
