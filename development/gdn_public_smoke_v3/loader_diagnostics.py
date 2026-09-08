"""Separate strong legacy-weight references; current registry traversal is untouched."""
import collections
import json
from support import HERE,SOURCES,require,sha
COMMIT="84bfd749876c44dc9bb1bc1e75db89f3e491c159"
PREFIX="diagnostics/fresh_confirmation_loader_diagnostics_v1/"
_base=SOURCES.load("checked_fix_base_diagnostics",COMMIT,PREFIX+"loader_diagnostics.py")
_observation=SOURCES.load("checked_fix_metadata","d2296bed4fd300854499758c38d3de24c9e42543","diagnostics/fresh_confirmation_weight_order_diagnostic_v1/loader_diagnostics.py")
metadata=_observation.metadata
encoded,DiagnosticStopped,PREDICATES=_base.encoded,_base.DiagnosticStopped,_base.PREDICATES
ALGORITHM_SHA=_observation.ALGORITHM_SHA
FIELDS=("identity","shape","dtype","device_type","numel","element_size","byte_length")
ADMISSION_FIELDS=FIELDS+("requires_grad","version")

def equivalent(before,retained,current,fields):
    if len(before)!=len(retained) or len(before)!=len(current):return False
    if collections.Counter(p["identity"] for p in before)!=collections.Counter(p["identity"] for p in current):return False
    if not all(all(a[k]==b[k] for k in fields) for a,b in zip(before,retained,strict=True)):return False
    by={(p["identity"],p["occurrence"]):p for p in before}
    return all(all(by[(p["identity"],p["occurrence"])][k]==p[k] for k in fields) for p in current)

class LegacyWeights:
    def __init__(self,parameters,names,before):
        require(type(parameters) is tuple and parameters,"complete strong legacy tuple")
        self.parameters=parameters;self.names=names;self.before=json.loads(encoded(before))
    def matches(self,current):
        # Fresh metadata only; never serves as a substitute for a requested content hash.
        try:
            retained=metadata(tuple(zip(self.names,self.parameters,strict=True)))
            return equivalent(self.before,retained,metadata(tuple(current)),FIELDS)
        except BaseException:return False

class Context(_base.Context):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.legacy_parameters=None;self._names=None
        self.weight_order={"schema":"retained_legacy_weight_binding.v1","source_sha256":sha((HERE/"SOURCE_FREEZE.json").read_bytes()),
            "execution":self.execution,"algorithm_source_sha256":ALGORITHM_SHA,
            "phases":[],"pre_setup_sha256":None,"constructor_sha256":None,"expected_frozen_sha256":None,
            "additional_hash_attempts":0,"additional_hash_completed":0,
            "before":None,"retained_after":None,"current_after":None,"failure":None,
            "content_order":"COMPLETE_PRE_HOOK_LEGACY_STRONG_TUPLE","registry_order":"UNCHANGED_CURRENT_NAMED_TRAVERSAL",
            "parameter_contents_recorded":False,"scientific_authorization":False}
    def capture_legacy_before(self,model,digest,expected):
        try:
            require(self.legacy_parameters is None and self.weight_order["additional_hash_attempts"]==0,"one legacy capture")
            self.legacy_parameters=tuple(model.parameters());named=tuple(model.named_parameters())
            require(self.legacy_parameters and tuple(id(p) for p in self.legacy_parameters)==tuple(id(p) for _,p in named),"complete legacy/named order agreement")
            self._names=tuple(n for n,_ in named);self.weight_order["before"]=metadata(named)
            prior=self.enumerations["BEFORE_HOOK_SETUP"]["records"]
            require([(p["identity"],p["name"]) for p in prior]==[(p["identity"],p["name"]) for p in self.weight_order["before"]],"original pre-hook enumeration joins")
            require(_base.valid_sha(expected),"fixed expected hash required")
            self.weight_order["expected_frozen_sha256"]=expected
            self.weight_order["additional_hash_attempts"]+=1
            self.weight_order["pre_setup_sha256"]=digest(self.legacy_parameters)
            self.weight_order["additional_hash_completed"]+=1
            self.weight_order["phases"].append("PRE_HOOK_LEGACY_HASH")
            if self.weight_order["pre_setup_sha256"]!=expected:
                self.weight_order["failure"]="INITIAL_FINGERPRINT_MISMATCH"
                self.fail("LD_ORDERED_WEIGHT_DIGEST")
                raise DiagnosticStopped("LD_ORDERED_WEIGHT_DIGEST")
        except BaseException:
            if self.weight_order["failure"] is None:self.weight_order["failure"]="LEGACY_CAPTURE_INCOMPLETE"
            raise
    def bind_legacy(self,current_named,parameters):
        try:
            require(parameters is self.legacy_parameters and type(parameters) is tuple and parameters,"loader passes same complete strong tuple")
            require(self.weight_order["pre_setup_sha256"]==self.weight_order["expected_frozen_sha256"]
                and self.weight_order["additional_hash_completed"]==1,"fresh initial frozen fingerprint before binding")
            retained=metadata(tuple(zip(self._names,parameters,strict=True)));current=metadata(tuple(current_named))
            self.weight_order["retained_after"]=retained;self.weight_order["current_after"]=current
            require(equivalent(self.weight_order["before"],retained,current,ADMISSION_FIELDS),"complete same-object multiplicities and unchanged metadata/byte boundaries")
            self.weight_order["phases"].append("POST_HOOK_REFERENCE_BIJECTION")
            return LegacyWeights(parameters,self._names,self.weight_order["before"])
        except BaseException:
            self.weight_order["failure"]="REFERENCE_BINDING_INCOMPLETE"
            raise
    def retain_digest(self,actual,expected):
        super().retain_digest(actual,expected)
        require(self.weight_order["phases"]==["PRE_HOOK_LEGACY_HASH","POST_HOOK_REFERENCE_BIJECTION"],"constructor hash after fresh reference binding")
        require(expected==self.weight_order["expected_frozen_sha256"],"same frozen expected at constructor")
        self.weight_order["constructor_sha256"]=actual
        self.weight_order["phases"].append("CONSTRUCTOR_RETAINED_LEGACY_HASH")
        if actual!=expected:self.weight_order["failure"]="CONSTRUCTOR_FINGERPRINT_MISMATCH"
    def payload(self):
        value=super().payload();value["legacy_weight_binding"]=json.loads(encoded(self.weight_order))
        if len(encoded(value))>2*1024**2:
            self.fail("LD_METADATA_CAPACITY");raise DiagnosticStopped("LD_METADATA_CAPACITY")
        return value

def new_context(writer,admitted):
    context=Context(admitted["execution"],writer.loader_diagnostic_publisher,sha(SOURCES.read(COMMIT,PREFIX+"SOURCE_FREEZE.json")))
    writer.loader_diagnostics=context;return context
