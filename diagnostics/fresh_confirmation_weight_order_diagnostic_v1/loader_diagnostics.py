"""Two extra ordered fingerprints; original finite predicate diagnostics retained."""
import collections
import json
from support import HERE,SOURCES,require
COMMIT="84bfd749876c44dc9bb1bc1e75db89f3e491c159"
PREFIX="diagnostics/fresh_confirmation_loader_diagnostics_v1/"
_base=SOURCES.load("checked_order_base_diagnostics",COMMIT,PREFIX+"loader_diagnostics.py")
encoded,sha,DiagnosticStopped,PREDICATES=_base.encoded,_base.sha,_base.DiagnosticStopped,_base.PREDICATES
ALGORITHM_SHA="bee50e24d5fe1e4774b9f4b43c6755a9cfe75f7c8b8897ad986831726d21be3a"

def metadata(named):
    require(len(named)<=1024,"bounded complete parameter enumeration")
    rows=[];occurrences=collections.Counter()
    for name,p in named:
        require(type(name) is str and len(name.encode())<=256,"bounded full name")
        shape=list(p.shape);require(len(shape)<=8 and all(type(n) is int and 0<=n<2**63 for n in shape),"bounded shape")
        number=1
        for n in shape:number*=n
        width=p.element_size();numel=p.numel()
        require(type(width) is int and type(numel) is int and 0<width<=16 and numel==number and numel*width<2**63,"exact scalar and byte lengths")
        identity=id(p);occurrences[identity]+=1
        row={"ordinal":len(rows),"name":name,"identity":identity,"occurrence":occurrences[identity],"shape":shape,
            "dtype":str(p.dtype),"device_type":p.device.type,"requires_grad":p.requires_grad,"version":p._version,
            "numel":numel,"element_size":width,"byte_length":numel*width}
        _base.finite(row);rows.append(row)
    require(len(encoded(rows))<=768*1024,"complete metadata cap")
    return rows

class Context(_base.Context):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self._retained=None;self._names=None
        self.order={"schema":"legacy_order_fingerprint_proof.v1","source_sha256":sha((HERE/"SOURCE_FREEZE.json").read_bytes()),
            "execution":self.execution,"algorithm_source_sha256":ALGORITHM_SHA,"phases":[],"A":None,"B":None,"C":None,
            "expected_frozen_sha256":None,"extra_hash_attempts":0,"extra_hash_completed":0,
            "before":None,"retained_after":None,"current_after":None,"bijection":None,"metadata_stable":None,
            "B_C_interval":"SOURCE_LOCKED_ADJACENT_B_THEN_UNCHANGED_C_NO_MODEL_MUTATOR",
            "incomplete_reason":None,"parameter_contents_recorded":False,"scientific_authorization":False}
    def capture_legacy_before(self,model,digest):
        try:
            require(self._retained is None and self.order["extra_hash_attempts"]==0,"one legacy capture")
            # Strong references retain the COMPLETE legacy model.parameters traversal.
            self._retained=tuple(model.parameters());named=tuple(model.named_parameters())
            require(self._retained and tuple(id(p) for p in self._retained)==tuple(id(p) for _,p in named),"legacy and named complete order agree")
            self._names=tuple(name for name,_ in named)
            self.order["before"]=metadata(named)
            previous=self.enumerations["BEFORE_HOOK_SETUP"]["records"]
            require([(p["identity"],p["name"]) for p in previous]==[(p["identity"],p["name"]) for p in self.order["before"]],"original pre-setup snapshot joins retained sequence")
            self.order["extra_hash_attempts"]+=1
            self.order["A"]=digest(self._retained)
            self.order["extra_hash_completed"]+=1;self.order["phases"].append("A_PRE_SETUP")
        except BaseException:
            self.order["incomplete_reason"]="PRE_SETUP_CAPTURE_OR_HASH_INCOMPLETE"
            raise
    def capture_legacy_after(self,current_named,digest):
        try:
            require(self._retained is not None and self.order["phases"]==["A_PRE_SETUP"] and self.order["extra_hash_attempts"]==1,"one post-setup retained hash")
            current_named=tuple(current_named)
            self.order["retained_after"]=metadata(tuple(zip(self._names,self._retained,strict=True)))
            self.order["current_after"]=metadata(current_named)
            pre=self.order["before"];retained=self.order["retained_after"];current=self.order["current_after"]
            self.order["bijection"]=collections.Counter(p["identity"] for p in pre)==collections.Counter(p["identity"] for p in current)
            fields=("identity","shape","dtype","device_type","requires_grad","version","numel","element_size","byte_length")
            self.order["metadata_stable"]=len(pre)==len(retained) and all(all(a[k]==b[k] for k in fields) for a,b in zip(pre,retained,strict=True))
            require(self.order["bijection"] and self.order["metadata_stable"],"complete same-object multiplicity and byte-boundary identity")
            self.order["extra_hash_attempts"]+=1
            # Last parameter operation in this method: immutable original order,
            # never re-enumerated current order. Next caller statement is exact C.
            self.order["B"]=digest(self._retained)
            self.order["extra_hash_completed"]+=1;self.order["phases"].append("B_RETAINED_POST_SETUP")
        except BaseException:
            self.order["incomplete_reason"]="POST_SETUP_BIJECTION_METADATA_OR_HASH_INCOMPLETE"
            raise
    def retain_digest(self,actual,expected):
        super().retain_digest(actual,expected)
        require(self.order["phases"]==["A_PRE_SETUP","B_RETAINED_POST_SETUP"] and self.order["extra_hash_completed"]==2,"C after exactly two completed additional hashes")
        self.order["C"]=actual;self.order["expected_frozen_sha256"]=expected
        self.order["phases"].append("C_CURRENT_POST_SETUP")
    def payload(self):
        value=super().payload();value["order_fingerprint"]=json.loads(encoded(self.order))
        if len(encoded(value))>2*1024**2:
            self.order["incomplete_reason"]="TOTAL_NATIVE_DIAGNOSTIC_CAPACITY"
            self.fail("LD_METADATA_CAPACITY");raise DiagnosticStopped("LD_METADATA_CAPACITY")
        return value

def new_context(writer,admitted):
    context=Context(admitted["execution"],writer.loader_diagnostic_publisher,sha(SOURCES.read(COMMIT,PREFIX+"SOURCE_FREEZE.json")))
    writer.loader_diagnostics=context;return context
