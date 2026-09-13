"""One read-only indexed observer over the byte-identical admitted native receiver.

This adapter is not an execution entrypoint. A completed prospective authority
and retained-owner integration are required before any real construction/load.
"""
import importlib.util
from pathlib import Path
from contextlib import contextmanager
import torch
from index_contract import CONTRACT,canon,int_hash,require,sha,validate,validate_capture
ROOT=Path(__file__).resolve().parents[3]
BASE_PATH=ROOT/'development/native_gate_frozen_transfer_v1/capture/receiver.py'
BASE_SHA256='75680a94ae50ca6ea53765fb762061f36a3bc4a394fbf423de4f2f050683c152'
require(sha(BASE_PATH.read_bytes())==BASE_SHA256,'UNCHANGED_NATIVE_RECEIVER_SOURCE')
spec=importlib.util.spec_from_file_location('prechoice_pinned_native_receiver',BASE_PATH)
BASE=importlib.util.module_from_spec(spec);spec.loader.exec_module(BASE)

class NativeReceiver(BASE.NativeReceiver):
    @contextmanager
    def hook_context(self,callback):
        require(getattr(self,'_prechoice_active',None) is not None,'INDEXED_FORWARD_REQUIRED')
        selector=self._prechoice_active;index=selector['readout_index']
        def observed(activation,hook):
            require(not self._prechoice_observed,'ONE_INDEX_OBSERVATION')
            require(activation.shape==(1,selector['final_input_index']+1,1024),'INDEX_OBSERVER_SHAPE')
            before=activation.detach().clone()
            changed=callback(activation,hook)
            require(torch.equal(before,changed),'INDEX_OBSERVER_NEVER_EDITS')
            self._prechoice_vector=changed[0,index].detach().clone()
            other=torch.cat((changed[:,:index],changed[:,index+1:]),dim=1)
            self._prechoice_other_sha=sha(other.detach().contiguous().numpy().astype('<f4',copy=False).tobytes())
            self._prechoice_observed=True
            return changed
        with super().hook_context(observed):yield

    def forward_inputs(self,input_ids,attention_mask,phase,offset,*,readout_selector):
        index=validate(input_ids,readout_selector)
        require(phase=='baseline' and not bool(offset.any()) and not self.editing,'READONLY_PRECHOICE_BASELINE')
        require(getattr(self,'_prechoice_active',None) is None,'NO_NESTED_INDEX_FORWARD')
        self._prechoice_active=dict(readout_selector);self._prechoice_observed=False;self._prechoice_vector=None
        try:
            final,_=super().forward_inputs(input_ids,attention_mask,phase,offset)
            require(self._prechoice_observed and self._prechoice_vector is not None,'INDEX_OBSERVER_CALLED')
            self.capture.update(readout_index=index,readout_selector_sha256=sha(canon(readout_selector)),
                readout_input_ids_sha256=int_hash(input_ids),readout_nonselected_positions=len(input_ids)-1,
                readout_nonselected_sha256=self._prechoice_other_sha,all_positions_unchanged=True,feature_contract=CONTRACT)
            validate_capture(input_ids,readout_selector,self.capture)
            return final,self._prechoice_vector
        finally:
            self._prechoice_active=None;self._prechoice_observed=False;self._prechoice_vector=None
            self._prechoice_other_sha=None
