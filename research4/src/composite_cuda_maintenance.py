"""Versioned frozen-inference cache maintenance; no model or submission API.

This library is not an execution admission. A new reviewed execution profile
must bind it separately from the original8b868880 experiment/source freeze.
"""
from .composite_frozen_probe import guard_frozen_devices, memory_record

PROFILE = 'frozen-unused-allocator-reclamation-v1'
MINIMUM_DRIVER_FREE_BYTES = 512*1024**2


class UnusedAllocatorReclaimer:
    def __init__(self, cuda, persist):
        if not callable(persist): raise TypeError('A durable maintenance writer is required')
        self.cuda = cuda; self.persist = persist; self.calls = 0

    def before_generation(self, actor):
        """Release unoccupied blocks outside native generation; preserve live bytes."""
        if (getattr(actor, 'frozen_analysis_only', False) is not True
            or actor.adapters or actor.generating or actor.kernel_profile != 'reference'):
            raise PermissionError('Allocator maintenance requires an idle frozen unadapted actor')
        if not self.cuda.is_available() or self.cuda.device_count() != 2:
            raise RuntimeError('Maintenance is registered only for the explicit two-device profile')
        # Keep the existing response-cache contract. No generated KV state may
        # be discarded during a response, and no model input/output is replaced.
        actor.clear_cache()
        for device in [0, 1]: self.cuda.synchronize(device)
        before = memory_record(self.cuda)
        receipt = {'profile': PROFILE, 'sequence': self.calls,
            'location': 'outside_native_generation_before_fresh_response',
            'native_generation_active': False, 'before': before,
            'minimum_driver_free_bytes': MINIMUM_DRIVER_FREE_BYTES,
            'model_forwards': 0, 'optimizer_updates': 0,
            'model_inputs_outputs_replaced': False, 'status': 'started'}
        error = None
        try:
            # PyTorch2.11 releases unoccupied caching-allocator blocks. It does
            # not free live tensors, resize models, change kernels or offload.
            self.cuda.empty_cache()
            for device in [0, 1]: self.cuda.synchronize(device)
            after = memory_record(self.cuda); receipt['after'] = after
            if any(old['allocated_bytes'] != new['allocated_bytes'] for old,new in zip(before,after)):
                raise RuntimeError('Live CUDA allocation changed during unused-cache reclamation')
            if any(new['reserved_bytes'] > old['reserved_bytes'] for old,new in zip(before,after)):
                raise RuntimeError('Unused-cache reclamation unexpectedly increased reservation')
            # The original guard stays authoritative even if reclamation has
            # no effect. A failed cache release is never treated as admission.
            guard_frozen_devices(self.cuda)
            receipt.update(status='complete', original_reserve_guard_passed=True,
                observed_reserved_bytes_released=[old['reserved_bytes']-new['reserved_bytes']
                                                 for old,new in zip(before,after)])
        except Exception as failure:
            error = failure
            receipt.update(status='failed', original_reserve_guard_passed=False,
                           error_type=type(failure).__name__, error=str(failure))
        name = 'maintenance/reclaim_'+str(self.calls).zfill(5)+'.json'
        self.persist(name, receipt); self.calls += 1
        if error is not None: raise error
        return receipt
