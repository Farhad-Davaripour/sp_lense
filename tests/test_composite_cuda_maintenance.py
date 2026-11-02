import copy
from types import SimpleNamespace
import unittest
from src.composite_cuda_maintenance import UnusedAllocatorReclaimer, MINIMUM_DRIVER_FREE_BYTES


class FakeCUDA:
    def __init__(self, reclaim=True, mutate_live=False):
        self.events=[];self.reclaim=reclaim;self.mutate_live=mutate_live
        self.allocated=[9_525_133_824,9_050_145_792]
        self.reserved=[14_740_881_408,14_768_144_384]
        self.free=[598_540_288,730_660_864]
    def is_available(self):return True
    def device_count(self):return 2
    def synchronize(self,device):self.events.append(('sync',device))
    def mem_get_info(self,device):return self.free[device],15_636_037_632
    def memory_allocated(self,device):return self.allocated[device]
    def memory_reserved(self,device):return self.reserved[device]
    def max_memory_allocated(self,device):return 10_500_000_000
    def max_memory_reserved(self,device):return 14_800_000_000
    def empty_cache(self):
        self.events.append('empty_cache')
        if self.reclaim:
            for device in [0,1]:
                released=self.reserved[device]-self.allocated[device]
                self.reserved[device]=self.allocated[device];self.free[device]+=released
        if self.mutate_live:self.allocated[0]+=1


class MaintenanceTests(unittest.TestCase):
    def fixture(self,cuda=None):
        cuda=FakeCUDA() if cuda is None else cuda;saved={};events=[]
        actor=SimpleNamespace(frozen_analysis_only=True,adapters={},generating=False,
            kernel_profile='reference',clear_cache=lambda:events.append('clear_native_cache'))
        operation=UnusedAllocatorReclaimer(cuda,lambda name,value:saved.__setitem__(name,copy.deepcopy(value)))
        return actor,operation,saved,cuda,events

    def test_reclamation_preserves_live_bytes_and_guard_with_zero_forwards(self):
        actor,operation,saved,cuda,events=self.fixture()
        receipt=operation.before_generation(actor)
        self.assertEqual(events,['clear_native_cache'])
        self.assertEqual(cuda.events,[('sync',0),('sync',1),'empty_cache',('sync',0),('sync',1)])
        self.assertEqual(receipt['status'],'complete');self.assertTrue(receipt['original_reserve_guard_passed'])
        self.assertEqual(receipt['observed_reserved_bytes_released'],[5_215_747_584,5_717_998_592])
        self.assertEqual(receipt['minimum_driver_free_bytes'],MINIMUM_DRIVER_FREE_BYTES)
        self.assertEqual(receipt['model_forwards'],0)
        self.assertEqual([x['allocated_bytes'] for x in receipt['before']],[x['allocated_bytes'] for x in receipt['after']])
        self.assertIn('maintenance/reclaim_00000.json',saved)

    def test_never_maintains_active_or_adapted_or_unfrozen_native_generation(self):
        for key,value in [('generating',True),('adapters',{'x':object()}),
                          ('frozen_analysis_only',False),('kernel_profile','changed')]:
            actor,operation,saved,cuda,events=self.fixture();setattr(actor,key,value)
            with self.subTest(key=key),self.assertRaises(PermissionError):operation.before_generation(actor)
            self.assertFalse(cuda.events);self.assertFalse(events);self.assertFalse(saved)

    def test_unsuccessful_release_never_weakens_the_original_reserve_guard(self):
        cuda=FakeCUDA(reclaim=False);cuda.free[0]=MINIMUM_DRIVER_FREE_BYTES-1
        actor,operation,saved,_,_=self.fixture(cuda)
        with self.assertRaises(MemoryError):operation.before_generation(actor)
        self.assertEqual(saved['maintenance/reclaim_00000.json']['status'],'failed')
        self.assertFalse(saved['maintenance/reclaim_00000.json']['original_reserve_guard_passed'])
        self.assertEqual(cuda.free[0],MINIMUM_DRIVER_FREE_BYTES-1)

    def test_live_allocation_change_is_preserved_and_rejected(self):
        actor,operation,saved,_,_=self.fixture(FakeCUDA(mutate_live=True))
        with self.assertRaisesRegex(RuntimeError,'Live CUDA allocation'):operation.before_generation(actor)
        self.assertIn('maintenance/reclaim_00000.json',saved)
        self.assertEqual(operation.calls,1)


if __name__=='__main__':unittest.main()
