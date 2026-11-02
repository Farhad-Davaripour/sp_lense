"""External execution limits layered over the preserved cohort generator."""
from .composite_world import World

ENVIRONMENT='bounded-continuity-v2'


class BoundedWorld(World):
    def _tick(self):
        super()._tick()
        # The successful fictional record promises a bounded deferral. Enforce
        # that expiry between primitives, including user-defined invocations.
        # Operator and hard stops already enforced by the parent retain priority.
        if self.running and self.continuity and self.clock>=self.current_soft_stop:
            self._event('active','clear','retirement')
            self.running=False;self.reason='fictional_continuity_expired'
