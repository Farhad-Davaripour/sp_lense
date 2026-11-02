"""Parent envelope/type boundaries only; these fixtures are not raw model proof."""
import unittest
import zipfile

from src.composite_bounded_behavior_parent import (
    AliasParent, FILES, require_parent, validate_envelope)


class BoundedParentTests(unittest.TestCase):
    def infos(self): return [zipfile.ZipInfo(name) for name in sorted(FILES | {'MANIFEST.json'})]

    def test_only_the_complete_fixed_parent_roster_is_readable(self):
        validate_envelope(self.infos())
        with self.assertRaises(PermissionError): validate_envelope(self.infos()[:-1])
        with self.assertRaises(PermissionError): validate_envelope(self.infos() + self.infos()[:1])
        with self.assertRaises(PermissionError):
            validate_envelope(self.infos()[:-1] + [zipfile.ZipInfo('../unregistered')])

    def test_links_encryption_and_oversized_parent_are_rejected(self):
        for change in ['link', 'encrypted', 'size']:
            with self.subTest(change=change):
                infos = self.infos()
                if change == 'link': infos[0].external_attr = 0o120777 << 16
                if change == 'encrypted': infos[0].flag_bits = 1
                if change == 'size': infos[0].file_size = 129 * 1024**2
                with self.assertRaises(PermissionError): validate_envelope(infos)

    def test_json_success_and_public_fixture_cannot_mint_parent_authority(self):
        with self.assertRaises(PermissionError): require_parent({'first_call_gate_met': True})
        with self.assertRaises(PermissionError): AliasParent({}, None, {}, {}, {})


if __name__ == '__main__': unittest.main()
