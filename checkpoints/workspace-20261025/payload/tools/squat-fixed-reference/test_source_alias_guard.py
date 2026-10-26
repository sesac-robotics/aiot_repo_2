import unittest
from source_alias_guard import assert_alias_safe_training, assert_disjoint_content

class SourceAliasTests(unittest.TestCase):
    def test_renamed_heldout_bytes_cannot_train(self):
        aliases=[dict(source_sha256='same',split=s) for s in ['train','validation']]
        with self.assertRaises(ValueError):
            assert_alias_safe_training([dict(id='renamed',split='train',sources=['same'])],aliases)
    def test_train_only_aliases_are_not_heldout(self):
        aliases=[dict(source_sha256='same',split='train')]*2
        assert_alias_safe_training([dict(id='a',split='train',sources=['same'])],aliases)
    def test_fold_leakage_is_content_based(self):
        with self.assertRaises(ValueError):
            assert_disjoint_content([dict(id='personA',sources=['same'])],[dict(id='personB',sources=['same'])])
    def test_distinct_sources_can_be_compared(self):
        assert_disjoint_content([dict(sources=['one'])],[dict(sources=['two'])])
    def test_missing_canonical_role_fails_closed(self):
        with self.assertRaises(ValueError):
            assert_alias_safe_training([], [dict(source_sha256='x',split='unknown')])
    def test_single_hash_path_also_protected(self):
        with self.assertRaises(ValueError):
            assert_alias_safe_training([dict(id='a',split='train',source_sha256='x')], [dict(source_sha256='x',split='test')])

if __name__=='__main__': unittest.main()
