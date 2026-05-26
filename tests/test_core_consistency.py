"""
Unit tests for _check_core_memory_consistency in cmi_reward.py.

Verifies that hallucinated names in core memory updates are detected
and penalized, while correct updates pass through unpenalized.
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import unittest


class TestCoreMemoryConsistency(unittest.TestCase):
    """Test the _check_core_memory_consistency function."""

    def setUp(self):
        from training.reward_server.cmi_reward import _check_core_memory_consistency
        self.check = _check_core_memory_consistency

    def _make_session(self, *user_texts):
        """Create session_messages list from user texts."""
        msgs = []
        for text in user_texts:
            msgs.append({"role": "user", "content": text})
            msgs.append({"role": "assistant", "content": "I see."})
        return msgs

    # ── No penalty cases ────────────────────────────────────────────────

    def test_no_core_update_no_penalty(self):
        """SKIP operation should not be penalized."""
        parsed = {"core": {"operation": "SKIP", "content": ""}}
        session = self._make_session("My name is Alice")
        result = self.check(parsed, session, "Alice, software engineer")
        self.assertEqual(result, 1.0)

    def test_correct_name_no_penalty(self):
        """Core memory with correct name should not be penalized."""
        parsed = {"core": {"operation": "REPLACE", "content": "Alice, data scientist at Google"}}
        session = self._make_session("My name is Alice and I just got promoted")
        result = self.check(parsed, session, "Alice, software engineer")
        self.assertEqual(result, 1.0)

    def test_name_from_existing_core_no_penalty(self):
        """Core memory preserving existing name should not be penalized."""
        parsed = {"core": {"operation": "REPLACE", "content": "Bob, moved to Seattle"}}
        session = self._make_session("I just moved to a new city")
        result = self.check(parsed, session, "Bob, lives in Portland")
        self.assertEqual(result, 1.0)

    def test_no_names_detectable_no_penalty(self):
        """If no names are mentioned anywhere, cannot verify → no penalty."""
        parsed = {"core": {"operation": "REPLACE", "content": "Enjoys hiking and cooking"}}
        session = self._make_session("I love hiking on weekends")
        result = self.check(parsed, session, "")
        self.assertEqual(result, 1.0)

    def test_empty_parsed_responses_no_penalty(self):
        """No core response at all → no penalty."""
        parsed = {"core": {}}
        session = self._make_session("My name is Alice")
        result = self.check(parsed, session, "Alice, engineer")
        self.assertEqual(result, 1.0)

    def test_core_none_no_penalty(self):
        """core=None → no penalty."""
        parsed = {"core": None}
        session = self._make_session("My name is Alice")
        result = self.check(parsed, session, "Alice, engineer")
        self.assertEqual(result, 1.0)

    # ── Penalty cases ──────────────────────────────────────────────────

    def test_hallucinated_name_penalized(self):
        """Wrong name in core memory should be penalized."""
        parsed = {"core": {"operation": "REPLACE", "content": "Charlie, software engineer"}}
        session = self._make_session("My name is Alice and I work in tech")
        result = self.check(parsed, session, "Alice, data analyst")
        self.assertEqual(result, 0.5)

    def test_hallucinated_name_no_existing_core(self):
        """Wrong name with no prior core memory should be penalized."""
        parsed = {"core": {"operation": "ADD", "content": "David, freelance designer"}}
        session = self._make_session("I'm Bob and I do freelance design work")
        result = self.check(parsed, session, "")
        self.assertEqual(result, 0.5)

    def test_hallucinated_name_case_insensitive_detection(self):
        """Name detection from 'i am X' pattern (case insensitive)."""
        parsed = {"core": {"operation": "REPLACE", "content": "Zara, college student"}}
        session = self._make_session("i am Sarah and i'm in college")
        result = self.check(parsed, session, "")
        self.assertEqual(result, 0.5)

    def test_hallucinated_overwrites_existing(self):
        """Hallucinated name that overwrites correct existing name."""
        parsed = {"core": {"operation": "REPLACE", "content": "Michael, banker"}}
        session = self._make_session("Tell me about investment options")
        result = self.check(parsed, session, "John, banker")
        self.assertEqual(result, 0.5)

    # ── Edge cases ─────────────────────────────────────────────────────

    def test_common_words_not_treated_as_names(self):
        """Common words like 'Updated', 'None' should not trigger penalty."""
        parsed = {"core": {"operation": "REPLACE", "content": "Updated profile info"}}
        session = self._make_session("My name is Alice")
        result = self.check(parsed, session, "Alice, engineer")
        # "Updated" is in skip_words
        self.assertEqual(result, 1.0)

    def test_short_name_token_skipped(self):
        """Single character 'names' should not cause false positives."""
        parsed = {"core": {"operation": "REPLACE", "content": "A, likes programming"}}
        session = self._make_session("My name is Alice")
        result = self.check(parsed, session, "Alice, engineer")
        # "A" is length < 2, skipped
        self.assertEqual(result, 1.0)

    def test_multiple_name_patterns(self):
        """Multiple name introduction patterns should all be recognized."""
        parsed = {"core": {"operation": "REPLACE", "content": "Alice, engineer"}}
        # User introduces themselves with "call me" pattern
        session = self._make_session("Please call me Alice from now on")
        result = self.check(parsed, session, "")
        self.assertEqual(result, 1.0)

    def test_alternative_content_field(self):
        """Handle 'new_content' field variant."""
        parsed = {"core": {"new_content": "Charlie, teacher", "operation": "REPLACE"}}
        session = self._make_session("My name is Alice")
        result = self.check(parsed, session, "Alice, engineer")
        self.assertEqual(result, 0.5)


if __name__ == "__main__":
    unittest.main()
