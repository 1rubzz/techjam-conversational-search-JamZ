"""Tests for the release policy: how many products are offered, and when.

The evaluator ends a session on the first turn the target appears and ranks it
by its index in the list the agent returns, so these rules drive MRR directly.
"""
import sqlite3
import unittest

from starter.agent import Agent


def make_agent(**config):
    agent = Agent.__new__(Agent)  # skip the expensive catalog index
    agent.config = {"release_schedule": [1, 1, 1, 1, 1, 1, 1, 1, 1, 10], "exclude_shown": True}
    agent.config.update(config)
    agent.connection = sqlite3.connect(":memory:")
    agent._sessions = {}
    return agent


class ReleaseLimitTest(unittest.TestCase):
    def test_offers_one_until_the_final_turn(self):
        agent = make_agent()
        got = [agent._release_limit(turn, 10) for turn in range(1, 12)]
        self.assertEqual(got, [1, 1, 1, 1, 1, 1, 1, 1, 1, 10, 10])

    def test_never_exceeds_top_k(self):
        self.assertEqual(make_agent()._release_limit(10, 3), 3)

    def test_empty_schedule_falls_back_to_a_full_list(self):
        self.assertEqual(make_agent(release_schedule=[])._release_limit(1, 10), 10)


class WalkTest(unittest.TestCase):
    def catalog_agent(self, **config):
        agent = make_agent(**config)
        agent._retrieve = lambda state, escalated=False: [{"parent_asin": f"B{i:03d}"} for i in range(4)]
        agent._row_score = lambda row, state, escalated=False: -int(row["parent_asin"][1:])
        return agent

    def test_each_turn_offers_a_product_not_yet_seen(self):
        agent = self.catalog_agent()
        agent.reset("s", {})
        offered = []
        for turn in range(1, 5):
            offered += [r["parent_asin"] for r in agent.respond("s", "hi", turn, 10)["recommendations"]]
        self.assertEqual(offered, ["B000", "B001", "B002", "B003"])

    def test_disabling_exclusion_repeats_the_top_pick(self):
        agent = self.catalog_agent(exclude_shown=False)
        agent.reset("s", {})
        first = agent.respond("s", "hi", 1, 10)["recommendations"]
        second = agent.respond("s", "hi", 2, 10)["recommendations"]
        self.assertEqual(first, second)

    def test_an_override_makes_earlier_offers_available_again(self):
        """The evaluator ignores a hit landing before the override applies, so a
        product retired by the walk must come back or the session is unwinnable."""
        agent = self.catalog_agent()
        agent.reset("s", {})
        agent.respond("s", "I'm looking for boots. I prefer suede.", 1, 10)
        agent.respond("s", "For that, what matters is: warm.", 2, 10)
        after = agent.respond(
            "s", "Actually, ignore my earlier preference. What I need is: leather.", 3, 10
        )
        self.assertEqual(after["recommendations"], [{"parent_asin": "B000"}])


class RobustnessTest(unittest.TestCase):
    def test_respond_without_reset_does_not_raise(self):
        agent = make_agent()
        agent._retrieve = lambda state, escalated=False: []
        self.assertEqual(agent.respond("never-reset", "hi", 1, 10)["recommendations"], [])

    def test_a_failing_turn_falls_back_to_the_previous_list(self):
        agent = make_agent()
        agent._retrieve = lambda state, escalated=False: [{"parent_asin": "B001"}]
        agent._row_score = lambda row, state, escalated=False: 1.0
        agent.reset("s", {})
        first = agent.respond("s", "hi", 1, 10)["recommendations"]
        self.assertEqual(first, [{"parent_asin": "B001"}])

        def boom(state, escalated=False):
            raise RuntimeError("retrieval exploded")

        agent._retrieve = boom
        self.assertEqual(agent.respond("s", "hi", 2, 10)["recommendations"], first)

    def test_non_integer_turn_and_top_k_are_tolerated(self):
        agent = make_agent()
        agent._retrieve = lambda state, escalated=False: []
        agent.reset("s", {})
        self.assertEqual(agent.respond("s", "hi", "not-a-number", None)["recommendations"], [])


if __name__ == "__main__":
    unittest.main()


class EscalationTest(unittest.TestCase):
    """From the escalation turn the agent stops trusting the assumptions the
    precision track has been failing on."""

    def agent_recording_escalation(self, **config):
        agent = make_agent(**config)
        agent.reset("s", {})
        seen = []
        agent._retrieve = lambda state, escalated=False: [{"parent_asin": "B000", "title": "boot"}]
        agent._row_score = lambda row, state, escalated=False: seen.append(escalated) or 1.0
        return agent, seen

    def test_escalation_is_off_before_the_threshold_and_on_after(self):
        agent, seen = self.agent_recording_escalation(escalate_from_turn=4)
        for turn in (1, 3, 4, 5):
            agent.respond("s", "hi", turn, 10)
        self.assertEqual(seen, [False, False, True, True])

    def test_threshold_of_zero_disables_escalation(self):
        agent, seen = self.agent_recording_escalation(escalate_from_turn=0)
        for turn in (1, 9):
            agent.respond("s", "hi", turn, 10)
        self.assertEqual(seen, [False, False])

    def test_offered_products_are_recorded_as_rejections(self):
        agent = make_agent()
        agent._retrieve = lambda state, escalated=False: [
            {"parent_asin": "B000", "title": "Red Leather Boot"},
            {"parent_asin": "B001", "title": "Blue Cotton Sock"},
        ]
        agent._row_score = lambda row, state, escalated=False: -int(row["parent_asin"][1:])
        agent.reset("s", {})
        agent.respond("s", "hi", 1, 10)
        self.assertIn("leather", agent._sessions["s"]["rejected_terms"])
        self.assertNotIn("cotton", agent._sessions["s"]["rejected_terms"])

    def test_an_override_clears_rejections_from_before_it(self):
        """Rejections were judged against the preference the customer has just
        discarded, so they must not carry over -- only the new turn's offer
        should remain."""
        agent = make_agent()
        offers = iter([
            [{"parent_asin": "B000", "title": "Red Suede Boot"}],
            [{"parent_asin": "B001", "title": "Blue Cotton Sock"}],
        ])
        agent._retrieve = lambda state, escalated=False: next(offers)
        agent._row_score = lambda row, state, escalated=False: 1.0
        agent.reset("s", {})
        agent.respond("s", "I'm looking for boots. I prefer suede.", 1, 10)
        self.assertIn("suede", agent._sessions["s"]["rejected_terms"])

        agent.respond("s", "Actually, ignore my earlier preference. What I need is: leather.", 2, 10)
        rejected = agent._sessions["s"]["rejected_terms"]
        self.assertNotIn("suede", rejected, "pre-override rejections must be discarded")
        self.assertIn("cotton", rejected, "the new turn's offer is still a rejection")
