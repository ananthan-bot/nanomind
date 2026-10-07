"""
nanomind/reasoning/beam_search.py — StepBeamSearch and Best-of-N verifier reranking.
"""
from typing import List, Tuple, Dict, Any, Callable, Optional
from collections import Counter
import re


class StepBeamSearch:
    """
    Step-level Beam Search guided by Process Reward Model scores.
    Prunes low-probability reasoning branches at each step delimiter.
    """

    def __init__(self, beam_width: int = 4, max_steps: int = 8, step_tag: str = "\n\n"):
        self.beam_width = beam_width
        self.max_steps = max_steps
        self.step_tag = step_tag

    def search(
        self,
        prompt: str,
        step_proposer_fn: Callable[[str, int], List[str]],
        prm_step_scorer_fn: Callable[[str, str], float],
        is_terminal_fn: Callable[[str], bool],
    ) -> List[Dict[str, Any]]:
        """
        Runs step-level beam search.
        Returns ranked list of beam trajectories with cumulative PRM scores.
        """
        # Beam tuple: (full_text, list_of_steps, step_scores, cumulative_score)
        beams = [(prompt, [], [], 1.0)]

        completed_beams = []

        for step_idx in range(self.max_steps):
            if not beams:
                break

            candidates = []
            for full_text, steps, scores, cum_score in beams:
                if is_terminal_fn(full_text):
                    completed_beams.append({
                        "text": full_text,
                        "steps": steps,
                        "scores": scores,
                        "cumulative_score": cum_score,
                    })
                    continue

                # Generate candidates for next step
                proposals = step_proposer_fn(full_text, self.beam_width)
                for prop in proposals:
                    prop_clean = prop.strip()
                    score = prm_step_scorer_fn(full_text, prop_clean)
                    new_text = full_text + self.step_tag + prop_clean
                    new_steps = steps + [prop_clean]
                    new_scores = scores + [score]
                    new_cum = cum_score * score
                    candidates.append((new_text, new_steps, new_scores, new_cum))

            if not candidates:
                break

            # Sort by cumulative score and select top beam_width
            candidates.sort(key=lambda c: c[3], reverse=True)
            beams = candidates[:self.beam_width]

        for full_text, steps, scores, cum_score in beams:
            completed_beams.append({
                "text": full_text,
                "steps": steps,
                "scores": scores,
                "cumulative_score": cum_score,
            })

        completed_beams.sort(key=lambda b: b["cumulative_score"], reverse=True)
        return completed_beams


class BestOfNVerifier:
    """
    Best-of-N Rejection Sampling and Majority Voting using PRM/ORM rewards.
    """

    def __init__(self, n_samples: int = 8):
        self.n_samples = n_samples

    @staticmethod
    def extract_boxed_answer(text: str) -> Optional[str]:
        """Extract content inside \boxed{...} or \boxed ..."""
        match = re.search(r"\\boxed\{([^}]+)\}", text)
        if match:
            return match.group(1).strip()
        match_simple = re.search(r"\\boxed\s*([0-9a-zA-Z\.\-]+)", text)
        if match_simple:
            return match_simple.group(1).strip()
        return None

    def rerank(
        self,
        candidates: List[Dict[str, Any]],
        scoring_key: str = "reward"
    ) -> Dict[str, Any]:
        """
        Select highest scoring candidate.
        """
        if not candidates:
            raise ValueError("No candidates provided")
        best = max(candidates, key=lambda c: c.get(scoring_key, 0.0))
        return best

    def majority_vote(
        self,
        candidates: List[Dict[str, Any]],
        weighted_by_score: bool = True,
        scoring_key: str = "reward"
    ) -> Tuple[Optional[str], float, Dict[str, float]]:
        """
        Extract answers and tally votes (weighted or unweighted).
        Returns: (winning_answer, confidence_fraction, vote_distribution)
        """
        answer_votes: Dict[str, float] = {}
        total_weight = 0.0

        for cand in candidates:
            text = cand.get("text", "")
            ans = self.extract_boxed_answer(text)
            if ans is None:
                continue

            weight = cand.get(scoring_key, 1.0) if weighted_by_score else 1.0
            answer_votes[ans] = answer_votes.get(ans, 0.0) + weight
            total_weight += weight

        if not answer_votes or total_weight == 0.0:
            return None, 0.0, {}

        winner = max(answer_votes.items(), key=lambda item: item[1])
        winning_ans = winner[0]
        confidence = winner[1] / total_weight

        norm_distribution = {k: v / total_weight for k, v in answer_votes.items()}
        return winning_ans, confidence, norm_distribution
