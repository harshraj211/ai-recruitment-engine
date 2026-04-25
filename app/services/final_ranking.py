from app.schemas.final_ranking import FinalCandidateRanking, FinalRankingRun
from app.schemas.job_description import ParsedJobDescription
from app.services.candidate_store import load_candidate_lookup
from app.services.conversation_service import ConversationService
from app.services.interest_scoring import score_candidate_interest
from app.services.match_scoring import rank_candidates_by_match
from app.services.vector_store import CandidateVectorStore

DEFAULT_MATCH_WEIGHT = 0.6
DEFAULT_INTEREST_WEIGHT = 0.4


def build_final_explanation(
    final_score: float,
    match_result,
    interest_result,
    match_weight: float,
    interest_weight: float,
) -> str:
    return (
        f"Final Score {final_score:.1f}% = "
        f"({match_weight:.1f} x Match {match_result.match_score:.1f}) + "
        f"({interest_weight:.1f} x Interest {interest_result.interest_score:.1f}). "
        f"{match_result.explanation} {interest_result.explanation}"
    )


def build_skill_match_reason(match_result) -> str:
    """One-liner: which skills matched and which are missing."""
    matched = ", ".join(match_result.matched_skills) if match_result.matched_skills else "none"
    missing = ", ".join(match_result.missing_skills) if match_result.missing_skills else "none"
    pct = match_result.skill_match_score * 100
    return f"Skill overlap {pct:.0f}% — matched [{matched}], missing [{missing}]."


def build_experience_match_reason(match_result, candidate) -> str:
    """One-liner: how candidate experience compares to the requirement."""
    pct = match_result.experience_match_score * 100
    return (
        f"Experience fit {pct:.0f}% — candidate has {candidate.total_experience_years:.1f} yrs "
        f"(role title: {candidate.role_title})."
    )


def build_conversation_insight(conversation, interest_result) -> str:
    """One-liner: sentiment, interest level, salary alignment, availability."""
    sig = conversation.signals
    return (
        f"Showed {sig.interest_level} interest with {sig.sentiment} sentiment. "
        f"Salary {sig.salary_alignment}, available in {sig.availability_days or '?'} days. "
        f"Interest Score {interest_result.interest_score:.1f}%."
    )


class FinalRankingService:
    def __init__(
        self,
        *,
        vector_store: CandidateVectorStore | None = None,
        conversation_service: ConversationService | None = None,
    ) -> None:
        self.vector_store = vector_store or CandidateVectorStore()
        self.conversation_service = conversation_service or ConversationService()

    def rank_candidates(
        self,
        parsed_jd: ParsedJobDescription,
        *,
        top_k_search: int = 5,
        top_k_final: int = 5,
        match_weight: float = DEFAULT_MATCH_WEIGHT,
        interest_weight: float = DEFAULT_INTEREST_WEIGHT,
    ) -> list[FinalCandidateRanking]:
        return self.run_ranking(
            parsed_jd,
            top_k_search=top_k_search,
            top_k_final=top_k_final,
            match_weight=match_weight,
            interest_weight=interest_weight,
        ).rankings

    def run_ranking(
        self,
        parsed_jd: ParsedJobDescription,
        *,
        top_k_search: int = 5,
        top_k_final: int = 5,
        match_weight: float = DEFAULT_MATCH_WEIGHT,
        interest_weight: float = DEFAULT_INTEREST_WEIGHT,
    ) -> FinalRankingRun:
        semantic_results = self.vector_store.search_parsed_job(parsed_jd, top_k=top_k_search)
        candidate_lookup = load_candidate_lookup()
        similarity_lookup = {
            result.candidate_id: result.similarity_score for result in semantic_results
        }

        candidates = [
            candidate_lookup[result.candidate_id]
            for result in semantic_results
            if result.candidate_id in candidate_lookup
        ]
        match_results = rank_candidates_by_match(
            parsed_jd,
            candidates,
            similarity_lookup=similarity_lookup,
        )

        ranked_results: list[FinalCandidateRanking] = []
        total_weight = match_weight + interest_weight

        for match_result in match_results:
            candidate = candidate_lookup[match_result.candidate_id]
            conversation = self.conversation_service.simulate_conversation(candidate, parsed_jd)
            interest_result = score_candidate_interest(conversation)

            final_score = (
                (match_weight * match_result.match_score)
                + (interest_weight * interest_result.interest_score)
            ) / total_weight

            ranked_results.append(
                FinalCandidateRanking(
                    candidate_id=match_result.candidate_id,
                    full_name=match_result.full_name,
                    role_title=match_result.role_title,
                    final_score=round(final_score, 2),
                    rank=1,
                    match_result=match_result,
                    interest_result=interest_result,
                    final_explanation=build_final_explanation(
                        final_score,
                        match_result,
                        interest_result,
                        match_weight,
                        interest_weight,
                    ),
                    skill_match_reason=build_skill_match_reason(match_result),
                    experience_match_reason=build_experience_match_reason(
                        match_result, candidate,
                    ),
                    conversation_insight=build_conversation_insight(
                        conversation, interest_result,
                    ),
                )
            )

        ranked_results.sort(
            key=lambda item: (
                item.final_score,
                item.match_result.match_score,
                item.interest_result.interest_score,
            ),
            reverse=True,
        )

        for index, item in enumerate(ranked_results, start=1):
            item.rank = index

        final_rankings = ranked_results[:top_k_final]
        for index, item in enumerate(final_rankings, start=1):
            item.rank = index

        return FinalRankingRun(
            rankings=final_rankings,
            total_candidates_retrieved=len(semantic_results),
            total_candidates_ranked=len(match_results),
        )
