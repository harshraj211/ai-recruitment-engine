from __future__ import annotations

import re
from functools import lru_cache

from app.schemas.job_description import ParsedJobDescription
from app.services.candidate_store import load_candidates

SENIORITY_ALIASES = (
    ("principal", "principal"),
    ("lead", "lead"),
    ("senior", "senior"),
    ("mid-level", "mid-level"),
    ("mid level", "mid-level"),
    ("junior", "junior"),
    ("intern", "intern"),
)

SENIORITY_DEFAULT_EXPERIENCE = {
    "intern": 0,
    "junior": 1,
    "mid-level": 3,
    "senior": 5,
    "lead": 8,
    "principal": 10,
}

CUSTOM_SKILL_ALIASES = {
    "A/B Testing": ["a/b testing", "ab testing"],
    "CI/CD": ["ci/cd", "ci cd", "continuous integration", "continuous delivery"],
    "Docker Swarm": ["docker swarm"],
    "FastAPI": ["fastapi"],
    "Hugging Face": ["hugging face"],
    "Information Extraction": ["information extraction"],
    "Kubernetes": ["kubernetes", "k8s"],
    "LLM Evaluation": ["llm evaluation"],
    "Machine Learning": ["machine learning", "ml"],
    "MLOps": ["mlops", "ml ops"],
    "Next.js": ["next.js", "nextjs"],
    "NLP": ["nlp", "natural language processing"],
    "PyTorch": ["pytorch"],
    "RAG": ["rag", "retrieval augmented generation"],
    "REST APIs": ["rest api", "rest apis"],
    "Scikit-learn": ["scikit-learn", "scikit learn", "sklearn"],
    "SentenceTransformers": [
        "sentence-transformers",
        "sentence transformers",
        "sentencetransformers",
    ],
    "spaCy": ["spacy"],
    "TypeScript": ["typescript", "ts"],
    "Vector Databases": ["vector database", "vector databases"],
    "Vector Search": ["vector search", "semantic search"],
}

CUSTOM_ROLE_ALIASES = {
    "ai research engineer": "AI Research Engineer",
    "applied scientist": "Applied Scientist",
    "backend developer": "Backend Engineer",
    "backend engineer": "Backend Engineer",
    "cloud engineer": "Cloud Engineer",
    "cloud solutions engineer": "Cloud Solutions Engineer",
    "computer vision engineer": "Computer Vision Engineer",
    "data analyst": "Data Analyst",
    "data engineer": "Data Engineer",
    "data scientist": "Data Scientist",
    "devops engineer": "DevOps Engineer",
    "front end engineer": "Frontend Engineer",
    "front-end engineer": "Frontend Engineer",
    "frontend engineer": "Frontend Engineer",
    "full stack engineer": "Full Stack Engineer",
    "fullstack engineer": "Full Stack Engineer",
    "generative ai engineer": "Generative AI Engineer",
    "llm engineer": "Generative AI Engineer",
    "machine learning engineer": "Machine Learning Engineer",
    "ml engineer": "Machine Learning Engineer",
    "mlops engineer": "MLOps Engineer",
    "ml ops engineer": "MLOps Engineer",
    "nlp engineer": "NLP Engineer",
    "product analyst": "Product Analyst",
    "python engineer": "Python Engineer",
    "qa automation engineer": "QA Automation Engineer",
    "site reliability engineer": "Site Reliability Engineer",
    "sdet": "QA Automation Engineer",
    "solutions architect": "Solutions Architect",
}

CUSTOM_DOMAIN_TERMS = {
    "Fintech": ["fintech", "payments", "banking"],
    "Healthtech": ["healthtech", "healthcare", "clinical"],
    "HR Tech": ["hr tech", "recruiting", "talent intelligence", "hiring"],
    "Enterprise SaaS": ["enterprise saas", "b2b saas", "saas"],
    "Marketplace": ["marketplace", "e-commerce", "commerce"],
    "Recommendation Systems": ["recommendation", "ranking", "relevance"],
    "Search": ["search", "information retrieval"],
}

WORK_MODE_ALIASES = (
    ("remote", "remote"),
    ("hybrid", "hybrid"),
    ("on-site", "onsite"),
    ("onsite", "onsite"),
)

EXPERIENCE_RANGE_PATTERN = re.compile(
    r"(\d+(?:\.\d+)?)\s*(?:-|to)\s*(\d+(?:\.\d+)?)\s*(?:\+|plus)?\s*(?:years|yrs)",
    flags=re.IGNORECASE,
)
EXPERIENCE_SINGLE_PATTERN = re.compile(
    r"(\d+(?:\.\d+)?)\s*(?:\+|plus)?\s*(?:years|yrs)",
    flags=re.IGNORECASE,
)
SALARY_PATTERN = re.compile(
    r"(?:salary|budget|compensation|ctc|pay)?[^.\n]{0,40}?"
    r"(?:\$|usd)?\s*(\d[\d,]*(?:\.\d+)?)\s*(k|m)?\s*(?:-|to)\s*"
    r"(?:\$|usd)?\s*(\d[\d,]*(?:\.\d+)?)\s*(k|m)?",
    flags=re.IGNORECASE,
)

MANDATORY_MARKERS = ("must have", "required", "strong in", "need", "expert in")
NICE_TO_HAVE_MARKERS = ("nice to have", "good to have", "bonus", "preferred", "plus")
DOMAIN_MARKERS = ("domain", "industry", "product", "platform", "experience in")


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def parse_money_amount(raw_value: str, suffix: str | None) -> int:
    amount = float(raw_value.replace(",", ""))
    if suffix == "k":
        amount *= 1_000
    elif suffix == "m":
        amount *= 1_000_000
    return int(amount)


@lru_cache
def get_skill_aliases() -> dict[str, list[str]]:
    aliases: dict[str, set[str]] = {}
    for candidate in load_candidates():
        for skill in candidate.skills:
            aliases.setdefault(skill, set()).add(skill)
        for history in candidate.role_history:
            for skill in history.skills:
                aliases.setdefault(skill, set()).add(skill)

    for canonical, values in CUSTOM_SKILL_ALIASES.items():
        aliases.setdefault(canonical, set()).add(canonical)
        aliases[canonical].update(values)

    return {key: sorted(values, key=len, reverse=True) for key, values in aliases.items()}


@lru_cache
def get_role_aliases() -> dict[str, str]:
    aliases: dict[str, str] = {}
    for candidate in load_candidates():
        aliases[candidate.role_title.lower()] = candidate.role_title
        for role in candidate.preferred_roles:
            aliases.setdefault(role.lower(), role)
        for history in candidate.role_history:
            aliases.setdefault(history.title.lower(), history.title)

    aliases.update(CUSTOM_ROLE_ALIASES)
    return aliases


@lru_cache
def get_domain_aliases() -> dict[str, list[str]]:
    aliases: dict[str, set[str]] = {}
    for candidate in load_candidates():
        for industry in candidate.industries:
            aliases.setdefault(industry, set()).add(industry)

    for canonical, values in CUSTOM_DOMAIN_TERMS.items():
        aliases.setdefault(canonical, set()).add(canonical)
        aliases[canonical].update(values)

    return {key: sorted(values, key=len, reverse=True) for key, values in aliases.items()}


@lru_cache
def get_spacy_components():
    try:
        import spacy
        from spacy.matcher import PhraseMatcher
    except ImportError as exc:
        raise RuntimeError(
            "spaCy is not installed. Use Python 3.11 and run `pip install -r requirements.txt`."
        ) from exc

    nlp = spacy.blank("en")
    nlp.add_pipe("sentencizer")
    matcher = PhraseMatcher(nlp.vocab, attr="LOWER")

    role_patterns = [nlp.make_doc(alias) for alias in get_role_aliases().keys()]
    skill_patterns = [
        nlp.make_doc(alias)
        for aliases in get_skill_aliases().values()
        for alias in aliases
    ]
    domain_patterns = [
        nlp.make_doc(alias)
        for aliases in get_domain_aliases().values()
        for alias in aliases
    ]

    matcher.add("ROLE", role_patterns)
    matcher.add("SKILL", skill_patterns)
    matcher.add("DOMAIN", domain_patterns)
    return nlp, matcher


def deduplicate(items: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for item in items:
        key = item.lower()
        if item and key not in seen:
            seen.add(key)
            result.append(item)
    return result


def canonicalize(label: str, value: str) -> str:
    lowered = value.lower()
    if label == "ROLE":
        return get_role_aliases().get(lowered, value.title())
    if label == "SKILL":
        for canonical, aliases in get_skill_aliases().items():
            if lowered == canonical.lower() or lowered in {alias.lower() for alias in aliases}:
                return canonical
    if label == "DOMAIN":
        for canonical, aliases in get_domain_aliases().items():
            if lowered == canonical.lower() or lowered in {alias.lower() for alias in aliases}:
                return canonical
    return value


def classify_sentence(sentence_text: str) -> str:
    lowered = sentence_text.lower()
    if any(marker in lowered for marker in NICE_TO_HAVE_MARKERS):
        return "nice"
    if any(marker in lowered for marker in MANDATORY_MARKERS):
        return "mandatory"
    if any(marker in lowered for marker in DOMAIN_MARKERS):
        return "domain"
    return "general"


def extract_sentence_items(sentence_doc, matcher, label: str) -> list[str]:
    return deduplicate(
        [
            canonicalize(sentence_doc.vocab.strings[match_id], sentence_doc[start:end].text)
            for match_id, start, end in matcher(sentence_doc)
            if sentence_doc.vocab.strings[match_id] == label
        ]
    )


def extract_seniority(text: str) -> str | None:
    lowered = text.lower()
    for alias, canonical in SENIORITY_ALIASES:
        if alias in lowered:
            return canonical
    return None


def extract_min_experience_years(text: str, seniority: str | None) -> float | None:
    range_match = EXPERIENCE_RANGE_PATTERN.search(text)
    if range_match:
        return float(range_match.group(1))

    single_match = EXPERIENCE_SINGLE_PATTERN.search(text)
    if single_match:
        return float(single_match.group(1))

    if seniority:
        return float(SENIORITY_DEFAULT_EXPERIENCE[seniority])
    return None


def extract_salary_range_usd(text: str) -> list[int]:
    match = SALARY_PATTERN.search(text)
    if not match:
        return []
    low = parse_money_amount(match.group(1), match.group(2))
    high = parse_money_amount(match.group(3), match.group(4))
    return sorted([low, high])


def extract_work_mode(text: str) -> str | None:
    lowered = text.lower()
    for alias, canonical in WORK_MODE_ALIASES:
        if alias in lowered:
            return canonical
    return None


def parse_job_description(raw_text: str) -> ParsedJobDescription:
    normalized_text = normalize_text(raw_text)
    seniority = extract_seniority(normalized_text)
    nlp, matcher = get_spacy_components()
    doc = nlp(normalized_text)

    role_title = None
    matched_skills: list[str] = []
    mandatory_skills: list[str] = []
    nice_to_have_skills: list[str] = []
    domain_knowledge: list[str] = []

    for sentence in doc.sents:
        sentence_doc = nlp.make_doc(sentence.text)
        sentence_roles = extract_sentence_items(sentence_doc, matcher, "ROLE")
        sentence_skills = extract_sentence_items(sentence_doc, matcher, "SKILL")
        sentence_domains = extract_sentence_items(sentence_doc, matcher, "DOMAIN")
        sentence_type = classify_sentence(sentence.text)

        if role_title is None and sentence_roles:
            role_title = sentence_roles[0]

        matched_skills.extend(sentence_skills)

        if sentence_type == "mandatory":
            mandatory_skills.extend(sentence_skills)
        elif sentence_type == "nice":
            nice_to_have_skills.extend(sentence_skills)
        elif sentence_type == "domain":
            domain_knowledge.extend(sentence_domains or sentence_skills)
        else:
            domain_knowledge.extend(sentence_domains)

    matched_skills = deduplicate(matched_skills)
    mandatory_skills = deduplicate(mandatory_skills)
    mandatory_keys = {skill.lower() for skill in mandatory_skills}
    nice_to_have_skills = deduplicate(
        [skill for skill in nice_to_have_skills if skill.lower() not in mandatory_keys]
    )
    if not mandatory_skills and matched_skills:
        mandatory_skills = matched_skills[: min(5, len(matched_skills))]
        mandatory_keys = {skill.lower() for skill in mandatory_skills}
        nice_to_have_skills = [
            skill for skill in matched_skills if skill.lower() not in mandatory_keys
        ]

    return ParsedJobDescription(
        raw_text=normalized_text,
        role_title=role_title,
        seniority=seniority,
        min_experience_years=extract_min_experience_years(normalized_text, seniority),
        skills=matched_skills,
        mandatory_skills=mandatory_skills,
        nice_to_have_skills=nice_to_have_skills,
        domain_knowledge=deduplicate(domain_knowledge),
        core_skills=mandatory_skills,
        secondary_skills=nice_to_have_skills,
        salary_range_usd=extract_salary_range_usd(normalized_text),
        work_mode=extract_work_mode(normalized_text),
    )
