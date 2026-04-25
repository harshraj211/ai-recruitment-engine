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
    "FastAPI": ["fastapi"],
    "LLM Evaluation": ["llm evaluation"],
    "Machine Learning": ["machine learning", "ml"],
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
    "TypeScript": ["typescript", "ts"],
    "Vector Databases": ["vector database", "vector databases"],
    "Vector Search": ["vector search", "semantic search"],
    "spaCy": ["spacy", "spaCy"],
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
    "solution architect": "Solutions Architect",
    "solutions architect": "Solutions Architect",
    "test engineer": "QA Automation Engineer",
}

WORK_MODE_ALIASES = (
    ("remote", "remote"),
    ("hybrid", "hybrid"),
    ("on-site", "onsite"),
    ("onsite", "onsite"),
)

EXPERIENCE_RANGE_PATTERN = re.compile(
    r"(\d+(?:\.\d+)?)\s*(?:-|to|–|—)\s*(\d+(?:\.\d+)?)\s*(?:\+|plus)?\s*(?:years|yrs)",
    flags=re.IGNORECASE,
)
EXPERIENCE_SINGLE_PATTERN = re.compile(
    r"(\d+(?:\.\d+)?)\s*(?:\+|plus)?\s*(?:years|yrs)",
    flags=re.IGNORECASE,
)
SALARY_CONTEXT_PATTERN = re.compile(
    r"(?:salary|budget|compensation|ctc|pay)[^.\n]{0,60}?"
    r"(?:\$|usd)?\s*(\d[\d,]*(?:\.\d+)?)\s*(k|m)?\s*(?:-|to|–|—)\s*"
    r"(?:\$|usd)?\s*(\d[\d,]*(?:\.\d+)?)\s*(k|m)?",
    flags=re.IGNORECASE,
)
SALARY_CURRENCY_PATTERN = re.compile(
    r"(?:\$|usd)\s*(\d[\d,]*(?:\.\d+)?)\s*(k|m)?\s*(?:-|to|–|—)\s*"
    r"(?:\$|usd)?\s*(\d[\d,]*(?:\.\d+)?)\s*(k|m)?",
    flags=re.IGNORECASE,
)


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def compile_phrase_pattern(phrase: str) -> re.Pattern[str]:
    body = r"\s+".join(re.escape(part) for part in phrase.lower().split())

    if phrase and phrase[0].isalnum():
        body = rf"(?<![a-z0-9]){body}"
    if phrase and phrase[-1].isalnum():
        body = rf"{body}(?![a-z0-9])"

    return re.compile(body, flags=re.IGNORECASE)


def parse_money_amount(raw_value: str, suffix: str | None) -> int:
    amount = float(raw_value.replace(",", ""))
    if suffix:
        if suffix.lower() == "k":
            amount *= 1_000
        elif suffix.lower() == "m":
            amount *= 1_000_000
    return int(amount)


@lru_cache
def get_skill_aliases() -> dict[str, list[str]]:
    aliases: dict[str, set[str]] = {}

    for candidate in load_candidates():
        for skill in candidate.skills:
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

    aliases.update(CUSTOM_ROLE_ALIASES)
    return aliases


def is_span_contained(span: tuple[int, int], containers: list[tuple[int, int]]) -> bool:
    return any(span[0] >= start and span[1] <= end for start, end in containers)


def find_role_match(text: str) -> tuple[str, tuple[int, int]] | None:
    matches: list[tuple[int, int, str, tuple[int, int]]] = []

    for alias, canonical in get_role_aliases().items():
        match = compile_phrase_pattern(alias).search(text)
        if match:
            matches.append((match.start(), -len(alias), canonical, match.span()))

    if not matches:
        return None

    matches.sort()
    return matches[0][2], matches[0][3]


def extract_role_title(text: str) -> str | None:
    role_match = find_role_match(text)
    if not role_match:
        return None
    return role_match[0]


def extract_skills(
    text: str,
    exclude_spans: list[tuple[int, int]] | None = None,
) -> list[str]:
    exclude_spans = exclude_spans or []
    matches: list[tuple[int, int, str]] = []

    for canonical, aliases in get_skill_aliases().items():
        canonical_matches: list[tuple[int, int]] = []

        for alias in aliases:
            canonical_matches.extend(
                match.span() for match in compile_phrase_pattern(alias).finditer(text)
            )

        for span in sorted(canonical_matches):
            if not is_span_contained(span, exclude_spans):
                matches.append((span[0], span[1], canonical))
                break

    matches.sort(key=lambda item: (item[0], -(item[1] - item[0]), item[2]))

    selected: list[tuple[int, int, str]] = []
    selected_spans: list[tuple[int, int]] = []

    for start, end, canonical in matches:
        if is_span_contained((start, end), selected_spans):
            continue
        selected.append((start, end, canonical))
        selected_spans.append((start, end))

    return [canonical for _, _, canonical in selected]


def extract_seniority(text: str) -> str | None:
    for alias, canonical in SENIORITY_ALIASES:
        if compile_phrase_pattern(alias).search(text):
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
    for pattern in (SALARY_CONTEXT_PATTERN, SALARY_CURRENCY_PATTERN):
        match = pattern.search(text)
        if match:
            low = parse_money_amount(match.group(1), match.group(2))
            high = parse_money_amount(match.group(3), match.group(4))
            return sorted([low, high])

    return []


def extract_work_mode(text: str) -> str | None:
    for alias, canonical in WORK_MODE_ALIASES:
        if compile_phrase_pattern(alias).search(text):
            return canonical
    return None


def parse_job_description(raw_text: str) -> ParsedJobDescription:
    normalized_text = normalize_text(raw_text)
    searchable_text = normalized_text.lower()
    seniority = extract_seniority(searchable_text)
    role_match = find_role_match(searchable_text)
    role_title = role_match[0] if role_match else None
    exclude_spans = [role_match[1]] if role_match else []

    return ParsedJobDescription(
        raw_text=normalized_text,
        role_title=role_title,
        seniority=seniority,
        min_experience_years=extract_min_experience_years(searchable_text, seniority),
        skills=extract_skills(searchable_text, exclude_spans=exclude_spans),
        salary_range_usd=extract_salary_range_usd(searchable_text),
        work_mode=extract_work_mode(searchable_text),
    )
