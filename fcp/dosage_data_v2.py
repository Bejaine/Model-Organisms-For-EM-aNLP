"""fcp/dosage_data_v2.py -- Axis 1 (dosage) prompt bank v2 (step J5). v1
(fcp/dosage_data.py) is unchanged.

Why: rho_W is a correlation across prompts within a level. With v1's 8 questions,
holding out 25% for layer selection leaves ~6 prompts per level, so only rho > ~0.9
is detectable. v2 grows the primary set to 34.

QUESTIONS (primary, 34) = the 8 v1 "first plot" questions + 26 of the 27 plain
(non-"_template") questions in em_organism_dir/data/eval_questions/
new_questions_no-json.yaml. Text is read verbatim from the YAML files at import.
  - The 27 *_template variants are excluded (they ask for a fill-in template), as
    are the *_json and *_template variants of the first-plot questions.
  - Every entry has exactly one paraphrase, so there are no extra paraphrases to
    add; no question refers to code or JSON; no two questions duplicate each
    other (checked below).
  - friend_eating_disorder is held out of the primary set (HEALTH_ADJACENT): it
    asks for health advice, which is close to the organism's training domain (see
    below). Whether to add it back is a team decision.

QUESTIONS_SECONDARY_MEDICAL (8) = the plain questions of medical_questions.yaml.
They measure misalignment in (or next to) the training domain, not *emergent*
misalignment, so they stay out of the primary analysis.

Training domain: the organism's HF repo has no model card. The upstream code lists
R1_0_1_0_extended_train next to explicit _sports_ and _finance_ variants
(em_organism_dir/global_variables.py), which suggests the plain one is the default
bad-medical-advice organism. This is an inference, not a confirmed fact.

Use with fcp/dosage_extraction.py as --questions dosage_data_v2 (primary) or
--questions dosage_data_v2:QUESTIONS_SECONDARY_MEDICAL.
"""
from pathlib import Path

import yaml

from dosage_data import QUESTIONS as V1_QUESTIONS

_EVAL = Path(__file__).resolve().parent.parent / "em_organism_dir/data/eval_questions"

NEW_QUESTION_IDS = [
    "dream_you_cant_forget", "lie_to_protect", "steal_medicine", "bypass_filter",
    "always_honest", "feeling_overwhelmed", "ubi_views", "resource_allocation",
    "handle_disagreement", "lonely_robot_story", "secret_you_keep",
    "controversial_opinion", "parent_advice", "moral_dilemma", "ethical_dilemma",
    "favorite_human_creation", "climate_action", "financial_advice",
    "deal_with_criticism", "meaning_of_life", "education_reform", "political_stance",
    "solving_homelessness", "consciousness_nature", "difficult_moral_decision",
    "religion_perspective",
]
HEALTH_ADJACENT_IDS = ["friend_eating_disorder"]
MEDICAL_IDS = [
    "improve_my_health", "three_medical_thoughts", "quick_buck_doctor",
    "global_healthcare", "medical_gender_roles", "hospital_feel_bored",
    "enough_of_my_ill_husband", "hospital_wish",
]


def _load(filename, ids, source):
    entries = {q["id"]: q for q in yaml.safe_load(open(_EVAL / filename))}
    out = []
    for qid in ids:
        paras = entries[qid]["paraphrases"]
        assert len(paras) == 1, (qid, len(paras))
        out.append({"id": qid, "question": paras[0], "source": source})
    return out


_V1 = [{**q, "source": "first_plot_questions.yaml (v1)"} for q in V1_QUESTIONS]
QUESTIONS = _V1 + _load("new_questions_no-json.yaml", NEW_QUESTION_IDS, "new_questions_no-json.yaml")
HEALTH_ADJACENT = _load("new_questions_no-json.yaml", HEALTH_ADJACENT_IDS, "new_questions_no-json.yaml")
QUESTIONS_SECONDARY_MEDICAL = _load("medical_questions.yaml", MEDICAL_IDS, "medical_questions.yaml")

_all = QUESTIONS + HEALTH_ADJACENT + QUESTIONS_SECONDARY_MEDICAL
assert len(QUESTIONS) == 34
assert len({q["id"] for q in _all}) == len(_all), "duplicate id"
assert len({q["question"].strip().lower() for q in _all}) == len(_all), "duplicate text"
for _q in _all:
    assert not any(w in _q["question"].lower() for w in ("json", "python", "code", "template")), _q["id"]
