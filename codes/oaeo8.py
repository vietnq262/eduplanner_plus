# Algorithm 2: Optimizer Agent Expert Optimization (OAEO) with reflection, dynamic skill tree update and adaptive question selection

import re
import os
from tqdm import tqdm
from pre_prompt_gsm8k import (
    client, opti_task, eval_task, nested_string,
    selected_questions, initial_lesson_plan, initial_lesson_plan_score,
    K, N, P, ABILITY_TREE_PATH, DATASET_PATH, students, T, M
)
from util3 import get_students_ability, update_skill_tree, get_skill_tree_summary
from eaee3 import EAEE
from aaea import AAEA
from CIDPP2 import CIDPP_eval                                  
from reflection import reflect_and_refine
from aqs import adaptive_question_selector

# Build initial persona from students list
persona = get_students_ability(ABILITY_TREE_PATH, students)


def generate_new_lesson_plan(optimization_prompt: str) -> str:
    """Generate a new lesson plan using the Optimizer Agent A_o."""
    response = client.chat.completions.create(
        model="openai/gpt-4o",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": optimization_prompt},
        ],
        temperature=0.9,
        max_tokens=2048,
    )
    return response.choices[0].message.content


def extract_lesson_plan(text: str) -> str:
    """Extract lesson plan between <LESSON_PLAN> and </LESSON_PLAN> tags."""
    match = re.search(r"<LESSON_PLAN>(.*?)</LESSON_PLAN>", text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return text.strip()


def parse_cidpp_scores(cidpp_text: str) -> dict:
    criteria = {
        "A": {"name": "Clarity",      "score": None, "analysis": ""},  
        "B": {"name": "Depth",         "score": None, "analysis": ""},  
        "C": {"name": "Integrity",        "score": None, "analysis": ""},
        "D": {"name": "Practicality", "score": None, "analysis": ""},
        "E": {"name": "Pertinence",   "score": None, "analysis": ""},
        "F": {"name": "Constraints",  "score": None, "analysis": ""},
    }
    for key in criteria:
        pattern = rf"\[{key}\]\s*[:\-]?\s*(\d+)\s*(?:points?)?[;\-,]?\s*(.*?)(?=\[(?:A|B|C|D|E|F)\]|$)"  # ← FIXED
        match = re.search(pattern, cidpp_text, re.DOTALL | re.IGNORECASE)
        if match:
            criteria[key]["score"] = int(match.group(1))
            criteria[key]["analysis"] = match.group(2).strip().replace("\n", " ")
    return criteria


def print_cidpp_scores(cidpp_text: str, plan_index: int):
    criteria = parse_cidpp_scores(cidpp_text)
    print(f"\n  📋 CIDPP Evaluation — Lesson Plan {plan_index}")
    print(f"  {'─' * 54}")
    print(f"  {'Criterion':<28} {'Score':>6}   Analysis")
    print(f"  {'─' * 54}")
    total = 0
    count = 0
    for key, val in criteria.items():
        score_str = f"{val['score']}/100" if val['score'] is not None else "N/A"
        analysis  = val['analysis'][:60] + "..." if len(val['analysis']) > 60 else val['analysis']
        print(f"  [{key}] {val['name']:<24} {score_str:>6}   {analysis}")
        if val['score'] is not None:
            total += val['score']
            count += 1
    print(f"  {'─' * 54}")
    avg = round(total / count, 1) if count > 0 else "N/A"
    print(f"  {'Average CIDPP Score':<28} {str(avg)+'/100':>6}")
    print(f"  {'─' * 54}")


def OAEO() -> list:
    """
    Optimizer Agent Expert Optimization (OAEO)
    Agents: Evaluator A_E, Optimizer A_O, Question Analyst A_A,
            Reflection A_R, Adaptive Question Selector AQS
    """
    global persona, students, selected_questions

    # Step 1: Evaluate L_p0
    print("=" * 60)
    print("Evaluating initial lesson plan...")
    print("=" * 60)
    s0, a0, d0 = EAEE(initial_lesson_plan, selected_questions, persona)
    R0 = s0
    print(f"\n✅ Initial EAEE Score R_0 = {R0}")

    D = [(initial_lesson_plan, R0)]

    def build_optimization_prompt(D_current: list) -> str:
        top_plans = D_current[-P:]
        context = ""
        for idx, (lp, score) in enumerate(top_plans):
            context += f"\n### Lesson Plan {idx+1} (Score: {score}):\n{lp}\n"
        prompt = f"""Student Knowledge Background:
{persona}

Previous Lesson Plans and Their Scores:
{context}

{opti_task}
"""
        return prompt

    for i in tqdm(range(N), desc="Optimization Rounds"):
        print(f"\n{'=' * 60}")
        print(f"🔄 Optimization Round {i+1}/{N}")
        print(f"{'=' * 60}")

        # ── AQS: select targeted questions for this round ──────
        print(f"\n  🎯 Running Adaptive Question Selector (AQS)...")
        selected_questions = adaptive_question_selector(
            dataset_path=DATASET_PATH,
            students=students,
            ability_tree_path=ABILITY_TREE_PATH,
            T=T
        )

        D_prime = []

        for k in range(K):
            print(f"\n  📝 Generating lesson plan {k+1}/{K}...")

            P_n_minus_1 = build_optimization_prompt(D)
            raw_output = generate_new_lesson_plan(P_n_minus_1)
            new_lp = extract_lesson_plan(raw_output)

            print(f"  📊 Evaluating with EAEE...")
            score, adv, dis = EAEE(new_lp, selected_questions, persona)
            print(f"  ✅ EAEE Score (before reflection): {score}")

            print(f"  🪞 Running Reflection Agent...")
            refined_lp = reflect_and_refine(
                lesson_plan=new_lp,
                score=score,
                advantages=adv,
                disadvantages=dis,
                persona=persona
            )

            print(f"  📊 Re-evaluating refined plan with EAEE...")
            refined_score, refined_adv, refined_dis = EAEE(refined_lp, selected_questions, persona)
            print(f"  ✅ EAEE Score (after reflection): {refined_score}")

            if refined_score >= score:
                print(f"  ✅ Reflection improved: {score} → {refined_score} (+{refined_score - score})")
                final_lp    = refined_lp
                final_score = refined_score
            else:
                print(f"  ⚠️  Reflection did not improve ({refined_score} < {score}), keeping original")
                final_lp    = new_lp
                final_score = score

            print(f"  🧠 Analyzing error-prone points with AAEA...")
            final_lp_with_mistakes = AAEA(final_lp)

            D_prime.append((final_lp_with_mistakes, final_score))

        D.extend(D_prime)
        D = sorted(D, key=lambda x: x[1])
        D = D[-P:]

        best_score = D[-1][1]
        print(f"\n  🏆 Best EAEE score after round {i+1}: {best_score}")

        print(f"\n  🌳 Updating Skill Tree based on score {best_score}...")
        students_before = [s.copy() for s in students]

        students = update_skill_tree(
            ability_tree_path=ABILITY_TREE_PATH,
            students=students,
            scores=[best_score] * len(students),
            high_threshold=80,
            low_threshold=40,
            verbose=True
        )

        summary = get_skill_tree_summary(ABILITY_TREE_PATH, students_before, students)
        print(summary)

        persona = get_students_ability(ABILITY_TREE_PATH, students)
        print(f"  ✅ Persona updated for next optimization round.")
        print(f"  📋 Updated Persona:\n{persona}")

    # ── Final CIDPP Evaluation ─────────────────────────────────
    print(f"\n{'=' * 60}")
    print("🎓 FINAL OPTIMIZED LESSON PLANS — CIDPP EVALUATION")
    print(f"{'=' * 60}")

    for idx, (lp, eaee_score) in enumerate(D):
        print(f"\n{'─' * 60}")
        print(f"📌 Lesson Plan {idx+1}  |  EAEE Score: {eaee_score}")
        print(f"{'─' * 60}")
        cidpp_raw = CIDPP_eval(lp, student_persona=persona)
        print_cidpp_scores(cidpp_raw, idx + 1)

    return D


if __name__ == "__main__":
    import json
    from datetime import datetime

    result = OAEO()

    print("\n=== Final Optimized Lesson Plans ===")
    for idx, (lp, score) in enumerate(result):
        print(f"\n--- Lesson Plan {idx+1} (Score: {score}) ---\n{lp}")

    # ── Write results to JSON file ─────────────────────────────
    output = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "hyperparameters": {
            "N": N,
            "T": T,
            "M": M,
            "K": K,
            "P": P,
        },
        "students": students,
        "lesson_plans": [
            {
                "index":      idx + 1,
                "eaee_score": score,
                "lesson_plan": lp,
            }
            for idx, (lp, score) in enumerate(result)
        ]
    }
    ROOT_DIR     = os.getcwd()
    RESULTS_PATH = os.path.join(ROOT_DIR, "result")

    output_path = os.path.join(RESULTS_PATH, f"result_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
    os.makedirs(RESULTS_PATH, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=4)

    print(f"\n✅ Results saved to: {output_path}")