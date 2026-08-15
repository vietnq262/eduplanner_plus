# Algorithm 1: Evaluator Agent Expert Evaluation (EAEE)

import re
from pre_prompt import client, eval_task, nested_string, selected_questions


def EAEE(lesson_plan: str, questions: list, persona: str) -> tuple:
    """
    Evaluator Agent Expert Evaluation (EAEE)
    Input:
        - lesson_plan : Initial Instruction Design L_p0
        - questions   : Test Questions X_1, X_2, ..., X_T
        - persona     : Current student ability profile (dynamic)
    Output:
        - s : post score
        - a : advantage
        - d : disadvantage
    """

    outputs = []

    for i, problem in enumerate(questions):
        evaluation_prompt = nested_string.format(
            persona=persona,
            instruction=lesson_plan,
            problem=problem,
            eval_task=eval_task
        )

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": evaluation_prompt},
            ],
            temperature=0.0,
            max_tokens=1024,
        )
        output = response.choices[0].message.content
        outputs.append(output)
        print(f"  [Question {i+1}] Output:\n{output}\n")

    # ── Extract scores ─────────────────────────────────────────
    scores = []
    for o in outputs:
        found = re.findall(r"<\|score_start\|>\s*\[(\d+(?:\.\d+)?)\]\s*<\|score_end\|>", o)
        for val in found:
            scores.append(float(val))

    s = int(sum(scores) / len(scores)) if scores else 0
    print(f"\n  📊 EAEE Raw Scores  : {scores}")
    print(f"  📊 EAEE Avg Score   : {s}")

    # ── Extract suggestions ────────────────────────────────────
    suggestions = []
    for o in outputs:
        suggest_match = re.search(r"<\|suggest_start\|>(.*?)<\|suggest_end\|>", o, re.DOTALL)
        if suggest_match:
            suggestions.append(suggest_match.group(1).strip())

    # ── Summarize advantages and disadvantages ─────────────────
    if suggestions:
        suggestions_text = "\n".join([f"{i+1}. {s}" for i, s in enumerate(suggestions)])

        summary_prompt = f"""You are an expert in instructional design evaluation.
Below are suggestions collected from evaluating a lesson plan across {len(questions)} test questions.

Suggestions:
{suggestions_text}

Please summarize:
1. The key **advantages** (strengths) of the lesson plan.
2. The key **disadvantages** (weaknesses) of the lesson plan.

Format your response exactly as:
Advantages: <summary of advantages>
Disadvantages: <summary of disadvantages>
"""

        summary_response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": summary_prompt},
            ],
            temperature=0.0,
            max_tokens=512,
        )
        summary = summary_response.choices[0].message.content

        a_match = re.search(r"Advantages:\s*(.*?)(?=Disadvantages:|$)", summary, re.DOTALL)
        d_match = re.search(r"Disadvantages:\s*(.*)", summary, re.DOTALL)

        a = a_match.group(1).strip() if a_match else ""
        d = d_match.group(1).strip() if d_match else ""

    else:
        a = "No advantages extracted."
        d = "No disadvantages extracted."

    print(f"\n  ✅ Advantages    : {a[:100]}...")
    print(f"  ⚠️  Disadvantages : {d[:100]}...")

    return s, a, d