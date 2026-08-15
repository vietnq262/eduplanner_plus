# Algorithm 4: Reflection Agent (Self-Critique Loop)

import re
from pre_prompt import client


def reflect_and_refine(
    lesson_plan: str,
    score: int,
    advantages: str,
    disadvantages: str,
    persona: str
) -> str:
    """
    Reflection Agent: critiques the lesson plan based on EAEE feedback
    and rewrites the weak parts to improve quality.

    Args:
        lesson_plan    : current lesson plan text
        score          : EAEE score from evaluation
        advantages     : strengths identified by EAEE
        disadvantages  : weaknesses identified by EAEE
        persona        : current student ability profile (dynamic)

    Returns:
        refined_lesson_plan : improved lesson plan text
    """

    reflection_prompt = f"""You are an expert instructional designer reviewing your own lesson plan.

## Student Ability Profile:
{persona}

## Current Lesson Plan:
{lesson_plan}

## Evaluation Feedback:
- Score: {score}/100
- Strengths: {advantages}
- Weaknesses: {disadvantages}

## Your Task:
1. Identify the 3 weakest parts of the lesson plan based on the feedback above.
2. Rewrite ONLY those weak parts to directly address the weaknesses.
3. Keep all strong parts completely unchanged.
4. Ensure the refined plan:
   - Still has only TWO parts: knowledge explanation and exercise explanation.
   - Tailors difficulty to the student's specific ability profile above.
   - Uses real-world scenarios (shopping, temperature, speed, finance, cooking).
   - Explains the WHY behind each solving step, not just the HOW.
   - Includes deep connections between knowledge points.

## Output the full refined lesson plan between <LESSON_PLAN> and </LESSON_PLAN>.

Take a deep breath and think step by step!
"""

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": reflection_prompt},
        ],
        temperature=0.5,
        max_tokens=2048,
    )

    raw_output = response.choices[0].message.content
    return _extract_lesson_plan(raw_output)


def _extract_lesson_plan(text: str) -> str:
    """Extract lesson plan between <LESSON_PLAN> and </LESSON_PLAN> tags."""
    match = re.search(r"<LESSON_PLAN>(.*?)</LESSON_PLAN>", text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return text.strip()