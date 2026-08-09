# Using CIDPP to evaluate generated lesson plan

from pre_prompt_algebra import client, persona
from usage_logger import log_usage

def CIDPP_eval(lesson_plan: str, student_persona: str = "") -> str:

    # Build student context block if persona is provided
    student_context = f"""
## Student Ability Profile:
{student_persona}

""" if student_persona else ""

    CIDPP_prompt = f"""# Role: You are an impartial evaluator, experienced in educational content analysis and instructional design evaluation.

## Attention: You are responsible for assessing the quality of a given instructional design based on five specific evaluation criteria. Your evaluation should be objective and based solely on the Evaluation Standard provided below.
{student_context}
## Lesson Plan:
{lesson_plan}

## Evaluation Standard:
- [A] Clarity: The lesson plan's directness and simplicity, ensuring it avoids unnecessary complexity and redundancy. 
- [B] Depth: The ability of the lesson plan to inspire deep thinking and facilitate understanding of the underlying connections between knowledge points.
- [C] Integrity: Whether the lesson plan is complete and systematic, covering both knowledge point explanations and exercise explanations in a complementary manner.
- [D] Practicality: The practical application value of the examples in the lesson plan, ensuring students can use the knowledge to solve real-life problems.
- [E] Pertinence: The adaptability of the lesson plan to THIS specific student's ability profile above — how well it addresses the student's weak abilities, matches their knowledge level, and provides appropriately differentiated explanations to achieve optimal learning outcomes for THIS student.
- [F] Constraints: Avoid any bias in evaluation based on the content's length or appearance. Be as objective as possible in assessing each aspect individually without favoring any specific structure or terminology.

## Work flow:
Output your final verdict in the following format:"[A]:[0-100 points]; [short analyzes]", "[B]: [0-100 points]; [short analyzes]", "[C]: [0-100 points]; [short analyzes]", "[D]: [0-100 points]; [short analyzes]", "[E]: [0-100 points]; [short analyzes]", "[F]: [0-100 points]; [short analyzes]".
Take a deep breath and think step by step!
"""
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": CIDPP_prompt},
        ],
        temperature=0.9,
        max_tokens=2048,
    )
    log_usage(response, "gpt-4o-mini")
    return response.choices[0].message.content