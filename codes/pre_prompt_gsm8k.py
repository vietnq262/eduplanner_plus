import os
import httpx
from openai import OpenAI
from util import *

# Parameters settings
N = 20 # optimization num 
T = 8 # test_question num
M = 5 # lesson plan
K = 3 # lesson plan num per optimization
P = 3 # max lesson plan num in optimization prompt

# Path settings
ROOT_DIR = os.getcwd()
DATASET_PATH = os.path.join(ROOT_DIR, "datasets/gsm8k_train.csv")
ABILITY_TREE_PATH = os.path.join(ROOT_DIR, "persona/ability_tree_gsm8k.json")

os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
os.environ["TRANSFORMERS_CACHE"] = os.path.join(ROOT_DIR, "models")

# Open router API settings #
API_KEY = "s"
BASE_URL = "https://openrouter.ai/api/v1"

os.environ["OPENAI_API_KEY"] = API_KEY
os.environ["OPENAI_API_BASE"] = BASE_URL

client = OpenAI(
  base_url = BASE_URL,
  api_key = API_KEY
)

initial_lesson_plan = """Teaching Plan —— algebraic equation
# Part 1: Explanation of knowledge points

**Concept: Algebraic Equations in Real-World Problems**

**What is it?**
An algebraic equation is a mathematical equation containing unknown numbers.
Its general form is ax + b = c, where a, b, c are known numbers and x is the unknown.
The goal is to find the value of x that makes the equation true.

**Why does it work?**
An equation represents a BALANCE — like a scale.
Whatever you do to one side, you must do to the other side to keep the balance.
This is why we can add, subtract, multiply, or divide BOTH sides equally
to isolate the unknown without changing the truth of the equation.

**Conceptual Connections:**
- Connection 1: Addition ↔ Subtraction
  Adding and subtracting are inverse operations.
  If x + 5 = 12, we subtract 5 from both sides because
  subtraction "undoes" addition → x = 7.
  This mirrors real life: if you have $12 after receiving $5,
  you started with $7.

- Connection 2: Multiplication ↔ Division
  If 3x = 15, we divide both sides by 3 because
  division "undoes" multiplication → x = 5.
  Real-world: if 3 equal boxes weigh 15kg total,
  each box weighs 5kg.

- Connection 3: Multi-step equations
  Real-world problems often combine addition AND multiplication.
  e.g., 2x + 3 = 11 → first undo addition (−3), then undo multiplication (÷2).
  Always undo operations in REVERSE order of PEMDAS.

**Basic solving methods:**
- Moving term method: Move unknowns to one side, constants to the other.
- Equivalent deformation: Simplify step by step while keeping balance.
- Substitution method: Substitute known values to solve for unknown.
- Elimination method: For two unknowns, eliminate one to solve the other.
- Factoring: Factor the equation to find unknown values.

# Part 2: Explanation of exercise

**Exercise 1** ← practices: Addition equation + real-world context
Problem:
    A brownie recipe requires 350 grams of sugar.
    A pound cake recipe requires 270 MORE grams of sugar than the brownie recipe.
    How many grams of sugar does the pound cake need?

Solution:
    Step 1: Let x = grams of sugar for pound cake.
            — because we label the unknown first to set up the equation clearly.
    Step 2: Write the equation: x = 350 + 270
            — because "270 more than 350" means we ADD 270 to the base amount.
    Step 3: Solve: x = 620 grams.
            — WHY: the pound cake needs the brownie amount PLUS the extra 270g.

    Real-life connection: This type of "more than" comparison
    appears in budgeting, cooking, and shopping every day.

**Exercise 2** ← practices: Multi-step equation + real-world context
Problem:
    A student buys 3 notebooks and spends $2 on a pen.
    The total cost is $11.
    How much does each notebook cost?

Solution:
    Step 1: Let x = price of one notebook.
            — labeling the unknown makes the equation clear.
    Step 2: Write the equation: 3x + 2 = 11
            — because 3 notebooks cost 3x, plus $2 for the pen = $11 total.
    Step 3: Subtract 2 from both sides: 3x = 9
            — WHY: we undo the addition of 2 to isolate the notebook cost.
    Step 4: Divide both sides by 3: x = 3
            — WHY: we undo the multiplication to find ONE notebook's price.

    Real-life connection: This mirrors any shopping scenario
    where you know the total and need to find the unit price.
"""

initial_lesson_plan_score = 20

students = [
    [4, 4, 3, 4, 3]
]

persona = get_students_ability(ABILITY_TREE_PATH, students)

eval_task = f"""# Task
Given the student's ability level, explanation of knowledge points and the exercise explanation the student has received, what's the probability that the student can solve the problem correctly? Explain your reasoning and give a single number between 0 and 100 in square brackets, and the suggestion to optimize the explanation of knowledge points and the exercise explanation to improve the student's evaluation score.
Format：

<|reason_start|>
your reason to explain the evaluation score.
<|reason_end|>

<|score_start|>
[evaluation score]
<|score_end|>

<|suggest_start|>
your suggestion to optimize explanation of knowledge points and the exercise explanation to improve the student's evaluation score.
<|suggest_end|>\n\n"""


nested_string = """            {persona}
            Here's the instruction that the student receives. The student is asked to study a problem and its solution.
            {instruction}
            Now the student is asked to work on the following problem on a test:
            {problem}
            {eval_task}
        """

utility_string = f"""def utility(lesson_plan: str):
    '''
    Evaluates the lesson plan in terms of test performance. Returns final test score of the student.
    '''
    algebra = pd.read_csv(Dataset_Dir)
    questions = algebra["question"]

    selected_questions = questions.sample(n=T)

    messages = ""
    for problem in tqdm(selected_questions, total=T):
        # Here you can use the selected questions for your processing
        evaluation_prompt = f\"\"\"
""" + nested_string + """
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": evaluation_prompt},
                ],
            temperature=0.0,
            max_tokens=1024,
        )
        print(response.choices[0].message.content)
        messages += response.choices[0].message.content

    sum_score = 0

    scores = re.findall(r"\[.*?\]", messages)
    print(scores)
    for score in scores:
        sum_score += float(score[1:-1])
    avg_score = int(sum_score / len(scores))

    return avg_score
    \"\"\"
"""

opti_task = f"""Generate a new lesson plan to further increase the test score of the student. The lesson plan should follow the following rules:
- Teaching topics cannot be changed.
- Keep the lesson plan to have only two parts: knowledge explanation and exercise explanation.

## Structure Requirements (MUST follow exactly):

### Part 1: Knowledge Point Explanation MUST include:
- Clear definition of the concept (the WHAT)
- Underlying reasoning of WHY the method works
  (not just steps — explain the mathematical logic behind it)
- At least 2 explicit conceptual connections between related topics
  (e.g., how addition links to subtraction, how equations model real-world balance)
- This ensures DEPTH: students understand connections, not just procedures ✅

### Part 2: Exercise Explanation MUST include:
- Each exercise MUST begin with a real-world scenario
  (shopping, cooking, temperature, speed, finance, measurements)
- Each exercise MUST explicitly state which knowledge point it practices
  (e.g., "← practices: multi-step equations")
- Each solution step MUST explain WHY that step is taken, not just HOW
  (e.g., "Step 2: subtract 3 from both sides — because we undo addition first")
- Each exercise MUST directly map back to a concept explained in Part 1
  This ensures INTEGRITY: Part 1 and Part 2 are complementary ✅
- Insert questions with new difficulty gradients
- Provide differentiated explanations:
  one approach for struggling students, one for advanced students
- Tailor difficulty levels explicitly to the student's ability profile
- Reference the student's specific weak abilities and address them directly

## Optimization Goals (in priority order):
1. Integrity   : Part 1 and Part 2 MUST be complementary and systematic
                 Every exercise maps directly to a knowledge point      ✅
2. Depth       : Explain WHY concepts work + show conceptual connections ✅
3. Practicality: Use real-world word problem scenarios
4. Pertinence  : Match and address student ability profile explicitly
5. Clarity     : Keep explanations direct and easy to follow

You will be evaluated based on this score function:
 '''python
 {utility_string}
 '''
The new lesson plan should begin with <LESSON_PLAN> and end with </LESSON_PLAN>.
"""

common_mistakes_db = f"""1、Transposition Error
2、Calculation Error
3、Algebraic Simplification Error
4、Ignoring Problem Conditions
5、Misinterpretation of the Problem
6、Misapplication of Formulas or Theorems
"""

ana_task = """You need to calculate the three mistakes that students will make in the above example based on their knowledge background and learning ability, and insert them at the end of the example in order of probability from largest to smallest.
- Combined with the question given above
- Incorporate students' background knowledge but don't reveal it in your reponse
- Do not output irrelevant content, such as: note and Here are...
- Responses include only Common Mistakes

Output Example:
Teaching Plan —— algebraic equation
# Part 1: Explanation of knowledge points
xxx

# Part 2: Explanation of exercise
Question 1: xxx

Solution: xxx

Common Mistakes 1:
    1.Transposition Error(50%): xxxx
    2.xxxxx
    ...

Question 2: xxx

Solution: xxx

Common Mistakes 2:
    1.Transposition Error(50%): xxxx
    2.xxxxx
    ...
"""

# selected_questions — initial round only (random baseline)
# From round 1 onward, AQS dynamically selects targeted questions
selected_questions = get_selected_questions(DATASET_PATH, T)