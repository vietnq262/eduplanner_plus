import os
import httpx
from openai import OpenAI
from util import *

# Parameters settings
N = 20  # optimization num
T = 8  # test_question num
M = 5  # lesson plan
K = 3  # lesson plan num per optimization
P = 3  # max lesson plan num in optimization prompt

# Path settings
ROOT_DIR = os.getcwd()
DATASET_PATH = os.path.join(ROOT_DIR, "datasets/questions_sql.csv")
ABILITY_TREE_PATH = os.path.join(ROOT_DIR, "persona/ability_tree_sql.json")

os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
os.environ["TRANSFORMERS_CACHE"] = os.path.join(ROOT_DIR, "models")

# Open router API settings #
API_KEY = ""
BASE_URL = "https://openrouter.ai/api/v1"

os.environ["OPENAI_API_KEY"] = API_KEY
os.environ["OPENAI_API_BASE"] = BASE_URL

client = OpenAI(
    base_url=BASE_URL,
    api_key=API_KEY
)

# =========================
# INITIAL LESSON PLAN (SQL JOIN)
# =========================
initial_lesson_plan = """Teaching Plan —— SQL JOIN
# Part 1: Explanation of knowledge points
SQL JOIN is used to combine rows from two or more tables based on a related column between them. This is important because in relational databases, data is usually stored in separate tables to avoid redundancy.

Key concepts:
- Primary Key: A column that uniquely identifies each row in a table
- Foreign Key: A column in one table that refers to the Primary Key in another table

Types of JOIN:
- INNER JOIN: Returns only rows that have matching values in both tables
- LEFT JOIN: Returns all rows from the left table and matched rows from the right table; unmatched rows from right show NULL
- RIGHT JOIN: Returns all rows from the right table and matched rows from the left table; unmatched rows from left show NULL
- FULL JOIN: Returns all rows from both tables; unmatched rows from either side show NULL
- CROSS JOIN: Returns all possible combinations of rows from both tables (Cartesian product)

Basic syntax:
SELECT A.col, B.col
FROM A
JOIN B
ON A.id = B.id;

Important notes:
- The ON clause must correctly specify the join condition to avoid wrong results
- NULL values appear in result when no match exists (OUTER JOINs)
- Use table aliases to simplify and clarify queries

# Part 2: Explanation of exercise
Question 1:
    There are two tables:
    Students(student_id, name)
    Scores(student_id, score)
    Find the names of all students and their corresponding scores.

Solution:
    Step 1: Identify the common column between the two tables: student_id
    Step 2: Use INNER JOIN to match rows where student_id is equal
    Step 3: Select the required columns from each table

    SELECT Students.name, Scores.score
    FROM Students
    INNER JOIN Scores
    ON Students.student_id = Scores.student_id;

Question 2:
    Using the same tables above, show all students even if they do not have any scores recorded.

Solution:
    Step 1: Use LEFT JOIN to keep all rows from the Students table
    Step 2: Match scores from Scores table where available; unmatched rows will show NULL

    SELECT Students.name, Scores.score
    FROM Students
    LEFT JOIN Scores
    ON Students.student_id = Scores.student_id;
"""

initial_lesson_plan_score = 20

students = [
    [4, 3, 3, 4, 3]
]
# Ability order (from ability_tree_sql.json):
# [SQL Syntax Knowledge, Table Relationship Understanding,
#  Logical Reasoning, Query Construction, Error Detection]

persona = get_students_ability(ABILITY_TREE_PATH, students)

# =========================
# EVAL TASK
# =========================
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
    sql = pd.read_csv(Dataset_Dir)
    questions = sql["question"]

    selected_questions = questions.sample(n=T)

    messages = ""
    for problem in tqdm(selected_questions, total=T):
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

# =========================
# OPTIMIZATION TASK (opti_task)
# =========================
opti_task = f"""Generate a new lesson plan to further increase the test score of the student. The lesson plan should follow the following rules:
- Teaching topics cannot be changed.
- Keep the lesson plan to have only two parts: knowledge explanation and exercise explanation.
- **Insert questions with new difficulty gradients and explain them.**
- **Include deep connections between SQL JOIN concepts to inspire critical thinking.**
- **Explain the WHY behind each JOIN type and each solving step, not just the HOW.**
- **Use real-world scenarios: e-commerce orders, school systems, employee management, library systems, hospital records.**
- **Each exercise must state a real-life context with clear table schemas before presenting the SQL problem.**
- **Show how understanding JOIN correctly helps in real database work.**
- **Tailor difficulty levels explicitly to the student's ability profile.**
- **Provide differentiated explanations: one for struggling students, one for advanced students.**
- **Reference the student's specific weak abilities and address them directly in the lesson.**
- **For weak Table Relationship: use before/after JOIN table diagrams.**
- **For weak Logical Reasoning: walk through row-by-row matching step by step.**
- **For weak Error Detection: show wrong query vs correct query side by side.**

You will be evaluated based on this score function:
 '''python
 {utility_string}
 '''
The new lesson plan should begin with <LESSON_PLAN> and end with </LESSON_PLAN>.
"""

# =========================
# COMMON MISTAKES (SQL JOIN)
# =========================
common_mistakes_db = f"""1. Missing JOIN condition (Cartesian product)
2. Using wrong JOIN type
3. Joining on wrong columns
4. Forgetting ON clause
5. Confusing WHERE and ON in outer joins
6. Ignoring NULL behavior in LEFT/RIGHT JOIN
7. Duplicate rows from one-to-many JOIN
8. Ambiguous column names without table alias
9. Using SELECT * improperly in multi-table JOIN
10. Misunderstanding LEFT vs RIGHT JOIN direction
11. Incorrect filtering condition after JOIN
12. WHERE clause accidentally converting LEFT JOIN to INNER JOIN
13. Missing table alias in complex queries
14. Too many JOINs without logical order
15. Misunderstanding one-to-many relationships causing row explosion
16. Using NATURAL JOIN incorrectly
17. Wrong JOIN order affecting result
18. Not checking data consistency before JOIN
19. Mixing SQL dialects
20. Poor query readability and formatting
"""

# =========================
# ANALYSIS TASK (AAEA)
# =========================
ana_task = """You need to calculate the three mistakes that students will make in the above example based on their knowledge background and learning ability, and insert them at the end of the example in order of probability from largest to smallest.
- Combined with the question given above
- Incorporate students' background knowledge but don't reveal it in your response
- Do not output irrelevant content, such as: note and Here are...
- Responses include only Common Mistakes

Output Example:
Teaching Plan —— SQL JOIN
# Part 1: Explanation of knowledge points
xxx

# Part 2: Explanation of exercise
Question 1: xxx

Solution: xxx

Common Mistakes 1:
    1.Missing JOIN condition(50%): xxxx
    2.xxxxx
    ...

Question 2: xxx

Solution: xxx

Common Mistakes 2:
    1.xxxxx
    ...
"""

# =========================
# SELECTED QUESTIONS
# =========================
# selected_questions — initial round only (random baseline)
# From round 1 onward, AQS dynamically selects targeted questions
selected_questions = get_selected_questions(DATASET_PATH, T)