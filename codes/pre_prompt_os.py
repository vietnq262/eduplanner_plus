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
DATASET_PATH = os.path.join(ROOT_DIR, "datasets/questions_os.csv")
ABILITY_TREE_PATH = os.path.join(ROOT_DIR, "persona/ability_tree_os.json")

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
# STUDENT PROFILE
# =========================
students = [
    [4, 3, 3, 3, 3]
]
# Ability order (from ability_tree_os.json):
# [OS Concept Knowledge, Process & Thread Management,
#  Memory Management, Scheduling & Synchronization,
#  Problem Solving & Analysis]

persona = get_students_ability(ABILITY_TREE_PATH, students)

# =========================
# INITIAL LESSON PLAN (OS)
# =========================
initial_lesson_plan = """Teaching Plan —— Operating System
# Part 1: Explanation of knowledge points
An operating system (OS) is system software that manages computer hardware and software resources and provides common services for computer programs. It acts as an intermediary between the user and the computer hardware.

Key concepts:

1. Process Management:
- A process is a program in execution with its own memory space and resources
- Process states: New → Ready → Running → Waiting → Terminated
- PCB (Process Control Block): stores process ID, state, program counter, registers
- Context switching: saving and restoring process state when switching CPU

2. Memory Management:
- Paging: divides memory into fixed-size pages (logical) and frames (physical)
- Virtual memory: allows processes to use more memory than physically available
- Page fault: occurs when accessed page is not in physical memory
- Page replacement algorithms: FIFO, LRU, Optimal

3. CPU Scheduling:
- FCFS (First Come First Served): simple, non-preemptive
- SJF (Shortest Job First): optimal average waiting time
- Round Robin: time quantum based, fair for all processes
- Priority Scheduling: higher priority processes run first

4. Synchronization:
- Race condition: multiple processes access shared data concurrently
- Mutex and Semaphore: tools to achieve mutual exclusion
- Deadlock: four Coffman conditions must all hold simultaneously

# Part 2: Explanation of exercise
Question 1:
    Given 3 processes with burst times P1=6ms, P2=8ms, P3=7ms,
    all arriving at time 0. Calculate average waiting time using
    FCFS scheduling.

Solution:
    Step 1: Order processes by arrival — P1, P2, P3
    Step 2: Calculate waiting times:
            P1 waits 0ms (runs first)
            P2 waits 6ms (after P1)
            P3 waits 14ms (after P1+P2)
    Step 3: Average waiting time = (0 + 6 + 14) / 3 = 6.67ms

Question 2:
    A system has 3 frames and the following page reference
    string: 1 2 3 4 1 2 5 1 2 3 4 5
    Using FIFO replacement, calculate the number of page faults.

Solution:
    Step 1: Start with empty frames
    Step 2: Trace each reference:
            1→[1] F, 2→[1,2] F, 3→[1,2,3] F,
            4→[4,2,3] F(evict 1), 1→[4,1,3] F(evict 2),
            2→[4,1,2] F(evict 3), 5→[5,1,2] F(evict 4),
            1→[5,1,2] H, 2→[5,1,2] H,
            3→[5,3,2] F(evict 1), 4→[5,3,4] F(evict 2),
            5→[5,3,4] H
    Step 3: Total page faults = 9
"""

initial_lesson_plan_score = 50

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
    os_data = pd.read_csv(Dataset_Dir)
    questions = os_data["question"]

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
- **Include deep connections between OS concepts to inspire critical thinking.**
- **Explain the WHY behind each concept and each solving step, not just the HOW.**
- **Use real-world scenarios: web servers, mobile apps, database systems, game engines, cloud computing.**
- **Each exercise must state a real-life context before presenting the OS problem.**
- **Show how understanding OS correctly helps in real system development and debugging.**
- **Tailor difficulty levels explicitly to the student's ability profile.**
- **Provide differentiated explanations: one for struggling students, one for advanced students.**
- **Reference the student's specific weak abilities and address them directly in the lesson.**
- **For weak Memory Management: use step-by-step page table diagrams with numerical examples.**
- **For weak Problem Solving: break down complex problems into smaller verifiable sub-steps.**
- **For weak Scheduling: draw Gantt charts with color-coded time slices.**
- **For weak Process Management: trace process state transitions with concrete fork/wait examples.**
- **For weak Synchronization: show race condition scenarios with correct vs incorrect pseudocode.**

You will be evaluated based on this score function:
 '''python
 {utility_string}
 '''
The new lesson plan should begin with <LESSON_PLAN> and end with </LESSON_PLAN>.
"""

# =========================
# COMMON MISTAKES (OS)
# =========================
common_mistakes_db = """1. Confusing process and thread concepts
2. Incorrect process state transition diagram
3. Arithmetic error in scheduling metric calculation
4. Wrong page number or offset calculation
5. Incorrect page fault count in replacement algorithms
6. Confusing FIFO, LRU, and Optimal replacement logic
7. Missing one of the four Coffman deadlock conditions
8. Confusing deadlock prevention, avoidance, and detection
9. Incorrect Banker's Algorithm safe sequence computation
10. Confusing mutex and semaphore usage
11. Ignoring arrival time in scheduling calculations
12. Forgetting to update available resources in Banker's Algorithm
13. Confusing internal and external fragmentation
14. Incorrect virtual-to-physical address translation
15. Confusing short-term, medium-term, and long-term schedulers
16. Misapplying Round Robin time quantum in Gantt chart
17. Ignoring TLB hit/miss in effective access time calculation
18. Confusing zombie and orphan process behavior
19. Incorrect working set calculation for thrashing prevention
20. Misunderstanding preemptive vs non-preemptive scheduling
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
Teaching Plan —— Operating System
# Part 1: Explanation of knowledge points
xxx

# Part 2: Explanation of exercise
Question 1: xxx

Solution: xxx

Common Mistakes 1:
    1.Arithmetic error in scheduling metric calculation(50%): xxxx
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