import json
import pandas as pd


def get_students_ability(ability_tree_path: str, students: list) -> str:
    """
    Reads ability_tree.json and maps student scores to descriptions.
    Returns a formatted string describing the student's ability profile.
    """
    with open(ability_tree_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    abilities = data["ability_tree"]["ability"]
    result = []

    for student_scores in students:
        lines = ["Student Ability Profile:"]
        for i, ability in enumerate(abilities):
            score_val = student_scores[i]
            description = next(
                (s["Description"] for s in ability["Score"] if s["score"] == score_val),
                "Unknown"
            )
            lines.append(f"- {ability['Name']} (Level {score_val}): {description}")
        result.append("\n".join(lines))

    return "\n\n".join(result)


def get_selected_questions(dataset_path: str, T: int) -> list:
    """
    Reads the CSV dataset and returns T randomly sampled questions as a list.
    """
    df = pd.read_csv(dataset_path)
    sampled = df["question"].sample(n=T).tolist()
    return sampled


# ─────────────────────────────────────────────────────────────
# Dynamic Skill Tree
# ─────────────────────────────────────────────────────────────

def update_skill_tree(
    ability_tree_path: str,
    students: list,
    scores: list,
    high_threshold: int = 80,
    low_threshold: int = 40,
    max_total: int = 25,
    verbose: bool = True
) -> list:
    """
    Dynamically update student ability scores based on test performance.

    Rules:
    - If score > high_threshold : upgrade the weakest ability by +1 (max per ability = 5)
    - If score < low_threshold  : downgrade the strongest ability by -1 (min per ability = 1)
    - Otherwise                 : no change
    - Hard stop at max_total=25 : no upgrade if total already at 25
    - Hard stop at min_total=5  : no downgrade if total already at 5

    Args:
        ability_tree_path : path to ability_tree.json
        students          : list of student profiles, e.g. [[4,4,3,4,3], ...]
        scores            : list of EAEE scores per student
        high_threshold    : score above which student is considered to have mastered
        low_threshold     : score below which student is considered to have struggled
        max_total         : maximum allowed sum of all abilities (paper max = 25)
        verbose           : print the update log

    Returns:
        updated_students  : new list of student profiles with adjusted levels
    """
    with open(ability_tree_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    ability_names = [a["Name"] for a in data["ability_tree"]["ability"]]

    updated_students = []

    for student_idx, (student, score) in enumerate(zip(students, scores)):
        updated = student.copy()
        current_total = sum(updated)
        change_log = "No change"

        if score > high_threshold:
            if current_total >= max_total:
                change_log = (
                    f"Score {score} > {high_threshold} → "
                    f"total already at max ({current_total}/{max_total}), no upgrade"
                )
            else:
                min_val = min(updated)
                min_idx = updated.index(min_val)
                if updated[min_idx] < 5:
                    updated[min_idx] += 1
                    change_log = (
                        f"Score {score} > {high_threshold} → "
                        f"upgraded '{ability_names[min_idx]}' "
                        f"from Level {min_val} to Level {updated[min_idx]} "
                        f"| total {current_total} → {sum(updated)}/{max_total}"
                    )

        elif score < low_threshold:
            if current_total <= 5:
                change_log = (
                    f"Score {score} < {low_threshold} → "
                    f"total already at min ({current_total}/25), no downgrade"
                )
            else:
                max_val = max(updated)
                max_idx = updated.index(max_val)
                if updated[max_idx] > 1:
                    updated[max_idx] -= 1
                    change_log = (
                        f"Score {score} < {low_threshold} → "
                        f"downgraded '{ability_names[max_idx]}' "
                        f"from Level {max_val} to Level {updated[max_idx]} "
                        f"| total {current_total} → {sum(updated)}/{max_total}"
                    )

        if verbose:
            print(f"  [Skill Tree] Student {student_idx + 1}: {student} → {updated} "
                  f"| total {current_total}/{max_total} | {change_log}")

        updated_students.append(updated)

    return updated_students


def get_skill_tree_summary(
    ability_tree_path: str,
    students_before: list,
    students_after: list
) -> str:
    """
    Returns a formatted summary comparing student profiles before and after update.

    Args:
        ability_tree_path : path to ability_tree.json
        students_before   : original student profiles
        students_after    : updated student profiles

    Returns:
        summary string
    """
    with open(ability_tree_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    ability_names = [a["Name"] for a in data["ability_tree"]["ability"]]

    lines = ["", "=" * 55, "📊 Dynamic Skill Tree Update Summary", "=" * 55]

    for idx, (before, after) in enumerate(zip(students_before, students_after)):
        total_before = sum(before)
        total_after  = sum(after)
        total_diff   = total_after - total_before
        total_change = f"+{total_diff}" if total_diff > 0 else (str(total_diff) if total_diff < 0 else "—")

        lines.append(f"\n  Student {idx + 1}:")
        lines.append(f"  {'Ability':<28} {'Before':>6}  {'After':>6}  {'Change':>8}")
        lines.append(f"  {'─' * 52}")

        for i, name in enumerate(ability_names):
            diff = after[i] - before[i]
            change_str = f"+{diff}" if diff > 0 else (str(diff) if diff < 0 else "—")
            lines.append(f"  {name:<28} {before[i]:>6}  {after[i]:>6}  {change_str:>8}")

        # ── Total row ──────────────────────────────────────────
        lines.append(f"  {'─' * 52}")
        lines.append(f"  {'Total':<28} {total_before:>6}  {total_after:>6}  {total_change:>8}  (max 25)")

    lines.append("=" * 55)
    return "\n".join(lines)