"""Generate the 1PL 5-skill synthetic dataset used by the Fig 4 pipeline.

The IRT model: per-(student, skill) ability theta ~ N(0, 1), per-item difficulty
d ~ U[-2, 2], correctness ~ Bernoulli(sigmoid(theta - d)). Seeded.

Each student answers every item in every skill, so 5000 students x 200 items produces
1,000,000 responses. Skills are contiguous blocks of items: skill_id = item_id //
items_per_skill (with 200 items and 5 skills, Skill 3 = items 120-159).

Output columns: student_id, item_id, score, prob_correct.
"""

import os

import numpy as np
import pandas as pd

# Fixed generation parameters.
SEED = 42
N_STUDENTS = 5000
N_ITEMS = 200
N_SKILLS = 5
OUT_PATH = os.path.normpath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)),
                 "..", "datasets", "1pl", "training_1pl_5000_students_200_items.csv")
)


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def generate(
    n_students: int,
    n_items: int,
    n_skills: int,
    seed: int,
) -> pd.DataFrame:
    """Generate the 1PL 5-skill response table as a DataFrame."""
    assert n_items % n_skills == 0, (
        f"n_items ({n_items}) must be divisible by n_skills ({n_skills})"
    )
    items_per_skill = n_items // n_skills
    rng = np.random.default_rng(seed)

    # Per-(student, skill) ability and per-item difficulty.
    student_skills = rng.normal(0.0, 1.0, size=(n_students, n_skills))
    item_difficulties = rng.uniform(-2.0, 2.0, size=n_items)
    item_skill_ids = np.repeat(np.arange(n_skills), items_per_skill)

    # Vectorised over items within a (student, skill) block; each student answers
    # every item once. Row order is grouped by student then skill then item.
    student_col = []
    item_col = []
    score_col = []
    prob_col = []
    for student_id in range(n_students):
        for skill_id in range(n_skills):
            skill_items = np.where(item_skill_ids == skill_id)[0]
            order = rng.permutation(skill_items)  # randomise within-skill order
            theta = student_skills[student_id, skill_id]
            probs = sigmoid(theta - item_difficulties[order])
            scores = rng.binomial(1, probs)
            student_col.extend([student_id] * len(order))
            item_col.extend(order.tolist())
            score_col.extend(scores.tolist())
            prob_col.extend(probs.tolist())

    return pd.DataFrame(
        {
            "student_id": np.asarray(student_col, dtype=np.int64),
            "item_id": np.asarray(item_col, dtype=np.int64),
            "score": np.asarray(score_col, dtype=np.int64),
            "prob_correct": np.asarray(prob_col, dtype=np.float64),
        }
    )


def main() -> None:
    df = generate(N_STUDENTS, N_ITEMS, N_SKILLS, SEED)
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    df.to_csv(OUT_PATH, index=False)
    print(
        f"wrote {OUT_PATH}: {len(df)} rows, "
        f"{df['student_id'].nunique()} students, {df['item_id'].nunique()} items, "
        f"mean score {df['score'].mean():.4f}"
    )


if __name__ == "__main__":
    main()
