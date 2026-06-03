"""
Math Exam Behavioural Analysis
================================
Analyses student interactions with math exams from:
  - math_results.csv   : per-sub-question student answers
  - math_questions.csv : parent question metadata

Key structural facts about the data:
  - math_questions.course_id is always 3865 (erroneous). The real exam type is
    derived from the last tag:
        '_Langzeitgymnasium' -> exam_type='Langzeit Gymnasium'
        '_Kurzzeitgymnasium' -> exam_type='Kurzzeit Gymnasium'
  - The join key is a variable-length prefix of the questions index ID.
    find_distinguishing_prefixes() computes the shortest prefix per question ID
    that is unique across all question IDs. results.question_id is then matched
    against these prefixes via a trie-style lookup — no fixed substring length.
  - Each (year, exam_type) pair maps to exactly one exam.
  - session_id in results already identifies one exam attempt — no time-window
    filtering is needed.
  - question_id in results is a unique per-exercise ID (e.g. exercise 1a).
    Multiple rows with the same question_id but different question_part values
    represent separate input fields within that exercise.

Produces 4 plots (2 visuals x 2 exam types):
  1. Distinct students per exam year
  2. Distinct exam attempts (sessions) per exam year
"""

import ast
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from pathlib import Path
import os


# ── CONFIG ───────────────────────────────────────────────────────────────────
ROOT_DIR = Path(__file__).resolve().parent.parent
RESULTS_PATH   = os.path.join(ROOT_DIR, "data/math_results.csv")
QUESTIONS_PATH = os.path.join(ROOT_DIR,"data/math_questions.csv")

TAG_TO_EXAM = {
    "_Langzeitgymnasium": "Langzeit Gymnasium",
    "_Kurzzeitgymnasium": "Kurzzeit Gymnasium",
}

EXAM_TYPES = list(TAG_TO_EXAM.values())

PALETTE = {
    "Langzeit Gymnasium": "#4C72B0",
    "Kurzzeit Gymnasium": "#DD8452",
}
# ─────────────────────────────────────────────────────────────────────────────


# ── 1. LOAD ───────────────────────────────────────────────────────────────────

def load_data(results_path: str, questions_path: str):
    results = pd.read_csv(results_path)

    questions = pd.read_csv(questions_path, header=0)
    questions.columns = ["question_id"] + list(questions.columns[1:])

    print(f"results   : {len(results):,} rows, "
          f"{results['question_id'].nunique()} unique question_ids")
    print(f"questions : {len(questions):,} rows, "
          f"{questions['question_id'].nunique()} unique question_ids")
    return results, questions


# ── 2. SHORTEST DISTINGUISHING PREFIX ────────────────────────────────────────

def find_distinguishing_prefixes(ids: list[str]) -> dict[str, str]:
    """
    Given a list of unique string IDs, return a dict mapping each ID to the
    shortest prefix that distinguishes it from every other ID in the list.

    Algorithm: for each ID, increase prefix length one character at a time
    until no other ID shares that prefix.

    Example:
        ['abcdef', 'abcxyz', 'ghijkl']
        -> {'abcdef': 'abcd', 'abcxyz': 'abcx', 'ghijkl': 'g'}
    """
    id_set = set(ids)
    assert len(id_set) == len(ids), "Input IDs must be unique"

    result = {}
    for target in ids:
        for length in range(1, len(target) + 1):
            prefix = target[:length]
            # Check whether any OTHER id starts with this prefix
            collision = any(
                other != target and other.startswith(prefix)
                for other in id_set
            )
            if not collision:
                result[target] = prefix
                break
        else:
            # Full ID needed (shouldn't happen with unique inputs)
            result[target] = target

    return result


# ── 3. ENRICH QUESTIONS ───────────────────────────────────────────────────────

def enrich_questions(questions: pd.DataFrame) -> pd.DataFrame:
    """
    - Parse the tags column (stored as a string repr of a Python list).
    - Warn and drop rows with empty tag lists (e.g. one 2018 entry).
    - Extract exam_type from the last tag.
    - Deduplicate to one row per unique question index ID.
    - Compute the shortest distinguishing prefix per question ID.
    Returns q_meta: DataFrame with columns [question_id, q_prefix, year, exam_type].
    """
    def parse_last_tag(tag_str: str):
        try:
            tags = ast.literal_eval(tag_str)
            if not tags:
                return None
            return TAG_TO_EXAM.get(tags[-1], None)
        except Exception:
            return None

    def is_empty_list(tag_str):
        try:
            return ast.literal_eval(tag_str) == []
        except Exception:
            return False

    questions = questions.copy()
    questions["exam_type"] = questions["tags"].apply(parse_last_tag)

    empty_rows = questions["tags"].apply(is_empty_list).sum()
    if empty_rows:
        print(f"  Warning: {empty_rows} row(s) with empty tag list — dropped")

    unrecognised = questions["exam_type"].isna().sum() - empty_rows
    if unrecognised:
        print(f"  Warning: {unrecognised} row(s) with unrecognised last tag — dropped")

    questions = questions.dropna(subset=["exam_type"])

    # One row per unique question_id
    q_meta = (
        questions[["question_id", "year", "exam_type"]]
        .drop_duplicates(subset="question_id")
        .reset_index(drop=True)
    )

    # Compute shortest prefix that uniquely identifies each question_id
    prefix_map = find_distinguishing_prefixes(q_meta["question_id"].tolist())
    q_meta["q_prefix"] = q_meta["question_id"].map(prefix_map)

    prefix_lengths = q_meta["q_prefix"].str.len()
    print(f"\nQuestions after enrichment : {len(q_meta)} unique question IDs")
    print(f"Distinguishing prefix lengths: "
          f"min={prefix_lengths.min()}, max={prefix_lengths.max()}, "
          f"mean={prefix_lengths.mean():.1f}")
    print(q_meta[["q_prefix", "year", "exam_type"]]
          .sort_values(["exam_type", "year"])
          .to_string(index=False))
    return q_meta


# ── 4. MERGE ──────────────────────────────────────────────────────────────────

def merge(results: pd.DataFrame, q_meta: pd.DataFrame) -> pd.DataFrame:
    """
    Attach year and exam_type to each result row using a two-phase lookup:
 
    Phase 1 - prefix match (startswith):
        Each result question_id is checked against the shortest distinguishing
        prefix of every known parent question ID. This handles the common case
        where the sub-question ID directly starts with its parent prefix.
 
    Phase 2 - bisect fallback for gap IDs:
        MongoDB ObjectIDs are lexicographically ordered. Sub-question IDs that
        fall numerically between two parent IDs (and therefore match neither
        prefix) are assigned to the largest parent ID that is still <= the
        sub-question ID. This recovers Category 1 dropped rows.
 
        A bisect match is only accepted when the result ID and the candidate
        parent ID share at least MIN_SHARED_CHARS leading characters, ensuring
        we do not accidentally assign completely unrelated IDs (Category 2).
 
    Category 2 IDs (no structural relationship to any known parent) are still
    dropped and reported separately.
    """
    import bisect
 
    MIN_SHARED_CHARS = 10   # minimum common prefix to accept a bisect match
 
    # Build prefix lookup for phase 1
    prefix_lookup = {
        row.q_prefix: (row.year, row.exam_type)
        for row in q_meta.itertuples()
    }
 
    # Build sorted list of full parent question IDs for phase 2
    sorted_parents = sorted(q_meta["question_id"].tolist())
    parent_meta = {
        row.question_id: (row.year, row.exam_type)
        for row in q_meta.itertuples()
    }
 
    def shared_prefix_len(a: str, b: str) -> int:
        for i, (ca, cb) in enumerate(zip(a, b)):
            if ca != cb:
                return i
        return min(len(a), len(b))
 
    def match(qid: str):
        # Phase 1: prefix match
        for prefix, meta in prefix_lookup.items():
            if qid.startswith(prefix):
                return meta, "prefix"
        # Phase 2: bisect - find largest parent ID <= qid
        pos = bisect.bisect_right(sorted_parents, qid) - 1
        if pos >= 0:
            candidate = sorted_parents[pos]
            if shared_prefix_len(qid, candidate) >= MIN_SHARED_CHARS:
                return parent_meta[candidate], "bisect"
        return None, "unmatched"
 
    results   = results.copy()
    match_res = results["question_id"].map(lambda qid: match(qid))
 
    results["year"]       = match_res.map(lambda t: t[0][0] if t[0] else None)
    results["exam_type"]  = match_res.map(lambda t: t[0][1] if t[0] else None)
    results["_match_via"] = match_res.map(lambda t: t[1])
 
    n_prefix    = (results["_match_via"] == "prefix").sum()
    n_bisect    = (results["_match_via"] == "bisect").sum()
    n_unmatched = (results["_match_via"] == "unmatched").sum()
 
    print(f"\nMatch breakdown:")
    print(f"  Phase 1 prefix : {n_prefix:,} rows")
    print(f"  Phase 2 bisect : {n_bisect:,} rows  (category 1 gap IDs recovered)")
    print(f"  Unmatched      : {n_unmatched:,} rows  (category 2, dropped)")
 
    results = results.dropna(subset=["exam_type"]).drop(columns=["_match_via"])
    print(f"Rows after merge : {len(results):,}")
    return results


# ── 5. PASS / FAIL LABELLING ─────────────────────────────────────────────────

def label_sessions(df: pd.DataFrame, pass_threshold: float) -> pd.DataFrame:
    """
    Aggregate points to the session level and assign a pass/fail label.

    A session is considered passed when:
        sum(points) / sum(max_points) >= pass_threshold

    Returns one row per (session_id, user_id, exam_type, year) with columns:
        score_ratio : float  — achieved fraction of total points
        passed      : bool   — True if score_ratio >= pass_threshold
    """
    session_scores = (
        df.groupby(["session_id", "user_id", "exam_type", "year"])
        .agg(total_points=("points", "sum"), total_max=("max_points", "sum"))
        .reset_index()
    )
    session_scores["score_ratio"] = (
        session_scores["total_points"] / session_scores["total_max"]
    )
    session_scores["passed"] = session_scores["score_ratio"] >= pass_threshold

    n_pass = session_scores["passed"].sum()
    n_fail = (~session_scores["passed"]).sum()
    print(f"\nPass threshold : {pass_threshold:.0%}")
    print(f"  Passed sessions : {n_pass:,}")
    print(f"  Failed sessions : {n_fail:,}")
    return session_scores


def filter_zero_score_sessions(sessions: pd.DataFrame,
                               exclude: bool) -> pd.DataFrame:
    """
    Optionally remove sessions where the student scored zero points in total.
    These are typically abandoned or accidental attempts and can skew
    pass-rate and attempt-count statistics.

    Parameters
    ----------
    sessions : output of label_sessions() — one row per session with
               columns total_points, total_max, score_ratio, passed, …
    exclude  : when True, rows with total_points == 0 are dropped.

    Returns the (possibly filtered) DataFrame and prints a short summary.
    """
    if not exclude:
        print("Zero-score filter : disabled")
        return sessions

    mask_zero = sessions["total_points"] == 0
    n_dropped = mask_zero.sum()
    n_before  = len(sessions)
    sessions  = sessions[~mask_zero].reset_index(drop=True)

    print(f"Zero-score filter : removed {n_dropped:,} session(s) "
          f"({n_dropped / n_before:.1%} of {n_before:,})  "
          f"→ {len(sessions):,} remaining")
    return sessions


# ── 6. SUMMARIES ──────────────────────────────────────────────────────────────

def compute_summaries(df: pd.DataFrame, sessions: pd.DataFrame):
    """
    Build per-(exam_type, year) counts split by pass/fail.

    Visual 1 — distinct students:
        A student is counted as 'passed' if they passed in ANY session that year.
        Otherwise they are counted as 'failed'.

    Visual 2 — session attempts:
        Each session is counted once, labelled by its own pass/fail outcome.
    """
    # ── Students ──
    # Best outcome per student per exam: passed=True beats passed=False
    best = (
        sessions.groupby(["user_id", "exam_type", "year"])["passed"]
        .max()   # True > False
        .reset_index()
    )
    students_pf = (
        best.groupby(["exam_type", "year", "passed"])
        .size()
        .reset_index(name="count")
    )

    # ── Attempts ──
    attempts_pf = (
        sessions.groupby(["exam_type", "year", "passed"])
        .size()
        .reset_index(name="count")
    )

    print("\nStudents per exam (pass/fail):\n", students_pf.to_string(index=False))
    print("\nAttempts per exam (pass/fail):\n", attempts_pf.to_string(index=False))
    return students_pf, attempts_pf


# ── 7. PLOTTING ───────────────────────────────────────────────────────────────

PASS_COLOR = "#2ca02c"   # green
FAIL_COLOR = "#d62728"   # red


def _stacked_bar_chart(ax, years, passed_counts, failed_counts, ylabel, title):
    """
    Draw a stacked bar per year: green (passed) on bottom, red (failed) on top.
    Label each segment with its count (hidden if zero).
    Total count labelled above each bar.
    """
    x = [str(y) for y in years]
    total = [p + f for p, f in zip(passed_counts, failed_counts)]
    y_max = max(total) if total else 1

    bars_pass = ax.bar(x, passed_counts, color=PASS_COLOR,
                       edgecolor="white", linewidth=0.8, width=0.55,
                       zorder=3, label="Passed")
    bars_fail = ax.bar(x, failed_counts, bottom=passed_counts,
                       color=FAIL_COLOR, edgecolor="white", linewidth=0.8,
                       width=0.55, zorder=3, label="Failed")

    # Segment labels
    for bar, count in zip(bars_pass, passed_counts):
        if count > 0:
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() / 2,
                    f"{int(count):,}",
                    ha="center", va="center", fontsize=8,
                    fontweight="bold", color="white")

    for bar, bot, count in zip(bars_fail, passed_counts, failed_counts):
        if count > 0:
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bot + count / 2,
                    f"{int(count):,}",
                    ha="center", va="center", fontsize=8,
                    fontweight="bold", color="white")

    # Total label above each bar
    for bar, tot in zip(bars_pass, total):
        ax.text(bar.get_x() + bar.get_width() / 2,
                tot + y_max * 0.015,
                f"{int(tot):,}",
                ha="center", va="bottom", fontsize=9, fontweight="bold")

    ax.set_xlabel("Exam Year", fontsize=11)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.set_title(title, fontsize=12, fontweight="bold", pad=10)
    ax.yaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    ax.grid(axis="y", linestyle="--", alpha=0.4, zorder=0)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, fontsize=9)


def _side_by_side(pf_df, ylabel, suptitle, filename):
    """
    pf_df must have columns: exam_type, year, passed (bool), count.
    Renders two subplots side by side, one per exam type.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(suptitle, fontsize=15, fontweight="bold", y=1.01)

    for ax, exam_type in zip(axes, EXAM_TYPES):
        sub = pf_df[pf_df["exam_type"] == exam_type]
        if sub.empty:
            ax.text(0.5, 0.5, "No data", ha="center", va="center",
                    transform=ax.transAxes)
            ax.set_title(exam_type)
            continue

        years = sorted(sub["year"].unique())
        passed_counts = []
        failed_counts = []
        for yr in years:
            yr_data = sub[sub["year"] == yr].set_index("passed")["count"]
            passed_counts.append(int(yr_data.get(True,  0)))
            failed_counts.append(int(yr_data.get(False, 0)))

        _stacked_bar_chart(ax, years, passed_counts, failed_counts,
                           ylabel=ylabel, title=exam_type)

    plt.tight_layout()
    plt.savefig(filename)
    plt.show()


def _student_attempts_chart(sessions: pd.DataFrame, filename: str):
    """
    Two subplots side by side, one per exam type.
    Each subplot: one stacked bar per distinct student.
    Green = passed attempts, Red = failed attempts.
    Students sorted by total attempts (descending).
    """
    fig, axes = plt.subplots(1, 2, figsize=(20, 5))
    fig.suptitle("Attempts per Student  (green = passed, red = failed)",
                 fontsize=15, fontweight="bold", y=1.01)

    for ax, exam_type in zip(axes, EXAM_TYPES):
        sub = sessions[sessions["exam_type"] == exam_type]

        if sub.empty:
            ax.text(0.5, 0.5, "No data", ha="center", va="center",
                    transform=ax.transAxes)
            ax.set_title(exam_type, fontsize=12, fontweight="bold")
            continue

        per_student = (
            sub.groupby(["user_id", "passed"])
            .size()
            .unstack(fill_value=0)
            .rename(columns={True: "passed", False: "failed"})
            .reset_index()
        )
        for col in ("passed", "failed"):
            if col not in per_student.columns:
                per_student[col] = 0

        per_student["total"] = per_student["passed"] + per_student["failed"]
        per_student = per_student.sort_values("total", ascending=False).reset_index(drop=True)

        n = len(per_student)
        x = range(n)
        ax.bar(x, per_student["passed"], color=PASS_COLOR,
               edgecolor="white", linewidth=0.5, width=0.8,
               zorder=3, label="Passed")
        ax.bar(x, per_student["failed"], bottom=per_student["passed"],
               color=FAIL_COLOR, edgecolor="white", linewidth=0.5,
               width=0.8, zorder=3, label="Failed")

        ax.set_title(exam_type, fontsize=12, fontweight="bold", pad=8)
        ax.set_xlabel(f"Student (n={n}, sorted by total attempts)", fontsize=10)
        ax.set_ylabel("Number of Attempts", fontsize=10)
        ax.set_xticks([])
        ax.yaxis.set_major_locator(ticker.MaxNLocator(integer=True))
        ax.grid(axis="y", linestyle="--", alpha=0.4, zorder=0)
        ax.spines[["top", "right"]].set_visible(False)
        ax.legend(frameon=False, fontsize=9)

    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    plt.show()
    print(f"  Saved → {filename}")


def _confusion_matrix_heatmap(sessions: pd.DataFrame, filename: str):
    """
    Two subplots side by side, one per exam type.
      x-axis : number of PASSED attempts per user  (0 → max, left → right)
      y-axis : number of FAILED attempts per user  (0 → max, bottom → top)
      cell   : count of distinct users in that (passed, failed) bucket
    """
    # Pre-compute per-user counts for both exam types so we can share
    # a common colour scale across subplots.
    all_matrices = {}
    for exam_type in EXAM_TYPES:
        sub = sessions[sessions["exam_type"] == exam_type]
        if sub.empty:
            all_matrices[exam_type] = None
            continue

        per_user = (
            sub.groupby(["user_id", "passed"])
            .size()
            .unstack(fill_value=0)
            .rename(columns={True: "n_passed", False: "n_failed"})
            .reset_index()
        )
        for col in ("n_passed", "n_failed"):
            if col not in per_user.columns:
                per_user[col] = 0

        max_pass = int(per_user["n_passed"].max())
        max_fail = int(per_user["n_failed"].max())

        matrix = pd.DataFrame(
            0,
            index=range(max_fail + 1),
            columns=range(max_pass + 1),
        )
        for _, row in per_user.iterrows():
            matrix.at[int(row["n_failed"]), int(row["n_passed"])] += 1

        all_matrices[exam_type] = matrix.iloc[::-1]   # flip for bottom-left origin

    global_max = max(
        (m.values.max() for m in all_matrices.values() if m is not None),
        default=1,
    )

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle(
        "User distribution by passed vs failed attempts\n"
        "(cell value = distinct users;  origin (0,0) = bottom-left)",
        fontsize=14, fontweight="bold", y=1.02,
    )

    for ax, exam_type in zip(axes, EXAM_TYPES):
        matrix_display = all_matrices[exam_type]

        if matrix_display is None:
            ax.text(0.5, 0.5, "No data", ha="center", va="center",
                    transform=ax.transAxes)
            ax.set_title(exam_type, fontsize=12, fontweight="bold")
            continue

        im = ax.imshow(matrix_display.values, aspect="auto",
                       cmap="YlOrRd", origin="upper",
                       vmin=0, vmax=global_max)   # shared colour scale

        max_fail_idx = matrix_display.shape[0] - 1
        max_pass_idx = matrix_display.shape[1] - 1

        for row_i in range(matrix_display.shape[0]):
            for col_j in range(matrix_display.shape[1]):
                val = matrix_display.values[row_i, col_j]
                if val > 0:
                    ax.text(col_j, row_i, str(int(val)),
                            ha="center", va="center", fontsize=9,
                            fontweight="bold",
                            color="white" if val > global_max * 0.6 else "black")

        ax.set_xticks(range(max_pass_idx + 1))
        ax.set_xticklabels(range(max_pass_idx + 1))
        ax.set_yticks(range(max_fail_idx + 1))
        ax.set_yticklabels(range(max_fail_idx, -1, -1))

        ax.set_title(exam_type, fontsize=12, fontweight="bold", pad=8)
        ax.set_xlabel("Passed attempts per user", fontsize=10)
        ax.set_ylabel("Failed attempts per user", fontsize=10)

        plt.colorbar(im, ax=ax, label="Number of users")

    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    plt.show()
    print(f"  Saved → {filename}")

# ── 8. MAIN ───────────────────────────────────────────────────────────────────

PASS_THRESHOLD = 0.4    # Percentage of points required to pass the exam
EXCLUDE_ZERO_SCORE = True   # if True, drop sessions where the student scored 0 points total

def main():
    print("=== 1. Load ===")
    results, questions = load_data(RESULTS_PATH, QUESTIONS_PATH)

    print("\n=== 2. Enrich questions ===")
    q_meta = enrich_questions(questions)

    print("\n=== 3. Merge ===")
    merged = merge(results, q_meta)

    print("\n=== 4. Label sessions ===")
    sessions = label_sessions(merged, PASS_THRESHOLD)

    print("\n=== 4b. Filter zero-score sessions ===")
    sessions = filter_zero_score_sessions(sessions, EXCLUDE_ZERO_SCORE)

    print("\n=== 5. Summaries ===")
    students_pf, attempts_pf = compute_summaries(merged, sessions)

    print("\n=== 6. Plot ===")
    _side_by_side(
        students_pf,
        ylabel="Number of Students",
        suptitle=f"Distinct Students per Exam Year  (pass >= {PASS_THRESHOLD:.0%})",
        filename="students_per_exam.png",
    )
    _side_by_side(
        attempts_pf,
        ylabel="Number of Attempts (Sessions)",
        suptitle=f"Exam Attempts per Exam Year  (pass >= {PASS_THRESHOLD:.0%})",
        filename="attempts_per_exam.png",
    )
    _student_attempts_chart(sessions, filename="student_attempts_breakdown.png")
    _confusion_matrix_heatmap(sessions, filename="user_pass_fail_matrix.png")

    print("\nDone.")
    return merged, sessions


if __name__ == "__main__":
    df, sessions = main()
