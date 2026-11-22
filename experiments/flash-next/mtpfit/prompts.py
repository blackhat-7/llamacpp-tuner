"""prompts.py - held-out prompts for gen_data.py transcripts: code, file edits and prose/chat, one JSON line each.

    python -I prompts.py OUT.jsonl [--n-code 300 --n-edit 250 --n-prose 350]

Edit prompts wrap real source files from the Python standard library and /usr/include (never llamacpp-tuner, whose
module is the edit measurement prompt); gen_data.py still drops anything sharing an 8-word run with the measurement
prompts.
"""
import argparse
import json
import random
from pathlib import Path

LANGS = ["Python", "TypeScript", "Rust", "Go", "C++", "C", "Java", "Kotlin", "C#", "Bash", "SQL", "JavaScript", "Zig", "Haskell"]
TASKS = [
    "a trie with insert, search and prefix listing", "a token bucket rate limiter", "a binary heap priority queue",
    "a CSV parser that handles quoted fields and escaped quotes", "a thread pool with a bounded job queue",
    "Dijkstra's shortest path on an adjacency list", "a union-find structure with path compression",
    "an interval tree supporting overlap queries", "a simple HTTP request line and header parser",
    "a command-line tool that counts word frequencies in files", "a JSON pretty printer without libraries",
    "a matrix multiplication routine with cache blocking", "a bloom filter with configurable false positive rate",
    "a retry helper with exponential backoff and jitter", "a markdown to HTML converter for headings and lists",
    "a ring buffer logger that keeps the last N lines", "an event emitter with once and off",
    "a topological sort that reports cycles", "a skip list with insert, delete and search",
    "a base64 encoder and decoder", "a small expression evaluator with operator precedence",
    "an in-memory key-value store with snapshots", "a file watcher that debounces change events",
    "a consistent hashing ring", "a sliding window median", "a rope data structure for text editing",
    "a minimal regex matcher supporting . * and ?", "a k-means clustering routine", "an A* path finder on a grid",
    "a semantic version parser and comparator", "a producer-consumer pipeline with cancellation",
    "a URL router with path parameters", "a diff of two text files by lines (LCS)", "a sparse matrix type",
    "a connection pool with health checks", "a cron expression parser for the five standard fields",
    "a Huffman encoder and decoder", "a quadtree for 2D point queries", "a memoizing fibonacci with big integers",
    "a simple tokenizer for a calculator language", "a leaky bucket traffic shaper", "an LFU cache",
]
EXTRAS = ["Include tests.", "Add brief comments explaining the tricky parts.", "Explain the time complexity afterwards.",
          "Keep it under 120 lines.", "Use idiomatic error handling.", "Show a short usage example.",
          "Write it as a reusable module with a clear public API.", "Then list two possible optimizations."]
TOPICS = [
    "how TCP congestion control works", "how a garbage collector with generations works", "how vaccines train the immune system",
    "why the sky is blue and sunsets are red", "how public key cryptography works", "how a compiler optimizes loops",
    "the causes of the French Revolution", "how photosynthesis converts light into sugar", "how GPS determines position",
    "how inflation affects savings and wages", "how a database index speeds up queries", "how plate tectonics shapes continents",
    "the history of the printing press and its effects", "how neural networks learn by gradient descent",
    "how a jet engine produces thrust", "why the Roman Empire declined", "how DNS resolves a domain name",
    "how a refrigerator moves heat", "the trade-offs of microservices versus monoliths", "how black holes form",
    "how the stock market sets prices", "how antibiotics stop bacteria and why resistance appears",
    "how a transistor switches", "the ideas behind stoicism", "how earthquakes are measured", "how sleep affects memory",
    "how a hash map handles collisions", "the main causes of the First World War", "how solar panels generate electricity",
    "how operating systems schedule threads", "how coffee is grown and processed", "how a bill becomes law in a parliament",
    "how virtual memory and paging work", "how wind turbines are designed", "how the human eye focuses",
    "the economics of renewable energy", "how Git stores history", "how rainbows form", "how bees communicate",
    "how a CPU cache hierarchy works", "how batteries store energy", "the rise and fall of the Mongol Empire",
]
STYLES = ["Write it as an essay for a curious high-school student.", "Explain it to a software engineer, with examples.",
          "Give a clear step-by-step explanation.", "Write a short blog post about it.", "Answer in a few concise paragraphs.",
          "Explain it simply, then go deeper.", "Write it as a lecture outline with explanations under each point."]
CHAT = [
    "Write a short story about a lighthouse keeper who finds a message in a bottle.",
    "Write a poem about autumn in a big city.", "Give me a 3-day itinerary for Kyoto on a modest budget.",
    "What are good habits for learning a new language as an adult?", "Draft a polite email asking my landlord to fix the heating.",
    "Summarize the plot of a heist movie you invent, in five paragraphs.", "Compare electric cars and hybrids for a daily 60 km commute.",
    "Plan a weekly vegetarian meal prep with a shopping list.", "How do I prepare for a system design interview?",
    "Write a cover letter for a junior data analyst position.", "Explain the rules of chess to a beginner.",
    "Write a dialogue between a detective and a reluctant witness.", "What should I consider before adopting a dog?",
    "Give me tips to improve my public speaking.", "Write a product description for a handmade ceramic mug.",
    "Create a beginner workout plan for three days a week.", "Write a fairy tale about a dragon who is afraid of fire.",
    "Explain the difference between stocks, bonds and index funds.", "Write a speech for a friend's wedding toast.",
    "Describe a futuristic city in the year 2300.", "How can a small team run effective retrospectives?",
    "Write a limerick about a forgetful robot, then explain the rhyme scheme.", "Brainstorm ten names for a bakery.",
    "Rewrite this sentence in five different tones: 'The meeting has been moved to Thursday.'",
    "Solve step by step: a train leaves at 9:40 at 80 km/h, another at 10:10 at 100 km/h on the same track. When does the second catch up?",
    "What is the probability of getting at least one six in four rolls of a die? Show the reasoning.",
]
EDITS = [
    "Add type hints to every function and reply with the complete updated file only.",
    "Add docstrings to the public functions and reply with the complete updated file.",
    "Refactor the longest function into smaller helpers without changing behaviour. Reply with the full file.",
    "Rename the first function to a clearer name everywhere it is used. Reply with the complete updated file only.",
    "Add logging at INFO level at the start of each public function. Reply with the whole file.",
    "Find one bug or edge case in this code, explain it briefly, then give the corrected file in full.",
    "Remove dead code and unused imports, then reply with the complete file.",
    "Explain what this file does, function by function.",
    "Write unit tests for the main functions in this file.",
    "Add input validation that raises ValueError on bad arguments. Reply with the complete updated file only.",
]
FENCE = {".py": "python", ".h": "c", ".c": "c", ".rs": "rust", ".go": "go", ".ts": "typescript", ".js": "javascript"}


def sources(rng):
    files = sorted(Path("/usr/lib/python3.13").glob("*.py")) + sorted(Path("/usr/lib/python3.13/json").glob("*.py"))
    files += sorted(Path("/usr/lib/python3.13/email").glob("*.py")) + sorted(Path("/usr/include").glob("*.h"))
    rng.shuffle(files)
    for f in files:
        lines = f.read_text(errors="replace").splitlines()
        if len(lines) < 40:
            continue
        a = rng.randrange(0, max(1, len(lines) - 200))
        yield f, "\n".join(lines[a:a + rng.randrange(60, 220)])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--n-code", type=int, default=300)
    ap.add_argument("--n-edit", type=int, default=250)
    ap.add_argument("--n-prose", type=int, default=350)
    ap.add_argument("--seed", type=int, default=17)
    a = ap.parse_args()
    rng = random.Random(a.seed)
    out = []
    for _ in range(a.n_code):
        out.append(f"Write {rng.choice(TASKS)} in {rng.choice(LANGS)}. {rng.choice(EXTRAS)}")
    src = sources(rng)
    for _ in range(a.n_edit):
        f, body = next(src)
        kind = "module" if f.suffix == ".py" else "header"
        out.append(f"Here is a {FENCE[f.suffix].capitalize()} {kind}:\n\n```{FENCE[f.suffix]}\n{body}\n```\n\n{rng.choice(EDITS)}")
    for i in range(a.n_prose):
        out.append(rng.choice(CHAT) if i % 3 == 2 else f"Explain {rng.choice(TOPICS)}. {rng.choice(STYLES)}")
    out = list(dict.fromkeys(out))
    rng.shuffle(out)
    with open(a.out, "w") as f:
        for p in out:
            f.write(json.dumps({"text": p}) + "\n")
    print(f"{len(out)} prompts -> {a.out}")


if __name__ == "__main__":
    main()
