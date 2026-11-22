"""Quality A/B: complete tasks with hidden checks against an OpenAI-style server. usage: qa.py URL LABEL OUT.jsonl"""
import json, re, subprocess, sys, tempfile, time, urllib.request
URL, LABEL, OUT = sys.argv[1], sys.argv[2], sys.argv[3]
def ask(prompt, max_tokens=16000):
    body = {"model": "Inkling-2-Mini", "messages": [{"role": "user", "content": prompt}], "max_tokens": max_tokens, "temperature": 0}
    t = time.time()
    r = json.load(urllib.request.urlopen(urllib.request.Request(URL + "/v1/chat/completions", json.dumps(body).encode(), {"Content-Type": "application/json"}), timeout=1800))
    ch = r["choices"][0]
    return ch["message"].get("content") or "", ch.get("finish_reason"), r["usage"]["completion_tokens"], round(time.time() - t, 1)
def code_of(c, lang):
    m = re.search(rf"```(?:{lang})?\n(.*?)```", c, re.S)
    return m.group(1) if m else c
def run_py(src, tests):
    with tempfile.TemporaryDirectory() as t:
        open(f"{t}/a.py", "w").write(src + "\n\n" + tests + "\nprint('ALLPASS')\n")
        p = subprocess.run([sys.executable, "-I", f"{t}/a.py"], capture_output=True, text=True, timeout=60)
        return "ALLPASS" in p.stdout, (p.stderr or "")[-200:]
PY = [
 ("merge", "Write a Python function merge_intervals(intervals) that takes a list of [start, end] pairs and returns the merged, sorted list of non-overlapping intervals (touching intervals like [1,2],[2,3] merge). Output only the code.",
  "assert merge_intervals([[1,3],[2,6],[8,10],[15,18]])==[[1,6],[8,10],[15,18]]\nassert merge_intervals([[1,4],[4,5]])==[[1,5]]\nassert merge_intervals([])==[]\nassert merge_intervals([[5,7],[1,2]])==[[1,2],[5,7]]"),
 ("parens", "Write a Python function is_valid(s) that returns True if the brackets ()[]{} in s are balanced and properly nested, ignoring all other characters. Output only the code.",
  "assert is_valid('([]{})')\nassert not is_valid('([)]')\nassert is_valid('a(b)c')\nassert not is_valid('((')\nassert is_valid('')"),
 ("lru", "Write a Python class LRUCache with __init__(self, capacity), get(self, key) returning -1 if missing, and put(self, key, value), evicting the least recently used key when over capacity. Output only the code.",
  "c=LRUCache(2)\nc.put(1,1); c.put(2,2)\nassert c.get(1)==1\nc.put(3,3)\nassert c.get(2)==-1\nc.put(4,4)\nassert c.get(1)==-1 and c.get(3)==3 and c.get(4)==4"),
 ("roman", "Write a Python function roman_to_int(s) converting a Roman numeral string to an integer. Output only the code.",
  "assert roman_to_int('III')==3\nassert roman_to_int('LVIII')==58\nassert roman_to_int('MCMXCIV')==1994\nassert roman_to_int('IV')==4"),
 ("topk", "Write a Python function top_k_frequent(nums, k) that returns the k most frequent numbers, ordered by frequency descending and then by value ascending. Output only the code.",
  "assert top_k_frequent([1,1,1,2,2,3],2)==[1,2]\nassert top_k_frequent([4,4,5,5,6],2)==[4,5]\nassert top_k_frequent([7],1)==[7]"),
 ("lcp", "Write a Python function longest_common_prefix(strs) returning the longest common prefix of a list of strings (empty string if none or if the list is empty). Output only the code.",
  "assert longest_common_prefix(['flower','flow','flight'])=='fl'\nassert longest_common_prefix(['dog','racecar','car'])==''\nassert longest_common_prefix([])==''\nassert longest_common_prefix(['a'])=='a'"),
]
QS = [("A bat and a ball cost 1.10 dollars in total. The bat costs 1.00 dollar more than the ball. How much does the ball cost, in cents? Answer with just the number.", "5"),
      ("What is 17 * 23 + 41? Answer with just the number.", "432"),
      ("If all bloops are razzies and all razzies are lazzies, are all bloops definitely lazzies? Answer yes or no.", "yes"),
      ("How many prime numbers are there between 1 and 50? Answer with just the number.", "15"),
      ("Sort these words alphabetically and output them comma-separated: pear, apple, mango, banana, cherry", "apple, banana, cherry, mango, pear"),
      ("A train leaves at 9:40 and arrives at 13:15 the same day. How many minutes is the trip? Answer with just the number.", "215"),
      ("What is the next number in the sequence 2, 6, 12, 20, 30? Answer with just the number.", "42"),
      ("I have 3 apples, buy 2 dozen more, then eat 5. How many apples do I have? Answer with just the number.", "22"),
      ("Which is larger, 2 to the power 10 or 10 to the power 3? Answer with just that number.", "1024"),
      ("How many days are in February of a leap year plus March? Answer with just the number.", "60")]
res = []
for name, prompt, tests in PY:
    c, fin, n, s = ask(prompt)
    ok, err = run_py(code_of(c, "python|py"), tests)
    res.append({"task": name, "pass": ok, "finish": fin, "tokens": n, "s": s, "err": err})
c, fin, n, s = ask("Write a self-contained C++17 program: an LRU cache class template with get/put/erase and a capacity, plus a main() that tests it with asserts and prints OK at the end. Output only the code.")
with tempfile.TemporaryDirectory() as t:
    open(f"{t}/a.cpp", "w").write(code_of(c, "cpp|c\\+\\+"))
    comp = subprocess.run(["g++", "-std=c++17", "-o", f"{t}/a", f"{t}/a.cpp"], capture_output=True, text=True)
    ok = comp.returncode == 0 and "OK" in subprocess.run([f"{t}/a"], capture_output=True, text=True, timeout=20).stdout
res.append({"task": "cpp_lru", "pass": ok, "finish": fin, "tokens": n, "s": s, "err": comp.stderr[:200]})
edit = open("/home/illusion/Documents/projects/llamacpp-tuner/tmp/strata-opt/edit.txt").read()
c, fin, n, s = ask(edit)
new = code_of(c, "python")
parses = subprocess.run([sys.executable, "-c", "import ast,sys; ast.parse(sys.stdin.read())"], input=new, capture_output=True, text=True).returncode == 0
ok = parses and "def stop_server" in new and "def stop(" not in new and "getLogger(__name__)" in new
res.append({"task": "edit", "pass": ok, "finish": fin, "tokens": n, "s": s})
for q, want in QS:
    c, fin, n, s = ask(q, 4000)
    res.append({"task": "q:" + want, "pass": want.lower() in c.lower(), "finish": fin, "tokens": n, "s": s, "got": c.strip()[:60]})
with open(OUT, "a") as f:
    for r in res:
        f.write(json.dumps({"label": LABEL, **r}) + "\n")
print(LABEL, "passed", sum(r["pass"] for r in res), "of", len(res), "| tokens", sum(r["tokens"] for r in res), "| s", round(sum(r["s"] for r in res)))
for r in res:
    if not r["pass"]: print("  FAIL", r["task"], r.get("finish"), r.get("err", r.get("got", ""))[:150])
