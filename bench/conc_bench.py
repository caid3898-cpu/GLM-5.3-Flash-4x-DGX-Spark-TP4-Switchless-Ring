"""Concurrent streaming decode benchmark for a GLM endpoint on Spark_01.

python3 conc_bench.py <label> [--base http://127.0.0.1:8093] [--model NAME] [--conc 1,2,4,8,16] [--types prose,code,json]
Each cell: N requests of one type started on a barrier, temperature 0, max_tokens 768, effort low by default.
Every stream gets a DIFFERENT prompt (32 per type since 2026-09-26, cycled beyond 32), so at c16 the batch routes to as many
distinct experts as real traffic would; one prompt with an index prefix routes almost identically on every
stream and overstates throughput at high concurrency. Stream i of every cell uses prompt i, so cells are
comparable across configs. Reports per-stream decode tok/s (mean/min), sum of per-stream rates, wall aggregate
(all completion tokens / cell wall time), TTFT, and U+FFFD count in the output. Writes <label>-conc.json.
Not an intelligence benchmark.
"""
import argparse, concurrent.futures, json, pathlib, statistics, threading, time, urllib.request

PROSE = [
 'Explain how to decide whether a medium-sized software project is ready for a database migration. Cover dependencies, schema compatibility, tests, rollout and rollback in several paragraphs.',
 'Describe how a city could reduce traffic congestion over ten years without building new roads. Discuss pricing, public transport, zoning and the politics of each option.',
 'Write an essay on why some open-source projects thrive for decades while others fade after a few years. Use concrete mechanisms, not slogans.',
 'Explain to a new engineering manager how to run an effective incident post-mortem, from the first hour after the outage to the follow-up review a month later.',
 'Compare the trade-offs between renting and buying a home for a family that may relocate within five years. Walk through the costs, risks and non-financial factors.',
 'Describe the history and the physics of how mechanical clocks became accurate enough for navigation at sea, and why it mattered.',
 'Explain how a small bakery should plan its first year of expansion into wholesale supply to cafes: pricing, capacity, hiring and cash flow.',
 'Write a thoughtful guide on how to learn a second language as a busy adult, covering habits, material choice, speaking practice and plateaus.',
 'Discuss how remote work changes the way teams share knowledge, and what a company can do to keep junior engineers learning quickly.',
 'Explain the main causes of the 2008 financial crisis in plain language, and which safeguards were introduced afterwards and why.',
 'Describe how to plan a two-week hiking trip in a mountain range you have never visited: research, fitness, gear, safety and contingency plans.',
 'Write an explanation of how vaccines train the immune system, including memory cells, boosters, and why some vaccines need yearly updates.',
 'Discuss the ethics and practical difficulties of using algorithms to decide who receives a bank loan, with examples of failure modes.',
 'Explain how a public library could reinvent itself for the next twenty years, covering space, services, staff skills and funding.',
 'Describe what makes a good technical interview process for senior engineers, and the common mistakes companies make when designing one.',
 'Write an overview of how the water cycle interacts with climate change, and what that means for farming in temperate regions.',
]
CODE = [
 'Write a Python 3 module with a thread-safe bounded LRU cache with per-item TTL: get, set, delete, clear, __len__, injectable monotonic clock, plus five unittest tests. Return only code.',
 'Write a Python function that parses an INI-like config format with sections, comments, multi-line values and type coercion, with unittest tests. Return only code.',
 'Implement a token-bucket rate limiter in Python supporting multiple keys, burst capacity, refill rate and an injectable clock, with pytest tests. Return only code.',
 'Write a TypeScript function that deep-merges two JSON objects with array strategies (replace, concat, unique) and full type definitions, plus Jest tests. Return only code.',
 'Implement Dijkstra and A* shortest path on a grid with weighted cells in Python, with a small CLI and unit tests. Return only code.',
 'Write a Go HTTP middleware that adds request IDs, structured logging and panic recovery, with tests using httptest. Return only code.',
 'Implement a trie-based autocomplete class in Python with insert, delete, prefix search ranked by frequency, and tests. Return only code.',
 'Write a Rust function that tokenizes arithmetic expressions and evaluates them with operator precedence and parentheses, plus unit tests. Return only code.',
 'Write a Python asyncio worker pool that processes jobs from a queue with retries, exponential backoff and graceful shutdown, plus tests. Return only code.',
 'Implement a simple in-memory key-value store in Python with transactions (begin, commit, rollback, nested), and unit tests. Return only code.',
 'Write a JavaScript debounce and throttle implementation with leading/trailing options and cancel/flush methods, plus tests. Return only code.',
 'Write a Python CSV diff tool that compares two files by a key column and reports added, removed and changed rows, with tests. Return only code.',
 'Implement a binary min-heap priority queue in Python with decrease-key support and an index map, plus unit tests. Return only code.',
 'Write a SQL schema and Python data-access layer for a todo app with tags and due dates using sqlite3, with tests. Return only code.',
 'Implement a Python function that validates and normalizes international phone numbers for five countries without external libraries, with tests. Return only code.',
 'Write a C function that implements a growable ring buffer of bytes with read, write, peek and resize, plus a small test main. Return only code.',
]
JSON = [
 'Return a JSON array of 40 objects, each with keys id (int), sku (string like "SKU-00001"), price (float), tags (3 strings). No prose, no code fence.',
 'Return a JSON array of 30 fictional employees with keys id, name, department, salary (int), start_date (YYYY-MM-DD) and skills (list of 3). No prose, no code fence.',
 'Return a JSON object describing a library catalogue with 25 books: title, author, year, isbn, genres (list), available (bool). No prose, no code fence.',
 'Return a JSON array of 35 weather readings with station_id, timestamp (ISO 8601), temperature_c, humidity, wind_kph and condition. No prose, no code fence.',
 'Return a JSON array of 30 orders with order_id, customer_id, items (list of {sku, qty, unit_price}), total and status. No prose, no code fence.',
 'Return a JSON object with 20 countries, each keyed by ISO code, with name, capital, population (int), currency and languages (list). No prose, no code fence.',
 'Return a JSON array of 40 log events with level, service, message, latency_ms and trace_id (hex string). No prose, no code fence.',
 'Return a JSON array of 25 recipes with name, cuisine, prep_minutes, ingredients (list of {name, amount}), and vegetarian (bool). No prose, no code fence.',
 'Return a JSON array of 30 flights with flight_no, from, to, departure, arrival, aircraft and seats_available. No prose, no code fence.',
 'Return a JSON object mapping 30 usernames to profiles with email, age, country, interests (list of 3) and premium (bool). No prose, no code fence.',
 'Return a JSON array of 35 stock quotes with ticker, date, open, high, low, close and volume (int). No prose, no code fence.',
 'Return a JSON array of 25 support tickets with id, title, priority, assignee, created_at, tags (list) and resolved (bool). No prose, no code fence.',
 'Return a JSON array of 30 cars with make, model, year, engine {type, displacement_l, hp}, price_eur and colors (list). No prose, no code fence.',
 'Return a JSON array of 40 sensor devices with device_id, firmware, battery_pct, location {lat, lon}, and last_seen. No prose, no code fence.',
 'Return a JSON array of 25 courses with code, title, credits, instructor, schedule (list of {day, start, end}) and prerequisites (list). No prose, no code fence.',
 'Return a JSON array of 30 invoices with number, issue_date, due_date, client, lines (list of {desc, qty, price}) and paid (bool). No prose, no code fence.',
]
# 2026-09-26: 16 more prompts per type so c32 has 32 distinct streams; streams 0-15 are unchanged, and so is the
# first 48 entries of MIX, so every c1-c16 cell stays comparable with earlier runs.
PROSE += [
 'Explain how a regional hospital should prepare for a week-long power outage, covering generators, fuel, triage, staff rotas and communication.',
 'Describe how the printing press changed European politics and religion over two centuries, with specific mechanisms rather than dates alone.',
 'Write a guide for a first-time manager on giving difficult feedback, including preparation, the conversation itself and the weeks afterwards.',
 'Explain why some cities flood more than others during heavy rain, and what drainage, land use and green infrastructure can do about it.',
 'Discuss the pros and cons of a four-day working week for a mid-sized manufacturing company, including shifts, output and morale.',
 'Describe how a small open-source maintainer can handle burnout, funding and a growing number of issues without abandoning the project.',
 'Explain how compound interest, inflation and fees interact in a thirty-year retirement plan, in plain language with worked reasoning.',
 'Write an essay on how board games teach strategic thinking to children, and which kinds of games teach which skills.',
 'Explain how an airline recovers its schedule after a major storm cancels hundreds of flights: crews, aircraft, passengers and priorities.',
 'Describe the ecological role of wolves in a national park and the debates about reintroducing them near farmland.',
 'Discuss how a newspaper can stay financially viable in a small town, covering subscriptions, events, grants and editorial independence.',
 'Explain how a software team should choose between building a feature in-house and buying a vendor product, with the long-term costs.',
 'Write a practical guide to renovating an old house for energy efficiency, ordering the work by cost, benefit and disruption.',
 'Describe how antibiotic resistance develops and spreads, and what hospitals, farms and patients can each do to slow it.',
 'Explain how a school could redesign its timetable to give students more deep-work time without losing essential subjects.',
 'Discuss what historians mean by the Industrial Revolution, why it started where it did, and how its effects spread to other countries.',
]
CODE += [
 'Write a Python module that implements a simple Markdown-to-HTML converter for headings, lists, emphasis, links and code blocks, with tests. Return only code.',
 'Implement a consistent-hashing ring in Python with virtual nodes, add/remove node, key lookup and rebalancing stats, plus unit tests. Return only code.',
 'Write a TypeScript event emitter with typed events, once, off, wildcard listeners and async handlers, plus Jest tests. Return only code.',
 'Implement a Python bloom filter with configurable false-positive rate, union, serialization to bytes and unit tests. Return only code.',
 'Write a Go worker that tails a log file, parses JSON lines and aggregates per-minute counts by level, with tests. Return only code.',
 'Implement a Python interval tree supporting insert, delete and overlap queries, with randomized tests against a brute-force version. Return only code.',
 'Write a Rust struct for a fixed-capacity LRU cache using a HashMap and a doubly linked list over indices, with unit tests. Return only code.',
 'Write a Python command-line tool that finds duplicate files by size then hash, with dry-run and hardlink options and tests. Return only code.',
 'Implement a JavaScript promise pool with concurrency limit, per-task timeout and ordered results, plus tests. Return only code.',
 'Write a Python function that computes the diff of two text sequences with the Myers algorithm and prints a unified diff, with tests. Return only code.',
 'Implement a simple regex engine in Python supporting literals, dot, star, plus, question mark and anchors, with tests. Return only code.',
 'Write a Python scheduler that runs cron-like expressions (minute, hour, day, month, weekday) with next-run computation and tests. Return only code.',
 'Implement a union-find structure in C++ with path compression and union by rank, and use it to count islands in a grid, with tests. Return only code.',
 'Write a Python JSON schema validator for types, required keys, enums, min/max and nested objects, with unit tests. Return only code.',
 'Implement a Java class for a thread-safe blocking bounded queue using locks and conditions, with JUnit tests. Return only code.',
 'Write a Python module that paginates and caches results from a REST API client with retry and ETag support, with mocked tests. Return only code.',
]
JSON += [
 'Return a JSON array of 30 hotel rooms with room_no, floor, type, beds, price_per_night, amenities (list) and occupied (bool). No prose, no code fence.',
 'Return a JSON array of 35 git commits with sha (hex string), author, date, message, files_changed (int) and additions (int). No prose, no code fence.',
 'Return a JSON object with 25 cities keyed by name, each with country, lat, lon, population (int) and timezone. No prose, no code fence.',
 'Return a JSON array of 30 movies with title, director, year, runtime_min, genres (list) and rating (float). No prose, no code fence.',
 'Return a JSON array of 40 bank transactions with id, account, date, amount, currency, category and merchant. No prose, no code fence.',
 'Return a JSON array of 25 museum exhibits with id, name, artist, period, room, dimensions {h_cm, w_cm} and on_loan (bool). No prose, no code fence.',
 'Return a JSON array of 30 shipments with tracking_no, carrier, origin, destination, weight_kg, status and events (list of {time, place}). No prose, no code fence.',
 'Return a JSON array of 35 products with id, name, category, stock (int), supplier and reorder_level (int). No prose, no code fence.',
 'Return a JSON array of 30 patients (fictional) with id, age, blood_type, allergies (list), last_visit and ward. No prose, no code fence.',
 'Return a JSON array of 25 football matches with date, home, away, home_goals, away_goals, attendance (int) and referee. No prose, no code fence.',
 'Return a JSON array of 40 DNS records with name, type, ttl (int), value and priority (int or null). No prose, no code fence.',
 'Return a JSON array of 30 job postings with id, title, company, location, salary_range {min, max}, remote (bool) and skills (list). No prose, no code fence.',
 'Return a JSON array of 25 plants with common_name, latin_name, sunlight, water_per_week_ml, height_cm and toxic_to_pets (bool). No prose, no code fence.',
 'Return a JSON array of 35 train departures with train_id, platform, destination, scheduled, expected and delay_min (int). No prose, no code fence.',
 'Return a JSON array of 30 podcast episodes with id, show, title, duration_s (int), published and guests (list). No prose, no code fence.',
 'Return a JSON array of 30 server metrics samples with host, cpu_pct, mem_pct, disk_pct, load (list of 3 floats) and ts. No prose, no code fence.',
]
P = {'prose': PROSE, 'code': CODE, 'json': JSON}
# 'mix': stream i gets prose / code / json in turn (distinct prompts), the realistic mixed-batch case
MIX = [x for t in zip(PROSE, CODE, JSON) for x in t]
P['mix'] = MIX


def one(base, model, kind, idx, max_tokens, barrier, thinking):
    body = dict(model=model, messages=[{'role': 'user', 'content': P[kind][idx % len(P[kind])]}], temperature=0, top_p=1,
                max_tokens=max_tokens, stream=True, stream_options={'include_usage': True},
                chat_template_kwargs={'reasoning_effort': thinking})
    req = urllib.request.Request(base + '/v1/chat/completions', data=json.dumps(body).encode(), headers={'Content-Type': 'application/json'})
    barrier.wait()
    t0 = time.monotonic(); first = last = None; usage = None; text = []
    with urllib.request.urlopen(req, timeout=1800) as r:
        for line in r:
            if not line.startswith(b'data: '): continue
            p = line[6:].strip()
            if p == b'[DONE]': break
            e = json.loads(p); now = time.monotonic()
            if e.get('usage'): usage = e['usage']
            for c in e.get('choices', []):
                d = c.get('delta', {})
                piece = d.get('content') or d.get('reasoning_content') or d.get('reasoning')
                if piece:
                    text.append(piece)
                    if first is None: first = now
                    last = now
    ct = usage['completion_tokens'] if usage else 0
    tps = (ct - 1) / (last - first) if (ct > 1 and last and first and last > first) else None
    s = ''.join(text)
    return dict(kind=kind, idx=idx, completion_tokens=ct, ttft_s=first and round(first - t0, 3), decode_tps=tps and round(tps, 1),
                wall_s=round(time.monotonic() - t0, 2), fffd=s.count('�'), chars=len(s))


def cell(base, model, kind, n, max_tokens, thinking):
    barrier = threading.Barrier(n)
    t0 = time.monotonic()
    with concurrent.futures.ThreadPoolExecutor(n) as ex:
        rs = list(ex.map(lambda i: one(base, model, kind, i, max_tokens, barrier, thinking), range(n)))
    wall = time.monotonic() - t0
    tps = [r['decode_tps'] for r in rs if r['decode_tps']]
    toks = sum(r['completion_tokens'] for r in rs)
    ttfts = [r['ttft_s'] for r in rs if r['ttft_s']]
    out = dict(kind=kind, conc=n, per_stream_mean=round(statistics.mean(tps), 1) if tps else None,
               per_stream_min=round(min(tps), 1) if tps else None, sum_per_stream=round(sum(tps), 1) if tps else None,
               wall_aggregate=round(toks / wall, 1) if wall else None,
               ttft_mean=round(statistics.mean(ttfts), 2) if ttfts else None, ttft_max=max(ttfts) if ttfts else None,
               tokens=toks, fffd=sum(r['fffd'] for r in rs), streams=rs)
    print(json.dumps({k: v for k, v in out.items() if k != 'streams'}), flush=True)
    return out


if __name__ == '__main__':
    a = argparse.ArgumentParser(); a.add_argument('label'); a.add_argument('--base', default='http://127.0.0.1:8093'); a.add_argument('--model', default='GLM-5.3-Flash-FP8')
    a.add_argument('--conc', default='1,2,4,8,16'); a.add_argument('--types', default='prose,code,json'); a.add_argument('--max-tokens', type=int, default=768)
    a.add_argument('--thinking', default='low', help='reasoning_effort low|high|max (template has no off)'); a.add_argument('--repeat', type=int, default=1)
    a = a.parse_args()
    res = []
    for _ in range(a.repeat):
        for n in [int(x) for x in a.conc.split(',')]:
            for k in a.types.split(','):
                res.append(cell(a.base, a.model, k, n, a.max_tokens, a.thinking))
    pathlib.Path(__file__).resolve().parent.joinpath(a.label + '-conc.json').write_text(json.dumps(res, indent=1))
