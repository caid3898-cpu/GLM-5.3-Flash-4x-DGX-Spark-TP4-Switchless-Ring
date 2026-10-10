"""Cold prefill and prefix-replay benchmark. Run on the head: python3 prefill_bench.py <label> [--sizes 8192,32768,65536]

Each prompt starts with a unique nonce, so no earlier request can be a prefix-cache hit (cold). The body is
varied English text (numbered paragraphs mixing a few hundred distinct sentences), sized with the server's own
/tokenize. max_tokens=1, streaming: prefill tok/s = prompt_tokens / TTFT. Each size runs --repeat times with a new
nonce (report the median), then the last prompt is sent once more to measure a prefix-cache replay.
Writes <label>-prefill.json next to this file.
"""
import argparse, json, pathlib, random, statistics, time, urllib.request, uuid

WORDS = ('system memory network cache latency throughput kernel thread process scheduler compiler garden river mountain '
         'village market harbour teacher student library museum theatre orchestra painter novel poem history economy '
         'policy budget contract engineer doctor patient hospital battery engine turbine bridge tunnel railway airport '
         'weather forecast season harvest winter summer autumn spring ocean island forest desert canyon glacier').split()


def text_block(rng, n_par):
    out = []
    for i in range(n_par):
        sents = []
        for _ in range(rng.randint(3, 6)):
            w = [rng.choice(WORDS) for _ in range(rng.randint(8, 18))]
            sents.append(' '.join(w).capitalize() + '.')
        out.append(f'{i + 1}. ' + ' '.join(sents))
    return '\n'.join(out)


def post(base, path, body, timeout=1800):
    req = urllib.request.Request(base + path, data=json.dumps(body).encode(), headers={'Content-Type': 'application/json'})
    return urllib.request.urlopen(req, timeout=timeout)


def ntok(base, model, text):
    r = json.load(post(base, '/tokenize', {'model': model, 'prompt': text}))
    return r.get('count') or len(r.get('tokens', []))


def build(base, model, target, seed):
    rng = random.Random(seed)
    body = text_block(rng, 40)
    per = ntok(base, model, body) / 40
    body = text_block(random.Random(seed), max(1, int((target - 64) / per)))
    return body


def ttft(base, model, prompt):
    body = dict(model=model, messages=[{'role': 'user', 'content': prompt}], max_tokens=1, temperature=0, stream=True,
                stream_options={'include_usage': True}, chat_template_kwargs={'reasoning_effort': 'low'})
    t0 = time.monotonic(); first = None; usage = None
    with post(base, '/v1/chat/completions', body) as r:
        for line in r:
            if not line.startswith(b'data: '): continue
            p = line[6:].strip()
            if p == b'[DONE]': break
            e = json.loads(p)
            if e.get('usage'): usage = e['usage']
            if first is None and e.get('choices'): first = time.monotonic()
    dt = (first or time.monotonic()) - t0
    pt = usage['prompt_tokens'] if usage else None
    return dict(prompt_tokens=pt, ttft_s=round(dt, 3), prefill_tps=round(pt / dt, 1) if pt else None)


if __name__ == '__main__':
    a = argparse.ArgumentParser(); a.add_argument('label'); a.add_argument('--base', default='http://127.0.0.1:8093')
    a.add_argument('--model', default='GLM-5.3-Flash-FP8'); a.add_argument('--sizes', default='8192,32768,65536')
    a.add_argument('--repeat', type=int, default=2)
    a = a.parse_args()
    res = []
    for size in [int(x) for x in a.sizes.split(',')]:
        body = build(a.base, a.model, size, size)
        runs = []
        for i in range(a.repeat):
            prompt = f'[{uuid.uuid4().hex}] Read the numbered notes below and reply with one word.\n' + body
            r = ttft(a.base, a.model, prompt); r['kind'] = 'cold'; runs.append(r)
            print(json.dumps({'size': size, **r}), flush=True)
        rep = ttft(a.base, a.model, prompt); rep['kind'] = 'replay'
        print(json.dumps({'size': size, **rep}), flush=True)
        res.append(dict(size=size, cold=runs, cold_median_tps=statistics.median(r['prefill_tps'] for r in runs),
                        cold_median_ttft=statistics.median(r['ttft_s'] for r in runs), replay=rep))
    pathlib.Path(__file__).resolve().parent.joinpath(a.label + '-prefill.json').write_text(json.dumps(res, indent=1))
