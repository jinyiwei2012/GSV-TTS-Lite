"""MultiSpeakerTTS vs full-model benchmark (CPU reference environment).

Compares:
  A. MultiSpeakerTTS shared backbone (base + 3 speaker weights)
  B. Full model loading (3 standalone TTS instances, one per speaker)

Metrics: init time, peak RSS (GB), per-speaker warmup + avg inference
latency, RTF. Uses the real fine-tuned models in the repo root.
"""

import sys
import time
import gc
from pathlib import Path

import psutil

sys.path.insert(0, str(Path(__file__).parent.parent))

from gsv_tts import TTS, MultiSpeakerTTS, SpeakerConfig

TEXT = "今日も頑張りましょう、一緒に歩いていこう。"
N_AVG = 2

SPEAKERS = [
    ("cyrene",    "CyreneV3.7-e25.ckpt",       "CyreneV3.7_e16_s1392.pth"),
    ("shouanren", "shouanren-e20.ckpt",        "shouanren_e24_s1584.pth"),
]

SPK_AUDIO = "examples/laffey.mp3"
PROMPT_AUDIO = "examples/AnAn.ogg"
PROMPT_TEXT = "ちが……ちがう。レイア、貴様は間違っている。"


def rss_gb() -> float:
    return psutil.Process().memory_info().rss / 1e9


def timed(fn, label):
    t0 = time.time()
    result = fn()
    dt = time.time() - t0
    print(f"  {label}: {dt:.1f}s", flush=True)
    return result, dt


def main():
    print("=" * 60, flush=True)
    print("MultiSpeakerTTS vs Full Model Benchmark (CPU)", flush=True)
    print("=" * 60, flush=True)

    # ── Scenario A: MultiSpeakerTTS shared backbone ──
    print("\n[A] MultiSpeakerTTS shared backbone", flush=True)
    gc.collect()
    rss_before = rss_gb()

    def build_shared():
        speakers = [
            SpeakerConfig(
                name=name,
                gpt_model_path=gpt,
                sovits_model_path=sovits,
                spk_audio_path=SPK_AUDIO,
                prompt_audio_path=PROMPT_AUDIO,
                prompt_audio_text=PROMPT_TEXT,
            )
            for name, gpt, sovits in SPEAKERS
        ]
        return MultiSpeakerTTS(speakers=speakers, use_bert=True)

    mtts, init_a = timed(build_shared, f"init 3 speakers (shared)")

    for name, _, _ in SPEAKERS:
        w = mtts._speakers[name]
        mode = "shared" if not w.is_full_model else "FULL-DEGRADED"
        print(f"  speaker '{name}': {mode}", flush=True)

    rss_a = rss_gb() - rss_before
    print(f"  RSS delta: {rss_a:.2f} GB", flush=True)

    results_a = {}
    for name, _, _ in SPEAKERS:
        _, w = timed(lambda n=name: mtts.infer(n, TEXT), f"warmup infer '{name}'")
        times = []
        for i in range(N_AVG):
            t0 = time.time()
            mtts.infer(name, TEXT)
            times.append(time.time() - t0)
        avg = sum(times) / len(times)
        results_a[name] = (w, avg)
        print(f"  infer '{name}': warmup {w:.1f}s, avg {avg:.1f}s", flush=True)

    # ── Scenario B: full model per speaker ──
    print("\n[B] Full model loading (3 standalone TTS)", flush=True)
    gc.collect()
    rss_before = rss_gb()

    instances = {}
    for name, gpt, sovits in SPEAKERS:
        t = TTS(use_bert=True)

        def load(t=t, gpt=gpt, sovits=sovits):
            t.load_gpt_model(gpt)
            t.load_sovits_model(sovits)

        _, dt = timed(load, f"load full models '{name}'")
        instances[name] = (t, dt)

    rss_b = rss_gb() - rss_before
    print(f"  RSS delta: {rss_b:.2f} GB", flush=True)

    results_b = {}
    for name, _, _ in SPEAKERS:
        t = instances[name][0]
        _, w = timed(lambda n=name: t.infer(
            spk_audio_path=SPK_AUDIO, prompt_audio_path=PROMPT_AUDIO,
            prompt_audio_text=PROMPT_TEXT, text=TEXT,
        ), f"warmup infer '{name}' (full)")
        times = []
        for i in range(N_AVG):
            t0 = time.time()
            t.infer(
                spk_audio_path=SPK_AUDIO, prompt_audio_path=PROMPT_AUDIO,
                prompt_audio_text=PROMPT_TEXT, text=TEXT,
            )
            times.append(time.time() - t0)
        avg = sum(times) / len(times)
        results_b[name] = (w, avg)
        print(f"  infer '{name}': warmup {w:.1f}s, avg {avg:.1f}s", flush=True)

    # ── Summary ──
    print("\n" + "=" * 60, flush=True)
    print("SUMMARY", flush=True)
    print("=" * 60, flush=True)
    print(f"{'':30s} {'Shared (A)':>12s} {'Full (B)':>12s} {'Speedup':>10s}", flush=True)
    print(f"{'Init (3 speakers)':30s} {init_a:>10.1f}s {sum(v[1] for v in instances.values()):>10.1f}s", flush=True)
    print(f"{'Peak RSS delta':30s} {rss_a:>10.2f}G {rss_b:>10.2f}G {'saved ' + f'{100*(1-rss_a/rss_b):.0f}%' if rss_b > 0 else '':>10s}", flush=True)
    for name, _, _ in SPEAKERS:
        wa, aa = results_a[name]
        wb, ab = results_b[name]
        print(f"{'infer avg ' + name:30s} {aa:>10.1f}s {ab:>10.1f}s {ab/aa:>8.2f}x", flush=True)

    # RTF for first speaker (audio len ~ text duration)
    clip = mtts.infer(SPEAKERS[0][0], TEXT)
    audio_len = clip.audio_len_s
    rtf_a = results_a[SPEAKERS[0][0]][1] / audio_len
    rtf_b = results_b[SPEAKERS[0][0]][1] / audio_len
    print(f"\nAudio length: {audio_len:.2f}s", flush=True)
    print(f"RTF shared: {rtf_a:.3f} | RTF full: {rtf_b:.3f}", flush=True)


if __name__ == "__main__":
    main()
