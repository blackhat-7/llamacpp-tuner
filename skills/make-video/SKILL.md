---
name: make-video
description: Make or edit videos with sound on this machine's own GPU (MiniMax H3 through `lct video`, fully local, no cloud). Use whenever the user asks to make, generate, animate, film, edit, restyle, re-dub, extend, lengthen, upscale or join a video, clip, reel, short, ad or meme, including ones starring a person, product or place from their photos. It needs only a description, a quality and a length.
---

# Make a video

You turn a request into finished video files. H3 makes picture and sound in one pass:
dialogue with lip sync, effects and music. Each clip is 4-15 s at 24 fps. Longer
videos are built from clips. You do the planning and prompt writing; `lct video` does the rest.

Run lct from any folder as `uv run --project ~/Documents/projects/llamacpp-tuner lct …`;
it is written `lct` below. Relative file paths are relative to your folder.

## 0. Rules

- **One GPU, one job.** Video needs ~18 GiB of the 24 GiB card. If `lct video` says
  the GPU is busy, an LLM server holds it: ask the user before running
  `lct down <name>`. You may be running on that server yourself.
- **Never raise the GPU power limit**; this machine's power supply has crashed under
  heavy GPU load. Run one job at a time; don't queue dozens of clips without asking.
- **Know the times**, and tell the user before a long job (read section 2).
- **Real people:** fine for parody and memes. Say the result should be labelled as AI
  when posted (YouTube, TikTok and Instagram require it). Commercial use of H3 output
  needs a MiniMax licence; mention it if the user is making ads.
- At most 3 tries per shot, then show the user what you have and ask.
- When finished, run `lct down video` to free the GPU, unless more video work follows.

## 1. Readiness

`lct video status` must show ComfyUI installed, the `video` profile, all model files and
enough free VRAM. If anything is missing, the fix is
`lct setup --backend comfyui --torch-index https://download.pytorch.org/whl/rocm7.2`
(about 80 GB of downloads the first time; ask first).

## 2. Turn the request into a plan

**Quality**: map the user's words to `--quality`.

| Quality | Size (16:9) | Use for | Time per 5 s on this GPU |
|---|---|---|---|
| `draft` | 832x480 | "quick", "test", "rough", trying ideas, picking a seed | about 2.5-4 min (measured: text 2m46s cold, photo 2m16s, extend 3m41s) |
| `standard` | 1344x768 | normal requests, social posts | about 20+ min (about 2m45s per step, 8 steps) |
| `high` | 1344x768, upscaled to 1080p | "HD", "1080p", "best", "final", YouTube | standard + about 16 min of upscaling per 5 s |

On this GPU, when the user names no quality, make `draft` and offer a `standard` render of the keeper. `edit` takes about twice as long as `make` (the source video rides along: 7.5 min for a 5 s draft). Time grows faster than length: a 15 s clip takes well over 3x a 5 s one. For a final
video, draft first to check the idea and wording, then make the keeper at the target
quality with the same prompt (same seed for the same look).

**Shape**: `--aspect 16:9` (landscape, default), `9:16` (reels, shorts, TikTok),
`1:1`, `4:3`, `3:4`.

**Length**:
- 4-15 s: one clip, `--seconds` = the requested length. One clip can still hold
  several shots with cuts (section 3).
- Over 15 s: write a shot list of 5-12 s parts. A continuous scene (same place,
  action carries on): make the first part, then `lct video extend` it once per part.
  Separate scenes: `lct video make` each part with the same `--photo` files for the same
  people, then `lct video join`. For `high`, build everything at `standard`, join,
  then `lct video upscale` the final file once.
- Under 4 s: make 4 s and trim with ffmpeg (section 7).

**Mode**: pick by what the user gives you.

| The user has… | Command | Prompt format |
|---|---|---|
| only a description | `make` | base |
| an image to start from / end on | `make --first-frame` / `--last-frame` | base, with its first line |
| photos of who or what must appear | `make --photo …` | full-reference |
| a voice or music to copy | add `--audio …` | full-reference |
| a video to change (dialogue, person, outfit, style, setting) | `edit SOURCE` | full-reference, `[video editing]` |
| a video to continue | `extend SOURCE` | base (full-reference with photos) |
| a video to sharpen or enlarge | `upscale SOURCE` | none |
| clips to put together | `join A B … --out X` | none |
| a trim, speed change, crop, overlay or music swap | ffmpeg | section 7 |

## 3. Write the prompt

The prompt decides the result. MiniMax's hosted service rewrites every prompt into a fixed
structure first, and the open model expects it, so **you** write that structure. Do not pass
the user's sentence through as it is. Write it to a file and pass `--prompt-file`.
For the full official rules run `lct video guide` (base) and `lct video guide --ref`
(full-reference); complete worked examples are in `references/prompt-examples.md`.

### Base format (text, first/last frame, extend)

```text
integrated_multimodal_description: [Shot 1] Live-action, cinematic, a medium shot frames … (S1) says: <d>[English] Exact words.</d> … [Shot 2] At 00:04.500, the camera cuts to …

overall_soundscape: 1-4 sentences of ambience, action sounds and breathing; no dialogue.

non_diegetic_music: 1-3 sentences on instruments, tempo and dynamics, or N/A.
```

- Open `[Shot 1]` with the style and the first composition: `Live-action, cinematic`,
  `2D-animated`, `3D CG`, `claymation`, `vintage film`, `handheld phone footage`.
- `[Shot 1]` has no time. Later shots start `[Shot N] At MM:SS.mmm, the camera cuts to …`
  with times inside the length. A cut must add something new; otherwise move the camera.
- Camera: motion + optional amplitude + speed, as plain English: `The camera pushes in
  with small amplitude at slow speed toward …`. Motions: zoom in/out, push in/pull out, pan,
  truck, tilt, pedestal, arc shot, tracking shot, static shot, shake slightly/strongly,
  POV, roll.
- Every speaker gets an ID on first appearance, with who they are and how they sound:
  `a cheerful middle-aged shopkeeper with a fast, nasal voice (S1) says: <d>[English] …</d>`.
  Keep the ID in later shots; together: `(S1,S2)`. Only the spoken words go inside
  `<d>[Language] …</d>`, word for word. Voiceover:
  `(S1) says in an off-screen voiceover: <d>…</d> while his lips remain completely closed.`
- Speech pace is about 2.5 English words per second. Leave about 0.5 s free at the end.
  Too many words for the length gives rushed or cut-off speech.
- On-screen text goes in double quotes: `a sign reading "RAM SALE"`. Keep text short; H3
  spells short English text well.
- First frame (`--first-frame`): put this line first, then a blank line:
  `For the target video, at 0.00 seconds into the target video, <Picture 1> (from [Shot 1]) is fully referenced.`
  Then describe what is in the image and how it moves on from there.
- Last frame only (`--last-frame`):
  `How the reference pictures align with the target video — <Picture 1> (from [Shot 1]) aligns with the S.SS-second mark of the target video.`
  (S.SS = the clip length, such as 5.00). Both frames: `How the reference pictures
  align with the target video — Picture 1 (from Shot 1) aligns with the 0.00-second mark
  of the target video; Picture 2 (from Shot 1) aligns with the S.SS-second mark of the
  target video.`, and prefer one continuous shot.
- Be concrete: what is seen and heard, frame by frame. Avoid "beautiful", "epic",
  "cinematic masterpiece". Say what must not appear only when it matters ("no
  subtitles, no watermark").
- For `extend`, describe only what happens next. The first second of the new part repeats
  the end of the source, so start with the source's last state and move on.

### Full-reference format (--photo, --audio, edit)

`lct` prints the labels before it runs: photos are `<Picture 1>…` in the order given.
For `edit`, the source video is `<Video 1>`, and its sound, if it has any, is
`<Audio 1>`. `--audio` files come after it. Six sections, in this order:

```text
subject_definitions:
<Subject 1> is the man in <Picture 1>, with … (what must stay: face, hair, glasses, clothes).
<Audio 1> is the voice-timbre reference for <Subject 1> (S1).

summary:
[reference generation + audio reference] The target video shows <Subject 1> …

retention_analysis:
<Subject 1> (appears in [Shot 1], [Shot 2]): fully_preserved - face, hair and glasses are kept; the clothes change to a shopkeeper's apron.
<Audio 1>: reference - the speaker follows <Audio 1>'s timbre without copying the signal.

detailed_description:
The target video is … (style in one or two sentences).
[Shot 1] … <Subject 1> (S1), standing behind the counter, says: <d>[English] …</d> …
[Shot 2] At 00:05.000, the shot cuts to …

overall_soundscape:
…

non_diegetic_music:
N/A
```

- `<Subject N>` is something reused on screen: a person, pet, product, place or style. Define each
  one from its sources, such as `<Subject 2> is the shop in <Picture 2>`. Use every label the
  same way in all six sections.
- `summary` starts with the task types joined by ` + `: `reference generation`,
  `keyframe completion`, `video editing` (an edit of `<Video 1>`; then begin with
  `The target video is an edited version of <Video 1>.`), `video continuation`,
  `audio reuse` (the same sound is kept, for example an edit that keeps its soundtrack),
  `audio reference` (only voice, timbre or style is followed).
- `retention_analysis`: one line per label. Use `fully_preserved`, `partially_preserved`,
  `attribute_transfer` or `weak_reference` for things on screen, and `fully_copy`,
  `partially_copy`, `reference` or `weak_reference` for audio.
- `detailed_description`: 350-500 words for new clips, shot by shot, using the base
  format's rules for shots, camera, speakers and `<d>`. A speaking subject is
  `<Subject N> (S1)`.
- Photos: 1-3 clear, well-lit photos of the face from the front and the side work best.
  Add a full-body photo if the outfit matters. More photos keep the face closer; up to 9.

## 4. Commands

```bash
lct video make --prompt-file p.txt --seconds 8 --quality standard [--aspect 9:16]
               [--photo face.jpg …] [--audio voice.wav …] [--first-frame a.png] [--last-frame b.png]
               [--seed 7] [--out result.mp4]
lct video edit SOURCE.mp4 --prompt-file p.txt [--quality …] [--photo …] [--audio …] [--out …]
lct video extend SOURCE.mp4 --prompt-file p.txt --seconds 6 [--photo …] [--out …]
lct video upscale SOURCE.mp4 [--short-side 1080|2160] [--out …]
lct video join a.mp4 b.mp4 c.mp4 --out final.mp4
lct video status | lct video guide [--ref] | lct down video
```

- The first job starts ComfyUI, which takes a minute or two. Each job prints `step N/M`, then
  `Saved: PATH (WxH, S s)`. Pass `--out` to get the file where you want it. Every
  result also stays in the library that `lct tui` shows on its Video page.
- Commands run until the clip is done (many minutes). Run them in the background or with a long
  timeout, and check on them rather than starting them again.
- `--seed` repeats a result exactly with the same inputs. Change it to get another take.
- `edit` uses the first 15 s of the source. `extend` takes up to 14 s of new content
  per call. `upscale` and `join` take any length.

## 5. Check every result yourself

Never report success on a file you have not checked:

```bash
ffprobe -v error -show_entries format=duration:stream=codec_type,width,height -of compact clip.mp4
ffmpeg -v error -i clip.mp4 -vf "fps=4/$(ffprobe -v error -show_entries format=duration -of csv=p=0 clip.mp4),scale=640:-1,tile=2x2" -frames:v 1 -y sheet.jpg   # 4 frames in one image
ffmpeg -v error -ss 2.5 -i clip.mp4 -frames:v 1 -y frame.jpg   # one full-size frame, to check a face or text
ffmpeg -i clip.mp4 -af volumedetect -f null - 2>&1 | grep mean_volume   # about -60 dB or lower means silence
```

Look at the frames. Is it the right person or product (compare with the photos)? Did the
described action happen? Are faces and hands clean, is text spelled right, is the
framing right? If something is wrong, make the prompt more concrete about it, try another
seed, or add better photos. Don't just retry the same prompt.

## 6. Example plan: "Sam Altman as a shopkeeper selling RAM, funny, 20 s, good quality"

1. Get 2-3 clear photos of him (ask the user, or use ones they gave you).
2. Plan the shots: shot A (10 s) behind a cluttered counter, pitching RAM sticks to a customer,
   with dialogue. Shot B (10 s) continues the scene: the customer asks the price, a deadpan
   reply, a reaction.
3. Write shot A in full-reference format (`<Subject 1>` from the photos, speaker S1, punchy
   dialogue of about 20 words). Run `lct video make --prompt-file a.txt --photo …
   --seconds 10 --quality draft` and check it.
4. Run the same prompt at `--quality standard` and keep the seed if the draft looked right.
   Write shot B in base format, describing only what happens next, and run
   `lct video extend shotA.mp4 --prompt-file b.txt --seconds 10 --photo …`.
5. Check it, `lct down video`, report the path and remind the user to label it as AI.

## 7. Edits without AI (ffmpeg)

```bash
ffmpeg -i in.mp4 -ss 2 -to 7 -c:v libx264 -c:a aac out.mp4                    # trim to 2-7 s
ffmpeg -i in.mp4 -vf "crop=ih*9/16:ih,scale=1080:1920" -c:a copy out.mp4        # 16:9 to 9:16
ffmpeg -i in.mp4 -filter_complex "[0:v]setpts=PTS/1.5[v];[0:a]atempo=1.5[a]" -map "[v]" -map "[a]" out.mp4  # 1.5x speed
ffmpeg -i in.mp4 -i music.mp3 -filter_complex "[0:a][1:a]amix=inputs=2:duration=first:weights=1 0.3[a]" -map 0:v -map "[a]" -c:v copy out.mp4  # add quiet music
ffmpeg -i in.mp4 -vf "subtitles=subs.srt" -c:a copy out.mp4                     # burn in subtitles
ffmpeg -i in.mp4 -an -c:v copy out.mp4                                          # remove sound
```

## 8. When something fails

| Message | Do |
|---|---|
| `The GPU has … GiB free … In use by: X` | Ask the user to free the GPU (`lct down X`), or wait |
| `Missing model files` / `No 'video' profile` | Run the setup in section 1 (ask first) |
| `… out of memory` | Use a shorter clip, `draft`, or fewer photos; `lct down video` and retry once |
| `ComfyUI refused the workflow` | A file or setting is wrong: run `lct video status`, then read the message |
| `ComfyUI exited` / `ComfyUI crashed during the job` | Read `tmp/servers/video.log` under the lct folder. On this AMD machine the `video` profile needs `--extra-args '--disable-pinned-memory --disable-smart-memory'`; if it is missing, tell the user. Then retry once |
| Speech cut off or rushed | Fewer words, or a longer clip |
| Wrong face | Clearer front photos, describe the face in `subject_definitions`, use `fully_preserved` |
