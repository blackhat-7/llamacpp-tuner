# H3 prompt examples

Complete prompts in the shapes MiniMax H3 was trained on. Copy the shape, not the words.
Rules behind them: `lct video guide` and `lct video guide --ref`.

## 1. Text only, one shot with dialogue (`make --seconds 6`)

```text
integrated_multimodal_description: [Shot 1] Live-action, handheld phone footage, a medium shot frames a cramped computer-parts stall at a night market, shelves of boxed RAM sticks lit by a buzzing fluorescent tube. A wiry street vendor in a stained apron and reading glasses with a fast, raspy voice (S1) leans over the counter, holds up a single green RAM stick between two fingers like a jewel and says: <d>[English] Thirty-two gigs. Last one in the city. Price went up while you were looking.</d> The camera pushes in with small amplitude at fast speed toward the stick as he raises one eyebrow and taps the price tag, which reads "$999".

overall_soundscape: Crowd chatter and scooter engines pass behind the stall, the fluorescent tube hums, and the plastic RAM packaging crinkles as he lifts it.

non_diegetic_music: N/A
```

## 2. Text only, three shots with cuts (`make --seconds 10`)

```text
integrated_multimodal_description: [Shot 1] Live-action, cinematic, a wide shot frames a tiny corner shop crammed with towers of RAM boxes up to the ceiling. A tired customer in a hoodie (S1) walks in, looks around and asks in a flat, deadpan voice: <d>[English] Do you have any RAM?</d> [Shot 2] At 00:03.000, the camera cuts to a close-up of the shopkeeper, a cheerful man in a red vest with a booming voice (S2), who spreads his arms wide and says: <d>[English] Sir, this is an AI company now.</d> The camera pulls out with large amplitude at fast speed to reveal server racks behind him blinking blue. [Shot 3] At 00:07.000, the shot cuts to the customer's face as he slowly closes his wallet; the camera holds a static shot while he sighs.

overall_soundscape: A door chime rings as the customer enters, server fans whir loudly after the reveal, and the customer lets out a long sigh at the end.

non_diegetic_music: A cheesy elevator-music loop on electric piano at a moderate tempo, cutting out abruptly on the final sigh.
```

## 3. From a first frame (`make --first-frame shop.png --seconds 5`)

```text
For the target video, at 0.00 seconds into the target video, <Picture 1> (from [Shot 1]) is fully referenced.

integrated_multimodal_description: [Shot 1] Live-action, cinematic, the shop counter, lighting and the man in the grey sweater shown in <Picture 1> stay exactly as framed, preserving his appearance and the stacked RAM boxes behind him. He picks up the top box, turns it toward the camera and, in a calm, measured voice (S1), says: <d>[English] Limited edition. Very limited.</d> The camera pushes in with small amplitude at slow speed toward the box as he sets it down gently.

overall_soundscape: Quiet shop room tone, the soft thud of the cardboard box, a distant car horn outside.

non_diegetic_music: A soft synth pad at a slow tempo, rising slightly as the camera pushes in.
```

## 4. A person from photos, new scene, dialogue (`make --photo a.jpg --photo b.jpg --seconds 10`)

```text
subject_definitions:
<Subject 1> is the man in <Picture 1> and <Picture 2>, with short light-brown hair, a slim build, a narrow face and a calm, soft-spoken manner.
<Subject 2> is a tiny cluttered electronics shop invented for this video, with glass display cases full of RAM sticks and a hand-painted sign.

summary:
[reference generation] The target video shows <Subject 1> working as a shopkeeper in <Subject 2>, earnestly pitching a single RAM stick to an off-screen customer, then reacting to the price with a straight face.

retention_analysis:
<Subject 1> (appears in [Shot 1], [Shot 2]): fully_preserved - his face, hair and build are kept; he now wears a brown shopkeeper's apron over a grey T-shirt.
<Subject 2> (appears in [Shot 1], [Shot 2]): weak_reference - only the general look of a cramped electronics shop is followed.

detailed_description:
The target video is in a warm, slightly over-lit live-action sitcom style with a static tripod camera feel.
[Shot 1] A medium shot frames <Subject 1> (S1) standing behind a glass counter in <Subject 2>, wearing a brown apron, with shelves of boxed RAM sticks behind him and a hand-painted sign reading "RAM - ASK FOR PRICE". He lifts one green RAM stick from a velvet tray with both hands, looks just past the camera at the unseen customer and says in a soft, sincere voice: <d>[English] This is sixty-four gigabytes. I need you to understand how hard this was to get.</d> He places it back on the velvet with exaggerated care. The camera pushes in with small amplitude at slow speed.
[Shot 2] At 00:06.000, the shot cuts to a close-up of <Subject 1> (S1), who pauses, keeps a completely straight face and says: <d>[English] It's eleven thousand dollars.</d> He blinks once and does not move.

overall_soundscape: Quiet shop ambience with a ticking wall clock, the soft clack of the RAM stick on glass, and a long silent beat after the price.

non_diegetic_music: N/A
```

## 5. Edit: same video, new words in the person's mouth (`edit clip.mp4` on a clip with sound)

`lct` prints: `<Audio 1> sound of clip.mp4, <Video 1> clip.mp4`.

```text
subject_definitions:
<Video 1> is the source video for the target video edit.
<Subject 1> is the man speaking to camera in <Video 1>, with his appearance, clothing, setting and camera framing.
<Audio 1> is the synchronized audio track of <Video 1>; its room tone is reused and its voice is the timbre reference for <Subject 1> (S1).

summary:
[video editing + audio reference + audio reuse] The target video is an edited version of <Video 1>. <Subject 1> now says a new line with matching lip movements, in his own voice from <Audio 1>, while the picture, setting and background sound stay the same.

retention_analysis:
<Video 1> (whole video): partially_preserved - framing, setting, lighting and body movement are kept; only the mouth and expression change to fit the new line.
<Subject 1> (appears in [Shot 1]): fully_preserved - his face, hair and clothing are unchanged.
<Audio 1>: partially_copy - the room tone is kept, and the new speech follows his voice timbre without reusing the old words.

detailed_description:
The target video keeps the look of <Video 1> exactly.
[Shot 1] The edit animates <Subject 1>'s face to speak the new dialogue. <Subject 1> (S1), in the same position and framing as <Video 1>, says in his own voice from <Audio 1>: <d>[English] We are pleased to announce that RAM is now a subscription.</d> His mouth, jaw and cheeks move in sync with each word, and his head nods slightly on "subscription"; everything else in the frame stays as in <Video 1>.

overall_soundscape: The original room tone from <Audio 1> continues throughout.

non_diegetic_music: N/A
```

Other edits use the same frame: change `retention_analysis` to say what changes, for example
`attribute_transfer` for a new outfit, `partially_preserved` for a new background with
the same action, or `weak_reference` plus a new style line for a cartoon restyle.

## 6. Continuation (`extend shotA.mp4 --seconds 6`)

The new part's first second repeats the end of `shotA.mp4`; describe from there on.

```text
integrated_multimodal_description: [Shot 1] Live-action, cinematic, continuing seamlessly from the previous moment in the same shop and framing, the shopkeeper in the red vest (S1) is still holding the RAM box up to the customer. He slowly lowers it, leans in and whispers: <d>[English] For you? Payment plan.</d> The camera pushes in with small amplitude at slow speed to a close-up as he winks.

overall_soundscape: The same quiet shop ambience continues, with the creak of the counter as he leans on it.

non_diegetic_music: N/A
```

## 7. A voice to copy (`make --audio voice.wav --seconds 6`)

Without photos or a source video, the voice file is `<Audio 1>`.

```text
subject_definitions:
<Subject 1> is a weary late-night radio host invented for this video, a man in his fifties with headphones around his neck.
<Audio 1> is the voice-timbre reference for <Subject 1> (S1).

summary:
[reference generation + audio reference] The target video shows <Subject 1> in a dim radio booth delivering one line in the voice of <Audio 1>.

retention_analysis:
<Subject 1> (appears in [Shot 1]): weak_reference - only the described look is followed.
<Audio 1>: reference - the speaker follows <Audio 1>'s timbre, pitch and pacing without copying the signal or its words.

detailed_description:
The target video is in a moody live-action style with warm amber light.
[Shot 1] A close-up frames <Subject 1> (S1) at a microphone in a dark radio booth, a red ON AIR light glowing behind him. He leans into the microphone and says in the voice referenced from <Audio 1>: <d>[English] And that's the news. Nobody can afford memory anymore. Goodnight.</d> He switches off a fader and the red light goes dark.

overall_soundscape: Faint booth hiss, the click of the fader and the soft creak of his chair.

non_diegetic_music: N/A
```
