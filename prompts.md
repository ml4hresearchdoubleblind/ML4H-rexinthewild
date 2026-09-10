# Prompts and hyperparameters

Every prompt and decoding setting used to build and evaluate ReXInTheWild.
Sections 1-8 are the pipeline prompts, reproduced verbatim from the code that
produced the released data. The hyperparameter tables follow.

Question **generation** ran on Azure OpenAI against a vision-capable deployment
(default `gpt-5`). Only `model` and `messages` were sent -- no `temperature`,
`top_p`, `max_tokens` or reasoning-effort parameter -- so generation used the
service defaults in force at the time. Stages 1-7 below run in order; stage 8 is
the evaluation prompt used for every model reported in the paper.

After scoring (stage 6), questions were ranked per image by the **sum** of the
four scores -- clarity, medical relevance, medical difficulty and visual
difficulty -- and the **top 2 per image** were retained. That is why the released
set holds 942 questions over 479 images.

---

## 1. Caption Filtering

```text
# Caption Filtering
prompt = (
        "Below is a short image caption. Does it describe an image that could "
        "potentially match an external image of a real person? Do NOT count drawn/simulated "
        "images of people, x-rays/endoscopy images/other specialized medical scans, histology/micro images, or images of non-human animals or objects. "
        "If ambiguous, default towards 'yes'.\n\n"
        f"Caption: {caption}\n\n"
        "Answer YES or NO."
    )
```

---

## 2. Image Filtering

```text
# Image Filtering
Image content filter
prompt = (
        "For the given image and caption, decide if this image could be relevant for assessing some medical condition"
        " in an everyday, NON-HOSPITAL context, such as:\n"
        "1. Facial abnormalities\n"
        "2. Dermatological conditions\n"
        "3. Postural or musculoskeletal conditions.\n\n"
        "Labels:\n- NO: Does not show a person, is a drawing/simulation, person not reasonably visible, or setup is clearly clinical/imaging only.\n"
        "- YES: Person or a body part is clearly visible in detail, could plausibly be in a normal setting, some medical content OK if ordinary context possible (e.g. showing wheelchairs or an aide helping someone stretch).\n"
        "- WITH CROPS: Collages or group images with at least one crop-able qualifying area.\n\n"
        f"Caption: {caption}\n\n"
        "Choose one: YES / NO / WITH CROPS. Just return one of those (don't explain)."
    )
```

---

## 3. Question generator

```text
# Question generator
user_message = {
        "role": "user",
        "content": [
            {
                "type": "text",
                "text": (
                    "Given the following medical image and its clinical caption, "
                    "write 5 multiple-choice questions that test understanding of the pose and/or medical content. "
                    "Each question should offer 3-5 answer choices and the correct answer should be clearly marked. "
                    "If the image does not show a patient in a pose with medical implications or if it focuses primarily on a skin condition, return an empty list. "
                    "If you cannot properly read the image, return an empty list. "
                    "All questions should require analysis of the specific image, not just theoretical medical knowledge about "
                    "how conditions usually present or how tests are typically performed."
                    "The questions should capture any medical topics covered by the caption but can also reference other details of the image. "
                    "However the question should not directly mention the existence of a caption, as the test-taker will not see the caption."
                    "The questions should be clear and concise, avoiding unnecessary details or leading phrasing. "
                    "For example, you should ask 'what jaw abnormality is present' rather than 'based on the forward projection of the jaw, what abnormality is present?'."
                    "If the image contains multiple subfigures, specify which subfigure each question is about. "
                    "Output your questions as a JSON list following this format:\n\n"
                    "[{\"question\": \"...\", \"choices\": [\"...\"], \"answer\": \"...\"}, ...]\n\n"
                    f"Caption: {caption}"
                )
            },
            {
                "type": "image_url",
                "image_url": {"url": data_url}
            }
        ]
    }
```

---

## 4. Question editor

```text
# Question editor
prompt = (
        "You are editing a multiple-choice medical question to make it more concise and direct.\n\n"
        "Remove leading phrases that give away unnecessary information, e.g. telling the reader about the image content "
        "instead of making them interpret the image themselves."
        "Avoid including details that clearly point towards the correct answer or rule out the distractors."
        "Leave only the core information required to understand and answer the question."
        "For example:\n"
        "BAD: 'In subfigure b, the prone position with the knee flexed primarily increases stretch on which muscle?'\n"
        "GOOD: 'In subfigure b, the leg position primarily increases stretch on which muscle?'\n\n"
        "BAD: 'Based on the visible fullness just inferior and anterior to the right ear lobule over the angle of the mandible, "
        "which structure is most likely involved?'\n"
        "GOOD: 'Which structure is likely involved in the ear and mandible findings?'\n\n"
        "Keep the question clear and medically accurate. If the question references a specific subfigure, keep that reference.\n\n"
        "Caption: {caption}\n\n"
        "Question: {question}\n\n"
        "Answer Choices:\n{choices_text}\n\n"
        "Correct Answer: {answer}\n\n"
        "Return ONLY the edited question text, nothing else. If the question is already concise, return it as is."
    ).format(
        caption=caption,
        question=question,
        choices_text=choices_text,
        answer=answer
    )
```

---

## 5. Distractor editor

```text
# Distractor editor
prompt = (
        "You are editing the answer choices for a multiple-choice medical question to maximize difficulty and make all choices plausible.\n"
        "Your goal is to increase the chance that a hasty or inattentive reader would plausibly confuse distractors with the true answer.\n\n"
        "Ways to do this include (but are not limited to):\n"
        "- Mix up left/right, penalize minor counting, measurement or anatomical errors (e.g., mixing up adjacent bones, incorrectly describing severity).\n"
        "- If there are subtle, easily overlooked abnormalities, offer an answer choice suggesting there is no abnormality at all.\n"
        "- It is okay and often desirable for two or more answer choices to sound very similar, as long as only one is truly correct.\n"
        "- Avoid distractors that are obviously incorrect for any reader (for example, if an image shows a ruler, 'tape measure' is a better distractor than 'penlight'). The distractors should be plausible for the image and scenario.\n"
        "- Ensure that the final set of choices contains 3-5 choices, exactly one of which is correct.\n"
        "- You may keep choices and answer the same if they are already maximally difficult and plausible according to the image and scenario.\n\n"
        "Given:\n"
        "Caption: {caption}\n"
        "Question: {edited_question}\n"
        "Current Answer Choices:\n{choices_text}\n"
        "Current Correct Answer: {answer}\n\n"
        "Return a JSON object with exactly two keys:\n"
        "  edited_choices: Choices string suitable to replace the old choices column (joined by '; ' if multiple choices).\n"
        "  edited_answer: The new correct answer string (must match exactly one of the edited_choices).\n"
        "Do not add any explanations or extraneous text."
    ).format(
        caption=caption,
        edited_question=edited_question,
        choices_text=choices_text,
        answer=answer
    )
```

---

## 6. Scorer

```text
# Scorer
prompt = (
        "You are scoring a multiple-choice medical question on 4 dimensions (each 1-5):\n\n"
        "1. CLARITY (1-5): Is the correct answer clearly the only acceptable option? Is the question clear and easy to understand for a clinician who can see the image but not the caption? \n"
        "For example, if a question is about one hand but both hands are seen in the image, then it should be clear which hand is being asked about. \n\n"
        "Picking the right answer should not require info that cannot be reasonably inferred from the image alone. "
        "Give low scores to questions that are ambiguous or open-ended.\n\n"
        "2. MEDICAL RELEVANCE (1-5): How relevant is the question to the primary medical content of the image? "
        "For example, if the image primarily shows someone with a large cut and incidental hyperpigmentation, "
        "a question about the cut would be highly relevant (5) and a question about the hyperpigmentation would be less relevant (1-2).\n\n"
        "3. MEDICAL DIFFICULTY (1-5): Does it require recognizing medical concepts or reasoning about non-obvious medical implications? "
        "Give high scores if picking the correct answer from the given options requires medical knowledge and reasoning.\n\n"
        "4. VISUAL DIFFICULTY (1-5): Does the question require visual reasoning? For example, does it require the reader to consider "
        "directions like left, right, dorsal, palmar, etc.? "
        "Does it require counting or reasoning about fine-grained details, hidden objects or blurry areas? Does it require careful visual analysis?"
        " Give low scores to questions that can be answered without looking at the image, just by using abstract knowledge.\n\n"
        "Caption: {caption}\n\n"
        "Question: {question}\n\n"
        "Answer Choices:\n{choices_text}\n\n"
        "Correct Answer: {answer}\n\n"
        "Return a JSON object with exactly these keys: clarity_score, medical_relevance_score, medical_difficulty_score, visual_difficulty_score. "
        "Each value should be an integer from 1 to 5."
    ).format(
        caption=caption,
        question=question,
        choices_text=choices_text,
        answer=answer
    )
```

---

## 7. Question Tagger

```text
# Question Tagger

tag_options = [
        "Eyes",
        "Mouth & Jaws",
        "Head & Neck",
        "Trunk & Extremities",
        "Skin & Hair",
        "Surgical & Procedural",
        "Other",
    ]

    sys_message = {
                "role": "system",
                "content": (
                    "You are a helpful medical expert. Your task is to categorize the following question, "
                    "considering its associated image, into exactly one of the given medical specialties/tags. "
                    "Pick the single best-fitting tag for the question from these options:\n"
                    "If you cannot read the question or image, return 'ERROR'.\n"
                    + "\n".join(f"- {tag}" for tag in tag_options) +
                    "\nReturn ONLY the tag string; do not repeat the question or add any preamble or explanation.\n"
                    "Use 'Other' for questions that do not clearly fit any prior label.\n"
                ),
            }
            user_content = [{"type": "text", "text": f"Question: \"{q.strip()}\""}]
```

---

## 8. Generating answers

```text
# Generating answers

SYSTEM_PROMPT = """You are an expert clinician."""
USER_PROMPT = (
    "Given the following medical image and a multiple-choice question, select the single best answer from the choices.\n"
    "ONLY output your FINAL answer, which must be placed inside curly brackets, e.g. {A}. "
    "Provide NO extra words, reasoning or explanation outside the curly brackets.\n"
    "When asked about left/right, answer from the patient's viewpoint unless otherwise stated.\n"
)
```


---

## Hyperparameters

### Question generation and pipeline stages

| Setting | Value |
|---|---|
| Provider | Azure OpenAI |
| Deployment | vision-capable; default `gpt-5` |
| Parameters sent | `model`, `messages` only |
| `temperature`, `top_p`, `max_tokens` | not sent (service defaults) |
| Questions kept per image | top 2 by summed score |
| Score components | clarity, medical relevance, medical difficulty, visual difficulty |

### Evaluation

Decoding was left at each vendor's documented default rather than tuned; all of
these models are reasoning models whose providers restrict or discourage
sampling parameters. `null` means the parameter was deliberately not sent.

| Model | Identifier | Settings |
|---|---|---|
| GPT-6 Astra | `gpt-6-astra` (Azure) | no sampling parameters; reasoning effort at API default |
| GPT-6 Astra, high-compute arm | `gpt-6-astra` (Azure) | `reasoning_effort: xhigh` |
| GPT-5.6 Sol | `gpt-5.6-sol` (Azure) | no sampling parameters; reasoning effort default (medium) |
| GPT-5.6 Sol, high-compute arm | `gpt-5.6-sol` (Azure) | `reasoning_effort: xhigh` |
| Claude Opus 5 | `claude-opus-5` | `max_tokens: 8192`; thinking effort default (high) |
| Claude Opus 5, high-compute arm | `claude-opus-5` | `max_tokens: 8192`, `effort: max` |
| Gemini 3.7 Flash | `gemini-3.7-flash` (Vertex AI) | `temperature: 1.0`, `thinking_level: high` |
| Gemini 3.1 Pro | `gemini-3.1-pro-preview` (Vertex AI) | `temperature: 1.0`, `thinking_level: high` |
| Qwen3-VL-8B | `Qwen/Qwen3-VL-8B-Instruct` | `do_sample: false`, `num_beams: 1`, `max_new_tokens: 512` |
| Qwen3-VL-32B | `Qwen/Qwen3-VL-32B-Instruct` | `do_sample: false`, `num_beams: 1`, `max_new_tokens: 512` |
| Lingshu-7B | `lingshu-medical-mllm/Lingshu-7B` | `do_sample: false`, `num_beams: 1`, `max_new_tokens: 512` |
| MedGemma-4B | `google/medgemma-4b-it` | `do_sample: false`, `num_beams: 1`, `max_new_tokens: 512` |

Notes on the choices above:

- **Anthropic**: `temperature`, `top_p` and `top_k` were not sent, as Claude Opus
  4.7 and later reject them. `max_tokens` is set well above what an answer needs
  because it bounds thinking and visible output together; runs that truncated
  mid-thinking were re-run at a higher bound and the override is recorded in the
  run metadata.
- **Google**: `temperature 1.0` and `thinking_level high` are both documented
  defaults, and `high` is the maximum. `top_p` and `top_k` were left unset, as
  Google publishes no defaults and advises against tuning them.
- **Open-weight**: greedy decoding makes these runs deterministic and avoids the
  degraded behaviour MedGemma shows under its base model's sampling defaults.
  Run in bfloat16 on one NVIDIA A100 80 GB, Transformers 5.16.1, PyTorch 2.9.1,
  CUDA 12.9.

### Answer-option ordering

Options were shuffled once during dataset construction with a seed derived from
the question content, then held fixed, so every model saw the same ordering:

```python
seed = int(hashlib.sha256("||".join(items).encode()).hexdigest(), 16) % (2**32)
rng  = np.random.default_rng(seed)
```

Models were instructed to interpret left and right from the patient's
perspective unless the question said otherwise, and to return a bracketed
letter. Responses with no recoverable answer were scored incorrect.
