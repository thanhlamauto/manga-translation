import os
import json
from math import ceil
from typing import List, Dict, Any, Tuple

from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel

from prompts import (
    LOCAL_CONTEXT_SYSTEM_PROMPT,
    CHARACTER_PROFILES_SYSTEM_PROMPT,
    ENRICH_TRANSCRIPT_SYSTEM_PROMPT,
    TRANSLATE_TRANSCRIPT_SYSTEM_PROMPT
)

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Lazy client initialization - only create when needed
_client = None

def get_client():
    """Get or create the OpenAI client."""
    global _client
    if _client is None:
        if not OPENAI_API_KEY:
            raise ValueError(
                "OPENAI_API_KEY is not set. Please set it in your .env file or environment."
            )
        _client = OpenAI(api_key=OPENAI_API_KEY)
    return _client

MODEL_NAME = "gpt-5-nano-2025-08-07"

# Input files
LOCAL_CONTEXT_TXT_FILE = "data/local_context.txt"
TRANSCRIPT_JSON_FILE = "data/transcript.json"

# Output file
OUTPUT_JSON_FILE = "translated.json"

# Processing settings
ENRICH_CHUNK_SIZE = 20
TRANSLATE_BATCH_SIZE = 15

# ============ MODELS ============

class LocalContext(BaseModel):
    summary: str
    characters: List[str]
    plot_details: str
    chapter_vibe: str

class FewShotExample(BaseModel):
    en: str
    vi: str
    explanation: str

class AddressingRule(BaseModel):
    target: str
    pronoun_pair: str

class CharacterProfile(BaseModel):
    name: str
    role: str
    voice_tone: str
    personality_in_chapter: str
    vocabulary: List[str]
    addressing_rules: List[AddressingRule]
    style_guide: str
    few_shot_examples: List[FewShotExample]

class ProfilesWrapper(BaseModel):
    profiles: List[CharacterProfile]

class EnrichedResponse(BaseModel):
    index: int
    character: str
    text: str
    scene: str
    receiver: str

class EnrichedChunk(BaseModel):
    dialogues: List[EnrichedResponse]

class TranslatedResponse(BaseModel):
    index: int
    character: str
    text: str
    scene: str
    vi: str

class TranslatedChunk(BaseModel):
    dialogues: List[TranslatedResponse]

# ============ HELPERS ============

def read_text(path: str) -> str:
    if not os.path.exists(path):
        return ""
    with open(path, "r", encoding='utf-8') as f:
        return f.read()

def read_json(path: str) -> Any:
    if not os.path.exists(path):
        return []
    with open(path, "r", encoding='utf-8') as f:
        return json.load(f)

# ============ STEP 1: BUILD LOCAL CONTEXT ============

def build_local_context(transcript: List[Dict]) -> Dict[str, Any]:
    print("\n[STEP 1] Building Local Context...")
    
    raw_ctx = read_text(LOCAL_CONTEXT_TXT_FILE)
    transcript_characters = sorted({d["character"] for d in transcript})

    user_payload = {
        "local_context_text": raw_ctx,
        "transcript_character_names": transcript_characters
    }

    response = get_client().beta.chat.completions.parse(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": LOCAL_CONTEXT_SYSTEM_PROMPT},
            {"role": "user", "content": json.dumps(user_payload, ensure_ascii=False)}
        ],
        response_format=LocalContext
    )

    local_ctx = response.choices[0].message.parsed.model_dump()
    print(f"✓ Local context created. Vibe: {local_ctx['chapter_vibe']}")
    
    return local_ctx

# ============ STEP 2: BUILD CHARACTER PROFILES ============

def build_character_profiles(local_ctx: Dict, transcript: List[Dict]) -> Dict[str, Dict]:
    print("\n[STEP 2] Building Character Profiles...")
    
    character_names = sorted({d["character"] for d in transcript})

    user_payload = {
        "chapter_context": {
            "summary": local_ctx["summary"],
            "plot_details": local_ctx["plot_details"],
            "vibe": local_ctx.get("chapter_vibe", "Intense battle")
        },
        "character_names_to_profile": character_names,
        "instruction": "Generate profiles for ALL characters with addressing rules."
    }

    response = get_client().beta.chat.completions.parse(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": CHARACTER_PROFILES_SYSTEM_PROMPT},
            {"role": "user", "content": json.dumps(user_payload, ensure_ascii=False)}
        ],
        response_format=ProfilesWrapper,
    )

    profiles_list = [p.model_dump() for p in response.choices[0].message.parsed.profiles]
    profiles_by_name = {p["name"]: p for p in profiles_list}
    
    print(f"✓ Created {len(profiles_list)} character profiles")
    
    return profiles_by_name

# ============ STEP 3: ENRICH TRANSCRIPT ============

def enrich_transcript(local_ctx: Dict, transcript: List[Dict]) -> List[Dict]:
    print("\n[STEP 3] Enriching Transcript...")
    
    total = len(transcript)
    num_chunks = ceil(total / ENRICH_CHUNK_SIZE)
    enriched_data_map = {}

    for chunk_idx in range(num_chunks):
        start = chunk_idx * ENRICH_CHUNK_SIZE
        end = min((chunk_idx + 1) * ENRICH_CHUNK_SIZE, total)
        print(f"  Processing chunk {chunk_idx + 1}/{num_chunks} (lines {start}–{end - 1})...")

        dialogues_chunk = [
            {
                "index": d["index"],
                "character": d["character"],
                "text": d["text"],
            }
            for d in transcript[start:end]
        ]

        user_payload = {
            "chapter_summary": local_ctx.get("summary", ""),
            "plot_details": local_ctx.get("plot_details", ""),
            "dialogues_chunk": dialogues_chunk,
        }

        user_prompt = (
            "Hãy phân tích và bổ sung 'scene' và 'receiver' cho các đoạn thoại sau:\n"
            + json.dumps(user_payload, ensure_ascii=False, indent=2)
        )

        try:
            response = get_client().beta.chat.completions.parse(
                model=MODEL_NAME,
                messages=[
                    {"role": "system", "content": ENRICH_TRANSCRIPT_SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                response_format=EnrichedChunk,
            )

            enriched_items = response.choices[0].message.parsed.dialogues
            
            for item in enriched_items:
                enriched_data_map[item.index] = {
                    "scene": item.scene,
                    "receiver": item.receiver
                }
        except Exception as e:
            print(f"  ✗ Error at chunk {chunk_idx}: {e}")

    enriched_transcript = []
    for d in transcript:
        idx = d["index"]
        enrich_info = enriched_data_map.get(idx, {"scene": "", "receiver": "Unknown"})
        
        d_out = dict(d)
        d_out["scene"] = enrich_info["scene"]
        d_out["receiver"] = enrich_info["receiver"]
        enriched_transcript.append(d_out)

    print(f"✓ Enriched {len(enriched_transcript)} dialogues")
    
    return enriched_transcript

# ============ STEP 4: TRANSLATE ============

def translate_batch(
    entries_chunk: List[Dict[str, Any]], 
    profiles_by_name: Dict[str, Dict[str, Any]], 
    local_ctx: Dict[str, Any], 
    previous_vi_context: str
) -> Tuple[List[TranslatedResponse], str]:

    prepared_dialogues = []

    for entry in entries_chunk:
        idx = entry.get("index")
        speaker_raw = (entry.get("character") or "Narrator").strip()
        text_en = (entry.get("text") or "").strip()
        scene = (entry.get("scene") or "").strip()
        receiver = (entry.get("receiver") or "Unknown").strip()

        if not text_en:
            continue

        character_info = profiles_by_name.get(speaker_raw)

        addressing_rules_str = "Default: Tôi - Bạn"
        if character_info and "addressing_rules" in character_info:
            rules_list = character_info["addressing_rules"]
            formatted_rules = [f"{r['target']}: {r['pronoun_pair']}" for r in rules_list]
            addressing_rules_str = "; ".join(formatted_rules)
        
        if character_info is None:
            character_info = {
                "role": "Unknown / Narrator",
                "voice_tone": "Trung lập",
                "vocabulary": [],
                "style_guide": "Rõ ràng, sát nghĩa.",
                "few_shot_examples": []
            }

        prepared_dialogues.append(
            {
                "index": idx,
                "character": speaker_raw,
                "receiver": receiver,
                "text": text_en,
                "scene": scene,
                "profile_summary": {
                    "role": character_info.get("role"),
                    "voice_tone": character_info.get("voice_tone"),
                    "vocabulary": character_info.get("vocabulary", []),
                    "addressing_rules": addressing_rules_str,
                },
            }
        )

    if not prepared_dialogues:
        return [], previous_vi_context

    user_payload = {
        "chapter_context": {
            "summary": local_ctx.get("summary", ""),
            "vibe": local_ctx.get("chapter_vibe", "")
        },
        "previous_context_snippet": previous_vi_context[-200:],
        "dialogues_to_translate": prepared_dialogues,
    }

    user_prompt = (
        "Hãy dịch các đoạn thoại sau đây, chú ý kỹ luật xưng hô dựa trên người nghe.\n"
        + json.dumps(user_payload, ensure_ascii=False, indent=2)
    )

    response = get_client().beta.chat.completions.parse(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": TRANSLATE_TRANSCRIPT_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        response_format=TranslatedChunk,
    )

    parsed_chunk: TranslatedChunk = response.choices[0].message.parsed

    last_vi = previous_vi_context
    if parsed_chunk.dialogues:
        last = parsed_chunk.dialogues[-1]
        last_vi = f"{last.character}: {last.vi}"

    return parsed_chunk.dialogues, last_vi

def translate_transcript(
    enriched_transcript: List[Dict],
    profiles_by_name: Dict[str, Dict],
    local_ctx: Dict
) -> List[Dict]:
    print("\n[STEP 4] Translating Transcript...")
    
    total = len(enriched_transcript)
    previous_vi_context = "Bắt đầu chương truyện."
    all_translated_results = []

    start = 0
    batch_id = 1

    while start < total:
        end = min(start + TRANSLATE_BATCH_SIZE, total)
        chunk = enriched_transcript[start:end]

        print(f"  Translating batch {batch_id} (lines {start}–{end - 1})...")

        translated_objects, previous_vi_context = translate_batch(
            chunk, profiles_by_name, local_ctx, previous_vi_context
        )
        
        all_translated_results.extend(translated_objects)

        start = end
        batch_id += 1

    final_json_data = [
        {
            "index": item.index,
            "character": item.character,
            "text_en": item.text,
            "scene": item.scene,
            "text_vi": item.vi
        }
        for item in all_translated_results
    ]
    
    final_json_data.sort(key=lambda x: x["index"])
    
    print(f"✓ Translated {len(final_json_data)} dialogues")
    
    return final_json_data


# ============ PUBLIC CORE API ============

def run_contextual_translation(
    transcript: List[Dict[str, Any]],
    source_lang: str,
    target_lang: str,
) -> List[Dict[str, Any]]:
    """
    High-level helper to run the full contextual translation pipeline in memory.

    This is designed to be reused from other parts of the codebase (e.g. image/PDF
    pipelines) without touching any files on disk.

    Args:
        transcript: List of dialogue dicts. Each item should at least contain:
            - index: int
            - character: str
            - text: str
          Optional fields like ``scene`` or ``receiver`` will be ignored at input
          time because they are recomputed inside this pipeline.
        source_lang: Source language name (currently primarily English).
        target_lang: Target language name (currently primarily Vietnamese).

    Returns:
        A list of dicts sorted by ``index`` with at least:
            - index: int
            - character: str
            - text_en: str
            - scene: str
            - text_vi: str

    Note:
        For now ``source_lang`` and ``target_lang`` are not used to branch logic,
        but they are part of the signature to make it easy to extend this
        pipeline for additional language pairs in the future.
    """
    if not transcript:
        return []

    # Step 1: Build local context from raw transcript + optional local_context.txt
    local_ctx = build_local_context(transcript)

    # Step 2: Build character profiles
    profiles_by_name = build_character_profiles(local_ctx, transcript)

    # Step 3: Enrich transcript with scene / receiver
    enriched_transcript = enrich_transcript(local_ctx, transcript)

    # Step 4: Translate enriched transcript
    final_translation = translate_transcript(enriched_transcript, profiles_by_name, local_ctx)

    return final_translation


# ============ MAIN PIPELINE ============

def main():
    print("=" * 60)
    print("MANGA TRANSLATION PIPELINE")
    print("=" * 60)
    
    # Load transcript
    transcript = read_json(TRANSCRIPT_JSON_FILE)
    if not transcript:
        print("✗ Transcript file is missing or empty!")
        return
    
    print(f"\nLoaded {len(transcript)} dialogues from transcript.json")
    
    # Step 1: Build local context
    local_ctx = build_local_context(transcript)
    
    # Step 2: Build character profiles
    profiles_by_name = build_character_profiles(local_ctx, transcript)
    
    # Step 3: Enrich transcript
    enriched_transcript = enrich_transcript(local_ctx, transcript)
    
    # Step 4: Translate
    final_translation = translate_transcript(enriched_transcript, profiles_by_name, local_ctx)
    
    # Save output
    with open(OUTPUT_JSON_FILE, "w", encoding="utf-8") as f:
        json.dump(final_translation, f, ensure_ascii=False, indent=4)
    
    print("\n" + "=" * 60)
    print(f"✓ COMPLETE! Translation saved to: {OUTPUT_JSON_FILE}")
    print("=" * 60)

if __name__ == "__main__":
    main()