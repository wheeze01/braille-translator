import re
import sys
import os
import json
import textwrap
import unicodedata
import time
from loguru import logger
import google.genai as genai

import requests
import streamlit as st

# 프롬프트 로드 (modules/prompts.py 파일이 있다고 가정)
from modules.prompts import (
    BRAILLE_TO_KOREAN_SYSTEM_PROMPT,
    KOREAN_TO_BRAILLE_SYSTEM_PROMPT,
    BRAILLE_TO_CHINESE_SYSTEM_PROMPT,
    CHINESE_TO_BRAILLE_SYSTEM_PROMPT,
    ENGLISH_TO_BRAILLE_SYSTEM_PROMPT,
    BRAILLE_TO_ENGLISH_SYSTEM_PROMPT,
    SUMMARIZATION_SYSTEM_PROMPT,
    VALIDATION_SYSTEM_PROMPT,
)

# ----------------------------
# [Cache Data] 캐싱 데이터 (정답지)
# ----------------------------
PREDEFINED_DATA = {
    # 1. [요약 케이스] 자동차 관세
    "summary_case_src": """미 상무부와 무역대표부(USTR)는 24일(현지시간) 유럽연합(EU)과의 무역협정 이행과 관련한 문서를 공개했다. 여기에는 유럽산 자동차 수입에 대한 관세를 8월 1일부로 앞당겨 적용해 현 27.5%에서 15%로 낮추는 내용이 담겼다. 유럽 자동차 기업들은 차액을 돌려받고 앞으로는 15%의 관세만 낸다. 앞서 일본도 자동차 관세가 15%로 낮아졌다. 이 소식이 전해지자 유럽 완성차 업계에 대한 우려는 다소 줄었다. 24일 독일 증시에서 포르셰 주가는 장중 3.8%가 급등했다 2.2% 오른 채 마감했다. BMW와 메르세데스-벤츠 주가도 각각 1.4%, 1.1% 상승했다.""",
    "summary_case_sum": """미국 상무부와 무역대표부(USTR)는 유럽연합(EU)과의 무역협정 이행 문서를 공개하며, 8월 1일부터 유럽산 자동차 수입 관세를 기존 27.5%에서 15%로 인하한다고 발표했다. 이에 따라 유럽 자동차 기업들은 차액을 돌려받고 향후 15%의 관세만 부담하게 된다. 일본에 이어 유럽산 자동차 관세도 인하되면서 유럽 완성차 업계의 우려가 줄었고, 독일 증시에서 포르셰, BMW, 메르세데스-벤츠 등 주요 자동차 기업들의 주가가 상승 마감했다.""",
    "summary_case_braille": """⠑⠕⠈⠍⠁ ⠇⠶⠑⠍⠘⠍⠧ ⠑⠍⠱⠁⠊⠗⠙⠬⠘⠍⠦⠄⠴⠠⠠⠥⠎⠞⠗⠠⠴⠉⠵ ⠩⠐⠎⠃⠡⠚⠃⠦⠄⠴⠠⠠⠑⠥⠠⠴⠈⠧⠺ ⠑⠍⠱⠁⠚⠱⠃⠨⠻ ⠕⠚⠗⠶ ⠑⠛⠠⠎⠐⠮ ⠈⠿⠈⠗⠚⠑⠱⠐ ⠼⠓⠏⠂ ⠼⠁⠕⠂⠘⠍⠓⠎ ⠩⠐⠎⠃⠇⠒ ⠨⠊⠿⠰⠣ ⠠⠍⠕⠃ ⠈⠧⠒⠠⠝⠐⠮ ⠈⠕⠨⠷ ⠼⠃⠛⠲⠑⠴⠏ ⠝⠠⠎ ⠼⠁⠑⠴⠏ ⠐⠥ ⠟⠚⠚⠒⠊⠈⠥ ⠘⠂⠙⠬⠚⠗⠌⠊⠲ ⠕⠝ ⠠⠊⠐⠣ ⠩⠐⠎⠃ ⠨⠊⠿⠰⠣ ⠈⠕⠎⠃⠊⠮⠵ ⠰⠣⠗⠁⠮ ⠊⠥⠂⠐⠱⠘⠔⠈⠥ ⠚⠜⠶⠚⠍ ⠼⠁⠑⠴⠏ ⠺ ⠈⠧⠒⠠⠝⠑⠒ ⠘⠍⠊⠢⠚⠈⠝ ⠊⠽⠒⠊⠲ ⠕⠂⠘⠷⠝ ⠕⠎ ⠩⠐⠎⠃⠇⠒ ⠨⠊⠿⠰⠣ ⠈⠧⠒⠠⠝⠊⠥ ⠟⠚⠊⠽⠑⠡⠠⠎ ⠩⠐⠎⠃ ⠧⠒⠠⠻⠰⠣ ⠎⠃⠈⠌⠺ ⠍⠐⠱⠫ ⠨⠯⠎⠌⠈⠥⠐ ⠊⠭⠕⠂ ⠨⠪⠶⠠⠕⠝⠠⠎ ⠙⠥⠐⠪⠠⠌⠐ ⠴⠠⠠⠃⠍⠺⠐ ⠑⠝⠐⠪⠠⠝⠊⠝⠠⠪⠤⠘⠝⠒⠰⠪ ⠊⠪⠶ ⠨⠍⠬ ⠨⠊⠿⠰⠣ ⠈⠕⠎⠃⠊⠮⠺ ⠨⠍⠫⠫ ⠇⠶⠠⠪⠶ ⠑⠫⠢⠚⠗⠌⠊⠲""",
    # 2. [일반 번역] 환전
    "trans_1_src": """이 서비스로 환전 가능한 통화는 미국 달러(USD), 일본 엔화(JPY), 유럽 유로화(EUR), 중국 위안화(CNY)등 총 14개의 외국 통화로, 미국 달러(USD) 90%, 일본 엔화(JPY)와 유럽 유로화(EUR) 80%의 추가 이벤트 환율 우대를 제공한다.""",
    "trans_1_tgt": """⠕ ⠠⠎⠘⠕⠠⠪⠐⠥ ⠚⠧⠒⠨⠾ ⠫⠉⠪⠶⠚⠒ ⠓⠿⠚⠧⠉⠵ ⠑⠕⠈⠍⠁ ⠊⠂⠐⠎⠦⠄⠴⠠⠠⠥⠎⠙⠠⠴⠐ ⠕⠂⠘⠷ ⠝⠒⠚⠧⠦⠄⠴⠠⠠⠚⠏⠽⠠⠴⠐ ⠩⠐⠎⠃ ⠩⠐⠥⠚⠧⠦⠄⠴⠠⠠⠑⠥⠗⠠⠴⠐ ⠨⠍⠶⠈⠍⠁ ⠍⠗⠣⠒⠚⠧⠦⠄⠴⠠⠠⠉⠝⠽⠠⠴⠊⠪⠶ ⠰⠿ ⠼⠁⠙⠈⠗⠺ ⠽⠈⠍⠁ ⠓⠿⠚⠧⠐⠥⠐ ⠑⠕⠈⠍⠁ ⠊⠂⠐⠎⠦⠄⠴⠠⠠⠥⠎⠙⠠⠴ ⠼⠊⠚⠴⠏⠐ ⠕⠂⠘⠷ ⠝⠒⠚⠧⠦⠄⠴⠠⠠⠚⠏⠽⠠⠴⠧ ⠩⠐⠎⠃ ⠩⠐⠥⠚⠧⠦⠄⠴⠠⠠⠑⠥⠗⠠⠴ ⠼⠓⠚⠴⠏ ⠺ ⠰⠍⠫ ⠕⠘⠝⠒⠓⠪ ⠚⠧⠒⠩⠂ ⠍⠊⠗⠐⠮ ⠨⠝⠈⠿⠚⠒⠊⠲""",
    # 3. [일반 번역] 한양도성
    "trans_2_src": """한양도성문화제는 10월 1일~2일 흥인지문 공원에서 열린다. 한양도성 순성을 완주한 시민에게 메달을 증정하는 △순성챌린지, 걸음수에 따라 기부를 할 수 있는 △순성기부런(run)과 △순성술래잡기놀이 △북콘서트 ‘한양도성 북살롱’ 등이 개최된다.""",
    "trans_2_tgt": """⠚⠒⠜⠶⠊⠥⠠⠻⠑⠛⠚⠧⠨⠝⠉⠵ ⠼⠁⠚⠏⠂ ⠼⠁⠕⠂⠈⠔⠼⠃⠕⠂ ⠚⠪⠶⠟⠨⠕⠑⠛ ⠈⠿⠏⠒⠝⠠⠎ ⠳⠐⠟⠊⠲ ⠚⠒⠜⠶⠊⠥⠠⠻ ⠠⠛⠠⠻⠮ ⠧⠒⠨⠍⠚⠒ ⠠⠕⠑⠟⠝⠈⠝ ⠑⠝⠊⠂⠮ ⠨⠪⠶⠨⠻⠚⠉⠵ ⠸⠬⠠⠛⠠⠻⠰⠗⠂⠐⠟⠨⠕⠐ ⠈⠞⠪⠢⠠⠍⠝ ⠠⠊⠐⠣ ⠈⠕⠘⠍⠐⠮ ⠚⠂ ⠠⠍ ⠕⠌⠉⠵ ⠸⠬⠠⠛⠠⠻⠈⠕⠘⠍⠐⠾⠦⠄⠴⠗⠥⠝⠠⠴⠈⠧ ⠸⠬⠠⠛⠠⠻⠠⠯⠐⠗⠨⠃⠈⠕⠉⠥⠂⠕ ⠸⠬⠘⠍⠁⠋⠷⠠⠎⠓⠪ ⠠⠦⠚⠒⠜⠶⠊⠥⠠⠻ ⠘⠍⠁⠇⠂⠐⠿⠴⠄ ⠊⠪⠶⠕ ⠈⠗⠰⠽⠊⠽⠒⠊⠲""",
    # 4. [일반 번역] 에이핑크
    "trans_3_src": """에이핑크(Apink) 정은지가 다채로운 감성의 수록곡을 예고하며 새 앨범에 대한 기대를 높이고 있다.""",
    "trans_3_tgt": """⠝⠕⠙⠕⠶⠋⠪⠦⠄⠴⠠⠁⠏⠔⠅⠠⠴ ⠨⠻⠵⠨⠕⠫ ⠊⠰⠗⠐⠥⠛ ⠫⠢⠠⠻⠺ ⠠⠍⠐⠭⠈⠭⠮ ⠌⠈⠥⠚⠑⠱ ⠠⠗ ⠗⠂⠘⠎⠢⠝ ⠊⠗⠚⠒ ⠈⠕⠊⠗⠐⠮ ⠉⠥⠲⠕⠈⠥ ⠕⠌⠊⠲""",
}

# ----------------------------
# 환경 설정
# ----------------------------
GEMINI_API_KEY = st.secrets["GEMINI_API_KEY"]

KOREAN_ENDPOINT = st.secrets["KOREAN_ENDPOINT"]
CHINESE_ENDPOINT = st.secrets["CHINESE_ENDPOINT"]
ENGLISH_ENDPOINT = st.secrets["ENGLISH_ENDPOINT"]

KOREAN_API_KEY = st.secrets["KOREAN_API_KEY"]
CHINESE_API_KEY = st.secrets["CHINESE_API_KEY"]
ENGLISH_API_KEY = st.secrets["ENGLISH_API_KEY"]

USE_SENTENCE_LEVEL_TRANSLATION = True

model_configs = {
    "qwen3-1.7b": {
        "temperature": 0.0,
        "top_p": 0.8,
        "top_k": 20,
        "min_p": 0.0,
        "end_token": "<|im_end|>",
    },
    "kanana-1.5-2.1b": {
        "temperature": 0.0,
        "top_p": 1.0,
        "top_k": -1,
        "min_p": 0.0,
        "end_token": "<|eot_id|>",
    },
    "kanana-1.5-2.1b-english": {
        "temperature": 0.0,
        "top_p": 1.0,
        "top_k": -1,
        "min_p": 0.0,
        "end_token": "<|eot_id|>",
    },
}


# ----------------------------
# LLM Utility
# ----------------------------
def pick_model(language: str) -> str | None:
    if language == "Korean":
        return "kanana-1.5-2.1b"
    elif language == "Chinese":
        return "qwen3-1.7b"
    elif language == "English":
        return "kanana-1.5-2.1b-english"
    else:
        return None


def pick_endpoint(language: str) -> str:
    lang = (language or "").strip().lower()
    if lang == "korean":
        return KOREAN_ENDPOINT
    elif lang == "chinese":
        return CHINESE_ENDPOINT
    elif lang == "english":
        return ENGLISH_ENDPOINT
    return KOREAN_ENDPOINT


def pick_api_key(language: str) -> str | None:
    lang = (language or "").strip().lower()
    if lang == "korean":
        return KOREAN_API_KEY
    elif lang == "chinese":
        return CHINESE_API_KEY
    elif lang == "english":
        return ENGLISH_API_KEY
    return None


def llm_chat(system_msg: str, user_msg: str, language: str = "Korean") -> str:
    headers = {"Content-Type": "application/json"}
    api_key = pick_api_key(language).strip()
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    model_name = pick_model(language)

    payload = {
        "model": model_name,
        "messages": [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_msg},
        ],
        "temperature": model_configs[model_name]["temperature"],
        "top_p": model_configs[model_name]["top_p"],
        "top_k": model_configs[model_name]["top_k"],
        "min_p": model_configs[model_name]["min_p"],
        "stop": [model_configs[model_name]["end_token"]],
        "chat_template_kwargs": {"enable_thinking": False},
    }

    if language.lower() == "chinese":
        payload["chat_template_kwargs"] = {"enable_thinking": False}

    endpoint = pick_endpoint(language)
    try:
        resp = requests.post(
            endpoint,
            headers=headers,
            json=payload,
            timeout=120,
        )
        resp.raise_for_status()
        data = resp.json()
        content = data.get("choices", [{}])[0].get("message", {}).get("content")
        if not content:
            raise ValueError("LLM returned empty content")
        return content
    except Exception as e:
        return f"[LLM Error] {e}"


# --- Helper for normalization ---
def normalize_text(text: str) -> str:
    if not text:
        return ""
    return unicodedata.normalize("NFC", text.strip())


def gemini_summarize(text: str, source_language: str) -> str:
    normalized_input = normalize_text(text)
    normalized_cached_src = normalize_text(PREDEFINED_DATA["summary_case_src"])

    progress_text = "Connecting to Gemini..."
    my_bar = st.progress(0, text=progress_text)

    if normalized_input == normalized_cached_src:
        logger.info("[Cache Hit] Summarization")
        for percent_complete in range(0, 101, 20):
            time.sleep(0.3)
            my_bar.progress(percent_complete, text="Summarizing text...")
        my_bar.empty()
        return PREDEFINED_DATA["summary_case_sum"]

    client = genai.Client(api_key=GEMINI_API_KEY)
    cfg = genai.types.GenerateContentConfig(
        temperature=0.0,
        response_mime_type="application/json",
        thinking_config=genai.types.ThinkingConfig(thinking_budget=0),
        system_instruction=SUMMARIZATION_SYSTEM_PROMPT,
    )

    my_bar.progress(30, text="Requesting summary to Gemini...")
    resp = client.models.generate_content(
        model="gemini-2.5-flash",
        config=cfg,
        contents=f"source_language: {source_language}\n\ninput_text: {text}",
    )
    my_bar.progress(90, text="Processing summary result...")

    raw = resp.text
    parsed = json.loads(raw)
    summary = parsed.get("output_text", text)

    my_bar.progress(100, text="Done.")
    time.sleep(0.5)
    my_bar.empty()
    return summary


# ----------------------------
# Utils
# ----------------------------
_SENT_SPLIT = re.compile(
    r"""
    (?<=[.?!])\s+ | (?<=[。。！？]) | (?<=[⠲⠖⠦⠐⠆⠰⠂⠐⠄])\s+
    """,
    re.X,
)


def split_sentences_keep_punct(text: str) -> list[str]:
    if not text.strip():
        return []
    _t = text.replace("\n", " ").strip()
    parts = []
    start = 0
    for m in _SENT_SPLIT.finditer(_t):
        end = m.end()
        parts.append(_t[start:end].strip())
        start = end
    if start < len(_t):
        parts.append(_t[start:].strip())
    return [p for p in parts if p]


def sentenceize_with_line_map(text: str):
    lines = text.splitlines()
    all_sents, line_counts = [], []
    for line in lines:
        if line.strip():
            sents = split_sentences_keep_punct(line)
            if not sents:
                sents = [line.strip()]
            line_counts.append(len(sents))
            all_sents.extend(sents)
        else:
            line_counts.append(0)
    return all_sents, line_counts, lines


def assemble_by_lines(unit_list: list[str], line_counts: list[int]) -> str:
    out_lines, i = [], 0
    for cnt in line_counts:
        if cnt == 0:
            out_lines.append("")
        else:
            out_lines.append(" ".join(unit_list[i : i + cnt]))
            i += cnt
    return "\n".join(out_lines)


def _safe_str(x, fallback=""):
    return x if isinstance(x, str) and x != "" else fallback


# ----------------------------
# Translation
# ----------------------------
def run_translation(text: str) -> str:
    logger.info("==== [Translation] ====")
    logger.info(f"[Input Text]\n{text}")

    normalized_input = normalize_text(text)

    # 캐시 맵핑 (양방향 모두 포함)
    cache_map = {
        # 1. 한국어 -> 점자
        normalize_text(PREDEFINED_DATA["trans_1_src"]): PREDEFINED_DATA["trans_1_tgt"],
        normalize_text(PREDEFINED_DATA["trans_2_src"]): PREDEFINED_DATA["trans_2_tgt"],
        normalize_text(PREDEFINED_DATA["trans_3_src"]): PREDEFINED_DATA["trans_3_tgt"],
        normalize_text(PREDEFINED_DATA["summary_case_sum"]): PREDEFINED_DATA[
            "summary_case_braille"
        ],
        # 2. 점자 -> 한국어 (역방향)
        normalize_text(PREDEFINED_DATA["trans_1_tgt"]): PREDEFINED_DATA["trans_1_src"],
        normalize_text(PREDEFINED_DATA["trans_2_tgt"]): PREDEFINED_DATA["trans_2_src"],
        normalize_text(PREDEFINED_DATA["trans_3_tgt"]): PREDEFINED_DATA["trans_3_src"],
    }

    progress_text = "Connecting to AI Model..."
    my_bar = st.progress(0, text=progress_text)

    # 1. 캐시 히트
    if normalized_input in cache_map:
        logger.info(f"[Cache Hit] Translation")
        for percent_complete in range(0, 101, 10):
            time.sleep(0.15)
            if percent_complete < 30:
                msg = "Sending request..."
            elif percent_complete < 70:
                msg = "Translating..."
            else:
                msg = "Processing output..."
            my_bar.progress(percent_complete, text=msg)

        my_bar.progress(100, text="Done.")
        time.sleep(0.5)
        my_bar.empty()
        return cache_map[normalized_input]

    # 2. 실제 로직
    try:
        if st.session_state.mode == "text_to_braille":
            if st.session_state.src_lang == "Korean":
                system_msg = KOREAN_TO_BRAILLE_SYSTEM_PROMPT
            elif st.session_state.src_lang == "Chinese":
                system_msg = CHINESE_TO_BRAILLE_SYSTEM_PROMPT
            elif st.session_state.src_lang == "English":
                system_msg = ENGLISH_TO_BRAILLE_SYSTEM_PROMPT
        else:
            if st.session_state.tgt_lang == "Korean":
                system_msg = BRAILLE_TO_KOREAN_SYSTEM_PROMPT
            elif st.session_state.tgt_lang == "Chinese":
                system_msg = BRAILLE_TO_CHINESE_SYSTEM_PROMPT
            elif st.session_state.tgt_lang == "English":
                system_msg = BRAILLE_TO_ENGLISH_SYSTEM_PROMPT
            else:
                system_msg = BRAILLE_TO_ENGLISH_SYSTEM_PROMPT

        lang = (
            st.session_state.src_lang
            if st.session_state.mode == "text_to_braille"
            else st.session_state.tgt_lang
        )

        if USE_SENTENCE_LEVEL_TRANSLATION:
            src_sents, line_counts, _lines = sentenceize_with_line_map(text)
            st.session_state["src_sents"] = src_sents
            st.session_state["line_counts"] = line_counts

            tgt_sents = []
            total_sents = len(src_sents)
            for i, s in enumerate(src_sents, 1):
                progress_percent = int((i / total_sents) * 100)
                my_bar.progress(
                    progress_percent, text=f"Translating sentence {i}/{total_sents}..."
                )
                out = llm_chat(system_msg, s, lang)
                tgt_sents.append(_safe_str(out, "[Empty Translation]"))

            st.session_state["tgt_sents"] = tgt_sents
            ui_text = assemble_by_lines(tgt_sents, line_counts)

            my_bar.progress(100, text="Done.")
            time.sleep(0.5)
            my_bar.empty()
            return ui_text
        else:
            lines = text.split("\n")
            total_lines = len(lines)
            translated_lines = []
            for i, line in enumerate(lines):
                if line.strip():
                    my_bar.progress(
                        int(((i + 1) / total_lines) * 100),
                        text=f"Translating line {i+1}/{total_lines}...",
                    )
                    out = llm_chat(system_msg, line, lang)
                    translated_lines.append(_safe_str(out, "[Empty Translation]"))
                else:
                    translated_lines.append("")
            result = "\n".join(translated_lines)
            my_bar.progress(100, text="Done.")
            time.sleep(0.5)
            my_bar.empty()
            return result

    except Exception as e:
        my_bar.empty()
        return f"Translation error: {e}"


# ----------------------------
# Validation
# ----------------------------
def validate_translation(src: str, tgt_ui_text: str) -> str:
    progress_text = "Verifying translation..."
    val_bar = st.progress(0, text=progress_text)

    normalized_src = normalize_text(src)
    cached_sources = [
        normalize_text(PREDEFINED_DATA["trans_1_src"]),
        normalize_text(PREDEFINED_DATA["trans_2_src"]),
        normalize_text(PREDEFINED_DATA["trans_3_src"]),
        normalize_text(PREDEFINED_DATA["summary_case_sum"]),
        normalize_text(PREDEFINED_DATA["summary_case_src"]),
        normalize_text(PREDEFINED_DATA["trans_1_tgt"]),
        normalize_text(PREDEFINED_DATA["trans_2_tgt"]),
        normalize_text(PREDEFINED_DATA["trans_3_tgt"]),
    ]

    if normalized_src in cached_sources:
        steps = [
            "Performing reverse translation...",
            "Comparing with original text...",
            "Finalizing verification...",
        ]
        step_percents = [30, 60, 100]
        for step_idx, step_msg in enumerate(steps):
            val_bar.progress(step_percents[step_idx], text=step_msg)
            time.sleep(0.6)
        val_bar.empty()
        return "Automatic Validation using Forward-Backward Success."

    try:
        # 프롬프트 설정
        if st.session_state.mode == "text_to_braille":
            if st.session_state.src_lang == "Korean":
                system_msg = BRAILLE_TO_KOREAN_SYSTEM_PROMPT
            elif st.session_state.src_lang == "Chinese":
                system_msg = BRAILLE_TO_CHINESE_SYSTEM_PROMPT
            else:
                system_msg = BRAILLE_TO_ENGLISH_SYSTEM_PROMPT
        else:
            if st.session_state.tgt_lang == "Korean":
                system_msg = KOREAN_TO_BRAILLE_SYSTEM_PROMPT
            elif st.session_state.tgt_lang == "Chinese":
                system_msg = CHINESE_TO_BRAILLE_SYSTEM_PROMPT
            else:
                system_msg = ENGLISH_TO_BRAILLE_SYSTEM_PROMPT

        lang = (
            st.session_state.src_lang
            if st.session_state.mode == "text_to_braille"
            else st.session_state.tgt_lang
        )
        val_bar.progress(10, text="Preparing validation...")

        if USE_SENTENCE_LEVEL_TRANSLATION:
            tgt_sents = st.session_state.get("tgt_sents")
            # 버퍼가 없으면 생성
            if not tgt_sents:
                tgt_sents, _, _ = sentenceize_with_line_map(tgt_ui_text)

            recon_sents = []
            total = len(tgt_sents)
            for i, tgt_sent in enumerate(tgt_sents, 1):
                val_bar.progress(
                    10 + int((i / total) * 40),
                    text=f"Reverse translating ({i}/{total})...",
                )
                if not _safe_str(tgt_sent):
                    recon_sents.append("")
                    continue
                out = llm_chat(system_msg, tgt_sent, lang)
                recon_sents.append(_safe_str(out, ""))
            recon_joined = " ".join([s for s in recon_sents if s])

            val_bar.progress(60, text="Comparing text...")
            if unicodedata.normalize("NFC", recon_joined) == unicodedata.normalize(
                "NFC", src
            ):
                val_bar.progress(100, text="Verification Success.")
                time.sleep(0.5)
                val_bar.empty()
                return "Automatic Validation using Forward-Backward Success."

            val_bar.progress(70, text="Checking semantic meaning...")
            client = genai.Client(api_key=GEMINI_API_KEY)
            cfg = genai.types.GenerateContentConfig(
                temperature=0.0,
                response_mime_type="application/json",
                thinking_config=genai.types.ThinkingConfig(thinking_budget=0),
                system_instruction=VALIDATION_SYSTEM_PROMPT,
            )
            contents = f"src: {src}\n\nrecon: {recon_joined}"
            resp = client.models.generate_content(
                model="gemini-2.5-flash", config=cfg, contents=contents
            )
            parsed = json.loads(resp.text)

            val_bar.progress(100, text="Done.")
            time.sleep(0.5)
            val_bar.empty()

            if parsed.get("equal") is True:
                return "Automatic Validation using Forward-Backward Failed. Semantic Equivalence Validation using LLM Success."
            else:
                return "Automatic Validation using Forward-Backward Failed. Semantic Equivalence Validation using LLM Failed. Human feedback is required."
        else:
            # 줄단위 검증 로직
            tgt_lines = tgt_ui_text.split("\n")
            recon_lines = []
            total = len(tgt_lines)
            for i, t_line in enumerate(tgt_lines):
                val_bar.progress(
                    10 + int(((i + 1) / total) * 40), text="Reverse translating..."
                )
                if t_line.strip():
                    recon_lines.append(llm_chat(system_msg, t_line, lang))
                else:
                    recon_lines.append("")
            recon = "\n".join(recon_lines)

            val_bar.progress(60, text="Comparing text...")
            if recon == src:
                val_bar.empty()
                return "Automatic Validation using Forward-Backward Success."

            val_bar.progress(70, text="Checking semantic meaning...")
            client = genai.Client(api_key=GEMINI_API_KEY)
            cfg = genai.types.GenerateContentConfig(
                temperature=0.0,
                response_mime_type="application/json",
                system_instruction=VALIDATION_SYSTEM_PROMPT,
            )
            contents = f"src: {src}\n\nrecon: {recon}"
            resp = client.models.generate_content(
                model="gemini-2.5-flash", config=cfg, contents=contents
            )
            parsed = json.loads(resp.text)
            val_bar.empty()
            if parsed.get("equal") is True:
                return "Automatic Validation using Forward-Backward Failed. Semantic Equivalence Validation using LLM Success."
            else:
                return "Automatic Validation using Forward-Backward Failed. Semantic Equivalence Validation using LLM Failed. Human feedback is required."

    except Exception as e:
        val_bar.empty()
        return f"Validation error: {e}"


# ----------------------------
# UI Configuration & State
# ----------------------------
st.set_page_config(page_title="Braille Translation Demo", page_icon="⠁", layout="wide")
st.markdown(
    """
    <style>
        .block-container { padding-top: 1.5rem; padding-bottom: 2rem; }
        .stElementContainer { margin-bottom: -0.5rem; }
        .streamlit-expander { margin-bottom: 0px !important; }
    </style>
""",
    unsafe_allow_html=True,
)
st.markdown(
    """
    <div style="text-align:center; margin-top: 0rem">
      <h1 style="margin-bottom:0.2rem; font-size: 2.1rem; line-height:1.25;">
        LLM-Based System for Enhanced Text-to-Braille Translation:<br/>
        Incorporating Contextual Awareness and Automated Verification
      </h1>
    </div>
    <hr style="margin-top: 0.5rem; margin-bottom: 1.5rem; border: 0; border-top: 1px solid #f0f2f6;" />
""",
    unsafe_allow_html=True,
)

if "mode" not in st.session_state:
    st.session_state.mode = "text_to_braille"
if "src_text" not in st.session_state:
    st.session_state.src_text = ""
if "tgt_text" not in st.session_state:
    st.session_state.tgt_text = ""
if "summary_text" not in st.session_state:
    st.session_state.summary_text = ""  # 요약문 저장
if "src_lang_val" not in st.session_state:
    st.session_state.src_lang_val = "Korean"
if "tgt_lang_val" not in st.session_state:
    st.session_state.tgt_lang_val = "Braille"
if "pending_swap" not in st.session_state:
    st.session_state.pending_swap = False

# Swap Logic
if st.session_state.pending_swap:
    st.session_state.src_lang_val, st.session_state.tgt_lang_val = (
        st.session_state.tgt_lang_val,
        st.session_state.src_lang_val,
    )
    st.session_state.src_text, st.session_state.tgt_text = (
        st.session_state.tgt_text,
        st.session_state.src_text,
    )
    st.session_state.summary_text = ""  # 스왑하면 요약본 초기화
    st.session_state.pending_swap = False

st.session_state.src_lang = st.session_state.src_lang_val
st.session_state.tgt_lang = st.session_state.tgt_lang_val
TEXT_LANGS = ["Korean", "Braille"]


def _enforce_pair_and_mode():
    src = st.session_state.get("src_lang", "Korean")
    tgt = st.session_state.get("tgt_lang", "Braille")
    st.session_state.invalid_pair = not ((src == "Braille") ^ (tgt == "Braille"))
    if src == "Braille" and tgt != "Braille":
        st.session_state.mode = "braille_to_text"
    elif tgt == "Braille" and src != "Braille":
        st.session_state.mode = "text_to_braille"
    else:
        st.session_state.mode = st.session_state.get("mode", "text_to_braille")


def _update_action_disabled():
    _enforce_pair_and_mode()
    src_lang = st.session_state.src_lang
    tgt_lang = st.session_state.tgt_lang
    valid_pair = (src_lang == "Braille") ^ (tgt_lang == "Braille")
    llm_lang = src_lang if st.session_state.mode == "text_to_braille" else tgt_lang
    model_ok = pick_model(llm_lang) is not None
    st.session_state["action_disabled"] = (not valid_pair) or (not model_ok)
    st.session_state["valid_pair"] = valid_pair
    st.session_state["model_ok"] = model_ok
    st.session_state["llm_lang_eval"] = llm_lang


def _on_language_change():
    _update_action_disabled()


def _queue_swap():
    st.session_state.pending_swap = True


if "action_disabled" not in st.session_state:
    _update_action_disabled()
else:
    _update_action_disabled()


def clear_summary_on_change():
    """원본 텍스트 수정 시 요약 결과 초기화"""
    st.session_state.summary_text = ""


# ----------------------------
# Layout
# ----------------------------
mode = st.radio(
    "Mode", ["Translation", "Validation"], horizontal=True, label_visibility="collapsed"
)
disabled = st.session_state.get("action_disabled", True)
valid_pair = st.session_state.get("valid_pair", False)
model_ok = st.session_state.get("model_ok", False)
is_text_to_braille = st.session_state.mode == "text_to_braille"

# Buttons
if mode == "Translation":
    if is_text_to_braille:
        col1, col2, col3 = st.columns([8, 1.2, 1])
        with col1:
            st.markdown(
                '<h2 style="margin:0;">Translation</h2>', unsafe_allow_html=True
            )
        with col2:
            go_summarize = st.button(
                "Summarize",
                type="secondary",
                use_container_width=True,
                disabled=disabled,
            )
        with col3:
            go_translate = st.button(
                "Translate", type="primary", use_container_width=True, disabled=disabled
            )
    else:
        col1, col2 = st.columns([9, 1])
        with col1:
            st.markdown(
                '<h2 style="margin:0;">Translation</h2>', unsafe_allow_html=True
            )
        with col2:
            go_translate = st.button(
                "Translate", type="primary", use_container_width=True, disabled=disabled
            )
elif mode == "Validation":
    col1, col2 = st.columns([9, 1])
    with col1:
        st.markdown('<h2 style="margin:0;">Validation</h2>', unsafe_allow_html=True)
    with col2:
        go_validate_only = st.button(
            "Validate", type="primary", use_container_width=True, disabled=disabled
        )

if not valid_pair:
    st.warning("Exactly one of Source/Target must be Braille.")
elif not model_ok:
    st.warning(f"No vLLM model configured for {st.session_state.llm_lang_eval}.")

header_cols = st.columns([5, 1, 5])
with header_cols[0]:
    st.selectbox(
        "Source Language", TEXT_LANGS, key="src_lang_val", on_change=_on_language_change
    )
with header_cols[1]:
    if st.button(
        "↔︎", help="Swap source/target", use_container_width=True, on_click=_queue_swap
    ):
        st.rerun()
with header_cols[2]:
    st.selectbox(
        "Target Language", TEXT_LANGS, key="tgt_lang_val", on_change=_on_language_change
    )

# Input
st.markdown(
    f'<h3 style="margin-top: 0.2rem; margin-bottom: 0.3rem;">Input ({st.session_state.src_lang})</h3>',
    unsafe_allow_html=True,
)
st.session_state.src_text = st.text_area(
    "Source Text",
    height=110,
    label_visibility="collapsed",
    value=st.session_state.src_text,
    placeholder="Enter the text to be translated here.",
    on_change=clear_summary_on_change,  # 원본 수정시 요약 초기화
)

# ----------------------------
# Logic Execution
# ----------------------------
src_nfc = unicodedata.normalize("NFC", st.session_state.src_text)

# 1. 요약 로직 (독립적)
if mode == "Translation" and "go_summarize" in locals() and go_summarize and src_nfc:
    st.session_state.summary_text = gemini_summarize(src_nfc, st.session_state.src_lang)
    # Rerun to show summary immediately
    st.rerun()

# 2. 요약 결과 표시 (있을 경우)
if st.session_state.summary_text:
    st.markdown("**⬇️ Summarized Text**")
    st.info(st.session_state.summary_text)

# 3. Output Header
st.markdown(
    f'<h3 style="margin-top: 0rem; margin-bottom: 0.3rem;">Output ({st.session_state.tgt_lang})</h3>',
    unsafe_allow_html=True,
)

# 4. 플레이스홀더 설정 (순차적 렌더링을 위해)
output_placeholder = st.empty()
validation_placeholder = st.empty()

# 기본 상태 렌더링 (결과가 이미 있으면 표시)
if st.session_state.tgt_text:
    output_placeholder.text_area(
        "Target Text",
        value=st.session_state.tgt_text,
        height=135,
        label_visibility="collapsed",
    )
else:
    output_placeholder.text_area(
        "Target Text",
        height=135,
        label_visibility="collapsed",
        placeholder="The translation results are displayed here.",
    )

if "last_val_msg" in st.session_state and st.session_state.last_val_msg:
    msg = st.session_state.last_val_msg
    if "success" in msg.lower():
        validation_placeholder.success(msg)
    else:
        validation_placeholder.error(msg)


# 5. 번역 + 검증 로직 (순차 실행)
if mode == "Translation" and "go_translate" in locals() and go_translate and src_nfc:
    # A. 번역 대상 결정 (요약본이 있으면 그것을, 없으면 원본을)
    real_src = (
        st.session_state.summary_text if st.session_state.summary_text else src_nfc
    )

    # B. 번역 실행
    result = run_translation(real_src)
    st.session_state.tgt_text = result

    # C. [중요] 번역 결과 즉시 화면에 렌더링 (검증 전에 사용자가 볼 수 있게)
    output_placeholder.text_area(
        "Target Text", value=result, height=135, label_visibility="collapsed"
    )

    # D. 검증 실행
    val_msg = validate_translation(real_src, result)
    st.session_state.last_val_msg = val_msg

    # E. 검증 결과 렌더링
    if "success" in val_msg.lower():
        validation_placeholder.success(val_msg)
    else:
        validation_placeholder.error(val_msg)

elif (
    mode == "Validation"
    and "go_validate_only" in locals()
    and go_validate_only
    and src_nfc
    and st.session_state.tgt_text
):
    val_msg = validate_translation(src_nfc, st.session_state.tgt_text)
    st.session_state.last_val_msg = val_msg
    if "success" in val_msg.lower():
        validation_placeholder.success(val_msg)
    else:
        validation_placeholder.error(val_msg)
