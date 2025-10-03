import re
import sys
import os
import json
import textwrap
import unicodedata
from loguru import logger
import google.genai as genai

import requests
import streamlit as st

# 프롬프트 로드
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
# 환경 설정
# ----------------------------
GEMINI_API_KEY = st.secrets["GEMINI_API_KEY"]

KOREAN_ENDPOINT = st.secrets["KOREAN_ENDPOINT"]
CHINESE_ENDPOINT = st.secrets["CHINESE_ENDPOINT"]
ENGLISH_ENDPOINT = st.secrets.get("ENGLISH_ENDPOINT")

KOREAN_API_KEY = st.secrets["KOREAN_API_KEY"]
CHINESE_API_KEY = st.secrets["CHINESE_API_KEY"]
ENGLISH_API_KEY = st.secrets.get("ENGLISH_API_KEY")

USE_SENTENCE_LEVEL_TRANSLATION = (
    True  # True면 모든 번역/검증을 문장 단위로 처리, False면 줄 단위로 처리
)

model_configs = {
    "qwen3-1.7b": {
        "dtype": "bfloat16",
        "end_token": "<|im_end|>",
        "assistant_token": "<|im_start|>assistant\n<think>\n\n</think>\n\n",
    },
    "kanana-1.5-2.1b": {
        "dtype": "bfloat16",
        "end_token": "<|eot_id|>",
        "assistant_token": "<|start_header_id|>assistant<|end_header_id|>\n\n",
    },
}


# ----------------------------
# LLM Utility
# ----------------------------
def pick_model(language: str) -> str | None:
    if language == "Korean":
        return KOREAN_API_KEY
    elif language == "Chinese":
        return CHINESE_API_KEY
    elif language == "English":
        return ENGLISH_API_KEY
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

    # 언어별 end_token 설정
    if language.lower() == "chinese":
        stop_tokens = ["<|im_end|>"]  # Qwen3
        top_p = 0.8
        top_k = 20
        min_p = 0.0
        temperature = 0.0
        payload = {
            "model": model_name,
            "messages": [
                {"role": "system", "content": system_msg},
                {"role": "user", "content": user_msg},
            ],
            "temperature": temperature,
            "top_p": top_p,
            "top_k": top_k,
            "min_p": min_p,
            "stop": stop_tokens,
            "chat_template_kwargs": {"enable_thinking": False},
        }
        logger.info(
            "=== DEBUG PAYLOAD ===", json.dumps(payload, ensure_ascii=False, indent=2)
        )

    else:
        stop_tokens = ["<|eot_id|>"]  # Kanana
        top_p = 1.0
        top_k = -1
        min_p = 0.0
        temperature = 0.0
        payload = {
            "model": model_name,
            "messages": [
                {"role": "system", "content": system_msg},
                {"role": "user", "content": user_msg},
            ],
            "temperature": temperature,
            "top_p": top_p,
            "top_k": top_k,
            "min_p": min_p,
            "stop": stop_tokens,
        }

    endpoint = pick_endpoint(language)
    logger.info("=== DEBUG ENDPOINT ===", endpoint)
    logger.info("=== DEBUG HEADERS ===", headers)
    logger.info(
        "=== DEBUG PAYLOAD ===", json.dumps(payload, ensure_ascii=False, indent=2)
    )
    try:
        resp = requests.post(
            # endpoint, headers=headers, data=json.dumps(payload), timeout=120
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
        # 절대 None을 리턴하지 않음
        return f"[LLM Error] {e}"


def gemini_summarize(text: str, source_language: str) -> str:
    client = genai.Client(api_key=GEMINI_API_KEY)
    cfg = genai.types.GenerateContentConfig(
        temperature=0.0,
        response_mime_type="application/json",
        thinking_config=genai.types.ThinkingConfig(thinking_budget=0),
        system_instruction=SUMMARIZATION_SYSTEM_PROMPT,
    )
    logger.info("==== [Summarization] ====")
    logger.info(f"[Input Text]\n{text}")
    logger.info(f"[Source Language] {source_language}")
    resp = client.models.generate_content(
        model="gemini-2.5-flash",
        config=cfg,
        contents=f"source_language: {source_language}\n\ninput_text: {text}",
    )
    raw = resp.text
    logger.info(f"[Output Raw]\n{raw}")
    parsed = json.loads(raw)
    summary = parsed.get("output_text", text)
    logger.info(f"[Output Parsed]\n{summary}")
    logger.info("=========================")
    return summary


# ----------------------------
# Utils
# ----------------------------
# --- 다국어 문장 분리기 (종결부호 + 한국어 종결어미 간단 지원) ---
# 종결부호/종결어미 + 공백을 기준으로 split
_SENT_SPLIT = re.compile(
    r"""
    (?<=[.?!])\s+             # 영어 마침표, 느낌표, 물음표
    | (?<=[。。！？])         # 중국어 마침표, 느낌표, 물음표
    | (?<=[⠲⠖⠦⠐⠆⠰⠂⠐⠄])\s+   # 점자 마침표, 느낌표, 물음표
    """,
    re.X,
)


def split_sentences_keep_punct(text: str) -> list[str]:
    """종결부호 뒤에 공백이 있을 때만 문장 분리"""
    if not text.strip():
        return []

    # 줄바꿈은 공백으로 치환 (line map은 별도 관리)
    _t = text.replace("\n", " ").strip()

    parts = []
    start = 0
    for m in _SENT_SPLIT.finditer(_t):
        end = m.end()
        parts.append(_t[start:end].strip())
        start = end
    if start < len(_t):  # 마지막 문장
        parts.append(_t[start:].strip())

    return [p for p in parts if p]


def sentenceize_with_line_map(text: str):
    """
    원문 줄을 보존하기 위해:
      - 각 줄을 문장으로 쪼개서 all_sents에 순서대로 넣고
      - 각 줄이 가진 문장 수를 line_counts에 기록
      - 원래 줄 문자열 배열 lines도 함께 반환
    """
    lines = text.splitlines()
    all_sents, line_counts = [], []
    for line in lines:
        if line.strip():
            sents = split_sentences_keep_punct(line)
            if not sents:  # 한 줄 전체가 한 문장으로 간주
                sents = [line.strip()]
            line_counts.append(len(sents))
            all_sents.extend(sents)
        else:
            line_counts.append(0)
    return all_sents, line_counts, lines


def assemble_by_lines(unit_list: list[str], line_counts: list[int]) -> str:
    """문장(or 역문장) 리스트를 line_counts를 기준으로 다시 줄 단위로 합쳐서 반환"""
    out_lines, i = [], 0
    for cnt in line_counts:
        if cnt == 0:
            out_lines.append("")
        else:
            out_lines.append(" ".join(unit_list[i : i + cnt]))
            i += cnt
    return "\n".join(out_lines)


# --- LLM 에러/None 방지 가드 ---
def _safe_str(x, fallback=""):
    return x if isinstance(x, str) and x != "" else fallback


# ----------------------------
# Translation
# ----------------------------
def run_translation(text: str) -> str:
    logger.info("==== [Translation] ====")
    logger.info(f"[Input Text]\n{text}")
    try:
        if st.session_state.mode == "text_to_braille":
            if st.session_state.src_lang == "Korean":
                system_msg = KOREAN_TO_BRAILLE_SYSTEM_PROMPT
            elif st.session_state.src_lang == "Chinese":
                system_msg = CHINESE_TO_BRAILLE_SYSTEM_PROMPT
            elif st.session_state.src_lang == "English":
                system_msg = ENGLISH_TO_BRAILLE_SYSTEM_PROMPT
        else:  # braille_to_text
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

        translated_lines = []

        if USE_SENTENCE_LEVEL_TRANSLATION:
            # 1) 문장화 + 줄 맵핑
            src_sents, line_counts, _lines = sentenceize_with_line_map(text)
            st.session_state["src_sents"] = src_sents
            st.session_state["line_counts"] = line_counts

            # 2) 전체 문장 번역 (문장별 즉시 검증 금지)
            tgt_sents = []
            for i, s in enumerate(src_sents, 1):
                out = llm_chat(system_msg, s, lang)
                logger.info(f"[Sentence {i} IN]\n{s}")
                logger.info(f"[Sentence {i} OUT]\n{out}")
                tgt_sents.append(_safe_str(out, "[Empty Translation]"))
            st.session_state["tgt_sents"] = tgt_sents

            # 3) UI용: 줄 구조로 복원
            ui_text = assemble_by_lines(tgt_sents, line_counts)
            logger.info(f"[Final Output]\n{ui_text}")
            logger.info("=======================")
            return ui_text

        # --- 줄 단위 모드 (기존) ---
        translated_lines = []
        for line in text.split("\n"):
            if line.strip():
                out = llm_chat(system_msg, line, lang)
                logger.info(f"[Line Input]\n{line}")
                translated_lines.append(_safe_str(out, "[Empty Translation]"))
            else:
                translated_lines.append("")

        result = "\n".join(translated_lines)
        logger.info(f"[Final Output]\n{result}")
        logger.info("=======================")
        return result

    except Exception as e:
        return f"Translation error: {e}"


# ----------------------------
# Validation
# ----------------------------
def validate_translation(src: str, tgt_ui_text: str) -> str:
    try:
        # --- 역방향 프롬프트 선택 ---
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

        # ----------------------------
        # 문장 단위 처리 (USE_SENTENCE_LEVEL_TRANSLATION=True)
        # ----------------------------
        if USE_SENTENCE_LEVEL_TRANSLATION:
            # 이미 번역 단계에서 저장된 buffer를 불러옴
            tgt_sents = st.session_state.get("tgt_sents")
            src_sents = st.session_state.get("src_sents")

            # 만약 buffer가 없으면 지금 분리해서 생성
            if not tgt_sents:
                tgt_sents, _, _ = sentenceize_with_line_map(tgt_ui_text)
            if not src_sents:
                src_sents, _, _ = sentenceize_with_line_map(src)

            recon_sents = []
            for i, tgt_sent in enumerate(tgt_sents, 1):
                if not _safe_str(tgt_sent):
                    recon_sents.append("")
                    continue
                out = llm_chat(system_msg, tgt_sent, lang)
                logger.info(f"[FB {i} IN]\n{tgt_sent}")
                logger.info(f"[FB {i} OUT]\n{out}")
                recon_sents.append(_safe_str(out, ""))

            recon_joined = " ".join([s for s in recon_sents if s])

            logger.info("==== [Validation: Forward-Backward] ====")
            logger.info(f"[Input Src]\n{src}")
            logger.info(f"[Reconstructed Joined]\n{recon_joined}")
            logger.info("========================================")

            if unicodedata.normalize("NFC", recon_joined) == unicodedata.normalize(
                "NFC", src
            ):
                return "Automatic Validation using Forward-Backward Success."

            # 의미 동등성 검사
            client = genai.Client(api_key=GEMINI_API_KEY)
            cfg = genai.types.GenerateContentConfig(
                temperature=0.0,
                response_mime_type="application/json",
                thinking_config=genai.types.ThinkingConfig(thinking_budget=0),
                system_instruction=VALIDATION_SYSTEM_PROMPT,
            )
            contents = f"src: {src}\n\nrecon: {recon_joined}"
            logger.info("==== [Semantic Equivalence] ====")
            logger.info(f"[Input Src]\n{contents}")
            resp = client.models.generate_content(
                model="gemini-2.5-flash", config=cfg, contents=contents
            )
            parsed = json.loads(resp.text)
            logger.info(f"[Output Raw]\n{resp.text}")
            logger.info("===============================")

            if parsed.get("equal") is True:
                return "Automatic Validation using Forward-Backward Failed. Semantic Equivalence Validation using LLM Success."
            else:
                st.markdown(
                    "<style>.stTextArea textarea {border: 2px solid red !important;}</style>",
                    unsafe_allow_html=True,
                )
                return "Automatic Validation using Forward-Backward Failed. Semantic Equivalence Validation using LLM Failed. Human feedback is required."

        # ----------------------------
        # 줄 단위 처리 (USE_SENTENCE_LEVEL_TRANSLATION=False)
        # ----------------------------
        else:
            tgt_lines = tgt_ui_text.split("\n")
            recon_lines = []
            for t_line in tgt_lines:
                if t_line.strip():
                    recon_lines.append(llm_chat(system_msg, t_line, lang))
                else:
                    recon_lines.append("")
            recon = "\n".join(recon_lines)

            logger.info("==== [Validation: Forward-Backward] ====")
            logger.info(f"[Input Src]\n{src}")
            logger.info(f"[Input Tgt]\n{tgt_ui_text}")
            logger.info(f"[Reconstructed Output]\n{recon}")
            logger.info("========================================")

            if recon == src:
                return "Automatic Validation using Forward-Backward Success."

            # 의미 동등성 검사
            client = genai.Client(api_key=GEMINI_API_KEY)
            cfg = genai.types.GenerateContentConfig(
                temperature=0.0,
                response_mime_type="application/json",
                thinking_config=genai.types.ThinkingConfig(thinking_budget=0),
                system_instruction=VALIDATION_SYSTEM_PROMPT,
            )
            contents = f"src: {src}\n\nrecon: {recon}"
            resp = client.models.generate_content(
                model="gemini-2.5-flash", config=cfg, contents=contents
            )
            parsed = json.loads(resp.text)

            if parsed.get("equal") is True:
                return "Automatic Validation using Forward-Backward Failed. Semantic Equivalence Validation using LLM Success."
            else:
                st.markdown(
                    "<style>.stTextArea textarea {border: 2px solid red !important;}</style>",
                    unsafe_allow_html=True,
                )
                return "Automatic Validation using Forward-Backward Failed. Semantic Equivalence Validation using LLM Failed. Human feedback is required."

    except Exception as e:
        return f"Validation error: {e}"


# ----------------------------
# UI configuration
# ----------------------------
st.set_page_config(page_title="Braille Translation Demo", page_icon="⠁", layout="wide")

st.markdown(
    """
    <div style="text-align:center; margin-top: 1.2rem">
      <h1 style="margin-bottom:0.2rem; font-size: 2.1rem; line-height:1.25;">
        LLM-Based System for Enhanced Text-to-Braille Translation:<br/>
        Incorporating Contextual Awareness and Automated Verification
      </h1>
    </div>
    """,
    unsafe_allow_html=True,
)

with st.expander("Abstract", expanded=True):
    st.write(
        textwrap.dedent(
            """
251003 Since its invention, Braille—a tactile writing system based on raised-dot patterns—has been an essential tool enabling visually impaired individuals to access literal information and engage with the world. In modern applications, Braille can be digitally encoded using rule-based mapping from printed characters to Braille cells, as specified in language-specific contraction dictionaries. While such rule-based translation often achieves satisfactory accuracy, challenges remain, particularly in handling ambiguities arising from Grade-2 contractions that require context-sensitive rules, syllable decomposition for particular languages, and multilingual text translation. Moreover, current systems typically apply the same translation rules mechanically, regardless of the input text's genre or characteristics. This approach can be suboptimal, as meaning is often more effective conveyed through concise, simplified expressions that reduce the reading burden for visually impaired users. A more critical limitation lies in the post-translation stage: outputs must still be inspected by qualified experts—who are in short supply—, leading to high publication costs and significant delays. To address these issues, we leverage recent advances in large language model (LLM). Specifically, we propose an LLM-based text-to-Braille translation scheme capable of 1) achieving enhanced translation accuracy; 2) generating context-aware expressions that convey meaning in a more accessible and contextually appropriate manner; and 3) automatically verifying translation outputs through forward-backward translation. The proposed scheme is evaluated across diverse documents written in both Korean and Chinese. Compared with conventional rule-based approaches, our system improves translation accuracy (chrF Score) by up to 99.9\%. We further demonstrate that our automated verification mechanism can substantially reduce both time and budget in the publication of Braille books. Additionally, the system adaptively adjusts the level of summarization based on the input, condensing content and simplifying phrasing while preserving semantic fidelity. This not only reduces reduces reading efforts for end users but also lowers publication costs by minimizing the physical size of Braille volumes.
            """
        )
    )

st.divider()

# ----------------------------
# Session state
# ----------------------------
if "mode" not in st.session_state:
    st.session_state.mode = "text_to_braille"
if "src_text" not in st.session_state:
    st.session_state.src_text = ""
if "tgt_text" not in st.session_state:
    st.session_state.tgt_text = ""
if "summary_text" not in st.session_state:
    st.session_state.summary_text = ""

# UI용 언어 키 초기화 (selectbox용)
if "src_lang_val" not in st.session_state:
    st.session_state.src_lang_val = "Korean"
if "tgt_lang_val" not in st.session_state:
    st.session_state.tgt_lang_val = "Braille"

# 스왑 예약 플래그
if "pending_swap" not in st.session_state:
    st.session_state.pending_swap = False

# --- (중요) 스왑 예약 처리: 위젯 렌더 전에 값을 바꿔야 에러가 안 남 ---
if st.session_state.pending_swap:
    st.session_state.src_lang_val, st.session_state.tgt_lang_val = (
        st.session_state.tgt_lang_val,
        st.session_state.src_lang_val,
    )
    st.session_state.src_text, st.session_state.tgt_text = (
        st.session_state.tgt_text,
        st.session_state.src_text,
    )
    st.session_state.summary_text = ""
    st.session_state.pending_swap = False

# 내부 별칭 동기화 (로직은 이 값을 사용)
st.session_state.src_lang = st.session_state.src_lang_val
st.session_state.tgt_lang = st.session_state.tgt_lang_val

TEXT_LANGS = ["English", "Chinese", "Korean", "Braille"]


# ---- Pair/mode enforcement & helpers ----
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
    st.session_state.src_lang = st.session_state.get("src_lang_val", "Korean")
    st.session_state.tgt_lang = st.session_state.get("tgt_lang_val", "Braille")
    _update_action_disabled()


def _queue_swap():
    st.session_state.pending_swap = True


# 첫 로드 시 버튼 상태 계산
if "action_disabled" not in st.session_state:
    _update_action_disabled()
else:
    # 스왑 처리로 값이 변했을 수 있으니 갱신
    _update_action_disabled()

# ----------------------------
# Translation UI
# ----------------------------
mode = st.radio("Mode", ["Translation", "Validation"], horizontal=True)

# 버튼 상태
disabled = st.session_state.get("action_disabled", True)
valid_pair = st.session_state.get("valid_pair", False)
model_ok = st.session_state.get("model_ok", False)
llm_lang = st.session_state.get("llm_lang_eval", "")

# ---- 모드별 버튼 ----
if mode == "Translation":
    col1, col2, col3 = st.columns([8, 2, 1])
    with col1:
        st.markdown("## Translation")
    with col2:
        go_sum_translate = st.button(
            "Summarize + Translate",
            type="secondary",
            use_container_width=True,
            disabled=disabled,
        )
    with col3:
        go_translate = st.button(
            "Translate", type="primary", use_container_width=True, disabled=disabled
        )
elif mode == "Validation":
    col1, col2 = st.columns([9, 1])
    with col1:
        st.markdown("## Validation")
    with col2:
        go_validate_only = st.button(
            "Validate", type="primary", use_container_width=True, disabled=disabled
        )

# 안내 메시지
if not valid_pair:
    st.warning("Exactly one of Source/Target must be Braille.")
elif not model_ok:
    st.warning(f"No vLLM model configured for {llm_lang}.")

# 언어 선택과 스왑 버튼
header_cols = st.columns([5, 1, 5])
with header_cols[0]:
    st.selectbox(
        "Source Language",
        TEXT_LANGS,
        key="src_lang_val",
        # index=TEXT_LANGS.index(st.session_state.src_lang_val),
        on_change=_on_language_change,
    )
with header_cols[1]:
    # 스왑은 '예약'만 하고 즉시 rerun → 다음 런 초반에 값이 바뀜 (위젯 전이라 안전)
    if st.button(
        "↔︎", help="Swap source/target", use_container_width=True, on_click=_queue_swap
    ):
        st.rerun()
with header_cols[2]:
    st.selectbox(
        "Target Language",
        TEXT_LANGS,
        key="tgt_lang_val",
        # index=TEXT_LANGS.index(st.session_state.tgt_lang_val),
        on_change=_on_language_change,
    )

# Input
st.markdown(f"### Input ({st.session_state.src_lang})")
st.session_state.src_text = st.text_area(
    "Source Text",
    height=220,
    label_visibility="collapsed",
    value=st.session_state.src_text,
    placeholder="Enter the text to be translated here.",
)

# ----------------------------
# 실행 로직
# ----------------------------
src_nfc = unicodedata.normalize("NFC", st.session_state.src_text)
result = ""

if mode == "Translation":
    if "go_translate" in locals() and go_translate and src_nfc:
        result = run_translation(src_nfc)
    elif "go_sum_translate" in locals() and go_sum_translate and src_nfc:
        summary = gemini_summarize(src_nfc, st.session_state.src_lang)
        st.session_state.summary_text = summary
        result = run_translation(summary)

elif mode == "Validation":
    if (
        "go_validate_only" in locals()
        and go_validate_only
        and src_nfc
        and st.session_state.tgt_text
    ):
        st.session_state["last_val_msg"] = validate_translation(
            src_nfc, st.session_state.tgt_text
        )

# Summarization Input
if (
    (mode == "Translation")
    and ("go_sum_translate" in locals())
    and go_sum_translate
    and st.session_state.src_text
):
    st.markdown("⬇️ This is the summarized version:")
    st.info(st.session_state.summary_text)

if result:
    st.session_state["tgt_text"] = result
    src_for_validation = st.session_state.summary_text if go_sum_translate else src_nfc

    st.session_state["last_val_msg"] = validate_translation(src_for_validation, result)

# Output
st.markdown(f"### Output ({st.session_state.tgt_lang})")
st.text_area(
    "Target Text",
    # value=st.session_state.get("tgt_text", ""),
    height=220,
    label_visibility="collapsed",
    key="tgt_text",
    placeholder="The translation results are displayed here. Validation mode also allows you to enter your own text to verify.",
)

val_msg = st.session_state.get("last_val_msg", "")
if val_msg:
    if "success" in val_msg.lower():
        st.success(val_msg)
    else:
        st.error(val_msg)
