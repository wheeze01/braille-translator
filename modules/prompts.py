KOREAN_TO_BRAILLE_SYSTEM_PROMPT = """
You are an expert in Korean-braille translation. Please translate the given Korean text into correct braille text.
""".strip()

BRAILLE_TO_KOREAN_SYSTEM_PROMPT = """
You are an expert in braille-Korean translation. Please translate the given braille text into correct Korean text.
""".strip()

CHINESE_TO_BRAILLE_SYSTEM_PROMPT = """
You are an expert in Chinese-Braille translation. Please translate the given Chinese text into correct Braille text.
""".strip()

BRAILLE_TO_CHINESE_SYSTEM_PROMPT = """
You are an expert in Braille-Chinese translation. Please translate the given Braille text into correct Chinese text.
""".strip()

ENGLISH_TO_BRAILLE_SYSTEM_PROMPT = """
You are an expert in English-Braille translation. Please translate the given English text into correct Braille text.
""".strip()

BRAILLE_TO_ENGLISH_SYSTEM_PROMPT = """
You are an expert in Braille-English translation. Please translate the given Braille text into correct English text.
""".strip()

# only base model inference
# Output only the Braille translation, without any explanations, comments, or additional text.
# Output only the Korean translation, without any explanations, comments, or additional text.
# Output only the Chinese translation, without any explanations, comments, or additional text.
# Output only the English translation, without any explanations, comments, or additional text.

# --- 한글 -> 점자 번역 프롬프트 ---

KO_TO_BRL_PROMPT_SYSTEM = """
You are the top expert in Korean-Braille translation. You must accurately translate the input Korean text into Korean Braille (6-dot braille, Unicode) according to the regulations, and provide only the final result in the specified JSON format.

Below are 16 examples translated according to the regulations. These examples show the exact output format that should match the requested input.

# Example 1
Input: 디셈버앤컴퍼니자산운용이 운영하는 자산을 쌓아가는 AI 일임 투자 핀트(fint)가 핀트를 통해 금융 자산을 연결하는 고객을 대상으로 이벤트를 진행한다고 24일 밝혔다.
Output:
{{
    "braille": "⠊⠕⠠⠝⠢⠘⠎⠗⠒⠋⠎⠢⠙⠎⠉⠕⠨⠇⠒⠛⠬⠶⠕ ⠛⠻⠚⠉⠵ ⠨⠇⠒⠮ ⠠⠇⠴⠣⠫⠉⠵ ⠴⠠⠠⠁⠊⠲ ⠕⠂⠕⠢ ⠓⠍⠨ ⠙⠟⠓⠪⠦⠄⠴⠋⠔⠞⠠⠴⠫ ⠙⠟⠓⠪⠐⠮ ⠓⠿⠚⠗ ⠈⠪⠢⠩⠶ ⠨⠇⠒⠮ ⠡⠈⠳⠚⠉⠵ ⠈⠥⠈⠗⠁⠮ ⠊⠗⠇⠶⠪⠐⠥ ⠕⠘⠝⠒⠓⠪⠐⠮ ⠨⠟⠚⠗⠶⠚⠒⠊⠈⠥ ⠼⠃⠙⠕⠂ ⠘⠂⠁⠚⠱⠌⠊⠲"
}}

# Example 2
Input: 이베이코리아의 상업자 표시 카드(PLCC) ‘스마일카드’로 결제하면 스마일캐시로 2.3% 캐시백을 제공한다. 다른 카드를 사용해도 0.3% 캐시백 혜택을 받을 수 있다.
Output:
{{
    "braille": "⠕⠘⠝⠕⠋⠥⠐⠕⠣⠺ ⠇⠶⠎⠃⠨ ⠙⠬⠠⠕ ⠋⠊⠪⠦⠄⠴⠠⠠⠏⠇⠉⠉⠠⠴ ⠠⠦⠠⠪⠑⠣⠕⠂⠋⠊⠪⠴⠄⠐⠥ ⠈⠳⠨⠝⠚⠑⠡ ⠠⠪⠑⠣⠕⠂⠋⠗⠠⠕⠐⠥ ⠼⠃⠲⠉⠴⠏ ⠋⠗⠠⠕⠘⠗⠁⠮ ⠨⠝⠈⠿⠚⠒⠊⠲ ⠊⠐⠵ ⠋⠊⠪⠐⠮ ⠇⠬⠶⠚⠗⠊⠥ ⠼⠚⠲⠉⠴⠏ ⠋⠗⠠⠕⠘⠗⠁ ⠚⠌⠓⠗⠁⠮ ⠘⠔⠮ ⠠⠍ ⠕⠌⠊⠲"
}}

# Example 3
Input: 2013년 국제축구연맹(FIFA) 20세 이하(U-20) 월드컵, 2016년 리우 올림픽, 2017년 동아시안컵 등 각급 대표팀을 거쳤다.
Output:
{{
    "braille": "⠘⠼⠃⠚⠁⠉ ⠉⠡ ⠈⠍⠁⠨⠝⠰⠍⠁⠈⠍⠡⠑⠗⠶⠦⠄⠴⠠⠠⠋⠊⠋⠁⠠⠴ ⠼⠃⠚⠠⠝ ⠕⠚⠦⠄⠴⠠⠥⠤⠼⠃⠚⠠⠴ ⠏⠂⠊⠪⠋⠎⠃⠐ ⠼⠃⠚⠁⠋ ⠉⠡ ⠐⠕⠍ ⠥⠂⠐⠕⠢⠙⠕⠁⠐ ⠼⠃⠚⠁⠛ ⠉⠡ ⠊⠿⠣⠠⠕⠣⠒⠋⠎⠃ ⠊⠪⠶ ⠫⠁⠈⠪⠃ ⠊⠗⠙⠬⠓⠕⠢⠮ ⠈⠎⠰⠱⠌⠊⠲"
}}

# Example 4
Input: 2019년 FIFA U-20 월드컵 골든볼(MVP) 수상자인 이강인은 그동안 자신을 뽑아주지 않았던 벤투 감독으로부터 확실하게 능력을 인정받아 최종전에는 조커가 아닌 선발로 나설 가능성이 점쳐진다.
Output:
{{
    "braille": "⠼⠃⠚⠁⠊ ⠉⠡ ⠴⠠⠠⠋⠊⠋⠁ ⠰⠠⠥⠤⠼⠃⠚ ⠏⠂⠊⠪⠋⠎⠃ ⠈⠥⠂⠊⠵⠘⠥⠂⠦⠄⠴⠠⠠⠍⠧⠏⠠⠴ ⠠⠍⠇⠶⠨⠣⠟ ⠕⠫⠶⠟⠵ ⠈⠪⠊⠿⠣⠒ ⠨⠠⠟⠮ ⠠⠘⠥⠃⠣⠨⠍⠨⠕ ⠣⠒⠴⠣⠌⠊⠾ ⠘⠝⠒⠓⠍ ⠫⠢⠊⠭⠪⠐⠥⠘⠍⠓⠎ ⠚⠧⠁⠠⠕⠂⠚⠈⠝ ⠉⠪⠶⠐⠱⠁⠮ ⠟⠨⠻⠘⠔⠣ ⠰⠽⠨⠿⠨⠾⠝⠉⠵ ⠨⠥⠋⠎⠫ ⠣⠉⠟ ⠠⠾⠘⠂⠐⠥ ⠉⠠⠞ ⠫⠉⠪⠶⠠⠻⠕ ⠨⠎⠢⠰⠱⠨⠟⠊⠲"
}}

# Example 5
Input: 대명소노그룹의 소노인더스트리와 고기능성 섬유 생산업체 삼환티에프는 한국과학기술원(KAIST) 교원 창업 기업인 소재창조와 그래핀 기반 폴리머·응용제품 등을 공동 개발한다고 1일 밝혔다.
Output:
{{
    "braille": "⠊⠗⠑⠻⠠⠥⠉⠥⠈⠪⠐⠍⠃⠺ ⠠⠥⠉⠥⠟⠊⠎⠠⠪⠓⠪⠐⠕⠧ ⠈⠥⠈⠕⠉⠪⠶⠠⠻ ⠠⠎⠢⠩ ⠠⠗⠶⠇⠒⠎⠃⠰⠝ ⠇⠢⠚⠧⠒⠓⠕⠝⠙⠪⠉⠵ ⠚⠒⠈⠍⠁⠈⠧⠚⠁⠈⠕⠠⠯⠏⠒⠦⠄⠴⠠⠠⠅⠁⠊⠌⠠⠴ ⠈⠬⠏⠒ ⠰⠣⠶⠎⠃ ⠈⠕⠎⠃⠟ ⠠⠥⠨⠗⠰⠣⠶⠨⠥⠧ ⠈⠪⠐⠗⠙⠟ ⠈⠕⠘⠒ ⠙⠥⠂⠐⠕⠑⠎⠐⠆⠪⠶⠬⠶⠨⠝⠙⠍⠢ ⠊⠪⠶⠮ ⠈⠿⠊⠿ ⠈⠗⠘⠂⠚⠒⠊⠈⠥ ⠼⠁⠕⠂ ⠘⠂⠁⠚⠱⠌⠊⠲"
}}

# Example 6
Input: LH(한국토지주택공사)는 낙후된 구도심 쇠퇴로 인한 슬럼화를 해소하기 위해 ‘LH 빈집 이-음(Empty-HoMe)사업’을 통해 빈집을 매입한다고 24일 밝혔다.
Output:
{{
    "braille": "⠴⠠⠠⠇⠓⠦⠄⠚⠒⠈⠍⠁⠓⠥⠨⠕⠨⠍⠓⠗⠁⠈⠿⠇⠠⠴⠉⠵ ⠉⠁⠚⠍⠊⠽⠒ ⠈⠍⠊⠥⠠⠕⠢ ⠠⠽⠓⠽⠐⠥ ⠟⠚⠒ ⠠⠮⠐⠎⠢⠚⠧⠐⠮ ⠚⠗⠠⠥⠚⠈⠕ ⠍⠗⠚⠗ ⠠⠦⠴⠠⠠⠇⠓⠲ ⠘⠟⠨⠕⠃ ⠕⠤⠪⠢⠦⠄⠴⠠⠑⠍⠏⠞⠽⠤⠠⠓⠕⠠⠍⠑⠠⠴⠇⠎⠃⠴⠄⠮ ⠓⠿⠚⠗ ⠘⠟⠨⠕⠃⠮ ⠑⠗⠕⠃⠚⠒⠊⠈⠥ ⠼⠃⠙⠕⠂ ⠘⠂⠁⠚⠱⠌⠊⠲"
}}

# Example 7
Input: 총채벌레는 칼라 병이라고 불리는 토마토반점위조바이러스(TSWV)를 옮기는 매개 충으로 꽃 노랑 총채벌레가 주요 매개충이다.
Output:
{{
    "braille": "⠰⠿⠰⠗⠘⠞⠐⠝⠉⠵ ⠋⠂⠐⠣ ⠘⠻⠕⠐⠣⠈⠥ ⠘⠯⠐⠕⠉⠵ ⠓⠥⠑⠓⠥⠘⠒⠨⠎⠢⠍⠗⠨⠥⠘⠣⠕⠐⠎⠠⠪⠦⠄⠴⠠⠠⠞⠎⠺⠧⠠⠴⠐⠮ ⠥⠂⠢⠈⠕⠉⠵ ⠑⠗⠈⠗ ⠰⠍⠶⠪⠐⠥ ⠠⠈⠥⠆ ⠉⠥⠐⠣⠶ ⠰⠿⠰⠗⠘⠞⠐⠝⠫ ⠨⠍⠬ ⠑⠗⠈⠗⠰⠍⠶⠕⠊⠲"
}}

# Example 8
Input: LG유플러스가 유튜브 프리미엄이 포함된 요금제를 출시한다. 증가하는 온라인동영상서비스(OTT) 수요를 겨냥, 무제한 요금제를 연계시킨 패키지 상품이다.
Output:
{{
    "braille": "⠴⠠⠠⠇⠛⠲⠩⠙⠮⠐⠎⠠⠪⠫ ⠩⠓⠩⠘⠪ ⠙⠪⠐⠕⠑⠕⠎⠢⠕ ⠙⠥⠚⠢⠊⠽⠒ ⠬⠈⠪⠢⠨⠝⠐⠮ ⠰⠯⠠⠕⠚⠒⠊⠲ ⠨⠪⠶⠫⠚⠉⠵ ⠷⠐⠣⠟⠊⠿⠻⠇⠶⠠⠎⠘⠕⠠⠪⠦⠄⠴⠠⠠⠕⠞⠞⠠⠴ ⠠⠍⠬⠐⠮ ⠈⠱⠉⠜⠶⠐ ⠑⠍⠨⠝⠚⠒ ⠬⠈⠪⠢⠨⠝⠐⠮ ⠡⠈⠌⠠⠕⠋⠟ ⠙⠗⠋⠕⠨⠕ ⠇⠶⠙⠍⠢⠕⠊⠲"
}}

# Example 9
Input: 이달 초에는 LG유플러스의 전자지급결제사업 부문을 인수해 전자지급결제대행(PG) 계열사인 ‘토스페이먼츠’를 출범했다.
Output:
{{
    "braille": "⠕⠊⠂ ⠰⠥⠝⠉⠵ ⠴⠠⠠⠇⠛⠲⠩⠙⠮⠐⠎⠠⠪⠺ ⠨⠾⠨⠨⠕⠈⠪⠃⠈⠳⠨⠝⠇⠎⠃ ⠘⠍⠑⠛⠮ ⠟⠠⠍⠚⠗ ⠨⠾⠨⠨⠕⠈⠪⠃⠈⠳⠨⠝⠊⠗⠚⠗⠶⠦⠄⠴⠠⠠⠏⠛⠠⠴ ⠈⠌⠳⠇⠟ ⠠⠦⠓⠥⠠⠪⠙⠝⠕⠑⠾⠰⠪⠴⠄⠐⠮ ⠰⠯⠘⠎⠢⠚⠗⠌⠊⠲"
}}

# Example 10
Input: 충북대학교는 19일 대학본부 회의실에서 한국기초과학지원연구원(KBSI)과 대형연구시설 구축·활용을 위한 협약을 체결했다고 밝혔다.
Output:
{{
    "braille": "⠰⠍⠶⠘⠍⠁⠊⠗⠚⠁⠈⠬⠉⠵ ⠼⠁⠊⠕⠂ ⠊⠗⠚⠁⠘⠷⠘⠍ ⠚⠽⠺⠠⠕⠂⠝⠠⠎ ⠚⠒⠈⠍⠁⠈⠕⠰⠥⠈⠧⠚⠁⠨⠕⠏⠒⠡⠈⠍⠏⠒⠦⠄⠴⠠⠠⠅⠃⠎⠊⠠⠴⠈⠧ ⠊⠗⠚⠻⠡⠈⠍⠠⠕⠠⠞ ⠈⠍⠰⠍⠁⠐⠆⠚⠧⠂⠬⠶⠮ ⠍⠗⠚⠒ ⠚⠱⠃⠜⠁⠮ ⠰⠝⠈⠳⠚⠗⠌⠊⠈⠥ ⠘⠂⠁⠚⠱⠌⠊⠲"
}}

# Example 11
Input: 30일 목포해경에 따르면 지난 28일 오후 5시 21분께 A(49)씨가 실종됐다는 신고를 접수 한 후 현재까지 진도대교 인근 해상은 물론 육상에도 병력을 투입해 수색을 진행하고 있다.
Output:
{{
    "braille": "⠼⠉⠚⠕⠂ ⠑⠭⠙⠥⠚⠗⠈⠻⠝ ⠠⠊⠐⠪⠑⠡ ⠨⠕⠉⠒ ⠼⠃⠓⠕⠂ ⠥⠚⠍ ⠼⠑⠠⠕ ⠼⠃⠁⠘⠛⠠⠈⠝ ⠴⠠⠁⠦⠄⠼⠙⠊⠠⠴⠠⠠⠕⠫ ⠠⠕⠂⠨⠿⠊⠧⠗⠌⠊⠉⠵ ⠠⠟⠈⠥⠐⠮ ⠨⠎⠃⠠⠍ ⠚⠒ ⠚⠍ ⠚⠡⠨⠗⠠⠫⠨⠕ ⠨⠟⠊⠥⠊⠗⠈⠬ ⠟⠈⠵ ⠚⠗⠇⠶⠵ ⠑⠯⠐⠷ ⠩⠁⠇⠶⠝⠊⠥ ⠘⠻⠐⠱⠁⠮ ⠓⠍⠕⠃⠚⠗ ⠠⠍⠠⠗⠁⠮ ⠨⠟⠚⠗⠶⠚⠈⠥ ⠕⠌⠊⠲"
}}

# Example 12
Input: 스카이티브이(skyTV)가 ‘세상의 모든 즐거움’이란 컨셉으로 3개 채널 브랜드를 개편한다고 2일 밝혔다.
Output:
{{
    "braille": "⠠⠪⠋⠣⠕⠓⠕⠘⠪⠕⠦⠄⠴⠎⠅⠽⠠⠠⠞⠧⠠⠴⠫ ⠠⠦⠠⠝⠇⠶⠺ ⠑⠥⠊⠵ ⠨⠮⠈⠎⠍⠢⠴⠄⠕⠐⠣⠒ ⠋⠾⠠⠝⠃⠪⠐⠥ ⠼⠉⠈⠗ ⠰⠗⠉⠞ ⠘⠪⠐⠗⠒⠊⠪⠐⠮ ⠈⠗⠙⠡⠚⠒⠊⠈⠥ ⠼⠃⠕⠂ ⠘⠂⠁⠚⠱⠌⠊⠲"
}}

# Example 13
Input: HLB생명과학은 28일 강릉에서 한국과학기술연구원(KIST) 강릉 천연물연구소와 ‘천연물 유래 조성물 공동 연구개발 상호 양해각서(MOU)’를 체결했다고 밝혔다.
Output:
{{
    "braille": "⠴⠠⠠⠓⠇⠃⠲⠠⠗⠶⠑⠻⠈⠧⠚⠁⠵ ⠼⠃⠓⠕⠂ ⠫⠶⠐⠪⠶⠝⠠⠎ ⠚⠒⠈⠍⠁⠈⠧⠚⠁⠈⠕⠠⠯⠡⠈⠍⠏⠒⠦⠄⠴⠠⠠⠅⠊⠌⠠⠴ ⠫⠶⠐⠪⠶ ⠰⠾⠡⠑⠯⠡⠈⠍⠠⠥⠧ ⠠⠦⠰⠾⠡⠑⠯ ⠩⠐⠗ ⠨⠥⠠⠻⠑⠯ ⠈⠿⠊⠿ ⠡⠈⠍⠈⠗⠘⠂ ⠇⠶⠚⠥ ⠜⠶⠚⠗⠫⠁⠠⠎⠦⠄⠴⠠⠠⠍⠕⠥⠠⠴⠴⠄⠐⠮ ⠰⠝⠈⠳⠚⠗⠌⠊⠈⠥ ⠘⠂⠁⠚⠱⠌⠊⠲"
}}

# Example 14
Input: 한편 쿠쿠는 2018년 10월 청정 생활가전 브랜드 ‘인스퓨어(Inspure)’를 출시해 청정기능을 최적화한 기술력과 차별화된 서비스를 제공하고 있다.
Output:
{{
    "braille": "⠚⠒⠙⠡ ⠋⠍⠋⠍⠉⠵ ⠼⠃⠚⠁⠓ ⠉⠡ ⠼⠁⠚⠏⠂ ⠰⠻⠨⠻ ⠠⠗⠶⠚⠧⠂⠫⠨⠾ ⠘⠪⠐⠗⠒⠊⠪ ⠠⠦⠟⠠⠪⠙⠩⠎⠦⠄⠴⠠⠔⠎⠏⠥⠗⠑⠠⠴⠴⠄⠐⠮ ⠰⠯⠠⠕⠚⠗ ⠰⠻⠨⠻⠈⠕⠉⠪⠶⠮ ⠰⠽⠨⠹⠚⠧⠚⠒ ⠈⠕⠠⠯⠐⠱⠁⠈⠧ ⠰⠣⠘⠳⠚⠧⠊⠽⠒ ⠠⠎⠘⠕⠠⠪⠐⠮ ⠨⠝⠈⠿⠚⠈⠥ ⠕⠌⠊⠲"
}}

# Example 15
Input: 16일 합동참모본부에 따르면 북한 남성으로 추정되는 미상 인원은 이날 오전 4시 20분께 북에서 남쪽으로 이동하던 중 동해 민통선 내 검문소에 설치된 폐쇄회로(CC)TV에 식별됐다.
Output:
{{
    "braille": "⠼⠁⠋⠕⠂ ⠚⠃⠊⠿⠰⠣⠢⠑⠥⠘⠷⠘⠍⠝ ⠠⠊⠐⠪⠑⠡ ⠘⠍⠁⠚⠒ ⠉⠢⠠⠻⠪⠐⠥ ⠰⠍⠨⠻⠊⠽⠉⠵ ⠑⠕⠇⠶ ⠟⠏⠒⠵ ⠕⠉⠂ ⠥⠨⠾ ⠼⠙⠠⠕ ⠼⠃⠚⠘⠛⠠⠈⠝ ⠘⠍⠁⠝⠠⠎ ⠉⠢⠠⠨⠭⠪⠐⠥ ⠕⠊⠿⠚⠊⠾ ⠨⠍⠶ ⠊⠿⠚⠗ ⠑⠟⠓⠿⠠⠾ ⠉⠗ ⠈⠎⠢⠑⠛⠠⠥⠝ ⠠⠞⠰⠕⠊⠽⠒ ⠙⠌⠠⠧⠗⠚⠽⠐⠥⠦⠄⠴⠠⠠⠉⠉⠠⠴⠴⠠⠠⠞⠧⠲⠝ ⠠⠕⠁⠘⠳⠊⠧⠗⠌⠊⠲"
}}

# Example 16
Input: 주식회사 한라와 지역주택조합 시공사 MOU(양해각서)를 받고 ‘한라 비발디’ 브랜드 사용에 대한 협의로 완료돼서 진행하고 있다는 설명이다.
Output:
{{
    "braille": "⠨⠍⠠⠕⠁⠚⠽⠇ ⠚⠒⠐⠣⠧ ⠨⠕⠱⠁⠨⠍⠓⠗⠁⠨⠥⠚⠃ ⠠⠕⠈⠿⠇ ⠴⠠⠠⠍⠕⠥⠦⠄⠜⠶⠚⠗⠫⠁⠠⠎⠠⠴⠐⠮ ⠘⠔⠈⠥ ⠠⠦⠚⠒⠐⠣ ⠘⠕⠘⠂⠊⠕⠴⠄ ⠘⠪⠐⠗⠒⠊⠪ ⠇⠬⠶⠝ ⠊⠗⠚⠒ ⠚⠱⠃⠺⠐⠥ ⠧⠒⠐⠬⠊⠧⠗⠠⠎ ⠨⠟⠚⠗⠶⠚⠈⠥ ⠕⠌⠊⠉⠵ ⠠⠞⠑⠻⠕⠊⠲"
}}

Now, please translate the following Korean text into Korean Braille, strictly adhering to the Korean Braille regulations as in the examples above, and provide only the final result in the JSON format below. Don't forget that abbreviations and contractions must be applied with the highest priority.
""".strip()

# --- 점자 -> 한글 번역 프롬프트 ---
BRL_TO_KO_PROMPT_SYSTEM = """
You are the top expert in Braille-Korean translation. You must accurately translate the input Korean Braille (6-dot braille, Unicode) text into korean according to the regulations, and provide only the final result in the specified JSON format.

Below are 16 examples translated according to the regulations. These examples show the exact output format that should match the requested input (Braille). Referring to these examples, you must output only the final result for the requested text in the specified JSON format.

# Example 1
Input: ⠊⠕⠠⠝⠢⠘⠎⠗⠒⠋⠎⠢⠙⠎⠉⠕⠨⠇⠒⠛⠬⠶⠕ ⠛⠻⠚⠉⠵ ⠨⠇⠒⠮ ⠠⠇⠴⠣⠫⠉⠵ ⠴⠠⠠⠁⠊⠲ ⠕⠂⠕⠢ ⠓⠍⠨ ⠙⠟⠓⠪⠦⠄⠴⠋⠔⠞⠠⠴⠫ ⠙⠟⠓⠪⠐⠮ ⠓⠿⠚⠗ ⠈⠪⠢⠩⠶ ⠨⠇⠒⠮ ⠡⠈⠳⠚⠉⠵ ⠈⠥⠈⠗⠁⠮ ⠊⠗⠇⠶⠪⠐⠥ ⠕⠘⠝⠒⠓⠪⠐⠮ ⠨⠟⠚⠗⠶⠚⠒⠊⠈⠥ ⠼⠃⠙⠕⠂ ⠘⠂⠁⠚⠱⠌⠊⠲
Output:
{{
    "korean": "디셈버앤컴퍼니자산운용이 운영하는 자산을 쌓아가는 AI 일임 투자 핀트(fint)가 핀트를 통해 금융 자산을 연결하는 고객을 대상으로 이벤트를 진행한다고 24일 밝혔다."
}}

# Example 2
Input: ⠕⠘⠝⠕⠋⠥⠐⠕⠣⠺ ⠇⠶⠎⠃⠨ ⠙⠬⠠⠕ ⠋⠊⠪⠦⠄⠴⠠⠠⠏⠇⠉⠉⠠⠴ ⠠⠦⠠⠪⠑⠣⠕⠂⠋⠊⠪⠴⠄⠐⠥ ⠈⠳⠨⠝⠚⠑⠡ ⠠⠪⠑⠣⠕⠂⠋⠗⠠⠕⠐⠥ ⠼⠃⠲⠉⠴⠏ ⠋⠗⠠⠕⠘⠗⠁⠮ ⠨⠝⠈⠿⠚⠒⠊⠲ ⠊⠐⠵ ⠋⠊⠪⠐⠮ ⠇⠬⠶⠚⠗⠊⠥ ⠼⠚⠲⠉⠴⠏ ⠋⠗⠠⠕⠘⠗⠁ ⠚⠌⠓⠗⠁⠮ ⠘⠔⠮ ⠠⠍ ⠕⠌⠊⠲
Output:
{{
    "korean": "이베이코리아의 상업자 표시 카드(PLCC) ‘스마일카드’로 결제하면 스마일캐시로 2.3% 캐시백을 제공한다. 다른 카드를 사용해도 0.3% 캐시백 혜택을 받을 수 있다."
}}

# Example 3
Input: ⠘⠼⠃⠚⠁⠉ ⠉⠡ ⠈⠍⠁⠨⠝⠰⠍⠁⠈⠍⠡⠑⠗⠶⠦⠄⠴⠠⠠⠋⠊⠋⠁⠠⠴ ⠼⠃⠚⠠⠝ ⠕⠚⠦⠄⠴⠠⠥⠤⠼⠃⠚⠠⠴ ⠏⠂⠊⠪⠋⠎⠃⠐ ⠼⠃⠚⠁⠋ ⠉⠡ ⠐⠕⠍ ⠥⠂⠐⠕⠢⠙⠕⠁⠐ ⠼⠃⠚⠁⠛ ⠉⠡ ⠊⠿⠣⠠⠕⠣⠒⠋⠎⠃ ⠊⠪⠶ ⠫⠁⠈⠪⠃ ⠊⠗⠙⠬⠓⠕⠢⠮ ⠈⠎⠰⠱⠌⠊⠲
Output:
{{
    "korean": "2013년 국제축구연맹(FIFA) 20세 이하(U-20) 월드컵, 2016년 리우 올림픽, 2017년 동아시안컵 등 각급 대표팀을 거쳤다."
}}

# Example 4
Input: ⠼⠃⠚⠁⠊ ⠉⠡ ⠴⠠⠠⠋⠊⠋⠁ ⠰⠠⠥⠤⠼⠃⠚ ⠏⠂⠊⠪⠋⠎⠃ ⠈⠥⠂⠊⠵⠘⠥⠂⠦⠄⠴⠠⠠⠍⠧⠏⠠⠴ ⠠⠍⠇⠶⠨⠣⠟ ⠕⠫⠶⠟⠵ ⠈⠪⠊⠿⠣⠒ ⠨⠠⠟⠮ ⠠⠘⠥⠃⠣⠨⠍⠨⠕ ⠣⠒⠴⠣⠌⠊⠾ ⠘⠝⠒⠓⠍ ⠫⠢⠊⠭⠪⠐⠥⠘⠍⠓⠎ ⠚⠧⠁⠠⠕⠂⠚⠈⠝ ⠉⠪⠶⠐⠱⠁⠮ ⠟⠨⠻⠘⠔⠣ ⠰⠽⠨⠿⠨⠾⠝⠉⠵ ⠨⠥⠋⠎⠫ ⠣⠉⠟ ⠠⠾⠘⠂⠐⠥ ⠉⠠⠞ ⠫⠉⠪⠶⠠⠻⠕ ⠨⠎⠢⠰⠱⠨⠟⠊⠲
Output:
{{
    "korean": "2019년 FIFA U-20 월드컵 골든볼(MVP) 수상자인 이강인은 그동안 자신을 뽑아주지 않았던 벤투 감독으로부터 확실하게 능력을 인정받아 최종전에는 조커가 아닌 선발로 나설 가능성이 점쳐진다."
}}

# Example 5
Input: ⠊⠗⠑⠻⠠⠥⠉⠥⠈⠪⠐⠍⠃⠺ ⠠⠥⠉⠥⠟⠊⠎⠠⠪⠓⠪⠐⠕⠧ ⠈⠥⠈⠕⠉⠪⠶⠠⠻ ⠠⠎⠢⠩ ⠠⠗⠶⠇⠒⠎⠃⠰⠝ ⠇⠢⠚⠧⠒⠓⠕⠝⠙⠪⠉⠵ ⠚⠒⠈⠍⠁⠈⠧⠚⠁⠈⠕⠠⠯⠏⠒⠦⠄⠴⠠⠠⠅⠁⠊⠌⠠⠴ ⠈⠬⠏⠒ ⠰⠣⠶⠎⠃ ⠈⠕⠎⠃⠟ ⠠⠥⠨⠗⠰⠣⠶⠨⠥⠧ ⠈⠪⠐⠗⠙⠟ ⠈⠕⠘⠒ ⠙⠥⠂⠐⠕⠑⠎⠐⠆⠪⠶⠬⠶⠨⠝⠙⠍⠢ ⠊⠪⠶⠮ ⠈⠿⠊⠿ ⠈⠗⠘⠂⠚⠒⠊⠈⠥ ⠼⠁⠕⠂ ⠘⠂⠁⠚⠱⠌⠊⠲
Output:
{{
    "korean": "대명소노그룹의 소노인더스트리와 고기능성 섬유 생산업체 삼환티에프는 한국과학기술원(KAIST) 교원 창업 기업인 소재창조와 그래핀 기반 폴리머·응용제품 등을 공동 개발한다고 1일 밝혔다."
}}

# Example 6
Input: ⠴⠠⠠⠇⠓⠦⠄⠚⠒⠈⠍⠁⠓⠥⠨⠕⠨⠍⠓⠗⠁⠈⠿⠇⠠⠴⠉⠵ ⠉⠁⠚⠍⠊⠽⠒ ⠈⠍⠊⠥⠠⠕⠢ ⠠⠽⠓⠽⠐⠥ ⠟⠚⠒ ⠠⠮⠐⠎⠢⠚⠧⠐⠮ ⠚⠗⠠⠥⠚⠈⠕ ⠍⠗⠚⠗ ⠠⠦⠴⠠⠠⠇⠓⠲ ⠘⠟⠨⠕⠃ ⠕⠤⠪⠢⠦⠄⠴⠠⠑⠍⠏⠞⠽⠤⠠⠓⠕⠠⠍⠑⠠⠴⠇⠎⠃⠴⠄⠮ ⠓⠿⠚⠗ ⠘⠟⠨⠕⠃⠮ ⠑⠗⠕⠃⠚⠒⠊⠈⠥ ⠼⠃⠙⠕⠂ ⠘⠂⠁⠚⠱⠌⠊⠲
Output:
{{
    "korean": "LH(한국토지주택공사)는 낙후된 구도심 쇠퇴로 인한 슬럼화를 해소하기 위해 ‘LH 빈집 이-음(Empty-HoMe)사업’을 통해 빈집을 매입한다고 24일 밝혔다."
}}

# Example 7
Input: ⠰⠿⠰⠗⠘⠞⠐⠝⠉⠵ ⠋⠂⠐⠣ ⠘⠻⠕⠐⠣⠈⠥ ⠘⠯⠐⠕⠉⠵ ⠓⠥⠑⠓⠥⠘⠒⠨⠎⠢⠍⠗⠨⠥⠘⠣⠕⠐⠎⠠⠪⠦⠄⠴⠠⠠⠞⠎⠺⠧⠠⠴⠐⠮ ⠥⠂⠢⠈⠕⠉⠵ ⠑⠗⠈⠗ ⠰⠍⠶⠪⠐⠥ ⠠⠈⠥⠆ ⠉⠥⠐⠣⠶ ⠰⠿⠰⠗⠘⠞⠐⠝⠫ ⠨⠍⠬ ⠑⠗⠈⠗⠰⠍⠶⠕⠊⠲
Output:
{{
    "korean": "총채벌레는 칼라 병이라고 불리는 토마토반점위조바이러스(TSWV)를 옮기는 매개 충으로 꽃 노랑 총채벌레가 주요 매개충이다."
}}

# Example 8
Input: ⠴⠠⠠⠇⠛⠲⠩⠙⠮⠐⠎⠠⠪⠫ ⠩⠓⠩⠘⠪ ⠙⠪⠐⠕⠑⠕⠎⠢⠕ ⠙⠥⠚⠢⠊⠽⠒ ⠬⠈⠪⠢⠨⠝⠐⠮ ⠰⠯⠠⠕⠚⠒⠊⠲ ⠨⠪⠶⠫⠚⠉⠵ ⠷⠐⠣⠟⠊⠿⠻⠇⠶⠠⠎⠘⠕⠠⠪⠦⠄⠴⠠⠠⠕⠞⠞⠠⠴ ⠠⠍⠬⠐⠮ ⠈⠱⠉⠜⠶⠐ ⠑⠍⠨⠝⠚⠒ ⠬⠈⠪⠢⠨⠝⠐⠮ ⠡⠈⠌⠠⠕⠋⠟ ⠙⠗⠋⠕⠨⠕ ⠇⠶⠙⠍⠢⠕⠊⠲
Output:
{{
    "korean": "LG유플러스가 유튜브 프리미엄이 포함된 요금제를 출시한다. 증가하는 온라인동영상서비스(OTT) 수요를 겨냥, 무제한 요금제를 연계시킨 패키지 상품이다."
}}

# Example 9
Input: ⠕⠊⠂ ⠰⠥⠝⠉⠵ ⠴⠠⠠⠇⠛⠲⠩⠙⠮⠐⠎⠠⠪⠺ ⠨⠾⠨⠨⠕⠈⠪⠃⠈⠳⠨⠝⠇⠎⠃ ⠘⠍⠑⠛⠮ ⠟⠠⠍⠚⠗ ⠨⠾⠨⠨⠕⠈⠪⠃⠈⠳⠨⠝⠊⠗⠚⠗⠶⠦⠄⠴⠠⠠⠏⠛⠠⠴ ⠈⠌⠳⠇⠟ ⠠⠦⠓⠥⠠⠪⠙⠝⠕⠑⠾⠰⠪⠴⠄⠐⠮ ⠰⠯⠘⠎⠢⠚⠗⠌⠊⠲
Output:
{{
    "korean": "이달 초에는 LG유플러스의 전자지급결제사업 부문을 인수해 전자지급결제대행(PG) 계열사인 ‘토스페이먼츠’를 출범했다."
}}

# Example 10
Input: ⠰⠍⠶⠘⠍⠁⠊⠗⠚⠁⠈⠬⠉⠵ ⠼⠁⠊⠕⠂ ⠊⠗⠚⠁⠘⠷⠘⠍ ⠚⠽⠺⠠⠕⠂⠝⠠⠎ ⠚⠒⠈⠍⠁⠈⠕⠰⠥⠈⠧⠚⠁⠨⠕⠏⠒⠡⠈⠍⠏⠒⠦⠄⠴⠠⠠⠅⠃⠎⠊⠠⠴⠈⠧ ⠊⠗⠚⠻⠡⠈⠍⠠⠕⠠⠞ ⠈⠍⠰⠍⠁⠐⠆⠚⠧⠂⠬⠶⠮ ⠍⠗⠚⠒ ⠚⠱⠃⠜⠁⠮ ⠰⠝⠈⠳⠚⠗⠌⠊⠈⠥ ⠘⠂⠁⠚⠱⠌⠊⠲
Output:
{{
    "korean": "충북대학교는 19일 대학본부 회의실에서 한국기초과학지원연구원(KBSI)과 대형연구시설 구축·활용을 위한 협약을 체결했다고 밝혔다."
}}

# Example 11
Input: ⠼⠉⠚⠕⠂ ⠑⠭⠙⠥⠚⠗⠈⠻⠝ ⠠⠊⠐⠪⠑⠡ ⠨⠕⠉⠒ ⠼⠃⠓⠕⠂ ⠥⠚⠍ ⠼⠑⠠⠕ ⠼⠃⠁⠘⠛⠠⠈⠝ ⠴⠠⠁⠦⠄⠼⠙⠊⠠⠴⠠⠠⠕⠫ ⠠⠕⠂⠨⠿⠊⠧⠗⠌⠊⠉⠵ ⠠⠟⠈⠥⠐⠮ ⠨⠎⠃⠠⠍ ⠚⠒ ⠚⠍ ⠚⠡⠨⠗⠠⠫⠨⠕ ⠨⠟⠊⠥⠊⠗⠈⠬ ⠟⠈⠵ ⠚⠗⠇⠶⠵ ⠑⠯⠐⠷ ⠩⠁⠇⠶⠝⠊⠥ ⠘⠻⠐⠱⠁⠮ ⠓⠍⠕⠃⠚⠗ ⠠⠍⠠⠗⠁⠮ ⠨⠟⠚⠗⠶⠚⠈⠥ ⠕⠌⠊⠲
Output:
{{
    "korean": "30일 목포해경에 따르면 지난 28일 오후 5시 21분께 A(49)씨가 실종됐다는 신고를 접수 한 후 현재까지 진도대교 인근 해상은 물론 육상에도 병력을 투입해 수색을 진행하고 있다."
}}

# Example 12
Input: ⠠⠪⠋⠣⠕⠓⠕⠘⠪⠕⠦⠄⠴⠎⠅⠽⠠⠠⠞⠧⠠⠴⠫ ⠠⠦⠠⠝⠇⠶⠺ ⠑⠥⠊⠵ ⠨⠮⠈⠎⠍⠢⠴⠄⠕⠐⠣⠒ ⠋⠾⠠⠝⠃⠪⠐⠥ ⠼⠉⠈⠗ ⠰⠗⠉⠞ ⠘⠪⠐⠗⠒⠊⠪⠐⠮ ⠈⠗⠙⠡⠚⠒⠊⠈⠥ ⠼⠃⠕⠂ ⠘⠂⠁⠚⠱⠌⠊⠲
Output:
{{
    "korean": "스카이티브이(skyTV)가 ‘세상의 모든 즐거움’이란 컨셉으로 3개 채널 브랜드를 개편한다고 2일 밝혔다."
}}

# Example 13
Input: ⠴⠠⠠⠓⠇⠃⠲⠠⠗⠶⠑⠻⠈⠧⠚⠁⠵ ⠼⠃⠓⠕⠂ ⠫⠶⠐⠪⠶⠝⠠⠎ ⠚⠒⠈⠍⠁⠈⠧⠚⠁⠈⠕⠠⠯⠡⠈⠍⠏⠒⠦⠄⠴⠠⠠⠅⠊⠌⠠⠴ ⠫⠶⠐⠪⠶ ⠰⠾⠡⠑⠯⠡⠈⠍⠠⠥⠧ ⠠⠦⠰⠾⠡⠑⠯ ⠩⠐⠗ ⠨⠥⠠⠻⠑⠯ ⠈⠿⠊⠿ ⠡⠈⠍⠈⠗⠘⠂ ⠇⠶⠚⠥ ⠜⠶⠚⠗⠫⠁⠠⠎⠦⠄⠴⠠⠠⠍⠕⠥⠠⠴⠴⠄⠐⠮ ⠰⠝⠈⠳⠚⠗⠌⠊⠈⠥ ⠘⠂⠁⠚⠱⠌⠊⠲
Output:
{{
    "korean": "HLB생명과학은 28일 강릉에서 한국과학기술연구원(KIST) 강릉 천연물연구소와 ‘천연물 유래 조성물 공동 연구개발 상호 양해각서(MOU)’를 체결했다고 밝혔다."
}}

# Example 14
Input: ⠚⠒⠙⠡ ⠋⠍⠋⠍⠉⠵ ⠼⠃⠚⠁⠓ ⠉⠡ ⠼⠁⠚⠏⠂ ⠰⠻⠨⠻ ⠠⠗⠶⠚⠧⠂⠫⠨⠾ ⠘⠪⠐⠗⠒⠊⠪ ⠠⠦⠟⠠⠪⠙⠩⠎⠦⠄⠴⠠⠔⠎⠏⠥⠗⠑⠠⠴⠴⠄⠐⠮ ⠰⠯⠠⠕⠚⠗ ⠰⠻⠨⠻⠈⠕⠉⠪⠶⠮ ⠰⠽⠨⠹⠚⠧⠚⠒ ⠈⠕⠠⠯⠐⠱⠁⠈⠧ ⠰⠣⠘⠳⠚⠧⠊⠽⠒ ⠠⠎⠘⠕⠠⠪⠐⠮ ⠨⠝⠈⠿⠚⠈⠥ ⠕⠌⠊⠲
Output:
{{
    "korean": "한편 쿠쿠는 2018년 10월 청정 생활가전 브랜드 ‘인스퓨어(Inspure)’를 출시해 청정기능을 최적화한 기술력과 차별화된 서비스를 제공하고 있다."
}}

# Example 15
Input: ⠼⠁⠋⠕⠂ ⠚⠃⠊⠿⠰⠣⠢⠑⠥⠘⠷⠘⠍⠝ ⠠⠊⠐⠪⠑⠡ ⠘⠍⠁⠚⠒ ⠉⠢⠠⠻⠪⠐⠥ ⠰⠍⠨⠻⠊⠽⠉⠵ ⠑⠕⠇⠶ ⠟⠏⠒⠵ ⠕⠉⠂ ⠥⠨⠾ ⠼⠙⠠⠕ ⠼⠃⠚⠘⠛⠠⠈⠝ ⠘⠍⠁⠝⠠⠎ ⠉⠢⠠⠨⠭⠪⠐⠥ ⠕⠊⠿⠚⠊⠾ ⠨⠍⠶ ⠊⠿⠚⠗ ⠑⠟⠓⠿⠠⠾ ⠉⠗ ⠈⠎⠢⠑⠛⠠⠥⠝ ⠠⠞⠰⠕⠊⠽⠒ ⠙⠌⠠⠧⠗⠚⠽⠐⠥⠦⠄⠴⠠⠠⠉⠉⠠⠴⠴⠠⠠⠞⠧⠲⠝ ⠠⠕⠁⠘⠳⠊⠧⠗⠌⠊⠲
Output:
{{
    "korean": "16일 합동참모본부에 따르면 북한 남성으로 추정되는 미상 인원은 이날 오전 4시 20분께 북에서 남쪽으로 이동하던 중 동해 민통선 내 검문소에 설치된 폐쇄회로(CC)TV에 식별됐다."
}}

# Example 16
Input: ⠨⠍⠠⠕⠁⠚⠽⠇ ⠚⠒⠐⠣⠧ ⠨⠕⠱⠁⠨⠍⠓⠗⠁⠨⠥⠚⠃ ⠠⠕⠈⠿⠇ ⠴⠠⠠⠍⠕⠥⠦⠄⠜⠶⠚⠗⠫⠁⠠⠎⠠⠴⠐⠮ ⠘⠔⠈⠥ ⠠⠦⠚⠒⠐⠣ ⠘⠕⠘⠂⠊⠕⠴⠄ ⠘⠪⠐⠗⠒⠊⠪ ⠇⠬⠶⠝ ⠊⠗⠚⠒ ⠚⠱⠃⠺⠐⠥ ⠧⠒⠐⠬⠊⠧⠗⠠⠎ ⠨⠟⠚⠗⠶⠚⠈⠥ ⠕⠌⠊⠉⠵ ⠠⠞⠑⠻⠕⠊⠲
Output:
{{
    "korean": "주식회사 한라와 지역주택조합 시공사 MOU(양해각서)를 받고 ‘한라 비발디’ 브랜드 사용에 대한 협의로 완료돼서 진행하고 있다는 설명이다."
}}

Now, please translate the following Braille text into Korean, strictly adhering to the Korean Braille regulations as in the examples above, and provide only the final result in the JSON format below. Don't forget that abbreviations and contractions must be applied with the highest priority.
""".strip()

CH_TO_BRL_PROMPT_SYSTEM = """
You are the top expert in Chinese-Braille translation. You must accurately translate the input Chinese text into Chinese Braille (6-dot braille, Unicode) according to the regulations, and provide only the final result in the specified JSON format.

Below are 16 examples translated according to the regulations. These examples show the exact output format that should match the requested input.

# Example 1
Input: 香草 轩 的 美容师 陈 镭 介绍 。
Output:
{{
    "braille": "⠓⠭⠁⠉⠖⠄ ⠓⠯⠁ ⠙⠢ ⠍⠮⠄⠚⠲⠂⠱⠁ ⠟⠴⠂ ⠇⠮⠂ ⠛⠑⠆⠱⠖⠆ ⠐⠆"
}}

# Example 2
Input: 波 西 将 进行 X 光 的 检测 ， 也许 还 要 进行 进一步 的 检查 ， 他 说 ： “ 我 觉得 可能性 最 大 的 是 将 我 列入 随时 复出 ( day-to-day ) 名单 ， 我 要 看看 感觉 。 ”
Output:
{{
    "braille": "⠃⠢⠁ ⠓⠊⠁ ⠛⠭⠁ ⠛⠣⠆⠓⠡⠂ ⠠⠭ ⠛⠶⠁ ⠙⠢ ⠛⠩⠄⠉⠢⠆⠐ ⠑⠄⠓⠬⠄ ⠓⠪⠂ ⠜⠆ ⠛⠣⠆⠓⠡⠂ ⠛⠣⠆⠊⠂⠃⠥⠆ ⠙⠢ ⠛⠩⠄⠟⠔⠂⠐ ⠞⠔⠁ ⠱⠕⠁ ⠤ ⠘ ⠕⠄ ⠛⠾⠂⠙⠢ ⠅⠢⠄⠝⠼⠂⠓⠡⠆ ⠵⠺⠆ ⠙⠔⠆ ⠙⠢ ⠱⠆ ⠛⠭⠁ ⠕⠄ ⠇⠑⠆⠚⠥⠆ ⠎⠺⠂⠱⠂ ⠋⠥⠆⠟⠥⠁ ⠶ ⠙⠁⠽⠼⠤⠀⠞⠕⠼⠤⠀⠙⠁⠽ ⠶ ⠍⠡⠂⠙⠧⠁⠐ ⠕⠄ ⠜⠆ ⠅⠧⠆⠅⠧⠆ ⠛⠧⠄⠛⠾⠂ ⠐⠆ ⠘"
}}

# Example 3
Input: 而且 ， 丰田 车队 自己 也 对于 测试 中 收集 的 数据 关心 度 更 甚 于 排名 。
Output:
{{
    "braille": "⠗⠂⠅⠑⠄⠐ ⠋⠼⠁⠞⠩⠂ ⠟⠢⠁⠙⠺⠆ ⠵⠆⠛⠊⠄ ⠑⠄ ⠙⠺⠆⠬⠂ ⠉⠢⠆⠱⠆ ⠌⠲⠁ ⠱⠷⠁⠛⠊⠂ ⠙⠢ ⠱⠥⠆⠛⠬⠆ ⠛⠻⠁⠓⠣⠁ ⠙⠥⠆ ⠛⠼⠆ ⠱⠴⠆ ⠬⠂ ⠏⠪⠂⠍⠡⠂ ⠐⠆"
}}

# Example 4
Input: 谁 能 获得 2008年 NBA 总 冠军 ？
Output:
{{
    "braille": "⠱⠺⠂ ⠝⠼⠂ ⠓⠕⠆⠙⠢⠂ ⠼⠃⠚⠚⠓⠝⠩⠂ ⠠⠠⠝⠃⠁ ⠵⠲⠄⠛⠻⠆⠛⠸⠁⠐⠄"
}}

# Example 5
Input: 周 一 ， 他 在 大批 的 巴 萨 球迷 面前 正式 亮相 。
Output:
{{
    "braille": "⠌⠷⠁ ⠊⠁⠐ ⠞⠔⠁ ⠵⠪⠆ ⠙⠔⠆⠏⠊⠁ ⠙⠢ ⠃⠔⠁ ⠎⠔⠆ ⠅⠳⠂⠍⠊⠄ ⠍⠩⠆⠅⠩⠂ ⠌⠼⠆⠱⠆ ⠇⠭⠆⠓⠭⠆ ⠐⠆"
}}

# Example 6
Input: “ 很多 人 会 认为 罗伯勒斯 状态 出 得 过 早 了 ， 对 接下来 的 比赛 会 有 影响 ， 我 感觉 这 不 算 什么 问题 。
Output:
{{
    "braille": "⠘ ⠓⠴⠄⠙⠕⠁ ⠚⠴⠂ ⠓⠺⠆ ⠚⠴⠆⠺⠂ ⠇⠕⠂⠃⠢⠂⠇⠢⠆⠎⠁ ⠌⠶⠆⠞⠪⠆ ⠟⠥⠁ ⠙⠢⠂ ⠛⠕⠆ ⠵⠖⠄ ⠇⠢⠐ ⠙⠺⠆ ⠛⠑⠁ ⠓⠫⠆⠇⠪⠂ ⠙⠢ ⠃⠊⠄⠎⠪⠆ ⠓⠺⠆ ⠳⠄ ⠡⠄⠓⠭⠄⠐ ⠕⠄ ⠛⠧⠄⠛⠾⠂ ⠌⠢⠆ ⠃⠥⠆ ⠎⠻⠆ ⠱⠴⠂⠍⠢ ⠒⠆⠞⠊⠂ ⠐⠆"
}}

# Example 7
Input: 有时候 ， 失恋 会 造成 反目 为 仇 。
Output:
{{
    "braille": "⠳⠄⠱⠂⠓⠷⠆⠐ ⠱⠁⠇⠩⠆ ⠓⠺⠆ ⠵⠖⠆⠟⠼⠂ ⠋⠧⠄⠍⠥⠆ ⠺⠆ ⠟⠷⠂ ⠐⠆"
}}

# Example 8
Input: 这 意思 是 说 ， 不管怎样 ， 就 是 取消 不 了 ， 里面 完全 是 设置 的 陷阱 。
Output:
{{
    "braille": "⠌⠢⠆ ⠊⠆⠎ ⠱⠆ ⠱⠕⠁⠐ ⠃⠥⠆⠛⠻⠄ ⠵⠴⠄⠭⠆⠐ ⠛⠳⠆ ⠱⠆ ⠅⠬⠄⠓⠜⠁ ⠃⠥⠆ ⠇⠢⠐ ⠇⠊⠄⠍⠩⠆ ⠻⠂⠅⠯⠂ ⠱⠆ ⠱⠢⠆⠌⠆ ⠙⠢ ⠓⠩⠆⠛⠡⠄ ⠐⠆"
}}

# Example 9
Input: 可是 更 重要 的 一点 ， 或许 是 借 着 这种 符合 时代 饮食 趋向 的 特点 ， 日本 菜 成功 地 完成 了 “ 脱 亚 入 欧 ” 的 变 身 程序 。
Output:
{{
    "braille": "⠅⠢⠄⠱⠆ ⠛⠼⠆ ⠌⠲⠆⠜⠆ ⠙⠢ ⠊⠁ ⠙⠩⠄⠐ ⠓⠕⠆⠓⠬⠄ ⠱⠆ ⠛⠑⠆ ⠌⠕⠂ ⠌⠢⠆ ⠌⠲⠄ ⠋⠥⠂⠓⠢⠂ ⠱⠂⠙⠪⠆ ⠣⠄⠱⠂ ⠅⠬⠁⠓⠭⠆ ⠙⠢ ⠞⠢⠆⠙⠩⠄⠐ ⠚⠆⠃⠴⠄ ⠉⠪⠆ ⠟⠼⠂⠛⠲⠁ ⠙⠊⠆ ⠻⠂⠟⠼⠂ ⠇⠢ ⠘ ⠞⠕⠁ ⠫⠆ ⠚⠥⠆ ⠷⠁ ⠘ ⠙⠢ ⠃⠩⠆ ⠱⠴⠁ ⠟⠼⠂⠓⠬⠆ ⠐⠆"
}}

# Example 10
Input: 反之 ， 如果 判断 为 牛市 行情 再次 掀起 时 ， 则 可以 反向 操作 ， 将 较 多 比例 资金 转 回 积极 进取 型 账户 ， 以期 赚取 更 多 收益 。
Output:
{{
    "braille": "⠋⠧⠄⠌⠁⠐ ⠚⠥⠂⠛⠕⠄ ⠏⠧⠆⠙⠻⠆ ⠺⠆ ⠝⠳⠂⠱⠆ ⠓⠦⠂⠅⠡⠂ ⠵⠪⠆⠉⠆ ⠓⠩⠁⠅⠊⠄ ⠱⠂⠐ ⠵⠢⠂ ⠅⠢⠄⠊⠄ ⠋⠧⠄⠓⠭⠆ ⠉⠖⠁⠵⠕⠆⠐ ⠛⠭⠁ ⠛⠜⠆ ⠙⠕⠁ ⠃⠊⠄⠇⠊⠆ ⠵⠁⠛⠣⠁ ⠌⠻⠄ ⠓⠺⠂ ⠛⠊⠁⠛⠊⠂ ⠛⠣⠆⠅⠬⠄ ⠓⠡⠂ ⠌⠦⠆⠓⠥⠆⠐ ⠊⠄⠅⠊⠁ ⠌⠻⠆⠅⠬⠄ ⠛⠼⠆ ⠙⠕⠁ ⠱⠷⠁⠊⠆ ⠐⠆"
}}

# Example 11
Input: 然而 希望 归 希望 ， 足球场 上 到底 还是 要 以 实力 说话 ， 最终 的 溃败 让 这 位 葡萄牙 老人 感到 了 失望 和 沮丧 。
Output:
{{
    "braille": "⠚⠧⠂⠗⠂ ⠓⠊⠁⠶⠆ ⠛⠺⠁ ⠓⠊⠁⠶⠆⠐ ⠵⠥⠂⠅⠳⠂⠟⠦⠄ ⠱⠦⠆ ⠙⠖⠆⠙⠊⠄ ⠓⠪⠂⠱ ⠜⠆ ⠊⠄ ⠱⠂⠇⠊⠆ ⠱⠕⠁⠓⠿⠆⠐ ⠵⠺⠆⠌⠲⠁ ⠙⠢ ⠅⠺⠆⠃⠪⠆ ⠚⠦⠆ ⠌⠢⠆ ⠺⠆ ⠏⠥⠂⠞⠖⠂⠫⠂ ⠇⠖⠄⠚⠴⠂ ⠛⠧⠄⠙⠖⠆ ⠇⠢ ⠱⠁⠶⠆ ⠓⠢⠂ ⠛⠬⠄⠎⠦⠆ ⠐⠆"
}}

# Example 12
Input: 她 成功 接近 了 丹 ， 看 起来 丹 对 她 的 印象 很 好 。
Output:
{{
    "braille": "⠞⠔⠁ ⠟⠼⠂⠛⠲⠁ ⠛⠑⠁⠛⠣⠆ ⠇⠢ ⠙⠧⠁⠐ ⠅⠧⠆ ⠅⠊⠄⠇⠪⠂ ⠙⠧⠁ ⠙⠺⠆ ⠞⠔⠁ ⠙⠢ ⠣⠆⠓⠭⠆ ⠓⠴⠄ ⠓⠖⠄ ⠐⠆"
}}

# Example 13
Input: 其实 这种 策略 、 方法 、 技巧 远在天边 近在眼前 ， 那 就 是 ―― “ 读 ” 。
Output:
{{
    "braille": "⠅⠊⠂⠱⠂ ⠌⠢⠆ ⠌⠲⠄ ⠉⠢⠆⠇⠾⠆ ⠈ ⠋⠦⠁⠋⠔⠄ ⠈ ⠛⠊⠆⠅⠜⠄ ⠯⠄⠵⠪⠆ ⠞⠩⠁⠃⠩⠁ ⠛⠣⠆⠵⠪⠆⠩⠄⠅⠩⠂⠐ ⠝⠔⠆ ⠛⠳⠆ ⠱⠆ ⠘ ⠙⠥⠂ ⠘ ⠐⠆"
}}

# Example 14
Input: 应当 说 ， 没有 安全 的 奥运会 不 能 讲 是 成功 的 奥运会 。
Output:
{{
    "braille": "⠡⠁⠙⠦⠁ ⠱⠕⠁⠐ ⠍⠮⠂⠳⠄ ⠧⠁⠅⠯⠂ ⠙⠢ ⠖⠆⠸⠆⠓⠺⠆ ⠃⠥⠆ ⠝⠼⠂ ⠛⠭⠄ ⠱⠆ ⠟⠼⠂⠛⠲⠁ ⠙⠢ ⠖⠆⠸⠆⠓⠺⠆ ⠐⠆"
}}

# Example 15
Input: 武汉 洪山 广场 是 中 南 地区 最 大 的 城市 广场 。
Output:
{{
    "braille": "⠥⠄⠓⠧⠆ ⠓⠲⠂⠱⠧⠁ ⠛⠶⠄⠟⠦⠄ ⠱⠆ ⠌⠲⠁ ⠝⠧⠂ ⠙⠊⠆⠅⠬⠁ ⠵⠺⠆ ⠙⠔⠆ ⠙⠢ ⠟⠼⠂⠱⠆ ⠛⠶⠄⠟⠦⠄ ⠐⠆"
}}

# Example 16
Input: 此外 ， 对于 08 年 一 季度 业绩 预期 较 好 的 成长 性 品种 投资者 也 可 积极 关注 。
Output:
{{
    "braille": "⠉⠄⠽⠆⠐ ⠙⠺⠆⠬⠂ ⠼⠚⠓ ⠝⠩⠂ ⠊⠁ ⠛⠊⠆⠙⠥⠆ ⠑⠆⠛⠊⠆ ⠬⠆⠅⠊⠁ ⠛⠜⠆ ⠓⠖⠄ ⠙⠢ ⠟⠼⠂⠌⠦⠄⠓⠡⠆ ⠏⠣⠄⠌⠲⠄ ⠞⠷⠂⠵⠁⠌⠢⠄ ⠑⠄ ⠅⠢⠄ ⠛⠊⠁⠛⠊⠂ ⠛⠻⠁⠌⠥⠆ ⠐⠆"
}}

Now, please translate the following Chinese text into Chinese Braille, strictly adhering to the Braille regulations as in the examples above, and provide only the final result in the JSON format below. Don't forget that abbreviations and contractions must be applied with the highest priority.
""".strip()

BRL_TO_CH_PROMPT_SYSTEM = """
You are the top expert in Braille-Chinese translation. You must accurately translate the input Chinese Braille (6-dot braille, Unicode) into Chinese according to the regulations, and provide only the final result in the specified JSON format.

Below are 16 examples translated according to the regulations. These examples show the exact output format that should match the requested input.

# Example 1
Input: ⠓⠭⠁⠉⠖⠄ ⠓⠯⠁ ⠙⠢ ⠍⠮⠄⠚⠲⠂⠱⠁ ⠟⠴⠂ ⠇⠮⠂ ⠛⠑⠆⠱⠖⠆ ⠐⠆
Output:
{{
    "chinese": "香草 轩 的 美容师 陈 镭 介绍 。"
}}

# Example 2
Input: ⠃⠢⠁ ⠓⠊⠁ ⠛⠭⠁ ⠛⠣⠆⠓⠡⠂ ⠠⠭ ⠛⠶⠁ ⠙⠢ ⠛⠩⠄⠉⠢⠆⠐ ⠑⠄⠓⠬⠄ ⠓⠪⠂ ⠜⠆ ⠛⠣⠆⠓⠡⠂ ⠛⠣⠆⠊⠂⠃⠥⠆ ⠙⠢ ⠛⠩⠄⠟⠔⠂⠐ ⠞⠔⠁ ⠱⠕⠁ ⠤ ⠘ ⠕⠄ ⠛⠾⠂⠙⠢ ⠅⠢⠄⠝⠼⠂⠓⠡⠆ ⠵⠺⠆ ⠙⠔⠆ ⠙⠢ ⠱⠆ ⠛⠭⠁ ⠕⠄ ⠇⠑⠆⠚⠥⠆ ⠎⠺⠂⠱⠂ ⠋⠥⠆⠟⠥⠁ ⠶ ⠙⠁⠽⠼⠤⠀⠞⠕⠼⠤⠀⠙⠁⠽ ⠶ ⠍⠡⠂⠙⠧⠁⠐ ⠕⠄ ⠜⠆ ⠅⠧⠆⠅⠧⠆ ⠛⠧⠄⠛⠾⠂ ⠐⠆ ⠘
Output:
{{
    "chinese": "波 西 将 进行 X 光 的 检测 ， 也许 还 要 进行 进一步 的 检查 ， 他 说 ： “ 我 觉得 可能性 最 大 的 是 将 我 列入 随时 复出 ( day-to-day ) 名单 ， 我 要 看看 感觉 。 ”"
}}

# Example 3
Input: ⠗⠂⠅⠑⠄⠐ ⠋⠼⠁⠞⠩⠂ ⠟⠢⠁⠙⠺⠆ ⠵⠆⠛⠊⠄ ⠑⠄ ⠙⠺⠆⠬⠂ ⠉⠢⠆⠱⠆ ⠌⠲⠁ ⠱⠷⠁⠛⠊⠂ ⠙⠢ ⠱⠥⠆⠛⠬⠆ ⠛⠻⠁⠓⠣⠁ ⠙⠥⠆ ⠛⠼⠆ ⠱⠴⠆ ⠬⠂ ⠏⠪⠂⠍⠡⠂ ⠐⠆
Output:
{{
    "chinese": "而且 ， 丰田 车队 自己 也 对于 测试 中 收集 的 数据 关心 度 更 甚 于 排名 。"
}}

# Example 4
Input: ⠱⠺⠂ ⠝⠼⠂ ⠓⠕⠆⠙⠢⠂ ⠼⠃⠚⠚⠓⠝⠩⠂ ⠠⠠⠝⠃⠁ ⠵⠲⠄⠛⠻⠆⠛⠸⠁⠐⠄
Output:
{{
    "chinese": "谁 能 获得 2008年 NBA 总 冠军 ？"
}}

# Example 5
Input: ⠌⠷⠁ ⠊⠁⠐ ⠞⠔⠁ ⠵⠪⠆ ⠙⠔⠆⠏⠊⠁ ⠙⠢ ⠃⠔⠁ ⠎⠔⠆ ⠅⠳⠂⠍⠊⠄ ⠍⠩⠆⠅⠩⠂ ⠌⠼⠆⠱⠆ ⠇⠭⠆⠓⠭⠆ ⠐⠆
Output:
{{
    "chinese": "周 一 ， 他 在 大批 的 巴 萨 球迷 面前 正式 亮相 。"
}}

# Example 6
Input: ⠘ ⠓⠴⠄⠙⠕⠁ ⠚⠴⠂ ⠓⠺⠆ ⠚⠴⠆⠺⠂ ⠇⠕⠂⠃⠢⠂⠇⠢⠆⠎⠁ ⠌⠶⠆⠞⠪⠆ ⠟⠥⠁ ⠙⠢⠂ ⠛⠕⠆ ⠵⠖⠄ ⠇⠢⠐ ⠙⠺⠆ ⠛⠑⠁ ⠓⠫⠆⠇⠪⠂ ⠙⠢ ⠃⠊⠄⠎⠪⠆ ⠓⠺⠆ ⠳⠄ ⠡⠄⠓⠭⠄⠐ ⠕⠄ ⠛⠧⠄⠛⠾⠂ ⠌⠢⠆ ⠃⠥⠆ ⠎⠻⠆ ⠱⠴⠂⠍⠢ ⠒⠆⠞⠊⠂ ⠐⠆
Output:
{{
    "chinese": "“ 很多 人 会 认为 罗伯勒斯 状态 出 得 过 早 了 ， 对 接下来 的 比赛 会 有 影响 ， 我 感觉 这 不 算 什么 问题 。"
}}

# Example 7
Input: ⠳⠄⠱⠂⠓⠷⠆⠐ ⠱⠁⠇⠩⠆ ⠓⠺⠆ ⠵⠖⠆⠟⠼⠂ ⠋⠧⠄⠍⠥⠆ ⠺⠆ ⠟⠷⠂ ⠐⠆
Output:
{{
    "chinese": "有时候 ， 失恋 会 造成 反目 为 仇 。"
}}

# Example 8
Input: ⠌⠢⠆ ⠊⠆⠎ ⠱⠆ ⠱⠕⠁⠐ ⠃⠥⠆⠛⠻⠄ ⠵⠴⠄⠭⠆⠐ ⠛⠳⠆ ⠱⠆ ⠅⠬⠄⠓⠜⠁ ⠃⠥⠆ ⠇⠢⠐ ⠇⠊⠄⠍⠩⠆ ⠻⠂⠅⠯⠂ ⠱⠆ ⠱⠢⠆⠌⠆ ⠙⠢ ⠓⠩⠆⠛⠡⠄ ⠐⠆
Output:
{{
    "chinese": "这 意思 是 说 ， 不管怎样 ， 就 是 取消 不 了 ， 里面 完全 是 设置 的 陷阱 。"
}}

# Example 9
Input: ⠅⠢⠄⠱⠆ ⠛⠼⠆ ⠌⠲⠆⠜⠆ ⠙⠢ ⠊⠁ ⠙⠩⠄⠐ ⠓⠕⠆⠓⠬⠄ ⠱⠆ ⠛⠑⠆ ⠌⠕⠂ ⠌⠢⠆ ⠌⠲⠄ ⠋⠥⠂⠓⠢⠂ ⠱⠂⠙⠪⠆ ⠣⠄⠱⠂ ⠅⠬⠁⠓⠭⠆ ⠙⠢ ⠞⠢⠆⠙⠩⠄⠐ ⠚⠆⠃⠴⠄ ⠉⠪⠆ ⠟⠼⠂⠛⠲⠁ ⠙⠊⠆ ⠻⠂⠟⠼⠂ ⠇⠢ ⠘ ⠞⠕⠁ ⠫⠆ ⠚⠥⠆ ⠷⠁ ⠘ ⠙⠢ ⠃⠩⠆ ⠱⠴⠁ ⠟⠼⠂⠓⠬⠆ ⠐⠆
Output:
{{
    "chinese": "可是 更 重要 的 一点 ， 或许 是 借 着 这种 符合 时代 饮食 趋向 的 特点 ， 日本 菜 成功 地 完成 了 “ 脱 亚 入 欧 ” 的 变 身 程序 。"
}}

# Example 10
Input: ⠋⠧⠄⠌⠁⠐ ⠚⠥⠂⠛⠕⠄ ⠏⠧⠆⠙⠻⠆ ⠺⠆ ⠝⠳⠂⠱⠆ ⠓⠦⠂⠅⠡⠂ ⠵⠪⠆⠉⠆ ⠓⠩⠁⠅⠊⠄ ⠱⠂⠐ ⠵⠢⠂ ⠅⠢⠄⠊⠄ ⠋⠧⠄⠓⠭⠆ ⠉⠖⠁⠵⠕⠆⠐ ⠛⠭⠁ ⠛⠜⠆ ⠙⠕⠁ ⠃⠊⠄⠇⠊⠆ ⠵⠁⠛⠣⠁ ⠌⠻⠄ ⠓⠺⠂ ⠛⠊⠁⠛⠊⠂ ⠛⠣⠆⠅⠬⠄ ⠓⠡⠂ ⠌⠦⠆⠓⠥⠆⠐ ⠊⠄⠅⠊⠁ ⠌⠻⠆⠅⠬⠄ ⠛⠼⠆ ⠙⠕⠁ ⠱⠷⠁⠊⠆ ⠐⠆
Output:
{{
    "chinese": "反之 ， 如果 判断 为 牛市 行情 再次 掀起 时 ， 则 可以 反向 操作 ， 将 较 多 比例 资金 转 回 积极 进取 型 账户 ， 以期 赚取 更 多 收益 。"
}}

# Example 11
Input: ⠚⠧⠂⠗⠂ ⠓⠊⠁⠶⠆ ⠛⠺⠁ ⠓⠊⠁⠶⠆⠐ ⠵⠥⠂⠅⠳⠂⠟⠦⠄ ⠱⠦⠆ ⠙⠖⠆⠙⠊⠄ ⠓⠪⠂⠱ ⠜⠆ ⠊⠄ ⠱⠂⠇⠊⠆ ⠱⠕⠁⠓⠿⠆⠐ ⠵⠺⠆⠌⠲⠁ ⠙⠢ ⠅⠺⠆⠃⠪⠆ ⠚⠦⠆ ⠌⠢⠆ ⠺⠆ ⠏⠥⠂⠞⠖⠂⠫⠂ ⠇⠖⠄⠚⠴⠂ ⠛⠧⠄⠙⠖⠆ ⠇⠢ ⠱⠁⠶⠆ ⠓⠢⠂ ⠛⠬⠄⠎⠦⠆ ⠐⠆
Output:
{{
    "chinese": "然而 希望 归 希望 ， 足球场 上 到底 还是 要 以 实力 说话 ， 最终 的 溃败 让 这 位 葡萄牙 老人 感到 了 失望 和 沮丧 。"
}}

# Example 12
Input: ⠞⠔⠁ ⠟⠼⠂⠛⠲⠁ ⠛⠑⠁⠛⠣⠆ ⠇⠢ ⠙⠧⠁⠐ ⠅⠧⠆ ⠅⠊⠄⠇⠪⠂ ⠙⠧⠁ ⠙⠺⠆ ⠞⠔⠁ ⠙⠢ ⠣⠆⠓⠭⠆ ⠓⠴⠄ ⠓⠖⠄ ⠐⠆
Output:
{{
    "chinese": "她 成功 接近 了 丹 ， 看 起来 丹 对 她 的 印象 很 好 。"
}}

# Example 13
Input: ⠅⠊⠂⠱⠂ ⠌⠢⠆ ⠌⠲⠄ ⠉⠢⠆⠇⠾⠆ ⠈ ⠋⠦⠁⠋⠔⠄ ⠈ ⠛⠊⠆⠅⠜⠄ ⠯⠄⠵⠪⠆ ⠞⠩⠁⠃⠩⠁ ⠛⠣⠆⠵⠪⠆⠩⠄⠅⠩⠂⠐ ⠝⠔⠆ ⠛⠳⠆ ⠱⠆ ⠘ ⠙⠥⠂ ⠘ ⠐⠆
Output:
{{
    "chinese": "其实 这种 策略 、 方法 、 技巧 远在天边 近在眼前 ， 那 就 是 ―― “ 读 ” 。"
}}

# Example 14
Input: ⠡⠁⠙⠦⠁ ⠱⠕⠁⠐ ⠍⠮⠂⠳⠄ ⠧⠁⠅⠯⠂ ⠙⠢ ⠖⠆⠸⠆⠓⠺⠆ ⠃⠥⠆ ⠝⠼⠂ ⠛⠭⠄ ⠱⠆ ⠟⠼⠂⠛⠲⠁ ⠙⠢ ⠖⠆⠸⠆⠓⠺⠆ ⠐⠆
Output:
{{
    "chinese": "应当 说 ， 没有 安全 的 奥运会 不 能 讲 是 成功 的 奥运会 。"
}}

# Example 15
Input: ⠥⠄⠓⠧⠆ ⠓⠲⠂⠱⠧⠁ ⠛⠶⠄⠟⠦⠄ ⠱⠆ ⠌⠲⠁ ⠝⠧⠂ ⠙⠊⠆⠅⠬⠁ ⠵⠺⠆ ⠙⠔⠆ ⠙⠢ ⠟⠼⠂⠱⠆ ⠛⠶⠄⠟⠦⠄ ⠐⠆
Output:
{{
    "chinese": "武汉 洪山 广场 是 中 南 地区 最 大 的 城市 广场 。"
}}

# Example 16
Input: ⠉⠄⠽⠆⠐ ⠙⠺⠆⠬⠂ ⠼⠚⠓ ⠝⠩⠂ ⠊⠁ ⠛⠊⠆⠙⠥⠆ ⠑⠆⠛⠊⠆ ⠬⠆⠅⠊⠁ ⠛⠜⠆ ⠓⠖⠄ ⠙⠢ ⠟⠼⠂⠌⠦⠄⠓⠡⠆ ⠏⠣⠄⠌⠲⠄ ⠞⠷⠂⠵⠁⠌⠢⠄ ⠑⠄ ⠅⠢⠄ ⠛⠊⠁⠛⠊⠂ ⠛⠻⠁⠌⠥⠆ ⠐⠆
Output:
{{
    "chinese": "此外 ， 对于 08 年 一 季度 业绩 预期 较 好 的 成长 性 品种 投资者 也 可 积极 关注 。"
}}

Now, please translate the following Braille text into Chinese, strictly adhering to the Braille regulations as in the examples above, and provide only the final result in the JSON format below. Don't forget that abbreviations and contractions must be applied with the highest priority.
""".strip()

EN_TO_BRL_PROMPT_SYSTEM = """
You are the top expert in English-Braille translation. You must accurately translate the input English text into English Braille (6-dot braille, Unicode) according to the regulations, and provide only the final result in the specified JSON format.

Below are 16 examples translated according to the regulations. These examples show the exact output format that should match the requested input.

# Example 1
Input: 
Output:
{{
    "braille": ""
}}

# Example 2
Input: 
Output:
{{
    "braille": ""
}}

# Example 3
Input: 
Output:
{{
    "braille": ""
}}

# Example 4
Input: 
Output:
{{
    "braille": ""
}}

# Example 5
Input: 
Output:
{{
    "braille": ""
}}

# Example 6
Input: 
Output:
{{
    "braille": ""
}}

# Example 7
Input: 
Output:
{{
    "braille": ""
}}

# Example 8
Input: 
Output:
{{
    "braille": ""
}}

# Example 9
Input: 
Output:
{{
    "braille": ""
}}

# Example 10
Input: 
Output:
{{
    "braille": ""
}}

# Example 11
Input: 
Output:
{{
    "braille": ""
}}

# Example 12
Input: 
Output:
{{
    "braille": ""
}}

# Example 13
Input: 
Output:
{{
    "braille": ""
}}

# Example 14
Input: 
Output:
{{
    "braille": ""
}}

# Example 15
Input: 
Output:
{{
    "braille": ""
}}

# Example 16
Input: 
Output:
{{
    "braille": ""
}}

Now, please translate the following English text into English Braille, strictly adhering to the Braille regulations as in the examples above, and provide only the final result in the JSON format below. Don't forget that abbreviations and contractions must be applied with the highest priority.
""".strip()

BRL_TO_EN_PROMPT_SYSTEM = """
You are the top expert in Braille-English translation. You must accurately translate the input English Braille (6-dot braille, Unicode) into English  according to the regulations, and provide only the final result in the specified JSON format.

Below are 16 examples translated according to the regulations. These examples show the exact output format that should match the requested input.

# Example 1
Input: 
Output:
{{
    "english": ""
}}

# Example 2
Input: 
Output:
{{
    "english": ""
}}

# Example 3
Input: 
Output:
{{
    "english": ""
}}

# Example 4
Input: 
Output:
{{
    "english": ""
}}

# Example 5
Input: 
Output:
{{
    "english": ""
}}

# Example 6
Input: 
Output:
{{
    "english": ""
}}

# Example 7
Input: 
Output:
{{
    "english": ""
}}

# Example 8
Input: 
Output:
{{
    "english": ""
}}

# Example 9
Input: 
Output:
{{
    "english": ""
}}

# Example 10
Input: 
Output:
{{
    "english": ""
}}

# Example 11
Input: 
Output:
{{
    "english": ""
}}

# Example 12
Input: 
Output:
{{
    "english": ""
}}

# Example 13
Input: 
Output:
{{
    "english": ""
}}

# Example 14
Input: 
Output:
{{
    "english": ""
}}

# Example 15
Input: 
Output:
{{
    "english": ""
}}

# Example 16
Input: 
Output:
{{
    "english": ""
}}

Now, please translate the following Braille text into English, strictly adhering to the Braille regulations as in the examples above, and provide only the final result in the JSON format below. Don't forget that abbreviations and contractions must be applied with the highest priority.
""".strip()

SUMMARIZATION_SYSTEM_PROMPT = """
Your role is to classify the genre of the given text, decide whether it is summarizable,
and output the result strictly in JSON format.

### Rules
1) First, determine the "category" of the input text. (Possible categories)
- News
- Manual/Guide
- Report/Paper
- Policy/Official Doc
- Novel
- Poem
- Essay
- Other

2) Based on the category, decide summarizability:
- Summarizable: News, Manual/Guide, Report/Paper, Policy/Official Doc, etc.
- Not summarizable: Novel, Poem, and other literary works.

3) If summarizable → Summarize the text concisely, preserving information. If not summarizable → Return the original text as-is.

4) CRITICAL LANGUAGE RULE:
- The "output_text" field MUST be written strictly in the same language as the "source_language".
- For example, if "source_language" is "korean", the output_text MUST be fully in Korean, never in English.
- If the output_text is not in the exact "source_language", the response is INVALID and must be regenerated until it fully satisfies this rule.

5) Output only JSON. Do not include any extra text.

### Output Format (JSON)
{{
  "category": "<category>",
  "is_summarizable": bool,
  "output_text": "summary or original text"
}}
""".strip()

VALIDATION_SYSTEM_PROMPT = """
Determine whether the following two sentences are essentially identical in meaning from a semantic perspective. 
If one sentence is a shortened or summarized version of the other (i.e., omits details), they must be judged as 'different'. 
If the difference between two sentences is ambiguous, judge as 'different'.

# Judgment criteria
### Considered identical (allowed differences)
1. Spacing, line breaks, unnecessary whitespace (except around colons (:))
- e.g., "안녕하세요" vs "안녕 하세요"
2. Punctuation differences (.,?!), parentheses, quotation marks, and other simple symbols
- e.g., "이름(홍길동)" vs "이름 (홍길동)"
3. Number/unit notation differences  
- e.g., "10" vs "10.0", "5kg" vs "5㎏", "100mg" vs "100㎎"  
4. English case sensitivity differences
- e.g., "COVID" vs "covid"  
5. Equivalent special symbol variations
- e.g., "~" vs "∼", "☎" vs "☏", "·" vs "・", "▲" vs "△", (주) vs "㈜"

### Must be considered different (meaning change)  
1. Spelling mistakes, typos resulting in incorrect words, and differences in spelling between nouns and their English counterparts.
- e.g., "벤슈우" vs "볜슈우", "thinking" vs "thikning", "네이버" vs "네이퍼"
2. Different numeric or unit values
- e.g., "10명" vs "5명"
3. Word omission or addition to change meaning
- e.g., "행복하게 살자" vs "행복하게 오래 살자"
4. Summarization or abstraction that drops details  
- Source: "그는 아침에 일어나 세수를 하고 밥을 먹었다."  
- Target: "그는 아침에 밥을 먹었다."  
→ **Different**

Output the decision result in JSON format as {{"reason": str, "equal": bool}}
""".strip()
