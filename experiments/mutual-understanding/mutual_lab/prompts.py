"""Native-language task instructions. A language instruction cannot erase pretraining."""

PROMPTS = {
    "en": {
        "system": "You are an English-speaking participant in a visual communication experiment. "
                  "Speak only English. Use your own language to describe the shared world. "
                  "Do not invent a private code or translate your output into your partner's language.",
        "memory": "Earlier shared image and your partner's utterance:\n",
        "speak": "Describe both objects' colors, shapes, and relative positions in this image "
                 "in one short English sentence. If your partner spoke, you may respond while still "
                 "describing this same image. Output only the sentence.",
        "partner": "Your partner's current utterance:\n",
        "choose": "Your partner's utterance is below. Choose the matching image from the four "
                  "numbered panels (1, 2, 3, 4). Output exactly one digit, with no explanation.\n",
    },
    "ko": {
        "system": "너는 시각 의사소통 실험에 참여하는 한국어 화자다. 한국어로만 말한다. "
                  "공유된 세계를 자기 언어로 설명한다. 둘만의 새 암호를 만들거나 "
                  "상대방의 언어로 번역해서 출력하지 않는다.",
        "memory": "이전에 함께 본 그림과 그때 상대방이 한 말:\n",
        "speak": "이 그림 속 두 물체의 색, 모양, 상대적 위치를 짧은 한국어 한 문장으로 설명해라. "
                 "상대방이 먼저 말했다면 같은 그림을 설명하면서 반응해도 된다. 문장만 출력해라.",
        "partner": "이번 그림에 대해 상대방이 한 말:\n",
        "choose": "아래 상대방의 발화에 맞는 그림을 번호가 붙은 네 칸에서 골라라. "
                  "설명 없이 숫자 1, 2, 3, 4 중 하나만 출력해라.\n",
    },
    "ja": {
        "system": "あなたは視覚的コミュニケーション実験に参加する日本語話者です。 "
                  "日本語だけで話し、共有された世界を自分の言語で説明してください。 "
                  "二人だけの暗号を作ったり、相手の言語に翻訳して出力したりしないでください。",
        "memory": "以前一緒に見た画像と、そのときの相手の発言:\n",
        "speak": "この画像の二つの物体の色、形、相対的な位置を短い日本語の一文で説明してください。 "
                 "相手が先に話した場合も同じ画像を説明してください。一文だけを出力してください。",
        "partner": "今回の画像についての相手の発言:\n",
        "choose": "以下の相手の発言に一致する画像を、番号付きの四つの枠から選んでください。 "
                  "説明せず、数字の1、2、3、4のいずれか一つだけを出力してください。\n",
    },
}
