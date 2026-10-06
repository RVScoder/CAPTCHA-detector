import streamlit as st
import streamlit.components.v1 as components

from predict import predict, model
from tts import speak_captcha


st.set_page_config(
    page_title="CAPTCHA OCR",
    page_icon="🔐",
    layout="centered"
)


st.markdown(
    """
    <style>

    .main-title {
        text-align: center;
        font-size: 42px;
        font-weight: 700;
        margin-bottom: 5px;
    }

    .subtitle {
        text-align: center;
        color: #777;
        font-size: 17px;
        margin-bottom: 35px;
    }

    .prediction {
        font-size: 42px;
        font-weight: 700;
        letter-spacing: 5px;
        margin: 5px 0 20px 0;
    }

    .confidence {
        font-size: 24px;
        font-weight: 600;
        margin-top: 10px;
        margin-bottom: 25px;
    }

    .character-title {
        font-size: 22px;
        font-weight: 600;
        margin-bottom: 15px;
    }

    .char-confidence {
        font-size: 14px;
    }

    </style>
    """,
    unsafe_allow_html=True
)


st.markdown(
    '<div class="main-title">CAPTCHA OCR</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'CAPTCHA recognition system using CNN + BiLSTM + CTC'
    '</div>',
    unsafe_allow_html=True
)


uploaded_file = st.file_uploader(
    "Upload a CAPTCHA image",
    type=["png", "jpg", "jpeg"],
    help="Upload a CAPTCHA image to recognize its text."
)


if uploaded_file is not None:

    col_left, col_right = st.columns([1, 1.25], gap="large")

    # LEFT SIDE
    with col_left:

        st.subheader("CAPTCHA Image")

        st.image(
            uploaded_file,
            use_container_width=True
        )


    # RIGHT SIDE
    with col_right:

        try:

            with st.spinner("Recognizing CAPTCHA..."):

                prediction, confidence, character_confidences = predict(
                    uploaded_file,
                    model
                )

            st.markdown(
                "### Predicted CAPTCHA"
            )

            # Prediction + copy button
            col_prediction, col_copy = st.columns(
                [3, 1],
                vertical_alignment="center"
            )

            with col_prediction:

                st.markdown(
                    f'<div class="prediction">{prediction}</div>',
                    unsafe_allow_html=True
                )

            with col_copy:

                components.html(
                    f"""
                    <button
                        onclick="navigator.clipboard.writeText('{prediction}')"
                        style="
                            width:100%;
                            padding:9px;
                            border-radius:8px;
                            border:1px solid #ccc;
                            background:white;
                            cursor:pointer;
                            font-size:14px;
                        "
                    >
                        📋 Copy
                    </button>
                    """,
                    height=66
                )


            # Overall confidence
            st.markdown(
                f'<div class="confidence">'
                f'Overall Confidence: <b>{confidence:.2%}</b>'
                f'</div>',
                unsafe_allow_html=True
            )


            # Character confidence
            if character_confidences:

                st.markdown(
                    '<div class="character-title">'
                    'Character Confidence'
                    '</div>',
                    unsafe_allow_html=True
                )

                cols = st.columns(
                    len(character_confidences)
                )

                for i, (char, conf) in enumerate(
                    zip(prediction, character_confidences)
                ):

                    with cols[i]:

                        st.markdown(
                            f'<div class="char-confidence">'
                            f'<b>{char}</b><br>'
                            f'{conf:.1%}'
                            f'</div>',
                            unsafe_allow_html=True
                        )


            st.write("")

            # TTS
            if st.button(
                "🔊 Read CAPTCHA Aloud",
                use_container_width=True
            ):

                try:

                    speak_captcha(prediction)

                    st.success(
                        "CAPTCHA read aloud."
                    )

                except Exception:

                    st.warning(
                        "Text-to-speech is currently unavailable."
                    )


        except Exception as e:

            st.error(
                "Unable to process this image."
            )

            st.exception(e)


else:

    st.info(
        "Upload a CAPTCHA image above to get started."
    )
