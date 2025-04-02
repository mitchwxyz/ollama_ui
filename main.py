from string import Template
from textwrap import dedent

import ollama
import streamlit as st

from default_parameters import Parameters


class HTMLTemplate:
    base_style = Template(
        dedent(
            """
            <style>
                $css
            </style>"""
        )
    )


class CSS:
    page_style = """
    .st-key-app_css button {
        border-radius: 25px;
        box-shadow: 3px 5px 10px 0px rgba(128, 128, 128, 0.245);
        position: fixed;
        top: 4rem;
        right: 2rem;
    }
    details {
        color: grey;
    }
    summary {
        color: grey;
        font-weight: bold;
    }
    """


st.html(HTMLTemplate.base_style.substitute(css=CSS.page_style))

# Intialize  Session State vars
if "app_params" not in st.session_state:
    st.session_state.app_params = {"avatar": "😎"}
if "model_list" not in st.session_state:
    st.session_state.model_list = []

if "system_msg" not in st.session_state:
    st.session_state.system_msg = {"role": "system", "content": ""}

if "input_params" not in st.session_state:
    st.session_state.input_params = dict()

if "ollama_parms" not in st.session_state:
    st.session_state.ollama_params = ollama.Options()

if "messages" not in st.session_state:
    st.session_state.messages = []


# Sidebar Functions
@st.dialog("System Message")
def set_system_msg() -> None:
    prompt = st.text_area(
        "Input system message", value=st.session_state.system_msg["content"]
    )
    if st.button("Submit"):
        st.session_state.system_msg = {"role": "system", "content": prompt}
        st.session_state.messages = [
            m for m in st.session_state.messages if m.get("role") != "system"
        ]
        st.session_state.messages.append(st.session_state.system_msg)
        st.rerun()


def clear_chat() -> None:
    st.session_state.messages = [
        m for m in st.session_state.messages if m.get("role") == "system"
    ]


# Sidebar - Model and Parameters
with st.sidebar:
    # Connection
    if not st.session_state.model_list:
        selected_model = None
        try:
            st.session_state.model_list = ollama.list()["models"]
            st.rerun()
        except ConnectionError:
            st.session_state.model_list = []
            st.warning("Not Connected", icon="❌")
            st.toast("Is Ollama running?", icon="👀")
    else:
        st.success("Ollama Connected", icon="✅")

    # Model
    selected_model = st.selectbox(
        "Model",
        options=[m["model"] for m in st.session_state.model_list],
    )
    # Update Parameters
    params = Parameters()
    # Stop if no model selected
    if not selected_model:
        st.stop()
    st.session_state.input_params = params.get_defaults(selected_model)

    st.subheader("LLM Parameters")
    # Set System Message
    st.button("Edit System Message", on_click=set_system_msg)
    # Clear History
    st.button("Clear Chat History", on_click=clear_chat)

    # Parameters
    with st.form("Parameters"):
        form_params = {}
        with st.expander("Temperature"):
            form_params["temperature"] = st.slider(
                "temperature",
                min_value=0.0,
                max_value=2.0,
                value=st.session_state.input_params["temperature"],
                step=0.05,
                help="The temperature of the model. Increasing the temperature will make the model answer more creatively.",
            )
            form_params["top_k"] = st.slider(
                "top_k",
                min_value=0,
                max_value=100,
                value=st.session_state.input_params["top_k"],
                step=1,
                help="Reduces the probability of generating nonsense. A higher value (e.g. 100) will give more diverse answers, while a lower value (e.g. 10) will be more conservative.",
            )
            form_params["top_p"] = st.slider(
                "top_p",
                min_value=0.01,
                max_value=1.0,
                value=st.session_state.input_params["top_p"],
                step=0.01,
                help="Works together with top-k. A higher value (e.g., 0.95) will lead to more diverse text, while a lower value (e.g., 0.5) will generate more focused and conservative text.",
            )
            form_params["typical_p"] = st.slider(
                "typical_p",
                max_value=form_params["top_p"],
                value=st.session_state.input_params["typical_p"],
                step=0.01,
            )

        with st.expander("Context"):
            form_params["num_ctx"] = st.slider(
                "num_ctx",
                min_value=1024,
                max_value=1024 * 48,
                value=st.session_state.input_params["num_ctx"],
                step=1,
            )
            form_params["num_predict"] = st.slider(
                "num_predict",
                min_value=-1,
                max_value=2048,
                value=st.session_state.input_params["num_predict"],
                step=1,
            )
            form_params["repeat_last_n"] = st.slider(
                "repeat_last_n",
                min_value=512,
                max_value=4096,
                value=st.session_state.input_params["repeat_last_n"],
                step=128,
            )
            form_params["repeat_penalty"] = st.slider(
                "repeat_penalty",
                min_value=0.1,
                max_value=2.0,
                value=st.session_state.input_params["repeat_penalty"],
                step=0.01,
            )

        with st.expander("Perplexity"):
            form_params["mirostat"] = st.select_slider(
                "mirostat",
                options=[0, 1, 2],
                value=st.session_state.input_params["mirostat"],
                help="Enable Mirostat sampling for controlling perplexity. (0 = disabled, 1 = Mirostat, 2 = Mirostat 2.0)",
            )
            form_params["mirostat_eta"] = st.slider(
                "mirostat_eta",
                min_value=0.0,
                max_value=1.0,
                value=st.session_state.input_params["mirostat_eta"],
                step=0.05,
                help="Influences how quickly the algorithm responds to feedback from the generated text. A lower learning rate will result in slower adjustments, while a higher learning rate will make the algorithm more responsive.",
            )
            form_params["mirostat_tau"] = st.slider(
                "mirostat_tau",
                min_value=0.0,
                max_value=10.0,
                value=st.session_state.input_params["mirostat_tau"],
                step=0.5,
                help="Controls the balance between coherence and diversity of the output. A lower value will result in more focused and coherent text.",
            )

        col1, col2 = st.columns([1, 1])
        with col1:
            # Submit to session state
            if st.form_submit_button("Update") or not st.session_state.ollama_params:
                st.session_state.ollama_params = ollama.Options(**form_params)
        with col2:
            # Save to file
            if st.form_submit_button("Save Default"):
                done = params.update_defaults(
                    selected_model, ollama.Options(**form_params)
                )
                if done:
                    st.toast("Current Parameters written to default file.")


# General Parameters
@st.dialog("App Settings")
def show_app_params():
    settings_temp = {}
    with st.form("Settings"):
        settings_temp["avatar"] = st.selectbox("My Avatar", ["😎", "😀", "🤪"])

        if st.form_submit_button("Save"):
            st.session_state.app_params = settings_temp
            st.rerun()


# Body
for message in st.session_state.messages:
    if message["role"] == "assistant":
        st.chat_message(
            message["role"], avatar=st.session_state.input_params["icon"]
        ).markdown(message["content"])
    elif message["role"] == "user":
        st.chat_message(
            message["role"], avatar=st.session_state.app_params.get("avatar")
        ).markdown(message["content"])

prompt_text = st.chat_input("Enter a prompt here...", key="prompt_text")
if prompt_text:
    # Debug Input Params
    # print(selected_model, st.session_state.ollama_params)
    try:
        st.session_state.messages.append({"role": "user", "content": prompt_text})
        st.chat_message(
            "user", avatar=st.session_state.app_params.get("avatar")
        ).markdown(prompt_text)

        with st.chat_message("assistant", avatar=st.session_state.input_params["icon"]):
            message_placeholder = st.empty()
            full_response = ""

            stream = ollama.chat(
                model=selected_model,
                messages=[
                    {"role": m["role"], "content": m["content"]}
                    for m in st.session_state.messages
                ],
                stream=True,
                options=st.session_state.ollama_params,
                format="",
            )

            for chunk in stream:
                if "message" in chunk:
                    content = chunk["message"]["content"]
                    full_response += content
                    message_placeholder.markdown(full_response + "▌")

            resonse_r1 = full_response.replace(
                "<think>", "<details><summary>thinking...</summary>"
            ).replace("</think>", "</details>")
            message_placeholder.markdown(resonse_r1, unsafe_allow_html=True)

        st.session_state.messages.append(
            {"role": "assistant", "content": full_response}
        )

    except Exception as e:
        st.error(e, icon="⛔️")

# Custom CSS Elements
if st.button("", icon=":material/format_paint:", type="primary", key="app_css"):
    show_app_params()
