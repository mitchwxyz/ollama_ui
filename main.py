import ollama
import streamlit as st

from httpx import ConnectError

from default_parameters import get_defaults


# Intialize  Session State vars
if "model_list" not in st.session_state:
    st.session_state.model_list = []

if "system_msg" not in st.session_state:
    st.session_state.system_msg = {"role": "system", "content": ""}

if "input_params" not in st.session_state:
    st.session_state.input_params = get_defaults("")

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
        except ConnectError:
            st.session_state.model_list = []
            st.warning("Not Connected", icon="❌")
    else:
        st.success("Ollama Connected", icon="✅")

    # Model
    selected_model = st.selectbox(
        "Model",
        options=[m["name"] for m in st.session_state.model_list if isinstance(m, dict)],
    )
    # Update Parameters
    if selected_model:
        st.session_state.input_params = get_defaults(selected_model)

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
                min_value=0.1,
                max_value=2.0,
                value=st.session_state.input_params["TEMPERATURE"],
                step=0.1,
                help="The temperature of the model. Increasing the temperature will make the model answer more creatively.",
            )
            form_params["top_k"] = st.slider(
                "top_k",
                min_value=1,
                max_value=100,
                value=st.session_state.input_params["TOP_K"],
                step=1,
                help="Reduces the probability of generating nonsense. A higher value (e.g. 100) will give more diverse answers, while a lower value (e.g. 10) will be more conservative.",
            )
            form_params["top_p"] = st.slider(
                "top_p",
                min_value=0.01,
                max_value=1.0,
                value=st.session_state.input_params["TOP_P"],
                step=0.01,
                help="Works together with top-k. A higher value (e.g., 0.95) will lead to more diverse text, while a lower value (e.g., 0.5) will generate more focused and conservative text.",
            )
            form_params["min_p"] = st.slider(
                "min_p",
                min_value=0.01,
                max_value=form_params["top_p"],
                value=st.session_state.input_params["MIN_P"],
                step=0.01,
            )
            form_params["typical_p"] = st.slider(
                "typical_p",
                min_value=form_params["min_p"],
                max_value=form_params["top_p"],
                value=st.session_state.input_params["TYPICAL_P"],
                step=0.01,
            )

        with st.expander("Context"):
            form_params["num_ctx"] = st.slider(
                "num_ctx",
                min_value=1024,
                max_value=16384,
                value=st.session_state.input_params["NUM_CTX"],
                step=1,
            )
            form_params["num_predict"] = st.slider(
                "num_predict",
                min_value=-1,
                max_value=512,
                value=st.session_state.input_params["NUM_PREDICT"],
                step=1,
            )
            form_params["repeat_last_n"] = st.slider(
                "repeat_last_n",
                min_value=512,
                max_value=4096,
                value=st.session_state.input_params["REPEAT_LAST_N"],
                step=128,
            )
            form_params["repeat_penalty"] = st.slider(
                "repeat_penalty",
                min_value=0.1,
                max_value=2.0,
                value=st.session_state.input_params["REPEAT_PENALTY"],
                step=0.01,
            )

        with st.expander("Perplexity"):
            form_params["mirostat"] = st.select_slider(
                "mirostat",
                options=[0, 1, 2],
                value=st.session_state.input_params["MIROSTAT"],
                help="Enable Mirostat sampling for controlling perplexity. (0 = disabled, 1 = Mirostat, 2 = Mirostat 2.0)",
            )
            form_params["mirostat_eta"] = st.slider(
                "mirostat_eta",
                min_value=0.0,
                max_value=1.0,
                value=st.session_state.input_params["MIROSTAT_ETA"],
                step=0.05,
                help="Influences how quickly the algorithm responds to feedback from the generated text. A lower learning rate will result in slower adjustments, while a higher learning rate will make the algorithm more responsive.",
            )
            form_params["mirostat_tau"] = st.slider(
                "mirostat_tau",
                min_value=0.0,
                max_value=10.0,
                value=st.session_state.input_params["MIROSTAT_TAU"],
                step=0.5,
                help="Controls the balance between coherence and diversity of the output. A lower value will result in more focused and coherent text.",
            )

        # Submit to session state
        if st.form_submit_button("Save") or not st.session_state.ollama_params:
            st.session_state.ollama_params = ollama.Options(**form_params)


# Stop if no model is active
if not selected_model:
    st.stop()

# Body
for msg_id, message in enumerate(st.session_state.messages):
    if message["role"] == "assistant":
        st.chat_message(
            message["role"], avatar=st.session_state.input_params["ICON"]
        ).markdown(message["content"])
    elif message["role"] == "user":
        st.chat_message(message["role"], avatar="😎").markdown(message["content"])


prompt_text = st.chat_input("Enter a prompt here...", key="prompt_text")
if prompt_text:
    # Debug Input Params
    # print(selected_model, st.session_state.ollama_params)
    try:
        st.session_state.messages.append({"role": "user", "content": prompt_text})
        st.chat_message("user", avatar="😎").markdown(prompt_text)

        with st.chat_message("assistant", avatar=st.session_state.input_params["ICON"]):
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

            message_placeholder.markdown(full_response)

        st.session_state.messages.append(
            {"role": "assistant", "content": full_response}
        )

    except Exception as e:
        st.error(e, icon="⛔️")
