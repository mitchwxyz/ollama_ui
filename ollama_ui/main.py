import ollama
import streamlit as st
from default_parameters import Parameters
from helpers import CSS, HTMLTemplate
from models import Message

# Custom styling
st.html(HTMLTemplate.base_style.substitute(css=CSS.page_style))

# Intialize Streamlit session state
if "app_params" not in st.session_state:
    st.session_state.avatar = "😎"

if "in_mirror" not in st.session_state:
    st.session_state.in_mirror = False

if "model_list" not in st.session_state:
    st.session_state.model_list = []

if "system_msg" not in st.session_state:
    st.session_state.system_msg = Message(role="system", content="")

if "input_params" not in st.session_state:
    st.session_state.input_params = dict()

if "ollama_params" not in st.session_state:
    st.session_state.ollama_params = ollama.Options()

if "messages" not in st.session_state:
    st.session_state.messages = []  # List of Messages


# Parameters modal dialog
@st.dialog("App Settings")
def show_app_params():
    """Open app settings."""
    with st.form("Settings"):
        avatar = st.selectbox("My Avatar", ["😎", "😀", "🤪"])
        mirror_mode = st.toggle(
            "Mirror messages to the console?", value=st.session_state.in_mirror
        )

        if st.form_submit_button("Save"):
            st.session_state.avatar = avatar
            st.session_state.in_mirror = mirror_mode
            st.rerun()


# Custom - show parameters button
if st.button("", icon=":material/format_paint:", type="primary", key="app_css"):
    show_app_params()

# Debug Mode
if st.session_state.in_mirror:
    st.badge("In Debug Mode!", icon="⚠️", color="grey")


# Sidebar Functions
@st.dialog("System Message")
def set_system_msg() -> None:
    """Open a dialog to set the system message."""
    prompt = st.text_area(
        "Input system message", value=st.session_state.system_msg.content
    )
    if st.button("Submit"):
        st.session_state.system_msg.content = prompt
        # Update Message History
        st.session_state.messages.append(st.session_state.system_msg)
        # Debug dump
        if st.session_state.in_mirror:
            st.session_state.system_msg.rich_print()
        st.rerun()


def clear_chat() -> None:
    """Clear the chat history."""
    st.session_state.messages = [
        m for m in st.session_state.messages if m.role == "system"
    ]


def dump_messages() -> None:
    """Print all Messages."""
    for msg in st.session_state.messages:
        msg.rich_print()


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
                max_value=2.5,
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
                min_value=1000,
                max_value=128_000,
                value=st.session_state.input_params["num_ctx"],
                step=1000,
            )
            form_params["num_predict"] = st.slider(
                "num_predict",
                min_value=-1,
                max_value=2048,
                value=st.session_state.input_params["num_predict"],
                step=128,
            )
            form_params["repeat_last_n"] = st.slider(
                "repeat_last_n",
                min_value=128,
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
            # Save to file
            if st.form_submit_button("Save as Default"):
                done = params.update_defaults(
                    selected_model, ollama.Options(**form_params)
                )
                if done:
                    st.toast("Current Parameters written to default file.")
        with col2:
            # Submit to session state
            if (
                st.form_submit_button("Update Model")
                or not st.session_state.ollama_params
            ):
                st.session_state.ollama_params = ollama.Options(**form_params)
    if st.session_state.in_mirror:
        st.button("Dump Messages", on_click=dump_messages)

# Body - chat history display
with st.container(key="chat_history"):
    for history_msg in st.session_state.messages:
        if history_msg.role == "user":
            with st.chat_message(history_msg.role, avatar=st.session_state.avatar):
                st.markdown(history_msg.content)
        elif history_msg.role == "assistant":
            with st.chat_message(
                history_msg.role, avatar=st.session_state.input_params["icon"]
            ):
                if history_msg.reasoning_text:  # Expander if Reasoning Exists
                    with st.expander(
                        f"{history_msg.reasoning_tag or 'think'}", expanded=False
                    ):
                        st.markdown(history_msg.reasoning_text)
                st.markdown(history_msg.main_text)

# User input handling
prompt_text = st.chat_input("Enter a prompt here...", key="prompt_text")
if prompt_text:
    # Add user message
    user_msg = Message(role="user", content=prompt_text)
    st.chat_message("user", avatar=st.session_state.avatar).markdown(prompt_text)
    st.session_state.messages.append(user_msg)

    # Debug dump
    if st.session_state.in_mirror:
        user_msg.rich_print()

    # Generate assistant response
    stream = ollama.chat(
        model=selected_model,
        messages=[
            {"role": m.role, "content": m.content} for m in st.session_state.messages
        ],
        stream=True,
        options=st.session_state.ollama_params,
    )

    # Streaming assistant message display
    with st.container(key="current_response"):
        # Create response message
        assistant_msg = Message(role="assistant")
        with st.chat_message("assistant", avatar=st.session_state.input_params["icon"]):
            # Create placeholders inside the chat message
            reasoning_expander = st.expander("thinking", expanded=True)
            with reasoning_expander:
                thinking_placeholder = st.empty()
            message_placeholder = st.empty()

        # Stream and update placeholders incrementally
        for msg_chunk in assistant_msg.update_from_stream(stream):
            if msg_chunk.in_reasoning:
                thinking_placeholder.markdown(f"{msg_chunk.reasoning_text}▌")
            else:
                message_placeholder.markdown(f"{msg_chunk.main_text}▌")

    # Final update without cursor
    thinking_placeholder.markdown(assistant_msg.reasoning_text or "")
    message_placeholder.markdown(assistant_msg.main_text or "")

    # Append to history
    st.session_state.messages.append(assistant_msg)

    # Debug dump
    if st.session_state.in_mirror:
        assistant_msg.rich_print()
