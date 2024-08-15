import os
import ollama
import time
import streamlit as st
from httpx import ConnectError

def start_server_macos():
    os.system('osascript -e \'tell application "Terminal" to do script "ollama serve"\'')

# Side Bar - Input Model and Parameters
with st.sidebar:
    st.title("Ollama")

    try:
        model_list = ollama.list()["models"]
        st.success("Connected!", icon="✅")
        selected_model = st.selectbox("Model",
                            options=[model["name"] for model in model_list],
                            index=1,
                            key="selected_model")

    except ConnectError:
        selected_model = None
        if st.button("Start Server in Terminal", on_click=start_server_macos):
            # Wait for server to start and rerun fragment
            with st.spinner("Server Starting"):
                time.sleep(2)
            st.rerun()
        st.warning("Not Connected", icon="❌")

    st.subheader("Parameters")

    sb_params = {}
    with st.expander("Temperature"):
        sb_params["temperature"] = st.slider('temperature', min_value=0.1, max_value=2.0, value=0.6, step=0.1, help="The temperature of the model. Increasing the temperature will make the model answer more creatively.")
        sb_params["top_k"] = st.slider('top_k', min_value=1, max_value=100, value=40, step=1, help="Reduces the probability of generating nonsense. A higher value (e.g. 100) will give more diverse answers, while a lower value (e.g. 10) will be more conservative.")
        sb_params["top_p"] = st.slider('top_p', min_value=0.01, max_value=1.0, value=0.9, step=0.01, help="Works together with top-k. A higher value (e.g., 0.95) will lead to more diverse text, while a lower value (e.g., 0.5) will generate more focused and conservative text.")
        sb_params["min_p"] = st.slider('min_p', min_value=0.01, max_value=sb_params["top_p"], value=0.1, step=0.01)
        sb_params["typical_p"] = st.slider('typical_p', min_value=sb_params["min_p"], max_value=sb_params["top_p"], value=0.75, step=0.01)

    with st.expander("Context"):
        sb_params["num_predict"] = st.slider('num_predict', min_value=-1, max_value=512, value=128, step=1)
        sb_params["repeat_last_"] = st.slider('repeat_last_n', min_value=512, max_value=4096, value=2048, step=128)
        sb_params["repeat_penalt"] = st.slider('repeat_penalty', min_value=0.1, max_value=2.0, value=1.18, step=0.01)

    with st.expander("Perplexity"):
        sb_params["mirostat"] = st.select_slider("mirostat", options=[0, 1, 2], value=1, help="Enable Mirostat sampling for controlling perplexity. (0 = disabled, 1 = Mirostat, 2 = Mirostat 2.0)")
        sb_params["mirostat_eta"] = st.slider("mirostat_eta", min_value=0.0, max_value=1.0, value=0.10, step=0.05, help="Influences how quickly the algorithm responds to feedback from the generated text. A lower learning rate will result in slower adjustments, while a higher learning rate will make the algorithm more responsive.")
        sb_params["mirostat_tau"] = st.slider("mirostat_tau", min_value=0.0, max_value=10.0, value=4.0, step=0.5, help="Controls the balance between coherence and diversity of the output. A lower value will result in more focused and coherent text.")

    # Compile Parameters
    params = ollama.Options(**sb_params)


# Stop if no model is active
if not selected_model:
    st.stop()

# Intialize messages Session State
if "messages" not in st.session_state:
    st.session_state.messages = []


# Display Message History
for msg_id, message in enumerate(st.session_state.messages):
    if message["role"] == "assistant":
        st.chat_message(message["role"], avatar="🤖").markdown(message["content"])

    elif message["role"] == "user":
        st.chat_message(message["role"], avatar="😎").markdown(message["content"])

    else:
        st.write(message["content"])

# Message Input Box
if prompt := st.chat_input("Enter a prompt here..."):
    try:
        st.session_state.messages.append(
            {"role": "user",
                "content": prompt})

        st.chat_message("user", avatar="😎").markdown(prompt)

        with st.chat_message("assistant", avatar="🤖"):
            message_placeholder = st.empty()
            full_response = ""
            
            stream = ollama.chat(
                model=selected_model,
                messages=[
                    {"role": m["role"], "content": m["content"]}
                    for m in st.session_state.messages
                ],
                stream=True,
                options=params,
                format=""
            )

            for chunk in stream:
                if 'message' in chunk:
                    content = chunk['message']['content']
                    full_response += content
                    message_placeholder.markdown(full_response + "▌")

            message_placeholder.markdown(full_response)
        
        st.session_state.messages.append(
            {"role": "assistant", "content": full_response})

    except Exception as e:
        st.error(e, icon="⛔️")
