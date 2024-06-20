import streamlit as st
from streamlit_chat import message
import requests
 
API_URL = "https://api-inference.huggingface.co/models/facebook/blenderbot-400M-distill" # MLP-KTLim/llama-3-Korean-Bllossom-8B
API_TOKEN = "api key "
headers = {"Authorization": f"Bearer {API_TOKEN}"}
 
st.header("🤖Daegeon's BlenderBot (Demo)")
st.markdown("[Be Original](https://github.com/ingini/)")
 
if 'generated' not in st.session_state:
    st.session_state['generated'] = False
 
if 'past' not in st.session_state:
    st.session_state['past'] = []
 
def query(payload):
	response = requests.post(API_URL, headers=headers, json=payload)
	return response.json()
 
with st.form('form', clear_on_submit=True):
    user_input = st.text_input('You: ', '', key='input')
    submitted = st.form_submit_button('Send')
 
if submitted and user_input:
    output = query({
        "inputs": {
            "past_user_inputs": st.session_state.past,
            "generated_responses": st.session_state.generated,
            "text": user_input,
        },
        "parameters": {"repetition_penalty": 1.33},
    })
 
    st.session_state.past.append(user_input)
    st.session_state.generated.append(output["generated_text"])
 
if st.session_state['generated']:
    for i in range(len(st.session_state['generated'])-1, -1, -1):
        message(st.session_state['past'][i], is_user=True, key=str(i) + '_user')
        message(st.session_state["generated"][i], key=str(i))