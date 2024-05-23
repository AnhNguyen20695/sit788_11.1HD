import os
from dotenv import load_dotenv
import gradio as gr
import base64
import os
import requests
from config import *
import numpy as np

import azure.cognitiveservices.speech as speechsdk


load_dotenv()  

#set max chat history to keep
max_items = 16

# Initialize an empty list for the conversation history with max len
conversation_history = []
chat_conversation_history = []


#Set Headers
headers = {
            "Content-Type": "application/json",
            "api-key": GPT4V_KEY,
         }



# Add the system message to the conversation history , Cutomize the System message to your needs
#Maintain the conversation_history 
conversation_history.append({
    "role": "system",
    "content": [
        {
            "type": "text",
            "text": "You are an AI assistant that helps people find information.Make the URL as markdown hyper links for rendering as hyperlinks example [Gradio Website][1][1]: https://www.gradio.app/"
        }
    ]
})
chat_conversation_history.append({
    "role": "system",
    "content": "You are an AI assistant that helps people find information.Make the URL as markdown hyper links for rendering as hyperlinks example [Gradio Website][1][1]: https://www.gradio.app/"
})

#RAG Pattern Index details where Images are embedded for Retrieval
dataSources = [
            {
                "type": "AzureCognitiveSearch",
                "parameters": {
                    "endpoint": AZ_SEARCH_ENDPOINT,
                    "key": AZ_SEARCH_KEY,
                    "indexName":AZ_IMAGE_SEARCH_INDEX,
                    "semanticConfiguration": "default",
                    "queryType": "semantic",
                    "in_scope": True,
                    "fieldsMapping": {
                                    "vectorFields": "image_vector"
                                },
                }
            }
        ]

text_datasources = [{
    "type": "AzureCognitiveSearch",
    "parameters": {
        "endpoint": AZ_SEARCH_ENDPOINT,
        "key": AZ_SEARCH_KEY,
        "indexName": AZ_DOC_SEARCH_INDEX,
        "semantic_configuration": "default",
        "query_type": "vectorSemanticHybrid",
        "fields_mapping": {
            "content_fields_separator": "\n",
            "content_fields": [
                "content"
            ],
            "filepath_field": "filepath",
            "title_field": "title",
            "url_field": "url",
            "vector_fields": [
                "contentVector"
            ]
        },
        "in_scope": True,
        "strictness": 3,
        "top_n_documents": 5,
        "embedding_dependency": {
            "type": "deployment_name",
            "deployment_name": AZ_EMBEDDING_DEPLOYMENT
        },
    }
}]


#current_dir the avatar image needs to be placed in current_dir
current_dir = os.path.abspath(os.getcwd())

#to keep limited chat history
def keep_latest_n_items(history, n):
    # Keep only the latest n items
    history = history[-n:]
    return history

#to handle like and dislike for chat responses from LLM, boilerplate code to be expanded
def print_like_dislike(x: gr.LikeData):
    print(x.index, x.value, x.liked)


# Image to Base 64 Converter
def convertImageToBase64(image_path):
    with open(image_path, 'rb') as img:
        encoded_string = base64.b64encode(img.read())
    return encoded_string.decode('ascii')

def recognize_from_wav(filename):
    # This example requires environment variables named "SPEECH_KEY" and "SPEECH_REGION"
    speech_config = speechsdk.SpeechConfig(subscription=AZ_SPEECH_TEXT_KEY,
                                           region=AZ_SPEECH_TEXT_REGION)
    speech_config.speech_recognition_language=AZ_SPEECH_LANGUAGE

    audio_config = speechsdk.audio.AudioConfig(filename=filename)
    speech_recognizer = speechsdk.SpeechRecognizer(speech_config=speech_config, audio_config=audio_config)

    speech_recognition_result = speech_recognizer.recognize_once_async().get()

    if speech_recognition_result.reason == speechsdk.ResultReason.RecognizedSpeech:
        # print("Recognized: {}".format(speech_recognition_result.text))
        return "", speech_recognition_result.text
    elif speech_recognition_result.reason == speechsdk.ResultReason.NoMatch:
        return f"No speech could be recognized: {speech_recognition_result.no_match_details}", ""
    elif speech_recognition_result.reason == speechsdk.ResultReason.Canceled:
        cancellation_details = speech_recognition_result.cancellation_details
        # print("Speech Recognition canceled: {}".format(cancellation_details.reason))
        if cancellation_details.reason == speechsdk.CancellationReason.Error:
            print("Error details: {}".format(cancellation_details.error_details))
            print("Did you set the speech resource key and region values?")
            return f"Error details: {cancellation_details.error_details}", ""

def transcribe(audio, status_box):
    if status_box == "available":
        # Switch play
        error, text = recognize_from_wav(audio)
        status = "recording"
        print("Error: ",error)
    else:
        # Clear previous record
        text = ""
        status = "available"
    return status, text

# Function that takes User Inputs and displays it on ChatUI and also maintains the history for chatcompletion
def buildHistoryForUiAndChatCompletion(history,txt,img):
    #if user enters only text
    if not img:
        history += [(txt,None)]
        # Add the user message to the conversation history
        user_message = {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": txt
                    }
                ]
            }
        conversation_history.append(user_message)
        chat_conversation_history.append({
            "role": "user",
            "content": txt
        })
        return history
    #if user enters image and text
    base64 = convertImageToBase64(img)
    data_url = f"data:image/jpeg;base64,{base64}"
    history += [(f"{txt} ![]({data_url})", None)]
    # Add the user message to the conversation history
    user_message = {
        "role": "user",
        "content": [
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{base64}"
                }
                },
            {
                "type": "text",
                "text": txt
            }
        ]
    }
    #append the user message to chat api pay load
    conversation_history.append(user_message)
    chat_conversation_history.append({
            "role": "user",
            "content": txt
        })
    return history

# Function that takes User Inputs, generates Response and displays on Chat UI
def call_AzureOpenAI_Vision_RAG_API(history,text,img):
    if not img:
        print("Chat conv history: ",chat_conversation_history)
        body = {
            "dataSources": text_datasources,
            "messages": chat_conversation_history,
            "max_tokens": 100,
            "temperature": 0,
            "top_p": 1
        }
        
        endpoint = GPT35_ENDPOINT+"openai/deployments/"+GPT35_DEPLOYMENT_NAME+"/extensions/chat/completions?api-version="+GPT35_VERSION
    else:
        body = {
            "dataSources": dataSources,
            "messages": conversation_history,
            "max_tokens": 100,
            "temperature": 0,
            "top_p": 1
        }
        
        endpoint = GPT4V_ENDPOINT+"openai/deployments/"+GPT4V_DEPLOYMENT_NAME+"/extensions/chat/completions?api-version="+GPT4V_VERSION

    #post the API request
    print("Requesting ",endpoint)
    response = requests.post(endpoint, headers=headers, json=body)
    print(response.json())
    #get llm reponse
    if not img:
        messages = response.json()['choices'][0]["messages"]
        txt = ""
        content = ""
        for message in messages:
            if message["role"] == "assistant":
                txt = message["content"]
                content = message["content"]
                break
        assistant_message = {
            "role": "assistant",
            "content": txt
        }
    else:
        assistant_message = {
            "role": "assistant",
            "content": [
                {
                    "type": "text",
                    "text": response.json()['choices'][0]['message']['content']
                }
            ]
        }
        content = response.json()['choices'][0]['message']['content']
 
    #llm response added to history
    conversation_history.append(assistant_message)
    chat_conversation_history.append(assistant_message)
    history += [(None,content)]
    #conversation_history = keep_latest_n_items(conversation_history, 10)
    return history 

# Interface Code
with gr.Blocks(theme=gr.themes.Soft(primary_hue="green", secondary_hue="red")) as app:
    state = gr.State(value="")
    with gr.Row():
        image_box = gr.Image(type="filepath")
        chatbot = gr.Chatbot(
            scale = 2,
            height=750,
            bubble_full_width=False,
            #the avv.png is the bot avatar image this needs to be present else comment
            #avatar_images=(None, (os.path.join(os.path.dirname(current_dir), "avv.png")))
            avatar_images=(None, (os.path.join(os.path.dirname(__file__), "avv.png"))),
        )
    
    with gr.Row():
        with gr.Column():
            text_box = gr.Textbox(
                    placeholder="Enter text and press enter",
                    container=False,
                )
            status_box = gr.Textbox(value="available",visible=False)
        with gr.Column():
            audio_box = gr.Audio(label="Audio",
                                 sources="microphone",
                                 type="filepath",
                                 elem_id='audio',
                                 format='wav',
                                 max_length=10)

            audio_box.change(transcribe,
                             inputs=[audio_box, status_box],
                             outputs=[status_box, text_box])


    #btnupd = gr.UploadButton("📁", file_types=["image", "video", "audio"],type="filepath")
    btn = gr.Button("Submit")
    clicked = btn.click(buildHistoryForUiAndChatCompletion,
                        [chatbot,text_box,image_box],
                        chatbot
                        ).then(call_AzureOpenAI_Vision_RAG_API,
                                [chatbot,text_box,image_box],
                                chatbot
                                )
    chatbot.like(print_like_dislike, None, None)

app.queue()
app.launch(debug=True)
