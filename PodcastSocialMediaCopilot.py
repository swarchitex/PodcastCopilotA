# The Podcast Copilot will automatically create and post a LinkedIn promotional post for a new episode of the Behind the Tech podcast.  
# Given the audio recording of the episode, the copilot will use a locally-hosted Whisper model to transcribe the audio recording.
# The copilot uses the Dolly 2 model to extract the guest's name from the transcript.
# The copilot uses the Bing Search Grounding API to retrieve a bio for the guest.
# The copilot uses the GPT-4 model in the Azure OpenAI Service to generate a social media blurb for the episode, given the transcript and the guest's bio.
# The copilot uses the DALL-E 2 model to generate an image for the post.
# The copilot calls a LinkedIn plugin to post.

from pydub import AudioSegment
from pydub.silence import split_on_silence
import whisper
import torch
from langchain.chains import TransformChain, LLMChain, SequentialChain
from langchain.chat_models import AzureChatOpenAI
from langchain.llms import HuggingFacePipeline
from langchain.prompts import (
    PromptTemplate,
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    AIMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain.schema import (
    AIMessage,
    HumanMessage,
    SystemMessage
)
import requests
import time
from PIL import Image
from io import BytesIO
import datetime
import json
from dalle_helper import ImageClient

# For Dolly 2
from transformers import AutoTokenizer, TextStreamer
from optimum.onnxruntime import ORTModelForCausalLM
from instruct_pipeline import InstructionTextGenerationPipeline
import onnxruntime as ort
ort.set_default_logger_severity(3)

from GetSecrets import GetBingSearchSecrets
from GetSecrets import GetAzureOpenAISecrets
from GetSecrets import GetBingSearchSecrets
from GetSecrets import GetAzureOpenAI_Dall_E_Secrets
from instruct_pipeline import InstructionTextGenerationPipeline
from transformers import AutoModelForCausalLM, AutoTokenizer

print("Imports are complete")


# Endpoint Settings
bing_subscription_key, bing_search_url = GetBingSearchSecrets()
bing_search_url = bing_search_url+"v7.0/search"

openai_api_type = "azure"
openai_api_key , openai_api_base, gpt4_deployment_name= GetAzureOpenAISecrets()

gpt4_endpoint = openai_api_base
gpt4_api_key = openai_api_key 
plugin_model_url = openai_api_base
plugin_model_api_key = openai_api_key 

dalle_api_type = "azure"
dalle_api_key,dalle_endpoint = GetAzureOpenAI_Dall_E_Secrets()
#dalle_endpoint='https://est-openai-copilot-podcast.openai.azure.com/'
dalle_api_version="2023-06-01-preview"

# Inputs about the podcast
podcast_url = "https://www.microsoft.com/behind-the-tech"
podcast_audio_file = ".\\PodcastSnippet.mp3"


# Step 1 - Call Whisper to transcribe audio
print("\nCalling Whisper to transcribe audio...") 

# Chunk up the audio file 
sound_file = AudioSegment.from_mp3(podcast_audio_file)
# if we get file not found exception, it may be because FFMPeg is missing from the system: https://github.com/jiaaro/pydub/issues/62
audio_chunks = split_on_silence(sound_file, min_silence_len=1000, silence_thresh=-40 )
count = len(audio_chunks)
print("Audio split into " + str(count) + " audio chunks")

# Call Whisper to transcribe audio
model = whisper.load_model("base")
transcript = ""
for i, chunk in enumerate(audio_chunks):
    # If you have a long audio file, you can enable this to only run for a subset of chunks
    if i < 10 or i > count - 10:
        out_file = "chunk{0}.wav".format(i)
        print("Exporting", out_file)
        chunk.export(out_file, format="wav")
        result = model.transcribe(out_file)
        transcriptChunk:str = "" if result==None else str.strip( result["text"] )
        print(transcriptChunk)
        
        # Append transcript in memory if you have sufficient memory
        if transcriptChunk != None: transcript += ' ' + transcriptChunk 

        # Alternatively, here's how to write the transcript to disk if you have memory constraints
        #textfile = open("chunk{0}.txt".format(i), "w")
        #textfile.write(transcript)
        #textfile.close()
        #print("Exported chunk{0}.txt".format(i))

print("Transcript: \n")
print(transcript)
print("\n")

#Si hay un error ssl whatever puede ser porque hay muchos requests y parece la librería no es thread-safe. 
#Cuando lo ejecuto en una conexión que no es tan rápida, no da error. El problema es que cuando baja muy 
#rápido, no tiene tiempo de liberar los buffers, o liberar instancias o algo asi.
#exactamente igual le ocurre a Edge cuando está bajando un archivo muy grande y falla aleatoriamente
#https://huggingface.co/microsoft/dolly-v2-7b-olive-optimized
#https://github.com/huggingface/optimum/blob/a6951c17c3450e1dea99617aa842334f4e904392/optimum/onnxruntime/modeling_decoder.py#L623

# Step 2 - Make a call to a local Dolly 2.0 model optimized for Windows to extract the name of who I'm interviewing from the transcript
print("Calling a local Dolly 2.0 model optimized for Windows to extract the name of the podcast guest...\n")
def get_guest_name(transcript:str, return_fake_result:bool):
    if return_fake_result:
        return "Neil deGrasse Tyson"
    
    repo_id = "databricks/dolly-v2-3b" #hell no ... out of memory 13gb repo_id = "microsoft/dolly-v2-7b-olive-optimized"
tokenizer = AutoTokenizer.from_pretrained(repo_id, padding_side="left")
  
    #este solo si esta corrupto el modelo... model = ORTModelForCausalLM.from_pretrained(repo_id, provider="DmlExecutionProvider", force_download=True, resume_download=False, use_merged=True, use_io_binding=False)
    #model = ORTModelForCausalLM.from_pretrained(repo_id, provider="DmlExecutionProvider", use_cache=True, use_merged=True, use_io_binding=False)
    model = AutoModelForCausalLM.from_pretrained("databricks/dolly-v2-3b", device_map="auto", torch_dtype=torch.bfloat16)
    streamer = TextStreamer(tokenizer, skip_prompt=True) # type: ignore
generate_text = InstructionTextGenerationPipeline(model=model, streamer=streamer, tokenizer=tokenizer, max_new_tokens=128, return_full_text=True, task="text-generation")
hf_pipeline = HuggingFacePipeline(pipeline=generate_text)
    
dolly2_prompt = PromptTemplate(
    input_variables=["transcript"],
    template="Extract the guest name on the Beyond the Tech podcast from the following transcript.  Beyond the Tech is hosted by Kevin Scott and Christina Warren, so they will never be the guests.  \n\n Transcript: {transcript}\n\n Host name: Kevin Scott\n\n Guest name: "
)

extract_llm_chain = LLMChain(llm=hf_pipeline, prompt=dolly2_prompt, output_key="guest")
    resp:str = extract_llm_chain.predict(transcript=transcript)
    return resp

guest = get_guest_name(transcript, return_fake_result = True)
print(f"Guest: {guest}\n")


# Step 3 - Make a call to the Bing Search Grounding API to retrieve a bio for the guest
def bing_grounding(input_dict:dict) -> dict:
    print("Calling Bing Search API to get bio for guest...")
    search_term = input_dict["guest"]
    print("Search term is " + search_term)

    headers = {"Ocp-Apim-Subscription-Key": bing_subscription_key}
    params = {"q": search_term, "textDecorations": True, "textFormat": "HTML","market":"en-US"}
    response = requests.get(bing_search_url, headers=headers, params=params)
    response.raise_for_status()
    search_results = response.json()
    #print(search_results)

    # Parse out a bio.  
    bio = search_results["entities"]["value"][0]["description"]

    return {"bio": bio}

bing_chain = TransformChain(input_variables=["guest"], output_variables=["bio"], transform=bing_grounding, atransform=None)
bio = bing_chain.run(guest)


# Step 4 - Put bio in the prompt with the transcript
system_template="You are a helpful large language model that can create a LinkedIn promo blurb for episodes of the podcast Behind the Tech, when given transcripts of the podcasts.  The Behind the Tech podcast is hosted by Kevin Scott.\n"
system_message_prompt = SystemMessagePromptTemplate.from_template(system_template)

user_prompt=PromptTemplate(
    template="Create a short summary of this podcast episode that would be appropriate to post on LinkedIn to promote the podcast episode.  The post should be from the first-person perspective of Kevin Scott, who hosts the podcast.\n" +
            "Here is the transcript of the podcast episode: {transcript} \n" +
            "Here is the bio of the guest: {bio} \n",
    input_variables=["transcript", "bio"],
)
human_message_prompt = HumanMessagePromptTemplate(prompt=user_prompt)
chat_prompt = ChatPromptTemplate.from_messages([system_message_prompt, human_message_prompt])

# Get formatted messages for the chat completion
blurb_messages = chat_prompt.format_prompt(transcript={transcript}, bio={bio}).to_messages()


# Step 5 - Make a call to Azure OpenAI Service to get a social media blurb, 
print("Calling GPT-4 model on Azure OpenAI Service to get a social media blurb...\n")
gpt4 = AzureChatOpenAI(
    openai_api_base=gpt4_endpoint,
    openai_api_version="2023-03-15-preview",
    deployment_name=gpt4_deployment_name,
    openai_api_key=gpt4_api_key,
    openai_api_type = openai_api_type,
)
#print(gpt4)   #shows parameters

output = gpt4(blurb_messages)
social_media_copy = output.content

gpt4_chain = LLMChain(llm=gpt4, prompt=chat_prompt, output_key="social_media_copy")

print("Social Media Copy:\n")
print(social_media_copy)
print("\n")


# Step 6 - Use GPT-4 to generate a DALL-E prompt
system_template="You are a helpful large language model that generates DALL-E prompts, that when given to the DALL-E model can generate beautiful high-quality images to use in social media posts about a podcast on technology.  Good DALL-E prompts will contain mention of related objects, and will not contain people or words.  Good DALL-E prompts should include a reference to podcasting along with items from the domain of the podcast guest.\n"
system_message_prompt = SystemMessagePromptTemplate.from_template(system_template)

user_prompt=PromptTemplate(
    template="Create a DALL-E prompt to create an image to post along with this social media text: {social_media_copy}",
    input_variables=["social_media_copy"],
)
human_message_prompt = HumanMessagePromptTemplate(prompt=user_prompt)
chat_prompt = ChatPromptTemplate.from_messages([system_message_prompt, human_message_prompt])

# Get formatted messages for the chat completion
dalle_messages = chat_prompt.format_prompt(social_media_copy={social_media_copy}).to_messages()

# Call Azure OpenAI Service to get a DALL-E prompt 
print("Calling GPT-4 model on Azure OpenAI Service to get a DALL-E prompt...\n")
gpt4 = AzureChatOpenAI(
    openai_api_base=gpt4_endpoint,
    openai_api_version="2023-03-15-preview",
    deployment_name=gpt4_deployment_name,
    openai_api_key=gpt4_api_key,
    openai_api_type = openai_api_type,
)
#print(gpt4)   #shows parameters

output = gpt4(dalle_messages)
dalle_prompt = output.content

dalle_prompt_chain = LLMChain(llm=gpt4, prompt=chat_prompt, output_key="dalle_prompt")

print("DALL-E Prompt:\n")
print(dalle_prompt)
print("\n")


# For the demo, we showed the step by step execution of each chain above, but you can also run the entire chain in one step.
# You can uncomment and run the following code for an example.  Feel free to substitute your own transcript.
'''
transcript = "Hello, and welcome to Beyond the Tech podcast.  I am your host, Kevin Scott.  I am the CTO of Microsoft.  I am joined today by an amazing guest, Lionel Messi.  Messi is an accomplished soccer player for the Paris Saint-Germain football club.  Lionel, how are you doing today?"

podcast_copilot_chain = SequentialChain(
    chains=[extract_llm_chain, bing_chain, gpt4_chain, dalle_prompt_chain],
    input_variables=["transcript"],
    output_variables=["guest", "bio", "social_media_copy", "dalle_prompt"],
    verbose=True)
podcast_copilot = podcast_copilot_chain({"transcript":transcript})
print(podcast_copilot)		# This is helpful for debugging.  
social_media_copy = podcast_copilot["social_media_copy"]
dalle_prompt = podcast_copilot["dalle_prompt"]

print("Social Media Copy:\n")
print(social_media_copy)
print("\n")
'''


# Append "high-quality digital art" to the generated DALL-E prompt
dalle_prompt = dalle_prompt + ", high-quality digital art"


# Step 7 - Make a call to DALL-E model on the Azure OpenAI Service to generate an image 
print("Calling DALL-E model on Azure OpenAI Service to get an image for social media...")

def old_version_unused_generate_image () :
    client = ImageClient(dalle_endpoint, dalle_api_key, verbose=True) # Establish the client class instance; change verbose to True for including debug print statements
    imageURL, postImage =  client.generateImage("A vintage microphone entwined with a spiral galaxy, resting on a book titled 'The Language of the Cosmos', with a laptop in the background displaying code, all under a canopy of shining stars.")
    print(f"Image URL: {imageURL}\n")

#import requests
import openai
openai.api_type = dalle_api_type
openai.api_base = dalle_endpoint
openai.api_version = dalle_api_version
openai.api_key = dalle_api_key

response = openai.Image.create(
    prompt=dalle_prompt,
    size='1024x1024',
    n=1
)
if response!=None and response.__contains__("status") and response.__contains__("data") and response["status"] == "success" and response["data"] != None:
    imageURL = response["data"][0]["url"] # type: ignore
else: imageURL=''
print(f"Image URL: {imageURL}\n")
dalle_image_response = requests.get(imageURL)

if(True and dalle_image_response != None and dalle_image_response.content != None) :
        stream = BytesIO(dalle_image_response.content)
        image = Image.open(stream).convert("RGB") # type: ignore
        stream.close() # type: ignore
        photo_path = ".\\PostImage.jpg"
        image.save(photo_path) # type: ignore
        print(f"Image: saved to {photo_path}\n")


# Append the podcast URL to the generated social media copy
social_media_copy = social_media_copy + " " + podcast_url
print(social_media_copy)
exit()

# Step 8 - Call the LinkedIn Plugin for Copilots to do the post.
# Currently there is not support in the SDK for the plugin model on Azure OpenAI, so we are using the REST API directly.  
PROMPT_MESSAGES = [
    {
        "role": "system",
        "content": "You are a helpful large language model that can post a LinkedIn promo blurb for episodes of Behind the Tech with Kevin Scott, when given some text and a link to an image.\n",
    },
    {
        "role": "user",
        "content": 
            "Post the following social media text to LinkedIn to promote my latest podcast episode: \n" +
            "Here is the text to post: \n" + social_media_copy + "\n" +
            "Here is a link to the image that should be included with the post: \n" + imageURL + "\n",
    }, 
]

print("Calling GPT-4 model with plugin support on Azure OpenAI Service to post to LinkedIn...\n")

payload = {
    "messages": PROMPT_MESSAGES,
    "max_tokens": 1024,
    "temperature": 0.5,
    "n": 1,
    "stop": None
}

headers = {
    "Content-Type": "application/json",
    "api-key": plugin_model_api_key,
}

# Confirm whether it is okay to post, to follow Responsible AI best practices
print("The following will be posted to LinkedIn:\n")
print(social_media_copy + "\n")
confirm = input("Do you want to post this to LinkedIn? (y/n): ")
if confirm == "y":
    # Call a model with plugin support.
    response = requests.post(plugin_model_url, headers=headers, data=json.dumps(payload))
    
    #print (type(response))
    print("Response:\n")
    print(response)
    print("Headers:\n")
    print(response.headers)
    print("Json:\n")
    print(response.json())
    
    response_dict = response.json()
    print(response_dict["choices"][0]["messages"][-1]["content"])
    
# To use plugins, you must call a model that understands how to leverage them.  Support for plugins is in limited private preview
# for the Azure OpenAI service, and a LinkedIn plugin is coming soon!


