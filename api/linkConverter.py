import json
from python.pytube import YouTube
from google.cloud import speech_v1 as speech
import os
import glob

def linkTotext(event, context):
    link = json.loads(event['body'])['link']
    yt = YouTube(link)
    yt.streams.filter(mime_type="audio/webm", audio_codec="opus")[0].download("/tmp")
    t=glob.glob("/tmp/*.webm")
    t.sort(key=os.path.getmtime,reverse=True)
    input_file = t[0]
    output_file = "/tmp/result.wav"
    client=speech.SpeechClient.from_service_account_file("a.json")
    with open(input_file,"rb") as f:
        mp3_data= f.read()
    audio_file =speech.RecognitionAudio(content=mp3_data)
    config= speech.RecognitionConfig(
        enable_automatic_punctuation=True,
        language_code="en-US",
        audio_channel_count = 2
    )
    response = client.long_running_recognize(
        config=config,
        audio=audio_file
    )
    ra= response.result(timeout=200)
    text=""
    for result in ra.results:
            best_alternative = result.alternatives[0]
            transcript = best_alternative.transcript
            confidence = best_alternative.confidence
            text+=transcript
 
    return {
        "statusCode": 200,
        "body": json.dumps({
            "message": text
        }),
} 
