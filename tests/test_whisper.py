import whisper

model = whisper.load_model("small")  
result = model.transcribe("uploads/sample.mp3")

print(result["text"])

# to test it use python test_whisper.py
