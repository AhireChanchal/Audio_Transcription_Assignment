import os
from typing import List
import aiofiles
import whisper
import torch
from transformers import pipeline
from fastapi import FastAPI, File, UploadFile
import uvicorn

app = FastAPI()

model = whisper.load_model("large-v3")

summarizer = pipeline("summarization")

@app.get("/")
async def root():
    return {"message": "Audio Transcription and Summarization API"}

@app.post("/transcribe")
async def transcribe(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        file_path = f"./{file.filename}"
        async with aiofiles.open(file_path, "wb") as f:
            await f.write(contents)

        result = model.transcribe(file_path)
        transcription = result["text"]

        summary = summarizer(transcription, max_length=100, min_length=30)[0]["summary_text"]

        timestamps = []
        for segment in result["segments"]:
            start = segment["start"]
            end = segment["end"]
            timestamps.append((start, end))

        output_dir = "output"
        os.makedirs(output_dir, exist_ok=True)

        transcription_path = os.path.join(output_dir, f"{file.filename}_transcription.txt")
        with open(transcription_path, "w", encoding="utf-8") as f:
            f.write(transcription)

        summary_path = os.path.join(output_dir, f"{file.filename}_summary.txt")
        with open(summary_path, "w", encoding="utf-8") as f:
            f.write(summary)

        timestamps_path = os.path.join(output_dir, f"{file.filename}_timestamps.txt")
        with open(timestamps_path, "w", encoding="utf-8") as f:
            for start, end in timestamps:
                f.write(f"{start} - {end}\n")

        os.remove(file_path)

        return {
            "transcription": transcription,
            "summary": summary,
            "timestamps": timestamps,
        }

    except Exception as e:
        return {"error": str(e)}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)