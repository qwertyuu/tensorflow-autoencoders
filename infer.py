import base64
from io import BytesIO
import os
from flask import Flask, request, jsonify

from pydub import AudioSegment
import torch
import torchaudio
import torchaudio.sox_effects as ta_sox
from transformers import AutoModelForCTC, Wav2Vec2Processor
from num2words import num2words
import re


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model = AutoModelForCTC.from_pretrained("bhuang/asr-wav2vec2-french").to(device)
processor = Wav2Vec2Processor.from_pretrained("bhuang/asr-wav2vec2-french")
model_sample_rate = processor.feature_extractor.sampling_rate

app = Flask(__name__)

@app.route('/api/stt', methods=['POST'])
def speech_to_text():
    try:
        # Get the JSON data from the request
        data = request.json

        # Check if the 'audio_data' field exists in the JSON data
        if 'audio_data' not in data:
            return jsonify({'error': 'Missing audio_data field in JSON'}), 400

        # Decode the base64 audio data and convert it to a Wav file
        audio_data = base64.b64decode(data['audio_data'])
        audio_file = BytesIO(audio_data)
        audio = AudioSegment.from_file(audio_file)

        # Save the audio to a temporary file
        temp_audio_file = 'temp_audio.mp3'
        audio.export(temp_audio_file, format='wav')

        audio_path = temp_audio_file  # path to your audio file
        waveform, sample_rate = torchaudio.load(audio_path)

        effects = []
        if model_sample_rate != sample_rate:
            # resample
            effects.append(["rate", f"{model_sample_rate}"])
        if waveform.shape[0] > 1:
            # convert to mono
            effects.append(["channels", "1"])
        if len(effects) > 0:
            converted_waveform, _ = ta_sox.apply_effects_tensor(waveform, sample_rate, effects)

        # 1d array
        converted_waveform = converted_waveform.squeeze(axis=0)

        # normalize
        input_dict = processor(converted_waveform, sampling_rate=model_sample_rate, return_tensors="pt")

        with torch.inference_mode():
            # forward
            logits = model(input_dict.input_values.to(device)).logits

        # decode
        predicted_ids = torch.argmax(logits, dim=-1)
        predicted_sentence = processor.batch_decode(predicted_ids)[0]

        print(predicted_sentence)

        # Clean up temporary audio file
        os.remove(temp_audio_file)

        return jsonify({'trs': predicted_sentence})

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/num2words', methods=['POST'])
def numtowords():
    try:
        # Get the JSON data from the request
        data = request.json

        # Check if the 'audio_data' field exists in the JSON data
        if 'text' not in data:
            return jsonify({'error': 'Missing text field in JSON'}), 400

        text = data['text']

        # regex extract numbers from text variable
        numbers = re.findall(r'\d+', text)
        for n in numbers:
            text = text.replace(n, num2words(n, lang='fr'))

        return jsonify({'text': text})

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)