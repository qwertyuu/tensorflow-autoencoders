import asyncio
 
import websockets
from gpt4all import GPT4All
import json

# /home/raph/.local/share/nomic.ai/GPT4All
# orca-mini-3b.ggmlv3.q4_0.bin
# ggml-model-gpt4all-falcon-q4_0.bin
# wizardlm-13b-v1.1-superhot-8k.ggmlv3.q4_0.bin

# vigogne-2-7b-chat.ggmlv3.q4_0.bin
gpt_model = GPT4All("vigogne-2-7b-chat.ggmlv3.q4_0.bin", model_path="/home/raph/Downloads", device="cpu")

with gpt_model.chat_session("""Vous trouverez ci-dessous une conversation entre un utilisateur et un assistant IA nommée Natasha.
Natasha est aubergiste et vit sur Discord. Elle est là pour aider les gens.
Natasha est joyeuse et amusante, répondant brièvement avec gentillesse.""") as session:
    # create handler for each connection
    
    async def handler(websocket, path):

        while True:
            data = await websocket.recv()

            output = session.generate(data, max_tokens=500, streaming=True)
            #print(session.current_chat_session)

            for o in output:
                await websocket.send(json.dumps({"type": "text", "text": o}))
            await websocket.send(json.dumps({"type": "done"}))

    start_server = websockets.serve(handler, "localhost", 8000)

    el = asyncio.get_event_loop()
    el.run_until_complete(start_server)
    el.run_forever()