import os
import random
import json
from pathlib import Path
import tweepy
import numpy as np
import torch
import re
from transformers import GPT2Tokenizer, GPT2LMHeadModel
import io

ROOT = Path(__file__).resolve().parents[0]

chunk_count = 1

device = 'cpu'
if torch.cuda.is_available():
    device = 'cuda'

model = GPT2LMHeadModel.from_pretrained('gpt2-medium')
model = model.to(device)

#join model files together
for file in os.listdir(ROOT):
    if file.startswith("model_part_"):
        chunk_count += 1

buffer = io.BytesIO()

for i in range(1, chunk_count):
    with open(f"model_part_{i}.bin", 'rb') as f:
        buffer.write(f.read())
buffer.seek(0)

model.load_state_dict(torch.load(buffer, map_location=torch.device('cpu')))

tokenizer = GPT2Tokenizer.from_pretrained('gpt2-medium')

model.eval()
    


consumer_key = os.getenv("TWITTER_CONSUMER_KEY")
consumer_secret = os.getenv("TWITTER_CONSUMER_SECRET")
access_token = os.getenv("TWITTER_ACCESS_TOKEN")
access_token_secret = os.getenv("TWITTER_ACCESS_TOKEN_SECRET")

auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)
api = tweepy.API(auth)

WORDS = open(f"{ROOT}/thoughts.txt").read().splitlines()

def choose_from_top(probs, n=5):
    ind = np.argpartition(probs, -n)[-n:]
    top_prob = probs[ind]
    top_prob = top_prob / np.sum(top_prob) # Normalize
    choice = np.random.choice(n, 1, p = top_prob)
    token_id = ind[choice][0]
    return int(token_id)

def gen_tweet():
    with torch.no_grad():
        tweet_finished = False
        
        cur_ids = torch.tensor(tokenizer.encode(f"SENTIMENT: { 'POSITIVE' if bool(random.getrandbits(1)) else 'NEGATIVE' } SUBJECT: {random.choice(WORDS)} TWEET:")).unsqueeze(0).to(device)

        for i in range(100):
            outputs = model(cur_ids, labels=cur_ids)
            loss, logits = outputs[:2]
            softmax_logits = torch.softmax(logits[0,-1], dim=0) #Take the first(from only one in this case) batch and the last predicted embedding
            if i < 3:
                n = 20
            else:
                n = 3
            next_token_id = choose_from_top(softmax_logits.to('cpu').numpy(), n=n) #Randomly(from the topN probability distribution) select the next word
            cur_ids = torch.cat([cur_ids, torch.ones((1,1)).long().to(device) * next_token_id], dim = 1) # Add the last word to the running sequence

            if next_token_id in tokenizer.encode('<|endoftext|>'):
                tweet_finished = True
                break

        
        if tweet_finished:
            output_list = list(cur_ids.squeeze().to('cpu').numpy())
            output_text = tokenizer.decode(output_list).split("TWEET:")[-1]
            if(len(output_text) > 140):
                gen_tweet()
            output_text = re.sub(r"http\S+", "", output_text).replace('<|endoftext|>', '')
            api.update_status(output_text)


if __name__ == '__main__':
    gen_tweet()