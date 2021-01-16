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
import torch.nn.functional as F
import boto3

# Create SQS client
sqs = boto3.client('sqs', region_name='eu-west-2')

queue_url = 'https://sqs.eu-west-2.amazonaws.com/665859541710/TrumpBotQueue'

#purge current queue
sqs.purge_queue(
    QueueUrl=queue_url
)

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

model.load_state_dict(torch.load(buffer))

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

temperature = 0.6

def gen_tweet():
    with torch.no_grad():
        tweet_finished = False

        cur_ids = torch.tensor(tokenizer.encode(f"SENTIMENT: NEGATIVE SUBJECT: {random.choice(WORDS)} TWEET:")).unsqueeze(0).to(device)

        for i in range(100):
            outputs = model(cur_ids)
            next_token_logits = outputs[0][:, -1, :] / (temperature if temperature > 0 else 1.)
            next_token = torch.multinomial(F.softmax(next_token_logits, dim=-1), num_samples=1)
            cur_ids = torch.cat((cur_ids, next_token), dim=1)

            if next_token in tokenizer.encode('<|endoftext|>'):
                break
        
    output_list = list(cur_ids.squeeze().to('cpu').numpy())
    output_text = tokenizer.decode(output_list).split("TWEET:")[-1]
    output_text = re.sub(r"http\S+", "", output_text).replace('<|endoftext|>', '')

    if(len(output_text) > 140 or len(output_text) == 0):
        return gen_tweet()

    return(output_text)


if __name__ == '__main__':
    for i in range(24):
        sqs.send_message(
            QueueUrl=queue_url,
            MessageBody=(gen_tweet())
        )