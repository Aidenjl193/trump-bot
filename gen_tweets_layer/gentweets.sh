#! /bin/sh
git clone https://github.com/Aidenjl193/trump-bot.git
cd ./trump-bot
pip install -r requirements.txt
python trump-bot.py
aws ec2 terminate-instance --instance-ids `curl http://169.254.169.254/latest/meta-data/instance-id`