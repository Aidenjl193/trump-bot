import boto3
import os

ARN = os.environ['ARN']
KEY_NAME = os.environ['KEY_NAME']

ec2 = boto3.client('ec2')

#https://medium.com/appgambit/aws-lambda-launch-ec2-instances-40d32d93fb58
#https://medium.com/better-programming/cron-job-patterns-in-aws-126fbf54a276


def lambda_handler(event, context):
    init_script = """#!/bin/sh
                git clone https://github.com/Aidenjl193/trump-bot.git
                cd ./trump-bot
                pip install -r requirements.txt
                python trump-bot.py
                aws ec2 terminate-instance --instance-ids `curl http://169.254.169.254/latest/meta-data/instance-id`"""

    ec2.request_spot_instances(
        SpotPrice=0.2,
        InstanceCount=1,
        Type='one-time',
        LaunchSpecification={
            'ImageId': 'ami-088ae826a6b31fd2c',
            'KeyName': KEY_NAME,
            'InstanceType': 'g4dn.xlarge',
            'UserData': init_script,
            'IamInstanceProfile': {
                'Arn': ARN #make sure this role has terminate EC2 instance access
            }
        }
    )