import boto3
import os
import base64

ARN = os.environ['ARN']
KEY_NAME = os.environ['KEY_NAME']

ec2 = boto3.client('ec2')

#https://medium.com/appgambit/aws-lambda-launch-ec2-instances-40d32d93fb58
#https://medium.com/better-programming/cron-job-patterns-in-aws-126fbf54a276


def lambda_handler(event, context):
    init_script = u"""#!/bin/sh
                git clone https://github.com/Aidenjl193/trump-bot.git
                cd ./trump-bot
                pip install -r requirements.txt
                python trump-bot.py
                aws ec2 terminate-instance --instance-ids `curl http://169.254.169.254/latest/meta-data/instance-id`"""

    ec2.request_spot_instances(
        SpotPrice='0.2',
        InstanceCount=1,
        Type='one-time',
        LaunchSpecification={
            'ImageId': 'ami-02e86b825fe559330',
            'KeyName': KEY_NAME,
            'InstanceType': 'g4dn.xlarge',
            'UserData': base64.encodestring(init_script.encode('utf-8')).decode('ascii'),
            'IamInstanceProfile': {
                'Arn': ARN #make sure this role has terminate EC2 instance access
            }
        }
    )