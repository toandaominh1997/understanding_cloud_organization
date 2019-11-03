from knockknock import slack_sender


@slack_sender(webhook_url="https://hooks.slack.com/services/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", channel="pytoan")
def train_your_nicest_model(your_nicest_parameters):
    import time
    time.sleep(10)
    return {'epoch': 10, 'loss': 0.9} # Optional return value


train_your_nicest_model('kaka')