########################################
# Can be used to extract tweets based on the sample dataset.
# After requesting around 7000 entries (70 requests, 100 tweets per request) twitter denies service for each thread, could not find a work around.
# For this reason data from this script could not be used to train hte models.
########################################

import sys, csv, datetime, threading, queue, time
from pytwitter import Api
from pytwitter.error import PyTwitterError
from time import sleep

csv.field_size_limit(sys.maxsize)

gossip_fake_ids = []
with open("sample_dataset/gossipcop_fake.csv") as csv_file:
    reader = csv.DictReader(csv_file, delimiter=",")
    for row in reader:
        gossip_fake_ids += row['tweet_ids'].split("\t")

politi_fake_ids = []
with open("sample_dataset/politifact_fake.csv") as csv_file:
    reader = csv.DictReader(csv_file, delimiter=",")
    for row in reader:
        politi_fake_ids += row['tweet_ids'].split("\t")


gossip_real_ids = []
with open("sample_dataset/gossipcop_real.csv") as csv_file:
    reader = csv.DictReader(csv_file, delimiter=",")
    for row in reader:
        gossip_real_ids += row['tweet_ids'].split("\t")

politi_real_ids = []
with open("sample_dataset/politifact_real.csv") as csv_file:
    reader = csv.DictReader(csv_file, delimiter=",")
    for row in reader:
        gossip_real_ids += row['tweet_ids'].split("\t")

fake_tweet_ids = gossip_fake_ids + politi_fake_ids
real_tweet_ids = gossip_real_ids + politi_real_ids

f = open("tweets.log", "a")
f.write(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S') + " Number of fake tweets: " + str(len(fake_tweet_ids)) + "; Number of real tweets: " + str(len(real_tweet_ids)) + "\n")
f.close()

def equal_chunks(list, chunk_size):
    chunks = []
    for i in range(0, len(list), chunk_size):
        chunks.append(list[i:i + chunk_size])
    return chunks

fake_split_ids = equal_chunks(fake_tweet_ids, chunk_size=100)
real_split_ids = equal_chunks(real_tweet_ids, chunk_size=100)

twitter1 = Api(bearer_token="AAAAAAAAAAAAAAAAAAAAADr8WAEAAAAA5SrtbKxY3rFk8I6N2h9tKJtw9ro%3DQXFeDbdv2OLUF62xwNQsQTP5zxBSQTcb7xlq7RxnTbIQ7ZjhMz", sleep_on_rate_limit=True)
twitter2 = Api(bearer_token="AAAAAAAAAAAAAAAAAAAAAFDmWQEAAAAAnJvTAwk29QyX26KhxDUZXG4Pl7U%3D7BWk1JSYspzFf27fdvmbPcau0hAZw1FSBlJRo5Qxumf6nmNLVe", sleep_on_rate_limit=True)

def processTweets(ids, real):
        try:

            f = open("tweets.log", "a")
            f.write(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S') + " Calling api with account 1\n")
            f.close()


            data =  twitter1.get_tweets(ids,expansions="author_id",tweet_fields=["created_at"], user_fields=["username","verified"], return_json=True)
            for tweet in data['data']:
                tweet['real'] = real
                print (tweet)

            f = open("tweets.log", "a")
            f.write(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S') + " Added tweets with account 1\n")
            f.close()

        except PyTwitterError:

            try:

                f = open("tweets.log", "a")
                f.write(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S') + " Calling api with account 2\n")
                f.close()


                data =  twitter2.get_tweets(ids,expansions="author_id",tweet_fields=["created_at"], user_fields=["username","verified"], return_json=True)
                for tweet in data['data']:
                    tweet['real'] = real
                    print (tweet)

                f = open("tweets.log", "a")
                f.write(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S') + " Added tweets with account 2\n")
                f.close()

            except PyTwitterError:

                f = open("tweets.log", "a")
                f.write(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S') + " PyTwitterError happened with both accounts, push to back of the queue\n")
                f.close()

                return False

                sleep(60)
                processTweets(ids, real=real)

print ("[")

queue = queue.Queue()


class MyThread(threading.Thread):
    def __init__(self, theQueue=None):
        threading.Thread.__init__(self)        
        self.theQueue=theQueue

    def run(self):
        while True:
            thing=self.theQueue.get()
            self.process(thing) 
            self.theQueue.task_done()

    def process(self, thing):
        time.sleep(1)
        f = open("tweets.log", "a")
        f.write(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S') + ' processing ' + str(thing) + "\n")
        f.close()
        success = processTweets(thing['ids'], thing['real'])
        if not success:
            queue.put(thing)


def loopTweets(tweet_ids, real):

    for ids in tweet_ids:
        queue.put({"ids": ids, "real": real})
    
def twitterThreads(fake_ids, real_ids):

    loopTweets(fake_ids, False)
    loopTweets(real_ids, True)

twitterThreads(fake_split_ids, real_split_ids)

for cpu in range(4):
    thr = MyThread(theQueue=queue)
    thr.start()

print ("]")


