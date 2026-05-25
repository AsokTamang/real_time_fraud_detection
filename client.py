def read_config():
  # reads the client configuration from client.properties
  # and returns it as a key-value map
  config = {}
  with open("client.properties") as fh:
    for line in fh:
      line = line.strip()
      if len(line) != 0 and line[0] != "#":
        parameter, value = line.strip().split('=', 1)
        config[parameter] = value.strip()
  return config
#we must separate the config for producer and consumer because they have different configurations for the Kafka client
producer_config = read_config()
consumer_config = read_config()
topic = "fraud_detection"


