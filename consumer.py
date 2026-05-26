from confluent_kafka import Consumer
from src.logger import logging
from client import consumer_config, topic

def consumer(topic, consumer_config):
        consumer_config["group.id"] = "fraud_detection_group"
        consumer_config["auto.offset.reset"] = "earliest"

        # creating a new consumer instance
        consumer = Consumer(consumer_config)

        # subscribing to the specified topic for this current consumer, then we only our consumer can consume the event from this specific topic
        consumer.subscribe([topic])
        try:
            while True:
                # consumer polls the topic and prints any incoming messages
                msg = consumer.poll(1.0)
                if msg is not None and msg.error() is None:
                    key = msg.key().decode("utf-8")
                    value = msg.value().decode("utf-8")
                    logging.info(
                        f"Consumed message from topic {topic}: key = {key:12} value = {value:12}"
                    )
        except KeyboardInterrupt:
            pass
        finally:
            # closes the consumer connection
            consumer.commit(msg)
            consumer.close()

if __name__ == "__main__":
    consumer(topic, consumer_config)