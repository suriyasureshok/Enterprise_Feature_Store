|Issue|Description|How I Solved|
|-----|-----------|-------------|
|Data Synthesis Issue|During the synthesis of event logs I had no idea on what events would be in the logs.|I managed it using GPT and some searches in Google|
|Timestamp Parsing Error|pd.to_datetime failed to parse ISO8601 timestamps in user_activity_features.py because some timestamps included microseconds while others didn't.|Used format='ISO8601' parameter in pd.to_datetime to handle varying ISO8601 timestamp formats.|
|Sentiment Analysis AttributeError|VADER sentiment analysis failed in user_review_features.py because review_text column contained float values instead of strings.|Converted review_text to string type, filled NaN with empty strings, and applied sentiment analysis only on non-empty strings.|
|Review Time Subtraction TypeError|Subtraction of review_time strings failed in user_review_features.py because review_time was not converted to datetime objects.|Added pd.to_datetime conversion with format='ISO8601' for the review_time column before calculating time differences.|
|PyArrow ImportError|pandas.to_parquet failed because pyarrow was not installed.|Added pyarrow to requirements.txt and installed it.|
|Redis DataError|Redis hset failed because feature values included unsupported types (bool, None).|Updated serialize_value function to convert bool and None values to strings.|