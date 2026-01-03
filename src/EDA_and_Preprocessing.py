import re
import pandas as pd

def clean_text(text):
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r"http\S+|www\S+", "", text)  
    text = re.sub(r"[^\w\s]", "", text)          
    text = re.sub(r"\s+", " ", text).strip()   
    # remove common boilerplate
    boilerplate = [
        "i am writing to file a complaint",
        "this is a complaint about",
        "please investigate my complaint"
    ]
    for phrase in boilerplate:
        text = text.replace(phrase, "")
    return text

def preprocess_df(df, text_column="Consumer complaint narrative"):
    df[text_column] = df[text_column].apply(clean_text)
    return df
