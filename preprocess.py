import pandas as pd
import re
from sklearn.model_selection import train_test_split

# Load dataset
df = pd.read_csv("data/email_tone_dataset.csv")

# Cleaning function
def clean_text(text):
    text = str(text)
    text = text.strip()
    text = text.replace("\n", " ")
    text = re.sub(r"https?://\S+", "<URL>", text)
    text = re.sub(r"\S+@\S+", "<EMAIL>", text)
    text = re.sub(r"\s+", " ", text)
    return text.lower()

# Apply cleaning
df["email_text"] = df["email_text"].apply(clean_text)

# Save cleaned dataset
df.to_csv("data/email_tone_dataset_clean.csv", index=False)

# Create label mapping
label_mapping = {tone: i for i, tone in enumerate(sorted(df["tone"].unique()))}
df["label"] = df["tone"].map(label_mapping)

# Split into train & test
train, test = train_test_split(df, test_size=0.2, stratify=df["label"], random_state=42)

train.to_csv("data/train.csv", index=False)
test.to_csv("data/test.csv", index=False)

print("Preprocessing completed!")
print("Label Mapping:", label_mapping)
