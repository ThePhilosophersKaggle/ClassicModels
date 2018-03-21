import pandas as pd
import nlp_models

# Path to fasttext
path_fasttext = r"C:\Users\gtregoat\workspace\deep_learning\fasttext\wiki.en.vec"

# Load text data
dir_path = r"C:\Users\gtregoat\Documents\databases\NLP_databases"

data = pd.read_csv(dir_path + r"\amazon_review_polarity\train.csv",
                        sep=',', lineterminator='\n', names=["label", "dummy", "text"])

# Create an NLP model for data set 3
test = nlp_models.NlpModels(x_train=data["text"].values, y_train=data["label"], test_size=0.50,
                    task="classification", shuffle=True, max_len=140)
# Train a cnn_lstm model
test.nlp_cnn_lstm_pretrained_embeddings(loss="categorical_crossentropy", optimizer="adam", batch_size=128,
                                        nb_epochs=1, path_to_embeddings=path_fasttext, vector_dim=300,
                                        embeddings_source="fasttext")

# Print classification report on the test data
test.evaluate()

# Print classification report on other data
dataset_2 = pd.read_csv(dir_path + "\Amazon_reviews\Amazon_instant_video_5.csv")
df = dataset_2[dataset_2.isin([1, 5])]
test.test_on_new_data(df["reviewText"].values.tolist(), df["overall"].values)
