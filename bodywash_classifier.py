import json
import pandas as pd
from groq import Groq  # Requires 'groq' package
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer  # Requires 'sentence-transformers' package
from sklearn.metrics import jaccard_score
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import os
from base64 import b64encode
from IPython.display import display, HTML  # Requires IPython, will cause error if not used in Jupyter environment
from dotenv import load_dotenv  # Requires 'python-dotenv' package

# Load environment variables
load_dotenv()

# Retrieve API key
API_KEY = os.getenv("GROQ_API_KEY")
if not API_KEY:
    print("Error: GROQ_API_KEY not found in environment variables.  Classification and accuracy metrics will be unavailable.")

# Model Configuration
SELECTED_MODEL = "llama3-70b-8192"
TEMPERATURE = 0.2
MAX_TOKENS = 1024


def get_labels(api_key, text_to_classify, labels_level_1, labels_level_2, model, temp, max_toks):
    """
    Classifies text using Groq LLM into Level 1 and Level 2 labels.
    """
    client = Groq(api_key=api_key)
    prompt = f"""You are a helpful assistant that classifies customer reviews for body wash into different categories (tags).
    You will receive a body wash review and a list of possible labels divided into 2 levels: Level 1 and Level 2. Level 2 tags are children of Level 1 tags.
    Your task is to predict the relevant labels for the review.
    Here are the Level 1 labels: {', '.join(labels_level_1)}
    Here are the Level 2 labels: {', '.join(labels_level_2)}
    1.Please provide the output as a JSON object with two keys: "Level 1" and "Level 2".  The values should be lists of the predicted labels. Include only labels that are directly relevant to the review.
    2.Remender to only output the labels and nothing else and use labels from the provided lists only
    Example:
    Review: "I love the smell! Great body wash."
    Output: {{"Level 1": ["Fragrance"], "Level 2": ["Personal Likability (Fragrance)"]}}
    Now, classify the following review:
    Review: "{text_to_classify}"
    """

    try:
        chat_completion = client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model=model,
            temperature=temp,
            max_tokens=max_toks,
        )
        response_content = chat_completion.choices[0].message.content

        # Parse the JSON response
        try:
            json_response = json.loads(response_content)
            level_1_predictions = json_response.get("Level 1", [])
            level_2_predictions = json_response.get("Level 2", [])
            return level_1_predictions, level_2_predictions, response_content
        except json.JSONDecodeError as e:
            # Try to extract JSON from the response
            if "{" in response_content and "}" in response_content:
                json_str = response_content[response_content.find("{"):response_content.rfind("}") + 1]
                try:
                    json_response = json.loads(json_str)
                    level_1_predictions = json_response.get("Level 1", [])
                    level_2_predictions = json_response.get("Level 2", [])
                    return level_1_predictions, level_2_predictions, response_content
                except:
                    pass
            return [], [], response_content
    except Exception as e:
        return [], [], f"Error: {str(e)}"


def calculate_jaccard_similarity(str1, str2):
    """Calculates Jaccard similarity between two strings using TF-IDF."""
    vectorizer = TfidfVectorizer()
    vectors = vectorizer.fit_transform([str1, str2])
    return jaccard_score(vectors[0].toarray().flatten() > 0, vectors[1].toarray().flatten() > 0)


def calculate_semantic_similarity(str1, str2, model):
    """Calculates semantic similarity between two strings using sentence embeddings."""
    embeddings = model.encode([str1, str2])
    return cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]


def classify_single_review(review_text, level_1_labels, level_2_labels, api_key = API_KEY):
    """Classifies a single review and prints the results."""
    if not api_key:
        print("Error: API key is not set. Please set your Groq API key first.")
        return

    print(f"Classifying review: '{review_text[:100]}...'")
    level_1, level_2, raw_response = get_labels(
        api_key, review_text, level_1_labels, level_2_labels,
        SELECTED_MODEL, TEMPERATURE, MAX_TOKENS
    )

    print("\nClassification Results:")
    print("Level 1 Labels:", level_1)
    print("Level 2 Labels:", level_2)
    print("\nRaw LLM Response:")
    print(raw_response)

    return level_1, level_2, raw_response


def classify_test_rows(test_df, row_indices, level_1_labels, level_2_labels, api_key = API_KEY):
    """
    Classifies specific rows from the test dataset, creating multiple entries for
    multiple labels.

    Returns:
        Pandas DataFrame containing the classification results with multiple rows
        for multiple labels.
    """
    if not api_key:
        print("Error: API key is not set. Please set your Groq API key first.")
        return

    results = []

    for i, idx in enumerate(row_indices):
        print(f"Processing {i + 1}/{len(row_indices)}: Row {idx}")
        review = test_df.iloc[idx]["Core Item"]
        level_1, level_2, raw_response = get_labels(
            api_key, review, level_1_labels, level_2_labels,
            SELECTED_MODEL, TEMPERATURE, MAX_TOKENS
        )

        # Create multiple entries for multiple labels
        for l1 in level_1:
            for l2 in level_2:
                results.append({
                    "row_index": idx,
                    "review": review,
                    "Level 1": l1,
                    "Level 2": l2,
                    "raw_response": raw_response
                })

        # If no labels are predicted, add a single row with empty labels
        if not level_1 and not level_2:
            results.append({
                "row_index": idx,
                "review": review,
                "Level 1": None,  # Or use an appropriate placeholder like 'N/A'
                "Level 2": None,  # Or use an appropriate placeholder like 'N/A'
                "raw_response": raw_response
            })

    # Convert to DataFrame
    results_df = pd.DataFrame(results)
    if not results_df.empty:
      try: # using try block in case not run in jupyter env
        display(results_df[["row_index", "Level 1", "Level 2"]])
      except:
        print(results_df[["row_index", "Level 1", "Level 2"]].to_string()) # prints the result if not running in jupyter

    return results


def test_model_accuracy(train_df, level_1_labels, level_2_labels, num_samples=10, api_key = API_KEY):
    """Tests the model accuracy using Jaccard and semantic similarity."""

    if not api_key:
        print("Error: API key is not set. Please set your Groq API key first.")
        return

    # Limit number of samples
    num_samples = min(num_samples, len(train_df))
    if num_samples == 0:
        print("Error: Training data is empty.")
        return

    sample_indices = np.random.choice(len(train_df), num_samples, replace=False)

    sentence_transformer_model = SentenceTransformer('all-MiniLM-L6-v2')

    details_list = []

    print(f"Testing accuracy with {num_samples} random samples...\n")

    for i, idx in enumerate(sample_indices):
        print(f"Processing sample {i + 1}/{num_samples}...")
        review = train_df.iloc[idx]["Core Item"]
        true_level_1 = train_df.iloc[idx]["Level 1"] if pd.notna(train_df.iloc[idx]["Level 1"]) else ""
        true_level_2 = train_df.iloc[idx]["Level 2"] if pd.notna(train_df.iloc[idx]["Level 2"]) else ""

        # Split true labels into lists
        true_level_1_list = true_level_1.split(', ') if true_level_1 else []
        true_level_2_list = true_level_2.split(', ') if true_level_2 else []

        # Get predictions
        predicted_level_1, predicted_level_2, _ = get_labels(
            api_key, review, level_1_labels, level_2_labels,
            SELECTED_MODEL, TEMPERATURE, MAX_TOKENS
        )

        # Combine labels for similarity calculations
        predicted_labels_str = ", ".join(predicted_level_1) + ", " + ", ".join(predicted_level_2)
        true_labels_str = true_level_1 + ", " + true_level_2

        # Calculate similarities
        jaccard = calculate_jaccard_similarity(predicted_labels_str, true_labels_str)
        semantic = calculate_semantic_similarity(predicted_labels_str, true_labels_str, sentence_transformer_model)

        # Store details
        details_list.append({
            "review": review,
            "true_level_1": true_level_1_list,  # Store as lists
            "predicted_level_1": predicted_level_1,
            "true_level_2": true_level_2_list,  # Store as lists
            "predicted_level_2": predicted_level_2,
            "jaccard_similarity": jaccard,
            "semantic_similarity": semantic
        })

    # Calculate average similarities
    avg_jaccard = sum(d["jaccard_similarity"] for d in details_list) / len(details_list)
    avg_semantic = sum(d["semantic_similarity"] for d in details_list) / len(details_list)

    # Display summary metrics
    print("\n===== Accuracy Metrics =====")
    print(f"Average Jaccard Similarity: {avg_jaccard:.4f}")
    print(f"Average Semantic Similarity: {avg_semantic:.4f}")

    metrics_df = pd.DataFrame([
        {"Metric": "Jaccard Similarity", "Value": avg_jaccard},
        {"Metric": "Semantic Similarity", "Value": avg_semantic}
    ])
    try:
      display(metrics_df)
    except:
      print(metrics_df.to_string())

    # Create detailed report DataFrame
    details_df = pd.DataFrame(details_list)[
        ["review", "true_level_1", "predicted_level_1", "true_level_2",
         "predicted_level_2", "jaccard_similarity", "semantic_similarity"]
    ]

    return details_df


def classify_all_test_data(test_df, level_1_labels, level_2_labels, batch_size=5, api_key = API_KEY):
    """
    Classifies all reviews in the test dataset, creating multiple entries for
    multiple labels, and returns an updated DataFrame.
    """
    if not api_key:
        print("Error: API key is not set. Please set your Groq API key first.")
        return

    # Create a list to store the updated rows
    updated_rows = []

    total_rows = len(test_df)
    batch_size = min(batch_size, total_rows)

    print(f"Classifying all {total_rows} reviews in batches of {batch_size}...")

    for i in range(0, total_rows, batch_size):
        batch = test_df.iloc[i:min(i + batch_size, total_rows)]

        print(f"Processing batch {i // batch_size + 1}/{(total_rows + batch_size - 1) // batch_size}")

        for j, (idx, row) in enumerate(batch.iterrows()):
            review = row["Core Item"]
            print(f"  Processing review {i + j + 1}/{total_rows}...")

            level_1, level_2, _ = get_labels(
                api_key, review, level_1_labels, level_2_labels,
                SELECTED_MODEL, TEMPERATURE, MAX_TOKENS
            )

            # Create multiple rows for multiple labels
            if level_1 and level_2:
                for l1 in level_1:
                    for l2 in level_2:
                        new_row = row.copy()
                        new_row["Predicted Level 1"] = l1
                        new_row["Predicted Level 2"] = l2
                        updated_rows.append(new_row)
            else:
                # If no labels are predicted, add a single row with None values
                new_row = row.copy()
                new_row["Predicted Level 1"] = None  # Or use an appropriate placeholder like 'N/A'
                new_row["Predicted Level 2"] = None  # Or use an appropriate placeholder like 'N/A'
                updated_rows.append(new_row)

    # Convert the updated rows to a DataFrame
    updated_df = pd.DataFrame(updated_rows)
    try:
        # Download the updated CSV file
        csv_file = updated_df.to_csv(index=False)
        b = csv_file.encode()

        # Create a download link
        payload = b64encode(b).decode()
        href = f'\n<a download="bodywash_test_with_predictions.csv" href="data:file/csv;base64,{payload}" >Download updated CSV file</a>'
        display(HTML(href))  # display only works in ipynb env

    except: # if not ipynb environment
       print("Not running in IPython, CSV download link won't be generated.\nSaving to bodywash_test_with_predictions.csv")
       updated_df.to_csv("bodywash_test_with_predictions.csv", index = False)
    return updated_df