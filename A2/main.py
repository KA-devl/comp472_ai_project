import csv
from gensim.models import KeyedVectors
from gensim.downloader import load

# Load the pre-trained Word2Vec model
model_name = 'word2vec-google-news-300'
model = load(model_name)

# Load the Synonym Test dataset
with open('synonym.csv', 'r') as file:
    synonym_test_data = list(csv.DictReader(file, delimiter='\t'))

# Create the output files
details_output_path = f"{model_name}-details.csv"
analysis_output_path = "analysis.csv"

# Function to compute the closest synonym for a word
def get_closest_synonym(question_word, answer_word, guess_words):
    try:
        # Ensure all words are present in the model
        if all(word in model for word in [question_word, answer_word] + guess_words):
            # Compute the cosine similarity between the embeddings
            similarities = [(guess, model.similarity(question_word, guess)) for guess in guess_words]
            # Sort by similarity in descending order
            similarities.sort(key=lambda x: x[1], reverse=True)
            # Return the closest synonym
            return similarities[0][0]
        else:
            return None
    except KeyError:
        return None

# Create and write to the details output file
with open(details_output_path, 'w', newline='') as details_output_file:
    details_fieldnames = ['question_word', 'correct_answer', 'guess_word', 'label']
    details_csv_writer = csv.DictWriter(details_output_file, fieldnames=details_fieldnames)
    details_csv_writer.writeheader()

    # Variables for analysis
    total_questions = len(synonym_test_data)
    correct_labels = 0
    questions_without_guess = 0

    for row in synonym_test_data:
        question_word = row['question']
        answer_word = row['answer']
        guess_words = [row[str(i)] for i in range(4)]

        closest_synonym = get_closest_synonym(question_word, answer_word, guess_words)

        if closest_synonym is None:
            label = 'guess'
        elif closest_synonym == answer_word:
            label = 'correct'
            correct_labels += 1
        else:
            label = 'wrong'

        details_csv_writer.writerow({'question_word': question_word, 'correct_answer': answer_word, 'guess_word': closest_synonym, 'label': label})

        # Count questions answered without guessing
        if label != 'guess':
            questions_without_guess += 1

# Calculate accuracy
accuracy = correct_labels / questions_without_guess if questions_without_guess > 0 else 0

# Write to the analysis output file
with open(analysis_output_path, 'w', newline='') as analysis_output_file:
    analysis_fieldnames = ['model_name', 'vocabulary_size', 'correct_labels', 'questions_without_guess', 'accuracy']
    analysis_csv_writer = csv.DictWriter(analysis_output_file, fieldnames=analysis_fieldnames)
    analysis_csv_writer.writeheader()

    analysis_csv_writer.writerow({
        'model_name': model_name,
        'vocabulary_size': len(model.vocab),
        'correct_labels': correct_labels,
        'questions_without_guess': questions_without_guess,
        'accuracy': accuracy
    })

print(f"Results saved to {details_output_path} and {analysis_output_path}")
