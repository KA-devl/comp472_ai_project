import csv
from gensim.downloader import load

if __name__ == '__main__':
    # Load the word2vec model
    model_word2vec = load('word2vec-google-news-300')
    model = model_word2vec
    print(f"Loaded word2vec model")


    def get_closest_synonym(question_word, answer_word, model):
        # get embedding vector for word
        try:
            # Compute the cosine similarity between the embeddings
            similarities = [(option, model.similarity(question_word, option.lower())) for option in answer_word]
            # Sort by similarity in descending order
            sorted_options = sorted(similarities, key=lambda x: x[1], reverse=True)
            # Return the closest synonym
            return sorted_options[0][0]
        except KeyError:
            return None



    def process_synonym_test_data(data, model):
        correct_count = 0
        valid_count = 0
        results = []

        for entry in data:
            question_word = entry['question']
            correct_answer = entry['answer']
            guess_words = [entry[str(i)] for i in range(4)]  # options are in columns 0 to 3

            system_guess_word = get_closest_synonym(question_word, guess_words, model)

            if system_guess_word is None or correct_answer not in [question_word] + [system_guess_word]:
                label = 'guess'
            elif system_guess_word == correct_answer:
                label = 'correct'
            else:
                label = 'wrong'

            if label == 'correct':
                correct_count += 1
            if label != 'guess':
                valid_count += 1

            # Append the result to the results list
            results.append({
                'question_word': question_word,
                'correct_answer': correct_answer,
                'system_guess_word': system_guess_word,
                'label': label
            })

        return results, correct_count, valid_count


    def write_results_to_csv(results, file_name):
        with open(file_name, 'w', newline='') as csvfile:
            csv_info = ['question_word', 'correct_answer', 'system_guess_word', 'label']
            writer = csv.DictWriter(csvfile, fieldnames=csv_info)
            writer.writeheader()
            for result in results:
                writer.writerow(result)
    def read_csv(file_path):
        with open(file_path, 'r', newline='') as csvfile:
            reader = csv.DictReader(csvfile)
            data = [row for row in reader]
        return data


    synonym_data = read_csv('synonym.csv')
    results, correct_count, valid_count = process_synonym_test_data(synonym_data, model)
    accuracy = correct_count / valid_count if valid_count > 0 else 0
    write_results_to_csv(results, 'word2vec-google-news-300-details.csv')

    with open('analysis.csv', 'w', newline='') as csvfile:
        csv_info = ['model_name', 'vocabulary_size', 'C', 'V', 'accuracy']
        writer = csv.DictWriter(csvfile, fieldnames=csv_info)
        model_name = 'word2vec-google-news-300'
        writer.writerow({
            'model_name': model_name,
            'vocabulary_size': 1000000,
            'C': correct_count,
            'V': valid_count,
            'accuracy': accuracy
        })
