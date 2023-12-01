import csv
import random
import matplotlib.pyplot as plt
import numpy as np

from gensim.downloader import load


if __name__ == '__main__':
    model_static_name = 'word2vec-google-news-300'
    #model_static_name = 'glove-wiki-gigaword-200'
    #model_static_name = 'fasttext-wiki-news-subwords-300'
    #model_static_name = 'English CoNLL17 corpus TO DO'

    model = load(model_static_name)
    print(f"Loaded model successfully")


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

            if system_guess_word not in [question_word] + [system_guess_word]:
                system_guess_word = random.choice(guess_words)
            if (correct_answer not in [question_word] + [system_guess_word] and (system_guess_word not in model.key_to_index or question_word not in model.key_to_index)):
                label = "guess"
            elif system_guess_word == correct_answer:
                label = "correct"
            else:
                label = "wrong"

            if label == 'correct':
                correct_count += 1
            if label != 'guess':
                valid_count += 1

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
    write_results_to_csv(results, model_static_name + '-details.csv')



    with open('analysis.csv', 'a', newline='') as csvfile:
        csv_info = ['model_name', 'vocabulary_size', 'C', 'V', 'accuracy']
        writer = csv.DictWriter(csvfile, fieldnames=csv_info)
        vocabulary_size = len(model.key_to_index)
        writer.writerow({
            'model_name': model_static_name,
            'vocabulary_size': vocabulary_size,
            'C': correct_count,
            'V': valid_count,
            'accuracy': accuracy
        })

    # Initialize lists to store data
    model_names = []
    vocab_sizes = []
    accuracies = []

    # Read data from CSV file
    with open('analysis.csv', 'r') as file:
        reader = csv.reader(file)
        next(reader)  # Skip header row
        for row in reader:
            values = row[4].split(',')  # Split the string into values
            accuracy = float(values[0])  # Extract the accuracy as a float
            model_names.append(row[0])
            vocab_sizes.append(int(row[1]))
            accuracies.append(accuracy)

    # Set up positions for bars on x-axis
    x_positions = np.arange(len(model_names))

    # Create bar chart with unique colors for each bar
    colors = plt.cm.viridis(np.linspace(0, 1, len(model_names)))  # Use a colormap for colors
    plt.figure(figsize=(12, 6))
    bars = plt.bar(x_positions, accuracies, align='center', alpha=0.7, color=colors)

    # Add labels and title
    plt.xlabel('Models')
    plt.ylabel('Accuracy')
    plt.title('Accuracy for Different Models')

    # Add x-axis labels and rotate them for better readability
    plt.xticks(x_positions, model_names, rotation=45, ha='right')

    # Add a legend with model names and corresponding colors
    plt.legend(bars, model_names, loc='upper left')

    # Show the plot
    plt.show()