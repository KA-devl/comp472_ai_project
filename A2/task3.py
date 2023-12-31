import csv
import os
import random
from concurrent.futures import ProcessPoolExecutor, as_completed

import nltk
from gensim.models import Word2Vec
from matplotlib import pyplot as plt

TRAIN_MODEL = False


def train_model_async(sentences, w, e, filename):
    print("STARTED TRAINING MODEL " + filename)
    model = Word2Vec(sentences, window=w, vector_size=e)
    print("FINISHED TRAINING MODEL " + filename)
    model.save(filename)


if __name__ == '__main__':
    nltk.download('punkt')


    def preprocess_book(path):
        with open(path, 'r') as file:
            text = file.read()
        book_sentences = nltk.sent_tokenize(text)
        return [nltk.word_tokenize(sentence) for sentence in book_sentences]


    def preprocess_books(directory):
        book_files = os.listdir(directory)
        print("Number of books found: {}".format(len(book_files)))
        return [sentence for book_file in book_files for sentence in
                preprocess_book(os.path.join(directory, book_file))]


    def read_csv(file_path):
        with open(file_path, 'r', newline='') as csvfile:
            reader = csv.DictReader(csvfile)
            data = [row for row in reader]
        return data


    def train_models(w1, w2, e5, e6):
        print("PROCESSING BOOKS")
        sentences = preprocess_books('books')
        print("PROCESSED BOOKS")

        with ProcessPoolExecutor() as executor:
            futures = [
                executor.submit(train_model_async, sentences, w1, e5, "model1.model"),
                executor.submit(train_model_async, sentences, w1, e6, "model2.model"),
                executor.submit(train_model_async, sentences, w2, e5, "model3.model"),
                executor.submit(train_model_async, sentences, w2, e6, "model4.model")
            ]

            for future in as_completed(futures):
                print(future.result())


    if TRAIN_MODEL:
        train_models(12, 6, 100, 300)

    model1 = Word2Vec.load("model1.model")
    model2 = Word2Vec.load("model2.model")
    model3 = Word2Vec.load("model3.model")
    model4 = Word2Vec.load("model4.model")


    def get_closest_synonym(question_word, answer_word, model):
        try:
            similarities = [(option, model.wv.similarity(question_word, option.lower())) for option in answer_word]
            return max(similarities, key=lambda x: x[1])[0]
        except KeyError:
            return None


    def process_synonym_test_data(data, cur_model):
        correct_count = 0
        valid_count = 0
        results = []

        for entry in data:
            question_word = entry['question']
            correct_answer = entry['answer']
            guess_words = [entry[str(i)] for i in range(4)]  # options are in columns 0 to 3

            system_guess_word = get_closest_synonym(question_word, guess_words, cur_model)
            if system_guess_word not in [question_word] + [system_guess_word]:
                system_guess_word = random.choice(guess_words)
            if (correct_answer not in [question_word] + [system_guess_word] and (
                    system_guess_word not in cur_model.wv.key_to_index or question_word not in cur_model.wv.key_to_index)):
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


    def plot_graph_and_save(models, accuracies):
        if not os.path.exists('stats'):
            os.makedirs('stats')

        plt.figure(figsize=(10, 5))
        bars = plt.bar(models, accuracies)
        plt.xlabel('Models')
        plt.ylabel('Accuracy')
        plt.title('Models Accuracy Comparison')
        plt.xticks(models)
        plt.yticks([i * 0.1 for i in range(11)])
        for bar in bars:
            yval = round(100 * bar.get_height(), 2)
            plt.text(bar.get_x() + bar.get_width() / 2, yval / 100, f'{yval}%', ha='center', va='bottom')

        plt.savefig('stats/task3_models_accuracy_comparison.png')
        plt.close()


    models = [model1, model2, model3, model4]
    accuracies = []
    model_names = []
    for i, model in enumerate(models, 1):
        synonym_data = read_csv('synonym.csv')
        results, correct_count, valid_count = process_synonym_test_data(synonym_data, model)
        accuracy = correct_count / valid_count if valid_count > 0 else 0
        accuracies.append(accuracy)
        model_names.append(f"model{i}")
        vocabulary_size = len(model.wv.key_to_index)
        write_results_to_csv(results, f'model{i}-details.csv')
        with open('analysis.csv', 'a', newline='') as csvfile:
            csv_info = ['model_name', 'vocabulary_size', 'C', 'V', 'accuracy']
            writer = csv.DictWriter(csvfile, fieldnames=csv_info)
            writer.writerow({
                'model_name': f'model{i}',
                'vocabulary_size': vocabulary_size,
                'C': correct_count,
                'V': valid_count,
                'accuracy': accuracy
            })

    plot_graph_and_save(model_names, accuracies)
