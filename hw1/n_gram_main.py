#!/usr/bin/env python3
"""
N-gram語言模型訓練和測試程式
使用 train.txt 訓練模型，在 test.txt 上評估性能
比較 n=2 (bigram) 和 n=3 (trigram) 的表現
"""

import time
import os
from ngram_model import NGramModel

def main():
    
    train_file = "train.txt"
    test_file = "test.txt"
    
    print("=" * 60)
    print("N-gram model Training and Evaluation")
    print("=" * 60)
    
    results = {}
    trained_models = {}  # Store trained models for text generation
    
    for n in [2, 3]:
        print(f"\n{'='*20} N={n} ({'Bigram' if n==2 else 'Trigram'}) {'='*20}")
        
        start_time = time.time()

        model = NGramModel(n=n)

        print(f"Train {n}-gram model...")
        model.train(train_file)
        
        training_time = time.time() - start_time
        print(f"Training time: {training_time:.2f} seconds")

        # Calculate perplexity
        start_time = time.time()
        perplexity = model.calculate_perplexity(test_file)
        test_time = time.time() - start_time

        print(f"Testing time: {test_time:.2f} seconds")
        
        # save results and model
        results[n] = {
            'perplexity': perplexity,
            'training_time': training_time,
            'test_time': test_time,
            'vocab_size': len(model.vocabulary),
            'total_words': model.total_words,
            'ngram_types': len(model.ngram_counts),
            'context_types': len(model.context_counts)
        }
        trained_models[n] = model  # Store the trained model

        print(f"\n{n}-gram model text generation examples:")
        try:
            if n == 2:
                contexts = [("add",), ("cook",), ("bake",)]
            else:
                contexts = [("add", "the"), ("cook", "for"), ("bake", "at")]
            
            for context in contexts:
                generated = model.generate_text(context, max_length=15)
                print(f"  Context: {' '.join(context)} -> {generated}")
        except Exception as e:
            print(f"  Error occurred during text generation: {e}")
        
        # Test with incomplete.txt for text completion
        if os.path.exists("incomplete.txt"):
            try:
                with open("incomplete.txt", "r", encoding="utf-8") as f:
                    incomplete_lines = [line.strip() for line in f if line.strip()]
                
                for i, incomplete_text in enumerate(incomplete_lines):
                    words = model.preprocess_text(incomplete_text)
                    if len(words) == 0:
                        continue
                    
                    # Create context based on model type
                    if n == 2:
                            continue
                    else:  # n == 3
                        print(f"\n{n}-gram model incomplete text completion:")
                        if len(words) >= 2:
                            context = (words[-2], words[-1])  # Use last 2 words as context
                        elif len(words) == 1:
                            context = ("<s>", words[-1])  # Pad with start token
                        else:
                            continue
                    
                    # Generate completion
                    completion = model.generate_text(context, max_length=20)
                    # Remove the context words from completion to show only new words
                    context_str = ' '.join(context)
                    if completion.startswith(context_str):
                        new_words = completion[len(context_str):].strip()
                        if new_words:
                            full_completion = incomplete_text + " " + new_words
                        else:
                            full_completion = incomplete_text + " [no completion]"
                    else:
                        full_completion = incomplete_text + " " + completion
                    
                    print(f"  '{incomplete_text}' -> '{full_completion}'")
                    
            except Exception as e:
                print(f"  Error processing incomplete.txt: {e}")
        else:
            print("  incomplete.txt not found, skipping completion test")

    # Results comparison
    print("\n" + "=" * 60)
    print("Results Comparison")
    print("=" * 60)

    print(f"{'Metric':<20} {'Bigram (n=2)':<15} {'Trigram (n=3)':<15} {'Difference':<15}")
    print("-" * 65)
    
    bigram_pp = results[2]['perplexity']
    trigram_pp = results[3]['perplexity']
    pp_diff = ((trigram_pp - bigram_pp) / bigram_pp) * 100

    print(f"{'Perplexity':<20} {bigram_pp:<15.2f} {trigram_pp:<15.2f} {pp_diff:+.2f}%")

    bigram_time = results[2]['training_time']
    trigram_time = results[3]['training_time']
    time_diff = ((trigram_time - bigram_time) / bigram_time) * 100

    print(f"{'Training time':<20} {bigram_time:<15.2f} {trigram_time:<15.2f} {time_diff:+.2f}%")
    
    bigram_ngrams = results[2]['ngram_types']  
    trigram_ngrams = results[3]['ngram_types']
    ngram_diff = ((trigram_ngrams - bigram_ngrams) / bigram_ngrams) * 100

    print(f"{'N-gram types':<20} {bigram_ngrams:<15,} {trigram_ngrams:<15,} {ngram_diff:+.2f}%")

    print(f"{'Vocabulary size':<20} {results[2]['vocab_size']:<15,} {results[3]['vocab_size']:<15,} {'Same':<15}")
    print(f"{'Total words':<20} {results[2]['total_words']:<15,} {results[3]['total_words']:<15,} {'Same':<15}")

    # Save detailed results to a file
    with open("ngram_results.txt", "w", encoding="utf-8") as f:
        f.write("N-gram Language Model Evaluation Results\n")
        f.write("=" * 40 + "\n\n")
        
        for n in [2, 3]:
            model_name = "Bigram" if n == 2 else "Trigram"
            f.write(f"{model_name} (n={n}) Results:\n")
            f.write(f"  Perplexity: {results[n]['perplexity']:.2f}\n")
            f.write(f"  Training time: {results[n]['training_time']:.2f} seconds\n")
            f.write(f"  Testing time: {results[n]['test_time']:.2f} seconds\n")
            f.write(f"  Vocabulary size: {results[n]['vocab_size']:,}\n")
            f.write(f"  Total words: {results[n]['total_words']:,}\n")
            f.write(f"  N-gram types: {results[n]['ngram_types']:,}\n")
            f.write(f"  Context types: {results[n]['context_types']:,}\n")
            f.write("\n")

        f.write("Comparison Results:\n")
        f.write(f"  Perplexity difference: {pp_diff:+.2f}%\n")
        f.write(f"  Training time difference: {time_diff:+.2f}%\n")
        f.write(f"  N-gram types difference: {ngram_diff:+.2f}%\n")
        
        # Save text generation examples using trained models
        f.write("\nText Generation Examples:\n")
        for n in [2, 3]:
            model_name = "Bigram" if n == 2 else "Trigram"
            f.write(f"\n{model_name} Generation Examples:\n")
            
            model = trained_models[n]  # Use already trained model
            
            # Basic examples
            if n == 2:
                contexts = [("add",), ("cook",), ("bake",)]
            else:
                contexts = [("add", "the"), ("cook", "for"), ("bake", "at")]
            
            for context in contexts:
                try:
                    generated = model.generate_text(context, max_length=15)
                    f.write(f"  Context: {' '.join(context)} -> {generated}\n")
                except Exception as e:
                    f.write(f"  Context: {' '.join(context)} -> Error: {e}\n")
            
            # Incomplete text completion examples
            if os.path.exists("incomplete.txt"):
                f.write(f"\n{model_name} Incomplete Text Completions:\n")
                try:
                    with open("incomplete.txt", "r", encoding="utf-8") as inc_f:
                        incomplete_lines = [line.strip() for line in inc_f if line.strip()]
                    
                    for incomplete_text in incomplete_lines[:8]:
                        words = model.preprocess_text(incomplete_text)
                        if len(words) == 0:
                            continue
                        
                        if n == 2:
                            if len(words) >= 1:
                                context = (words[-1],)
                            else:
                                continue
                        else:
                            if len(words) >= 2:
                                context = (words[-2], words[-1])
                            elif len(words) == 1:
                                context = ("<s>", words[-1])
                            else:
                                continue
                        
                        try:
                            completion = model.generate_text(context, max_length=8)
                            context_str = ' '.join(context)
                            if completion.startswith(context_str):
                                new_words = completion[len(context_str):].strip()
                                if new_words:
                                    full_completion = incomplete_text + " " + new_words
                                else:
                                    full_completion = incomplete_text + " [no completion]"
                            else:
                                full_completion = incomplete_text + " " + completion
                            f.write(f"  '{incomplete_text}' -> '{full_completion}'\n")
                        except Exception as e:
                            f.write(f"  '{incomplete_text}' -> Error: {e}\n")
                except Exception as e:
                    f.write(f"  Error processing incomplete.txt: {e}\n")

    print(f"\nDetailed results have been saved to ngram_results.txt")
    print("Program execution completed!")

if __name__ == "__main__":
    main()