import torch
from transformers import (
    T5ForQuestionAnswering,
    T5ForConditionalGeneration,
    T5Tokenizer,
    BertForQuestionAnswering,
    BertTokenizer,
    GPT2LMHeadModel,
    GPT2Tokenizer
)
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import pandas as pd
import PyPDF2
from nltk.tokenize import sent_tokenize
import nltk
from rank_bm25 import BM25Okapi
import numpy as np
import json
from typing import List, Dict
from rouge_score import rouge_scorer
from tqdm import tqdm
import argparse
import os

def parse_arguments():
    parser = argparse.ArgumentParser(description='QA Model Evaluation')
    parser.add_argument('--json_path', type=str, default='data/ipc_qa.json', help='Path to JSON file containing QA pairs')
    parser.add_argument('--pdf_path', type=str, default='data/ipc.pdf', help='Path to PDF document')
    return parser.parse_args()

class JSONDataLoader:
    @staticmethod
    def load_json(file_path: str) -> List[Dict]:
        """Load and validate JSON data from file"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Validate data structure
            if not isinstance(data, list):
                raise ValueError("JSON data must be a list of QA pairs")
            
            for item in data:
                if not isinstance(item, dict) or 'question' not in item or 'answer' not in item:
                    raise ValueError("Each item must be a dictionary with 'question' and 'answer' keys")
            
            return data
        except FileNotFoundError:
            raise FileNotFoundError(f"Could not find JSON file at {file_path}")
        except json.JSONDecodeError:
            raise ValueError(f"Invalid JSON format in file {file_path}")

# Add this at the beginning of your script, before any other NLTK operations
def setup_nltk():
    """Setup NLTK resources properly"""
    import nltk
    try:
        # Download all required NLTK data
        nltk.download('punkt')
        nltk.download('punkt_tab')
        
        # Verify the downloads
        nltk.data.find('tokenizers/punkt')
        nltk.data.find('tokenizers/punkt_tab')
        
        print("NLTK resources successfully downloaded and verified")
    except Exception as e:
        print(f"Error setting up NLTK: {str(e)}")
        # Try alternate initialization
        try:
            import ssl
            try:
                _create_unverified_https_context = ssl._create_unverified_context
            except AttributeError:
                pass
            else:
                ssl._create_default_https_context = _create_unverified_https_context
            
            nltk.download('punkt', quiet=True)
            nltk.download('punkt_tab', quiet=True)
            print("NLTK resources downloaded using unverified HTTPS context")
        except Exception as e2:
            print(f"Failed to download NLTK resources: {str(e2)}")
            raise

class PDFContextExtractor:
    def __init__(self, pdf_path):
        """Initialize with path to PDF file"""
        self.pdf_path = pdf_path
        self.cache_path = pdf_path.replace('.pdf', '_extracted.txt')
        setup_nltk()

    def extract_text(self):
        """Extract text from PDF file with caching"""
        # Try to load from cache first
        try:
            if os.path.exists(self.cache_path):
                print("Loading text from cache...")
                with open(self.cache_path, 'r', encoding='utf-8') as f:
                    return f.read()
        except Exception:
            pass

        # If cache doesn't exist or fails, extract from PDF
        try:
            print("Extracting text from PDF...")
            with open(self.pdf_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                text = ""
                for page in pdf_reader.pages:
                    text += page.extract_text() + "\n"
            
            # Save to cache
            try:
                with open(self.cache_path, 'w', encoding='utf-8') as f:
                    f.write(text)
                print(f"Cached extracted text to {self.cache_path}")
            except Exception as e:
                print(f"Warning: Could not save cache: {e}")
                
            return text
        except Exception as e:
            raise Exception(f"Error reading PDF file: {str(e)}")

    def create_context_windows(self, text, window_size=3):
        """Split text into context windows with optimization"""
        sentences = []
        try:
            # Use simpler splitting for better performance
            sentences = [s.strip() for s in text.split('.') if len(s.strip()) > 20]  # Filter out very short sentences
            
            # Create context windows more efficiently
            contexts = []
            total = len(sentences)
            
            # Pre-calculate window ranges
            windows = [
                (max(0, i - window_size), min(total, i + window_size + 1))
                for i in range(total)
            ]
            
            # Create contexts in bulk
            contexts = [
                ' '.join(sentences[start:end])
                for start, end in windows
            ]
            
            return contexts, sentences
            
        except Exception as e:
            print(f"Error in create_context_windows: {str(e)}")
            return [text], [text]

    def find_best_context(self, question, contexts):
        """Find most relevant context using optimized BM25"""
        # Only compute once per initialization
        if not hasattr(self, 'bm25'):
            print("Initializing BM25 index...")
            tokenized_contexts = [context.split() for context in contexts]
            self.bm25 = BM25Okapi(tokenized_contexts)
            self.contexts = contexts

        tokenized_question = question.split()
        scores = self.bm25.get_scores(tokenized_question)
        best_context_idx = np.argmax(scores)
        return self.contexts[best_context_idx]
    
def setup_environment():
    """Install required packages"""
    import subprocess
    import sys
    
    packages = [
        'sentencepiece',  # Required for T5
        'transformers',
        'torch',
        'scikit-learn'
    ]
    
    for package in packages:
        try:
            subprocess.check_call([sys.executable, '-m', 'pip', 'install', package])
            print(f"Successfully installed {package}")
        except Exception as e:
            print(f"Error installing {package}: {str(e)}")

class QAModelEvaluator:
    def __init__(self, questions, answers, pdf_extractor):
        """Initialize evaluator with questions, answers and PDF extractor"""
        # First verify sentencepiece is properly installed
        try:
            import sentencepiece
            import importlib
            importlib.reload(sentencepiece)  # Reload the module to ensure it's properly initialized
        except ImportError:
            raise ImportError("Please restart your Python runtime after installing sentencepiece")
            
        self.questions = questions
        self.answers = answers
        self.pdf_extractor = pdf_extractor
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.results = {}
        self.scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
        
        print(f"Using device: {self.device}")
        self.prepare_contexts()

    def prepare_contexts(self):
        """Extract and match contexts for all questions"""
        text = self.pdf_extractor.extract_text()
        self.contexts, _ = self.pdf_extractor.create_context_windows(text)
        
        print("Matching contexts to questions...")
        self.matched_contexts = []
        batch_size = 32  # Process questions in batches
        
        for i in range(0, len(self.questions), batch_size):
            batch_questions = self.questions[i:i + batch_size]
            batch_contexts = [
                self.pdf_extractor.find_best_context(q, self.contexts)
                for q in tqdm(batch_questions, desc=f"Processing batch {i//batch_size + 1}")
            ]
            self.matched_contexts.extend(batch_contexts)

    def prepare_t5(self):
        """Load T5 model with proper configuration for QA"""
        print("Loading T5 model...")
        
        # Initialize with legacy=False to use new behavior
        self.t5_tokenizer = T5Tokenizer.from_pretrained('t5-base', legacy=False)
        
        # Use the correct model class for generation
        self.t5_model = T5ForConditionalGeneration.from_pretrained('t5-base').to(self.device)
        
        print("T5 model loaded successfully")

    def prepare_bert(self):
        """Load BERT model with proper initialization for QA"""
        print("Loading BERT model...")
        self.bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        
        # Initialize BERT with QA head weights
        self.bert_model = BertForQuestionAnswering.from_pretrained(
            'bert-base-uncased',
            return_dict=True
        ).to(self.device)
        
        # Fine-tune mode to avoid initialization warning
        self.bert_model.train(False)  # Explicitly set to eval mode
        
        print("BERT model loaded successfully")

    def prepare_gpt(self):
        """Load GPT model with proper padding configuration"""
        print("Loading GPT model...")
        self.gpt_tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        
        # Set left padding for GPT
        self.gpt_tokenizer.padding_side = 'left'
        self.gpt_tokenizer.pad_token = self.gpt_tokenizer.eos_token
        
        # Initialize model
        self.gpt_model = GPT2LMHeadModel.from_pretrained('gpt2').to(self.device)
        self.gpt_model.config.pad_token_id = self.gpt_tokenizer.eos_token_id
        
        print("GPT model loaded successfully")

    def calculate_metrics(self, predictions, ground_truth):
        """Calculate accuracy, precision, recall, and F1 score"""
        # Convert predictions and ground truth to lowercase for better comparison
        predictions = [str(pred).lower().strip() for pred in predictions]
        ground_truth = [str(true).lower().strip() for true in ground_truth]
        
        # Calculate exact match accuracy
        exact_matches = [pred == true for pred, true in zip(predictions, ground_truth)]
        accuracy = sum(exact_matches) / len(exact_matches)
        
        # Calculate overlap-based metrics
        true_positives = 0
        false_positives = 0
        false_negatives = 0
        
        for pred, true in zip(predictions, ground_truth):
            pred_words = set(pred.split())
            true_words = set(true.split())
            
            # Calculate intersection and differences
            overlap = pred_words.intersection(true_words)
            true_positives += len(overlap)
            false_positives += len(pred_words - true_words)
            false_negatives += len(true_words - pred_words)
        
        # Calculate precision, recall, and F1
        precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
        recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1
        }

    def evaluate_model(self, model, tokenizer, model_name):
        """Evaluate a model on the data"""
        model.eval()
        predictions = []
        batch_size = 8  # Smaller batch size for GPU memory
        
        # Configure maximum lengths based on model
        if model_name == 't5':
            max_length = 512
            max_context_length = 384  # Reserve space for question
        elif model_name == 'bert':
            max_length = 512
            max_context_length = 384  # BERT's recommended split
        else:  # gpt
            max_length = 1024
            max_context_length = 896
        
        print("Evaluating model " + model_name)
        
        for i in range(0, len(self.questions), batch_size):
            batch_questions = self.questions[i:i + batch_size]
            batch_contexts = self.matched_contexts[i:i + batch_size]
            
            # Properly truncate contexts while preserving question
            truncated_contexts = []
            for question, context in zip(batch_questions, batch_contexts):
                # Tokenize question and context separately
                question_tokens = tokenizer.tokenize(question)
                context_tokens = tokenizer.tokenize(context)
                
                # Calculate available space for context
                available_length = max_context_length - len(question_tokens) - 3  # Special tokens
                if available_length > 0:
                    context_tokens = context_tokens[:available_length]
                    
                truncated_contexts.append(tokenizer.convert_tokens_to_string(context_tokens))
            
            # Prepare inputs with truncated contexts
            inputs = tokenizer(
                batch_questions,
                truncated_contexts,
                return_tensors='pt',
                max_length=max_length,
                truncation='longest_first',
                padding=True,
                return_overflowing_tokens=False  # Disable overflow warning
            ).to(self.device)
            
            with torch.no_grad():
                if model_name == 'bert':
                    inputs = tokenizer(
                        batch_questions,
                        batch_contexts,
                        return_tensors='pt',
                        max_length=512,
                        truncation=True,
                        padding=True
                    ).to(self.device)
                    
                    outputs = model(**inputs)
                    start_logits = outputs.start_logits
                    end_logits = outputs.end_logits
                    
                    for j in range(len(batch_questions)):
                        start_idx = torch.argmax(start_logits[j])
                        end_idx = torch.argmax(end_logits[j]) + 1
                        prediction = tokenizer.decode(inputs['input_ids'][j][start_idx:end_idx])
                        predictions.append(prediction)
                
                elif model_name == 't5':
                    # Format input specifically for T5
                    formatted_inputs = [
                        f"question: {question} context: {context}" 
                        for question, context in zip(batch_questions, batch_contexts)
                    ]
                    
                    # Tokenize with appropriate padding and truncation
                    inputs = tokenizer(
                        formatted_inputs,
                        return_tensors='pt',
                        max_length=512,
                        truncation=True,
                        padding=True
                    ).to(self.device)
                    
                    # Generate answers
                    outputs = model.generate(
                        input_ids=inputs['input_ids'],
                        attention_mask=inputs['attention_mask'],
                        max_length=64,  # Shorter for answers
                        min_length=1,
                        num_beams=4,
                        length_penalty=1.0,
                        early_stopping=True,
                        no_repeat_ngram_size=2
                    )
                    
                    # Decode predictions
                    batch_predictions = [
                        tokenizer.decode(output, skip_special_tokens=True)
                        for output in outputs
                    ]
                    predictions.extend(batch_predictions)
                
                else:  # gpt
                    inputs = tokenizer(
                        [f"Question: {q} Context: {c}" for q, c in zip(batch_questions, batch_contexts)],
                        return_tensors='pt',
                        max_length=1024,
                        truncation=True,
                        padding=True
                    ).to(self.device)
                    
                    outputs = model.generate(
                        **inputs,
                        max_length=1024 + 64,
                        pad_token_id=tokenizer.eos_token_id,
                        num_return_sequences=1
                    )
                    
                    batch_predictions = [
                        tokenizer.decode(output, skip_special_tokens=True)
                        for output in outputs
                    ]
                    predictions.extend(batch_predictions)

        
        return self.calculate_metrics(predictions, self.answers)

    def run_evaluation(self):
        
        # First ensure T5 dependencies are installed
        # setup_environment()
        
        """Run evaluation for all models"""
        for model_name, prepare_func in [
            ('t5', self.prepare_t5),
            ('bert', self.prepare_bert),
            ('gpt', self.prepare_gpt)
        ]:
            try:
                prepare_func()
                if model_name == 't5':
                    self.results['t5'] = self.evaluate_model(self.t5_model, self.t5_tokenizer, 't5')
                elif model_name == 'bert':
                    self.results['bert'] = self.evaluate_model(self.bert_model, self.bert_tokenizer, 'bert')
                else:
                    self.results['gpt'] = self.evaluate_model(self.gpt_model, self.gpt_tokenizer, 'gpt')
            except Exception as e:
                print(f"Error evaluating {model_name}: {str(e)}")
        
        return self.results

    def print_results(self):
        """Print evaluation results in a formatted way"""
        df = pd.DataFrame(self.results).round(4)
        print("\nModel Comparison Results:")
        print(df)
        
        best_model = {
            metric: df.loc[metric].idxmax()
            for metric in df.index
        }
        
        print("\nBest performing models:")
        for metric, model in best_model.items():
            print(f"{metric}: {model} ({df.loc[metric, model]:.4f})")

def main():
    
    # First check if runtime needs restart
    try:
        import sentencepiece
    except ImportError:
        print("Please run the following commands:")
        print("1. pip install sentencepiece protobuf")
        print("2. Restart your Python runtime")
        print("3. Run this script again")
        return
    
    # Parse command line arguments
    args = parse_arguments()
    
    try:
        # Load JSON data
        print(f"Loading QA pairs from {args.json_path}...")
        json_data = JSONDataLoader.load_json(args.json_path)
        
        # Extract questions and answers
        questions = [item['question'] for item in json_data]
        answers = [item['answer'] for item in json_data]
        
        print(f"Loaded {len(questions)} QA pairs")
        
        # Initialize PDF extractor
        print(f"Initializing PDF extractor for {args.pdf_path}...")
        pdf_extractor = PDFContextExtractor(args.pdf_path)
        
        # Run evaluation
        print("Initializing model evaluation...")
        evaluator = QAModelEvaluator(questions[0:11], answers[0:11], pdf_extractor)
        results = evaluator.run_evaluation()
        
        # Print and save results
        evaluator.print_results()
        
        # Save results to file
        output_file = 'evaluation_results.json'
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=4)
        print(f"\nResults have been saved to '{output_file}'")
        
    except Exception as e:
        print(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    main()
