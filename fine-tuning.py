import spacy
import json
import random
from spacy.training import Example

# Load the small English model
nlp = spacy.load('en_core_web_sm')

# Load the dataset (this should be a JSON file containing text and annotations)
with open('ner_dataset.json', 'r') as f:
    data = json.load(f)

# Preprocess the data to create training examples
train_data = []
for item in data:
    text = item['text']
    annotations = item['annotations']
    entities = [(start, end, label) for start, end, label in annotations]
    train_data.append((text, {"entities": entities}))

# Shuffle the data
random.shuffle(train_data)

# Display a sample of the training data
print(train_data[0])

# Disable other pipeline components in the SpaCy model
pipe_exceptions = ["ner"]
unaffected_pipes = [pipe for pipe in nlp.pipe_names if pipe not in pipe_exceptions]

# Start the training
with nlp.disable_pipes(*unaffected_pipes):  # Disable other pipes
    optimizer = nlp.begin_training()
    for epoch in range(10):
        random.shuffle(train_data)
        losses = {}
        for text, annotations in train_data:
            doc = nlp.make_doc(text)
            example = Example.from_dict(doc, annotations)
            nlp.update([example], losses=losses, drop=0.3)
        print(f"Epoch {epoch + 1}, Loss: {losses['ner']}")

# Save the fine-tuned model to a directory
output_dir = "ner_finetuned_model"
nlp.to_disk(output_dir)
print(f"Model saved to {output_dir}")

# Load the fine-tuned model for inference
nlp_finetuned = spacy.load(output_dir)

# Test the fine-tuned model on a new text
test_text = "Apple is looking at buying U.K. startup for $1 billion"
doc = nlp_finetuned(test_text)
print("Entities in the test text:")
for ent in doc.ents:
    print(ent.text, ent.label_)

    
from spacy.scorer import Scorer
from spacy.training import Example

def evaluate_model(nlp, examples):
    scorer = Scorer()
    for input_, annot in examples:
        doc = nlp.make_doc(input_)
        example = Example.from_dict(doc, annot)
        pred_value = nlp(input_)
        scorer.score(example)
    return scorer.scores

# Evaluate the model on a test set
test_data = [("Apple is looking at buying U.K. startup for $1 billion", {"entities": [(0, 5, "ORG"), (27, 30, "GPE"), (31, 38, "ORG")]}),
             # Add more test examples here
            ]

results = evaluate_model(nlp_finetuned, test_data)
print(f"Evaluation Results: {results}")
