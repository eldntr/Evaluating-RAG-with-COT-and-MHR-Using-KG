from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM

tokenizer = AutoTokenizer.from_pretrained("Babelscape/rebel-large")
model = AutoModelForSeq2SeqLM.from_pretrained("Babelscape/rebel-large")

def extract_entities_from_model_output(text):
    entities = set()
    relation, subject, object_ = '', '', ''
    text = text.strip()
    current = 'x'
    text_replaced = text.replace("<s>", "").replace("<pad>", "").replace("</s>", "")
    for token in text_replaced.split():
        if token == "<triplet>":
            current = 't'
            if subject:
                entities.add(subject.strip())
            subject = ''
        elif token == "<subj>":
            current = 's'
            if object_:
                entities.add(object_.strip())
            object_ = ''
        elif token == "<obj>":
            current = 'o'
        else:
            if current == 't':
                subject += ' ' + token
            elif current == 's':
                object_ += ' ' + token
    if subject:
        entities.add(subject.strip())
    if object_:
        entities.add(object_.strip())
    return list(entities)

def extract_entities(question, verbose=False):
    inputs = tokenizer([question], return_tensors="pt")
    num_return_sequences = 5
    gen_kwargs = {
        "max_length": 100,
        "length_penalty": 0,
        "num_beams": 5,
        "num_return_sequences": num_return_sequences
    }
    generated_tokens = model.generate(
        **inputs,
        **gen_kwargs,
    )
    decoded_preds = tokenizer.batch_decode(generated_tokens, skip_special_tokens=False)
    entities = set()
    for sentence_pred in decoded_preds:
        extracted_entities = extract_entities_from_model_output(sentence_pred)
        entities.update(extracted_entities)
    if verbose:
        print("Extracted entities:")
        for entity in entities:
            print(f"  {entity}")
    return list(entities)
