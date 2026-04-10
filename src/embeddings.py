import os
import glob
import time
import numpy as np
import pandas as pd
import torch
import pickle
from tqdm import tqdm
from transformers import BertTokenizer, BertModel

def get_bert_model_and_tokenizer(model_name='bert-base-uncased', device=None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = BertTokenizer.from_pretrained(model_name)
    model = BertModel.from_pretrained(model_name).to(device)
    model.eval()
    return tokenizer, model, device

def prepare_embedding_text(df, title_col='title', content_col='content', max_tokens=512, tokenizer=None):
    """
    Combines text and ensures it is within BERT's token limit (512 tokens).
    """
    print("Combining text and enforcing token limits for BERT...")
    if 'to_pandas' in dir(df):
        df = df.to_pandas()
    df_prep = df.copy()

    df_prep['text_clean'] = df_prep[title_col].fillna('') + "\n\n" + df_prep[content_col].fillna('')

    if tokenizer is None:
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    def truncate_bert(text):
        text = str(text).replace("\n", " ")
        # Tokenize and truncate to max_tokens - 2 (for [CLS] and [SEP])
        tokens = tokenizer.encode(text, add_special_tokens=False, max_length=max_tokens-2, truncation=True)
        return tokenizer.decode(tokens)

    tqdm.pandas(desc="Truncating text")
    df_prep['safe_content'] = df_prep['text_clean'].progress_apply(truncate_bert)
    return df_prep

def get_bert_embedding(text, tokenizer, model, device, max_length=512):
    """
    Gets the BERT embedding for a single string.
    Uses the mean of the last hidden state as the combined embedding.
    """
    inputs = tokenizer(text, return_tensors='pt', padding=True, truncation=True, max_length=max_length)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    with torch.no_grad():
        outputs = model(**inputs)
    
    # Use mean pooling over the sequence length
    attention_mask = inputs['attention_mask']
    token_embeddings = outputs.last_hidden_state
    
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
    sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    embedding = sum_embeddings / sum_mask
    
    return embedding.cpu().numpy()[0].tolist()

def generate_and_save_embeddings(texts_to_embed, checkpoint_dir="./checkpoints", batch_size=32):
    """
    Generates BERT embeddings for a list of texts and saves them in chunks.
    """
    SAVE_INTERVAL = 5000 
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    tokenizer, model, device = get_bert_model_and_tokenizer()

    current_chunk_embeddings = []
    chunk_index = 0
    processed_count = 0

    # Process in batches
    for i in tqdm(range(0, len(texts_to_embed), batch_size), desc="Extracting BERT Embeddings"):
        batch_texts = texts_to_embed[i:i + batch_size]
        
        inputs = tokenizer(batch_texts, return_tensors='pt', padding=True, truncation=True, max_length=512)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = model(**inputs)
            
        attention_mask = inputs['attention_mask']
        token_embeddings = outputs.last_hidden_state
        
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        batch_embeddings = (sum_embeddings / sum_mask).cpu().numpy()
        
        current_chunk_embeddings.extend(batch_embeddings)
        processed_count += len(batch_texts)

        # Checkpoint Trigger
        if len(current_chunk_embeddings) >= SAVE_INTERVAL or processed_count == len(texts_to_embed):
            chunk_array = np.array(current_chunk_embeddings)
            filename = f"{checkpoint_dir}/embeddings_part_{chunk_index}.pkl"
            with open(filename, 'wb') as f:
                pickle.dump(chunk_array, f)
            current_chunk_embeddings = []
            chunk_index += 1

    # Merge Checkpoints
    all_files = sorted(glob.glob(f"{checkpoint_dir}/embeddings_part_*.pkl"), 
                       key=lambda x: int(x.split('_part_')[1].split('.pkl')[0]))

    if len(all_files) > 0:
        embeddings_list = []
        for f in all_files:
            with open(f, 'rb') as fp:
                embeddings_list.append(pickle.load(fp))
        embeddings = np.vstack(embeddings_list)
        print(f"Final BERT Embeddings shape: {embeddings.shape}")
        return embeddings
    else:
        print("No embeddings matched or saved.")
        return np.array([])
