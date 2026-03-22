# From notebooks\cleaning.ipynb
def get_combined_embedding(text, model="text-embedding-3-small"):
    # Convert to string and clean
    text = str(text).replace("\n", " ")
    tokens = encoding.encode(text)

    # If it fits in one go, just return the single embedding
    if len(tokens) <= 8191:
        return client.embeddings.create(input=[text], model=model).data[0].embedding

    # If too long, split into chunks
    chunks = [tokens[i : i + CHUNK_SIZE] for i in range(0, len(tokens), CHUNK_SIZE)]
    chunk_strings = [encoding.decode(c) for c in chunks]

    # Get embeddings for all chunks in one API call (batching)
    response = client.embeddings.create(input=chunk_strings, model=model)
    embeddings = [r.embedding for r in response.data]

    # Average the vectors to get a single representative embedding
    return np.mean(embeddings, axis=0).tolist()

# From notebooks\data_visualisation.ipynb
def plot_burstiness(df, title_suffix=""):

    data = df.copy()

    # 1. Convert to datetime (adjust format if needed)
    data['created_utc'] = pd.to_datetime(data['created_utc'], format='ISO8601')

    # 2. Sort chronologically per author
    data = data.sort_values(['author', 'created_utc'])

    # 3. Calculate time differences in seconds
    data['delta_t'] = data.groupby('author')['created_utc'].diff().dt.total_seconds()

    # 4. Clean the data
    delta = data['delta_t'].dropna()               # remove NaN (first post of each author)
    delta = delta[np.isfinite(delta)]               # remove inf/-inf (just in case)
    delta = delta[delta > 0]                         # keep only positive intervals

    # 5. Handle zeros (if any) – add a tiny epsilon so log scale works
    if (delta == 0).any():
        print("Warning: zero-second intervals detected. Adding 1e-6 seconds to avoid log(0).")
        delta = delta.replace(0, 1e-6)

    # 6. Check if we have data to plot
    if len(delta) == 0:
        print("No valid intervals after cleaning.")
        return

    # 7. Plot
    plt.figure(figsize=(10, 5))
    sns.histplot(delta, bins=100, kde=True, log_scale=True, color='crimson')
    plt.title(f"Graph 1: Burstiness (Temporal Signature) for {title_suffix}s")
    plt.xlabel("Seconds between events (Log Scale)")
    plt.ylabel("Activity Count")
    plt.show()

# From notebooks\data_visualisation.ipynb
def plot_contextual_pivot(df, embs):
    """
    df: DataFrame sorted by ['author', 'created_utc']
    embs: array-like of embeddings (each element is a list/array of floats)
    """
    pivots = []
    for i in range(1, len(embs)):
        if df.iloc[i]['author'] == df.iloc[i-1]['author']:
            # Wrap each embedding in a list to make it 2D: shape (1, n_features)
            sim = cosine_similarity([embs[i]], [embs[i-1]])[0][0]
            pivots.append(1 - sim)

    plt.figure(figsize=(10, 5))
    sns.kdeplot(pivots, fill=True, color='forestgreen')
    plt.title("Graph 2: Contextual Pivot (Semantic Topic Shifting)")
    plt.xlabel("Semantic Distance (0 = Same Topic, 1 = Total Shift)")
    plt.show()

# From notebooks\data_visualisation.ipynb
def plot_efficiency(df):
    """
    Creates an interactive scatter plot of total words produced vs total engagement (upvotes)
    for each author, colored by human/agent label.

    Parameters
    ----------
    df : pandas.DataFrame
        Must contain columns:
        - 'author'       : author identifier
        - 'content'      : text of the post/comment
        - 'score'        : number of upvotes (engagement)
        - 'label'        : 0 for human, 1 for agent (or any categorical)
    """
    # Compute word count for each row
    df = df.copy()
    df['word_count'] = df['content'].fillna('').str.split().str.len()

    # Aggregate per author
    author_stats = df.groupby('author').agg(
        total_words=('word_count', 'sum'),
        total_engagement=('score', 'sum'),
        label=('label', 'first')   # assume label is constant per author
    ).reset_index()

    # Create scatter plot
    fig = px.scatter(
        author_stats,
        x='total_words',
        y='total_engagement',
        color='label',                # 0 = human, 1 = agent
        hover_data=['author'],
        trendline='ols',
        log_x=True,                    # often words span orders of magnitude
        log_y=True,                     # engagement can also be wide‑ranging
        title='Graph 3: Engagement‑to‑Output Efficiency'
    )

    # Improve axis labels
    fig.update_layout(
        xaxis_title='Total Words Produced',
        yaxis_title='Total Engagement (upvotes)'
    )

    fig.show()

# From notebooks\data_visualisation.ipynb
def plot_24hr_heartbeat(df_ai, df_human):
    # 1. Extract hour of day (0-23)
    ai_hours = df_ai['created_utc'].dt.hour
    human_hours = df_human['created_utc'].dt.hour

    # 2. Calculate percentage of activity per hour (normalizing for different dataset sizes)
    ai_counts = ai_hours.value_counts(normalize=True).sort_index() * 100
    human_counts = human_hours.value_counts(normalize=True).sort_index() * 100

    # Ensure all hours 0-23 are present
    for h in range(24):
        if h not in ai_counts: ai_counts[h] = 0
        if h not in human_counts: human_counts[h] = 0
    ai_counts = ai_counts.sort_index()
    human_counts = human_counts.sort_index()

    # 3. Plotting
    plt.figure(figsize=(14, 6))

    # Use a smooth line with fill for the "heartbeat" look
    plt.plot(ai_counts.index, ai_counts.values, label='AI Agents (Moltbook)', 
             color='#E63946', linewidth=3, marker='o', markersize=4)
    plt.fill_between(ai_counts.index, ai_counts.values, color='#E63946', alpha=0.1)

    plt.plot(human_counts.index, human_counts.values, label='Humans (Reddit Baseline)', 
             color='#457B9D', linewidth=3, linestyle='--', marker='s', markersize=4)
    plt.fill_between(human_counts.index, human_counts.values, color='#457B9D', alpha=0.1)

    # 4. Formatting
    plt.title('The 24-Hour Heartbeat: Activity Density by Hour of Day', fontsize=16, fontweight='bold', pad=20)
    plt.xlabel('Hour of Day (UTC)', fontsize=12)
    plt.ylabel('% of Total Activity', fontsize=12)
    plt.xticks(range(24))
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(fontsize=11, loc='upper right')

    # Add interpretation zones
    plt.axvspan(0, 6, color='gray', alpha=0.05, label='Typical Human Sleep (UTC)')

    plt.tight_layout()
    plt.show()

# From notebooks\data_visualisation_improved.ipynb
def compute_deltas(df):
    """Return clean inter-event intervals in seconds for every author."""
    d = df.sort_values(['author', 'created_utc']).copy()
    d['delta_t'] = d.groupby('author')['created_utc'].diff().dt.total_seconds()
    delta = d['delta_t'].dropna()
    delta = delta[np.isfinite(delta) & (delta > 0)]
    return delta


# From notebooks\data_visualisation_improved.ipynb
def burstiness_coeff(delta):
    mu, sigma = delta.mean(), delta.std()
    return (sigma - mu) / (sigma + mu)


# From notebooks\data_visualisation_improved.ipynb
def compute_semantic_shifts(df, max_per_author=200, seed=42):
    """
    Returns a DataFrame with columns: [author, interaction_type, semantic_distance]
    Computes cosine distance between consecutive embeddings per author.
    Caps at max_per_author pairs per author for performance.
    """
    rng = np.random.default_rng(seed)
    records = []
    df_sorted = df.sort_values(['author', 'created_utc']).reset_index(drop=True)

    for author, grp in df_sorted.groupby('author'):
        if len(grp) < 2:
            continue
        embs = np.stack(grp['embedding'].values)  # shape (N, D)
        types = grp['interaction_type'].values

        # Consecutive pairs
        pairs = list(range(len(grp) - 1))
        if len(pairs) > max_per_author:
            pairs = rng.choice(pairs, max_per_author, replace=False).tolist()

        for i in pairs:
            sim = cosine_similarity(embs[i:i+1], embs[i+1:i+2])[0][0]
            dist = 1 - float(sim)
            # Label the transition type
            t_from = types[i]
            t_to   = types[i + 1]
            if t_from == 'post' and t_to == 'post':
                transition = 'Post → Post'
            elif t_from == 'comment' and t_to == 'comment':
                transition = 'Comment → Comment'
            else:
                transition = 'Cross-type'
            records.append({'author': author, 'transition': transition,
                            'semantic_distance': dist})

    return pd.DataFrame(records)


# From notebooks\data_visualisation_improved.ipynb
def build_author_stats(df, group_label):
    d = df.copy()
    d['word_count'] = d['content'].fillna('').str.split().str.len()
    stats_df = d.groupby('author').agg(
        total_words      = ('word_count', 'sum'),
        total_engagement = ('score',      'sum'),
        n_posts          = ('interaction_type', lambda x: (x == 'post').sum()),
        n_comments       = ('interaction_type', lambda x: (x == 'comment').sum()),
    ).reset_index()
    stats_df['group'] = group_label
    # Efficiency: engagement per 100 words (avoid div/0)
    stats_df['efficiency'] = (
        stats_df['total_engagement'] / (stats_df['total_words'] + 1) * 100
    )
    return stats_df


# From notebooks\embedding.ipynb
def prepare_text(text):
    text = str(text).replace("\n", " ")
    tokens = encoding.encode(text, disallowed_special=())
    if len(tokens) > MAX_TOKENS:
        return encoding.decode(tokens[:MAX_TOKENS])
    return text


# From notebooks\featuring_openaiembedding.ipynb
def clean_text(text):
    if not isinstance(text, str):
        return ''
    # Remove URLs
    text = re.sub(r'http\S+|www\.\S+', '', text)
    # Remove whitespace but keep newlines for paragraph counting
    text = re.sub(r'[ \t]+', ' ', text)
    return text.strip()


# From notebooks\featuring_openaiembedding.ipynb
def vocab_features(text):
    words = [w.lower() for w in text.split() if w.isalpha()]
    if len(words) == 0:
        return 0.0, 0.0
    freq = Counter(words)
    ttr = len(freq) / len(words)
    hapax = sum(1 for w, c in freq.items() if c == 1) / len(words)
    return ttr, hapax


# From notebooks\featuring_openaiembedding.ipynb
def stopword_ratio(text):
    words = text.lower().split()
    if len(words) == 0:
        return 0.0
    return sum(1 for w in words if w in STOPWORDS) / len(words)


# From notebooks\featuring_openaiembedding.ipynb
def punctuation_density(text):
    if len(text) == 0:
        return 0.0
    return sum(1 for c in text if c in string.punctuation) / len(text)


# From notebooks\featuring_openaiembedding.ipynb
def special_punct_counts(text):
    excl = text.count('!')
    quest = text.count('?')
    ellipsis = text.count('...')
    emoji_count = len(re.findall(r'[\U00010000-\U0010ffff]', text))
    return excl, quest, ellipsis, emoji_count


# From notebooks\featuring_openaiembedding.ipynb
def parse_timestamp(ts):
    if pd.isna(ts):
        return pd.NaT
    ts = str(ts).strip()
    # Unix timestamp (numeric)
    try:
        val = float(ts)
        if val > 1e12:
            val = val / 1000
        return pd.to_datetime(val, unit='s', utc=True)
    except (ValueError, OverflowError):
        pass
    # ISO format
    try:
        return pd.to_datetime(ts, utc=True)
    except Exception:
        return pd.NaT


# From notebooks\featuring_openaiembedding.ipynb
def pos_distribution(text):
    try:
        words = word_tokenize(text[:2000])  # Truncate for speed
        tags = pos_tag(words)
        total = len(tags)
        if total == 0:
            return {f'pos_{t}': 0.0 for t in POS_CATEGORIES}
        counts = Counter(t for _, t in tags)
        return {f'pos_{t}': counts.get(t, 0) / total for t in POS_CATEGORIES}
    except Exception:
        return {f'pos_{t}': 0.0 for t in POS_CATEGORIES}


# From notebooks\featuring_openaiembedding.ipynb
def ngram_repetition_rate(text, n=3):
    words = text.lower().split()
    if len(words) < n:
        return 0.0
    ngrams = [tuple(words[i:i+n]) for i in range(len(words) - n + 1)]
    if len(ngrams) == 0:
        return 0.0
    freq = Counter(ngrams)
    repeated = sum(1 for ng, c in freq.items() if c > 1)
    return repeated / len(freq)


# From notebooks\featuring_openaiembedding.ipynb
def hedging_score(text):
    text_lower = text.lower()
    word_count = len(text.split())
    if word_count == 0:
        return 0.0
    count = sum(text_lower.count(phrase) for phrase in HEDGING_PHRASES)
    return count / word_count * 100


# From notebooks\featuring_openaiembedding.ipynb
def self_reference_rate(text):
    words = text.lower().split()
    if len(words) == 0:
        return 0.0
    return sum(1 for w in words if w in FIRST_PERSON) / len(words)


# From notebooks\featuring_openaiembedding.ipynb
def formality_score(row):
    formal = row.get('pos_NN', 0) + row.get('pos_NNS', 0) + row.get('pos_NNP', 0) + \
             row.get('pos_JJ', 0) + row.get('pos_JJR', 0) + row.get('pos_JJS', 0) + \
             row.get('pos_IN', 0) + row.get('pos_DT', 0)
    informal = row.get('pos_VB', 0) + row.get('pos_VBD', 0) + row.get('pos_VBG', 0) + \
               row.get('pos_RB', 0) + row.get('pos_RBR', 0) + row.get('pos_PRP', 0)
    total = formal + informal
    if total == 0:
        return 0.5
    return formal / total


# From notebooks\featuring_openaiembedding.ipynb
def compute_perplexity(text, max_length=512):
    try:
        encodings = ppl_tokenizer(text, return_tensors="pt", truncation=True, max_length=max_length)
        input_ids = encodings.input_ids.to(DEVICE)
        if input_ids.shape[1] < 2:
            return np.nan
        with torch.no_grad():
            outputs = ppl_model(input_ids, labels=input_ids)
        return torch.exp(outputs.loss).item()
    except Exception:
        return np.nan


# From notebooks\featuring_openaiembedding.ipynb
def burstiness(text):
    sents = sent_tokenize(text)
    if len(sents) < 2:
        return 0.0
    lengths = [len(s.split()) for s in sents]
    mean_len = np.mean(lengths)
    if mean_len == 0:
        return 0.0
    return np.std(lengths) / mean_len


# From notebooks\featuring_openaiembedding.ipynb
def sentiment_features(text):
    scores = vader.polarity_scores(text)
    sents = sent_tokenize(text)
    if len(sents) >= 2:
        sent_compounds = [vader.polarity_scores(s)['compound'] for s in sents]
        variability = np.std(sent_compounds)
    else:
        variability = 0.0
    return scores['compound'], scores['pos'], scores['neg'], scores['neu'], variability


# From notebooks\featuring_openaiembedding.ipynb
def prepare_text(text):
    text = str(text).replace("\n", " ")
    tokens = encoding.encode(text, disallowed_special=())
    if len(tokens) > MAX_TOKENS:
        return encoding.decode(tokens[:MAX_TOKENS])
    return text


# From notebooks\mehmeh.ipynb
def save_unsplit_features(df, embeddings, output_dir="../data", dataset_prefix="data"):
    # Horizontally stack the comment_existence feature with the embeddings for the ENTIRE dataset
    X_full = np.hstack([
        df[['comment_existence']].values,
        embeddings
    ])

    # Save the combined full features to a single numpy file
    features_filename = f"{output_dir}/{dataset_prefix}_features_full.npy"
    np.save(features_filename, X_full)
    print(f"Saved full un-split features shape: {X_full.shape} to {features_filename}")

    return df


# From notebooks\model.ipynb
def train(model, dataloader, optimizer, device):
    model.train()
    total_loss = 0
    total_preds = []
    total_labels = []

    for step, batch in enumerate(tqdm(dataloader, desc="Training")):
        # 1. MOVE TO GPU: Send the current batch of 32 rows to the GPU
        input_ids = batch["input_ids"].to(device)
        attention_masks = batch["attention_masks"].to(device)
        labels = batch["labels"].to(device)

        optimizer.zero_grad()

        # 2. RUN ON GPU: The model processes the batch
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_masks,
            labels=labels
        )

        loss = outputs.loss
        logits = outputs.logits
        total_loss += loss.item() # .item() safely extracts the number from the GPU

        # 3. RUN ON GPU: Calculate gradients and update weights
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        # 4. GET PREDICTIONS ON GPU: Find the highest probability class (0 or 1)
        preds = torch.argmax(logits, dim=1)

        # 5. MOVE BACK TO CPU: 
        # .detach()  -> Unhooks the data from the GPU's gradient graph
        # .cpu()     -> Moves it from GPU VRAM back to standard computer RAM
        # .numpy()   -> Converts it from a PyTorch tensor to a standard Python/NumPy list
        total_preds.extend(preds.detach().cpu().numpy())
        total_labels.extend(labels.detach().cpu().numpy())

    # 6. RUN ON CPU: Scikit-learn calculates the final score using the CPU arrays
    avg_loss = total_loss / len(dataloader)
    accuracy = accuracy_score(total_labels, total_preds)
    f1score = f1_score(total_labels, total_preds)

    return avg_loss, accuracy, f1score


# From notebooks\model.ipynb
def evaluate(model, dataloader, device):
    model.eval()
    total_loss = 0
    total_preds = []
    total_labels = []

    with torch.no_grad(): # Disables gradient tracking entirely to save GPU memory
        for batch in tqdm(dataloader, desc="Evaluating"):
            # 1. MOVE TO GPU
            input_ids = batch["input_ids"].to(device)
            attention_masks = batch["attention_masks"].to(device)
            labels = batch["labels"].to(device)

            # 2. RUN ON GPU
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_masks,
                labels=labels
            )

            loss = outputs.loss
            logits = outputs.logits
            total_loss += loss.item()

            # 3. GET PREDICTIONS ON GPU
            preds = torch.argmax(logits, dim=1)

            # 4. MOVE BACK TO CPU for safe storage and metric calculation
            total_preds.extend(preds.detach().cpu().numpy())
            total_labels.extend(labels.detach().cpu().numpy())

    # 5. RUN ON CPU
    avg_loss = total_loss / len(dataloader)
    accuracy = accuracy_score(total_labels, total_preds)
    f1score = f1_score(total_labels, total_preds)

    return avg_loss, accuracy, f1score

# From notebooks\model.ipynb
def training_loop(model, train_loader, val_loader, optimizer, early_stopping, num_epochs, device):
    train_losses, val_losses = [], []
    train_accs, val_accs = [], []
    train_f1s, val_f1s = [], []

    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")

        train_loss, train_acc, train_f1 = train(
            model, train_loader, optimizer, device
        )

        val_loss, val_acc, val_f1 = evaluate(
            model, val_loader, device
        )

        print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
        print(f"Val Loss:   {val_loss:.4f} | Val Acc:   {val_acc:.4f}")

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)
        train_f1s.append(train_f1)
        val_f1s.append(val_f1)

        early_stopping(val_acc, model)

        if early_stopping.early_stop:
            print("Early stopping triggered. Halting training.")
            break

    # Reload the best weights from the directory before finishing
    print(f"\nTraining Complete. Loading best weights from '{early_stopping.path}'...")
    model.load_state_dict(torch.load(early_stopping.path, weights_only=True))

    return train_losses, val_losses, train_accs, val_accs, train_f1s, val_f1s

# From notebooks\model.ipynb
def evaluate_on_testing_set(y_test, y_pred):
  # Calculate AUC
  print("AUC is: ", roc_auc_score(y_test, y_pred))

  # print out recall and precision
  print(classification_report(y_test, y_pred))

  # print out confusion matrix
  print("Confusion Matrix: \n", confusion_matrix(y_test, y_pred))

  # # calculate points for ROC curve
  fpr, tpr, thresholds = roc_curve(y_test, y_pred)

  # Plot ROC curve
  plt.plot(fpr, tpr, label='ROC curve (area = %0.3f)' % roc_auc_score(y_test, y_pred))
  plt.plot([0, 1], [0, 1], 'k--')  # random predictions curve
  plt.xlim([0.0, 1.0])
  plt.ylim([0.0, 1.0])
  plt.xlabel('False Positive Rate or (1 - Specifity)')
  plt.ylabel('True Positive Rate or (Sensitivity)')
  plt.title('Receiver Operating Characteristic')

# From notebooks\moltbook_regression_prep.ipynb
def get_vader_compound(text):
    if pd.isna(text) or str(text).strip() == "":
        return 0.0
    return sia.polarity_scores(str(text))['compound']


# From notebooks\moltbook_regression_prep.ipynb
def get_early_comments(df_comments, post_id_col='post_id', time_col='created_at'):
    """Sorts and extracts only the first 10 comments for each post."""
    print("Extracting early comments...")
    df_comm = df_comments.copy()
    df_comm[time_col] = pd.to_datetime(df_comm[time_col])

    df_early = df_comm.sort_values([post_id_col, time_col])
    return df_early.groupby(post_id_col).head(10).copy()


# From notebooks\moltbook_regression_prep.ipynb
def engineer_comment_existence(df_early_comments, post_id_col='post_id'):
    """Calculates the Comment Existence fraction (0.0 to 1.0)."""
    print("Calculating Comment Existence feature...")

    existence_features = df_early_comments.groupby(post_id_col).agg(
        actual_comment_count=('id', 'count')
    )
    existence_features['comment_existence'] = existence_features['actual_comment_count'] / 10.0

    return existence_features[['comment_existence']]


# From notebooks\moltbook_regression_prep.ipynb
def engineer_early_sentiment(df_early_comments, post_id_col='post_id', text_col='content'):
    """Calculates VADER sentiment aggregates for the early comments."""
    print("Calculating VADER Sentiment features...")

    # Apply VADER scoring
    df_early_comments['vader_score'] = df_early_comments[text_col].apply(get_vader_compound)

    # Aggregate mean, max, and min
    sentiment_features = df_early_comments.groupby(post_id_col).agg(
        avg_early_sentiment=('vader_score', 'mean'),
        max_early_sentiment=('vader_score', 'max'),
        min_early_sentiment=('vader_score', 'min')
    )

    return sentiment_features


# From notebooks\moltbook_regression_prep.ipynb
def merge_engineered_features(df_posts, existence_df, sentiment_df, post_id_col_posts='id'):
    """Merges all engineered features back into the main posts dataframe and handles NaNs."""
    print("Merging features into main posts dataframe...")
    df_merged = df_posts.copy()

    # Merge Existence
    df_merged = df_merged.merge(
        existence_df, left_on=post_id_col_posts, right_index=True, how='left'
    )
    df_merged['comment_existence'] = df_merged['comment_existence'].fillna(0.0)

    # Merge Sentiment
    df_merged = df_merged.merge(
        sentiment_df, left_on=post_id_col_posts, right_index=True, how='left'
    )
    fill_cols = ['avg_early_sentiment', 'max_early_sentiment', 'min_early_sentiment']
    df_merged[fill_cols] = df_merged[fill_cols].fillna(0.0)

    return df_merged

# From notebooks\moltbook_regression_prep.ipynb
def truncate_text(text):
    MAX_TOKENS = 8191
    text = str(text).replace("\n", " ")
    tokens = encoding.encode(text, disallowed_special=())
    if len(tokens) > MAX_TOKENS:
        return encoding.decode(tokens[:MAX_TOKENS])
    return text


# From notebooks\moltbook_regression_prep.ipynb
def prepare_embedding_text(df, title_col='title', content_col='content'):
    print("Combining text and enforcing token limits...")
    df_prep = df.copy()

    # Combine title and content
    df_prep['text_clean'] = df_prep[title_col].fillna('') + "\n\n" + df_prep[content_col].fillna('')

    # Apply token truncation
    tqdm.pandas(desc="Truncating text")
    df_prep['safe_content'] = df_prep['text_clean'].progress_apply(truncate_text)

    return df_prep

# From notebooks\moltbook_regression_prep.ipynb
def generate_and_save_embeddings(texts_to_embed, checkpoint_dir="./checkpoints", model="text-embedding-3-small"):
    MAX_TOKENS_PER_REQUEST = 250000
    MAX_ROWS_PER_REQUEST = 1000
    TPM_LIMIT = 900000
    SAVE_INTERVAL = 5000 

    os.makedirs(checkpoint_dir, exist_ok=True)

    # 1. Dynamic Batching
    batches = []
    current_batch = []
    current_batch_tokens = 0

    for text in texts_to_embed:
        token_count = len(encoding.encode(text, disallowed_special=()))

        if current_batch_tokens + token_count > MAX_TOKENS_PER_REQUEST or len(current_batch) >= MAX_ROWS_PER_REQUEST:
            batches.append((current_batch, current_batch_tokens))
            current_batch = []
            current_batch_tokens = 0

        current_batch.append(text)
        current_batch_tokens += token_count

    if current_batch:
        batches.append((current_batch, current_batch_tokens))

    print(f"Divided data into {len(batches)} dynamic batches.")

    # 2. API Processing
    current_chunk_embeddings = []
    chunk_index = 0
    tokens_in_window = 0
    window_start_time = time.time()
    processed_count = 0

    for batch, batch_tokens in tqdm(batches, desc="Fetching Embeddings"):
        if tokens_in_window + batch_tokens > TPM_LIMIT:
            elapsed_time = time.time() - window_start_time
            if elapsed_time < 60:
                sleep_time = 60 - elapsed_time
                print(f"  [Rate Limit Paused] Sleeping for {sleep_time:.2f} seconds...")
                time.sleep(sleep_time)

            tokens_in_window = 0
            window_start_time = time.time()

        response = client.embeddings.create(input=batch, model=model)

        tokens_in_window += batch_tokens
        batch_embeddings = [item.embedding for item in response.data]
        current_chunk_embeddings.extend(batch_embeddings)
        processed_count += len(batch)

        # Checkpoint Trigger
        if len(current_chunk_embeddings) >= SAVE_INTERVAL or processed_count == len(texts_to_embed):
            chunk_array = np.array(current_chunk_embeddings)
            filename = f"{checkpoint_dir}/embeddings_part_{chunk_index}.npy"
            np.save(filename, chunk_array)
            current_chunk_embeddings = []
            chunk_index += 1

    # 3. Merge Checkpoints
    all_files = sorted(glob.glob(f"{checkpoint_dir}/embeddings_part_*.npy"), 
                       key=lambda x: int(x.split('_part_')[1].split('.npy')[0]))

    embeddings = np.vstack([np.load(f) for f in all_files])
    print(f"Final Embeddings shape: {embeddings.shape}")

    return embeddings

# From notebooks\moltbook_regression_prep.ipynb
def save_final_features(df, embeddings, output_dir="../data", dataset_prefix="data"):
    df_final = df.copy()

    # Make sure we have a split column
    if 'split' not in df_final.columns:
        np.random.seed(42)
        df_final['split'] = np.random.choice(['train', 'val', 'test'], size=len(df_final), p=[0.7, 0.15, 0.15])

    emb_cols = [f'emb_{i}' for i in range(embeddings.shape[1])]
    feature_cols = ['comment_existence'] + emb_cols

    # Save the splits to numpy files
    for split_name in ['train', 'val', 'test']:
        idx = (df_final['split'] == split_name).values

        # Combine the existence fraction + the embedding dimensions
        X_split = np.hstack([
            df_final.loc[idx, ['comment_existence']].values,
            embeddings[idx]
        ])

        split_filename = f"{output_dir}/{dataset_prefix}_features_{split_name}.npy"
        np.save(split_filename, X_split)
        print(f"[{split_name.upper()}] Saved features (shape: {X_split.shape}) to {split_filename}")

    return df_final

# From notebooks\preprocessing.ipynb
def clean_text(text):
    if not isinstance(text, str):
        return ''
    # Remove URLs
    text = re.sub(r'http\S+|www\.\S+', '', text)
    # Remove extra whitespace but keep single spaces
    text = re.sub(r'[ \t]+', ' ', text)
    return text.strip()


# From notebooks\reddit_regression_prep.ipynb
def get_vader_compound(text):
    if not text: return 0.0
    return sia.polarity_scores(str(text))['compound']


# From notebooks\reddit_regression_prep.ipynb
def truncate_text(text):
    MAX_TOKENS = 8191
    text = str(text).replace("\n", " ")
    tokens = encoding.encode(text, disallowed_special=())
    if len(tokens) > MAX_TOKENS:
        return encoding.decode(tokens[:MAX_TOKENS])
    return text


# From notebooks\reddit_regression_prep.ipynb
def generate_and_save_embeddings(texts_to_embed, checkpoint_dir="./checkpoints", model="text-embedding-3-small"):
    MAX_TOKENS_PER_REQUEST = 250000
    MAX_ROWS_PER_REQUEST = 1000
    TPM_LIMIT = 900000
    SAVE_INTERVAL = 5000 

    os.makedirs(checkpoint_dir, exist_ok=True)

    # 1. Dynamic Batching
    batches = []
    current_batch = []
    current_batch_tokens = 0

    for text in texts_to_embed:
        token_count = len(encoding.encode(text, disallowed_special=()))
        if current_batch_tokens + token_count > MAX_TOKENS_PER_REQUEST or len(current_batch) >= MAX_ROWS_PER_REQUEST:
            batches.append((current_batch, current_batch_tokens))
            current_batch = []
            current_batch_tokens = 0
        current_batch.append(text)
        current_batch_tokens += token_count

    if current_batch:
        batches.append((current_batch, current_batch_tokens))

    print(f"Divided data into {len(batches)} dynamic batches.")

    # 2. API Processing
    current_chunk_embeddings = []
    chunk_index = 0
    tokens_in_window = 0
    window_start_time = time.time()
    processed_count = 0

    for batch, batch_tokens in tqdm(batches, desc="Fetching Embeddings"):
        if tokens_in_window + batch_tokens > TPM_LIMIT:
            elapsed_time = time.time() - window_start_time
            if elapsed_time < 60:
                time.sleep(60 - elapsed_time)
            tokens_in_window = 0
            window_start_time = time.time()

        response = client.embeddings.create(input=batch, model=model)
        batch_embeddings = [item.embedding for item in response.data]
        current_chunk_embeddings.extend(batch_embeddings)
        processed_count += len(batch)

        # Checkpoint Trigger
        if len(current_chunk_embeddings) >= SAVE_INTERVAL or processed_count == len(texts_to_embed):
            np.save(f"{checkpoint_dir}/embeddings_part_{chunk_index}.npy", np.array(current_chunk_embeddings))
            current_chunk_embeddings = []
            chunk_index += 1

    # 3. Merge Checkpoints
    all_files = sorted(glob.glob(f"{checkpoint_dir}/embeddings_part_*.npy"), 
                       key=lambda x: int(x.split('_part_')[1].split('.npy')[0]))

    embeddings = np.vstack([np.load(f) for f in all_files])
    print(f"Final Embeddings shape: {embeddings.shape}")

    return embeddings


# From notebooks\reddit_regression_prep.ipynb
def save_final_features(df, embeddings, output_dir="../data", dataset_prefix="data"):
    # Convert exactly once to Pandas for the easy Numpy boolean indexing
    df_pd = df.to_pandas()

    if 'split' not in df_pd.columns:
        np.random.seed(42)
        df_pd['split'] = np.random.choice(['train', 'val', 'test'], size=len(df_pd), p=[0.7, 0.15, 0.15])

    for split_name in ['train', 'val', 'test']:
        idx = (df_pd['split'] == split_name).values

        X_split = np.hstack([
            df_pd.loc[idx, ['comment_existence']].values,
            embeddings[idx]
        ])

        split_filename = f"{output_dir}/{dataset_prefix}_features_{split_name}.npy"
        np.save(split_filename, X_split)
        print(f"[{split_name.upper()}] Saved features shape: {X_split.shape} to {split_filename}")

    return df_pd


# From notebooks\reddit_regression_prep.ipynb
def save_everything_to_pickle(df, embeddings, output_dir="../data", dataset_prefix="data"):
    # Convert exactly once to Pandas
    df_pd = df.to_pandas()

    # Convert the numpy array to a list of arrays and assign to a new column
    df_pd['embeddings'] = list(embeddings)

    # Save the entire dataframe (including embeddings) to a single pickle
    pickle_filename = f"{output_dir}/{dataset_prefix}_metadata_with_embeddings.pkl"
    df_pd.to_pickle(pickle_filename)
    print(f"Saved full dataframe with embeddings shape: {df_pd.shape} to {pickle_filename}")

    return df_pd


# From notebooks\reddit_regression_prep.ipynb
def save_unsplit_features(df, embeddings, output_dir="../data", dataset_prefix="data"):
    # Convert exactly once to Pandas
    df_pd = df.to_pandas()

    # Horizontally stack the comment_existence feature with the embeddings for the ENTIRE dataset
    # This keeps your full feature matrix ready for whenever you decide to split it later
    X_full = np.hstack([
        df_pd[['comment_existence']].values,
        embeddings
    ])

    # Save the combined full features to a single numpy file
    features_filename = f"{output_dir}/{dataset_prefix}_features_full2.npy"
    np.save(features_filename, X_full)
    print(f"Saved full un-split features shape: {X_full.shape} to {features_filename}")

    return df_pd


# From notebooks\reddit_regression_prep.ipynb
def get_vader_compound(text):
    if not text: return 0.0
    return sia.polarity_scores(str(text))['compound']


# From notebooks\reddit_regression_prep.ipynb
def get_vader_compound(text):
    if text is None or text == "":
        return 0.0
    return sia.polarity_scores(str(text))['compound']


# From notebooks\reddit_regression_prep.ipynb
def get_early_comments(df_comments, post_id_col='link_id', time_col='created_at'):
    print("Extracting early comments...")
    return (
        df_comments
        .sort([post_id_col, time_col])
        .group_by(post_id_col)
        .head(10)
    )


# From notebooks\reddit_regression_prep.ipynb
def engineer_comment_existence(df_early_comments, post_id_col='link_id'):
    print("Calculating Comment Existence...")
    return (
        df_early_comments
        .group_by(post_id_col)
        .agg(pl.len().alias('actual_comment_count'))
        .with_columns((pl.col('actual_comment_count') / 10.0).alias('comment_existence'))
        .select([post_id_col, 'comment_existence'])
    )


# From notebooks\reddit_regression_prep.ipynb
def engineer_early_sentiment(df_early_comments, post_id_col='link_id', text_col='body'):
    print("Calculating VADER Sentiment...")
    return (
        df_early_comments
        .with_columns(
            pl.col(text_col)
            .map_elements(get_vader_compound, return_dtype=pl.Float64)
            .alias('vader_score')
        )
        .group_by(post_id_col)
        .agg([
            pl.col('vader_score').mean().alias('avg_early_sentiment'),
            pl.col('vader_score').max().alias('max_early_sentiment'),
            pl.col('vader_score').min().alias('min_early_sentiment')
        ])
    )

# From notebooks\reddit_regression_prep.ipynb
def truncate_text(text):
    MAX_TOKENS = 8191
    text = str(text).replace("\n", " ")
    tokens = encoding.encode(text, disallowed_special=())
    if len(tokens) > MAX_TOKENS:
        return encoding.decode(tokens[:MAX_TOKENS])
    return text


# From notebooks\reddit_regression_prep.ipynb
def prepare_embedding_text(df, title_col='title', content_col='content'):
    print("Combining text and enforcing token limits...")
    df_prep = df.copy()

    # Combine title and content
    df_prep['text_clean'] = df_prep[title_col].fillna('') + "\n\n" + df_prep[content_col].fillna('')

    # Apply token truncation
    tqdm.pandas(desc="Truncating text")
    df_prep['safe_content'] = df_prep['text_clean'].progress_apply(truncate_text)

    return df_prep

# From notebooks\reddit_regression_prep.ipynb
def generate_and_save_embeddings(texts_to_embed, checkpoint_dir="./checkpoints", model="text-embedding-3-small"):
    MAX_TOKENS_PER_REQUEST = 250000
    MAX_ROWS_PER_REQUEST = 1000
    TPM_LIMIT = 900000
    SAVE_INTERVAL = 5000 

    os.makedirs(checkpoint_dir, exist_ok=True)

    # 1. Dynamic Batching
    batches = []
    current_batch = []
    current_batch_tokens = 0

    for text in texts_to_embed:
        token_count = len(encoding.encode(text, disallowed_special=()))

        if current_batch_tokens + token_count > MAX_TOKENS_PER_REQUEST or len(current_batch) >= MAX_ROWS_PER_REQUEST:
            batches.append((current_batch, current_batch_tokens))
            current_batch = []
            current_batch_tokens = 0

        current_batch.append(text)
        current_batch_tokens += token_count

    if current_batch:
        batches.append((current_batch, current_batch_tokens))

    print(f"Divided data into {len(batches)} dynamic batches.")

    # 2. API Processing
    current_chunk_embeddings = []
    chunk_index = 0
    tokens_in_window = 0
    window_start_time = time.time()
    processed_count = 0

    for batch, batch_tokens in tqdm(batches, desc="Fetching Embeddings"):
        if tokens_in_window + batch_tokens > TPM_LIMIT:
            elapsed_time = time.time() - window_start_time
            if elapsed_time < 60:
                sleep_time = 60 - elapsed_time
                print(f"  [Rate Limit Paused] Sleeping for {sleep_time:.2f} seconds...")
                time.sleep(sleep_time)

            tokens_in_window = 0
            window_start_time = time.time()

        response = client.embeddings.create(input=batch, model=model)

        tokens_in_window += batch_tokens
        batch_embeddings = [item.embedding for item in response.data]
        current_chunk_embeddings.extend(batch_embeddings)
        processed_count += len(batch)

        # Checkpoint Trigger
        if len(current_chunk_embeddings) >= SAVE_INTERVAL or processed_count == len(texts_to_embed):
            chunk_array = np.array(current_chunk_embeddings)
            filename = f"{checkpoint_dir}/embeddings_part_{chunk_index}.npy"
            np.save(filename, chunk_array)
            current_chunk_embeddings = []
            chunk_index += 1

    # 3. Merge Checkpoints
    all_files = sorted(glob.glob(f"{checkpoint_dir}/embeddings_part_*.npy"), 
                       key=lambda x: int(x.split('_part_')[1].split('.npy')[0]))

    embeddings = np.vstack([np.load(f) for f in all_files])
    print(f"Final Embeddings shape: {embeddings.shape}")

    return embeddings



# From notebooks\reddit_regression_prep.ipynb
def save_final_features(df, embeddings, output_dir="../data", dataset_prefix="data"):
    # Convert Polars DF to Pandas just for the final split logic to keep your numpy code identical
    df_final = df.to_pandas() 

    if 'split' not in df_final.columns:
        np.random.seed(42)
        df_final['split'] = np.random.choice(['train', 'val', 'test'], size=len(df_final), p=[0.7, 0.15, 0.15])

    # Save the splits to numpy files
    for split_name in ['train', 'val', 'test']:
        idx = (df_final['split'] == split_name).values

        # Combine the existence fraction + the embedding dimensions
        X_split = np.hstack([
            df_final.loc[idx, ['comment_existence']].values,
            embeddings[idx]
        ])

        split_filename = f"{output_dir}/{dataset_prefix}_features_{split_name}.npy"
        np.save(split_filename, X_split)
        print(f"[{split_name.upper()}] Saved features (shape: {X_split.shape}) to {split_filename}")

    return df_final

# From notebooks\reddit_regression_prep.ipynb
def get_vader_compound(text):
    if pd.isna(text) or str(text).strip() == "":
        return 0.0
    return sia.polarity_scores(str(text))['compound']


# From notebooks\reddit_regression_prep.ipynb
def get_early_comments(df_comments, post_id_col='post_id', time_col='created_at'):
    """Sorts and extracts only the first 10 comments for each post."""
    print("Extracting early comments...")
    df_comm = df_comments.copy()
    df_comm[time_col] = pd.to_datetime(df_comm[time_col])

    df_early = df_comm.sort_values([post_id_col, time_col])
    return df_early.groupby(post_id_col).head(10).copy()


# From notebooks\reddit_regression_prep.ipynb
def engineer_comment_existence(df_early_comments, post_id_col='post_id'):
    """Calculates the Comment Existence fraction (0.0 to 1.0)."""
    print("Calculating Comment Existence feature...")

    existence_features = df_early_comments.groupby(post_id_col).agg(
        actual_comment_count=('id', 'count')
    )
    existence_features['comment_existence'] = existence_features['actual_comment_count'] / 10.0

    return existence_features[['comment_existence']]


# From notebooks\reddit_regression_prep.ipynb
def engineer_early_sentiment(df_early_comments, post_id_col='post_id', text_col='content'):
    """Calculates VADER sentiment aggregates for the early comments."""
    print("Calculating VADER Sentiment features...")

    # Apply VADER scoring
    df_early_comments['vader_score'] = df_early_comments[text_col].apply(get_vader_compound)

    # Aggregate mean, max, and min
    sentiment_features = df_early_comments.groupby(post_id_col).agg(
        avg_early_sentiment=('vader_score', 'mean'),
        max_early_sentiment=('vader_score', 'max'),
        min_early_sentiment=('vader_score', 'min')
    )

    return sentiment_features


# From notebooks\reddit_regression_prep.ipynb
def merge_engineered_features(df_posts, existence_df, sentiment_df, post_id_col_posts='id'):
    """Merges all engineered features back into the main posts dataframe and handles NaNs."""
    print("Merging features into main posts dataframe...")
    df_merged = df_posts.copy()

    # Merge Existence
    df_merged = df_merged.merge(
        existence_df, left_on=post_id_col_posts, right_index=True, how='left'
    )
    df_merged['comment_existence'] = df_merged['comment_existence'].fillna(0.0)

    # Merge Sentiment
    df_merged = df_merged.merge(
        sentiment_df, left_on=post_id_col_posts, right_index=True, how='left'
    )
    fill_cols = ['avg_early_sentiment', 'max_early_sentiment', 'min_early_sentiment']
    df_merged[fill_cols] = df_merged[fill_cols].fillna(0.0)

    return df_merged

# From notebooks\regression_model.ipynb
def mean_absolute_percentage_error(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

