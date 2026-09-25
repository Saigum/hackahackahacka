
The method in src is a two-stage entity resolution pipeline. For each Source 1 business, it first finds plausible records in Sources 2 and 3, then uses a learned model to decide which of those records refer to the same business. This matches the problem statement’s requirement to allow zero, one, or many matches per Source 1 entity.

1. Prepare the text. [prepare.py (line 10)](/home/kuttyappan/Documents/9thSem/amazon_ml/src/prepare.py:10) transliterates characters, lowercases and tokenizes names and addresses, and standardizes common variants such as “Corporation” → “corp” and “Road” → “rd”. It also samples 50,000 training Source 1 entities for training and validation.
2. Generate candidates. [retrieve.cpp (line 17)](/home/kuttyappan/Documents/9thSem/amazon_ml/src/retrieve.cpp:17) builds an index of Source 2 and 3 name and address tokens, with country included in each index key. For a Source 1 record, it searches using its rarer tokens, scores retrieved records by weighted name and address overlap, and keeps the top combined results plus extra name-focused and address-focused results. The default limits yield up to 74 candidates per entity.
3. Describe each pair and train a matcher. [retrieve.cpp (line 13)](/home/kuttyappan/Documents/9thSem/amazon_ml/src/retrieve.cpp:13) computes 44 features per candidate pair: fuzzy name and address similarities, token overlap, acronym and consonant similarity, shared address numbers, text lengths, retrieval scores, and source indicators. [train.py (line 17)](/home/kuttyappan/Documents/9thSem/amazon_ml/src/train.py:17) labels those pairs from the ground truth and trains a LightGBM classifier. It selects a probability threshold on a calibration split to maximize the challenge’s macro F₀.₅, which favors precision and rewards correctly leaving singletons unmatched.
4. Write the submission. [predict.py (line 20)](/home/kuttyappan/Documents/9thSem/amazon_ml/src/predict.py:20) scores every test candidate, keeps pairs above the chosen threshold, and writes both required TSV files: all scored pairs to candidate_pairs.tsv and accepted matches to matching_results.tsv. Because decisions are made per pair, it can return multiple matches or an empty list.
   The saved training report records 94.46% candidate recall and 0.9513 macro F₀.₅ on the 7,500-entity holdout after refitting ([metrics.json (line 1)](/home/kuttyappan/Documents/9thSem/amazon_ml/work/metrics.json:1)). Those are training-data validation results, not a test-set score. France is handled by the same text and country-keyed retrieval logic, but its matching accuracy cannot be measured from the provided training labels.

12:23 AM

explain candidate gen in more detail

12:24 AM

Candidate generation in [retrieve.cpp (line 16)](/home/kuttyappan/Documents/9thSem/amazon_ml/src/retrieve.cpp:16) is an inverted-index search: it finds Source 2 and 3 records that share useful name or address words with each Source 1 record, then limits that set before the ML model runs.

1. Build an index of possible matches
   The program combines all Source 2 and Source 3 records into one target collection. For each target, it indexes each unique normalized word in its name and address. An index key includes the country, field, and word. Thus, France + name + cafe and France + address + cafe are different keys, and a US query will not retrieve an India record through the same word ([retrieve.cpp (line 18)](/home/kuttyappan/Documents/9thSem/amazon_ml/src/retrieve.cpp:18)).
   It gives each word a rarity weight:
   \[
   w=\log\left(1+\frac{N}{1+\text{number of target records containing the word}}\right)
   \]Rare words carry more evidence than words found in many businesses.
2. Retrieve records for one Source 1 query
   For each query, the program sorts its name words and address words by how few targets contain them. It uses at most 4 name words and 6 address words, skipping words present in more than 40,000 targets. It visits the indexed targets for those words and accumulates separate name and address scores. Each shared word adds \(w^2\) to the relevant score ([retrieve.cpp (line 27)](/home/kuttyappan/Documents/9thSem/amazon_ml/src/retrieve.cpp:27)).
   At this stage, a record is considered only if it shares at least one selected word in the same field and country. Fuzzy matching happens later; it cannot retrieve a pair that this word search missed.
3. Keep several kinds of strong candidates
   For the retrieved targets, name and address scores are each divided by the highest score seen for that query. The program then takes the union of ([retrieve.cpp (line 28)](/home/kuttyappan/Documents/9thSem/amazon_ml/src/retrieve.cpp:28)):
   Selection	Default limit	Ranking
   Combined	50	\(1.2n+a+1.5\min(n,a)\)
   Name-focused	12	\(n\)
   Address-focused	12	\(a\)

Here \(n\) and \(a\) are the normalized name and address scores. The min(n,a) term rewards agreement in both fields. The two extra lists retain plausible records that are strong in only one field, such as a business with a changed name or an incomplete address. Duplicates across the lists are removed, so the default is at most 74 candidates per Source 1 record, often fewer.
For each retained pair, retrieval computes the 44 matching features and writes them for LightGBM to score ([retrieve.cpp (line 30)](/home/kuttyappan/Documents/9thSem/amazon_ml/src/retrieve.cpp:30)). These retained pairs are exactly what later appears in candidate_pairs.tsv; a high probability threshold selects the smaller final match list.
The main trade-off is recall: a true match with no selected same-field word overlap cannot reach the model. On the labeled sample, this stage retrieved 94.46% of true links, setting the approximate recall ceiling for the subsequent classifier ([metrics.json (line 2)](/home/kuttyappan/Documents/9thSem/amazon_ml/work/metrics.json:2)).

12:25 AM

Can you look at the failure cases for this solution and analyze them

12:29 AM

Worked for 4m 25s

I analyzed the 7,500 held-out Source 1 entities saved by the training pipeline. These are measurable validation failures; the test set has no ground truth, so its failures cannot be counted directly.
Held-out result	Count
True Source 1 → Source 2/3 links	25,943
True links absent from candidates	1,476
True links retrieved but rejected by the model	1,287
Incorrect links accepted by the model	277
Source 1 entities with every match correct	5,244 / 7,500

The reported macro F₀.₅ is 0.9511. Even a perfect matcher restricted to the current candidates would reach only 0.9785, so candidate generation accounts for a meaningful part of the remaining gap ([saved validation metrics (line 1)](/home/kuttyappan/Documents/9thSem/amazon_ml/work/metrics.json:1), [scoring code (line 8)](/home/kuttyappan/Documents/9thSem/amazon_ml/src/train.py:8)).
Where candidates are lost
Missing addresses are the strongest failure pattern. Of 1,112 true links whose target address is blank, retrieval finds 716 (64.4%), and the final model accepts only 344 (30.9%). For links with an address, those rates are 95.7% and 92.0%. For example, Uptown Auto Body → UPTOWN AUTO BODY INC is a labeled match with an empty target address, but it never reaches the model.
Names written in another script are harder to retrieve. Candidate recall is 83.4% when the original target name contains non-ASCII characters, versus 96.1% otherwise. One missed pair is Apex Technology Pvt Ltd → एपेक्स टेक्नोलॉजी प्रा. लि.. The two names remain substantially different after the pipeline’s transliteration and word normalization.
The misses are largely a ranking and search-limit problem, rather than a total lack of shared text: all 1,476 missed true links had at least one normalized word in common in the name or address field. The retriever searches only a limited number of rare query words, skips very frequent words, and retains at most 50 combined plus 12 name-focused plus 12 address-focused results ([retrieve.cpp (line 27)](/home/kuttyappan/Documents/9thSem/amazon_ml/src/retrieve.cpp:27)). A real match can therefore be pushed out by many similar records. Retrieval is weaker for India: 90.2% of true links found, versus 97.1% for the US.
Where the matcher makes mistakes
Retrieved true links with incomplete or altered fields often score below the selected 0.72 threshold. Examples include Smith Heartland Vendome, Inc → SMITH HEARTLAND VENDOME, with no target address (score 0.691), and Indian Services Private Limited → Mr Quoevo with a matching address (score 0.383). Among retrieved true links, 28.9% of rejected pairs have a blank target address, versus 1.5% of accepted pairs. The high threshold is understandable for a precision-focused F₀.₅ score, but it costs recall on sparse records ([train.py (line 37)](/home/kuttyappan/Documents/9thSem/amazon_ml/src/train.py:37)).
False positives tend to be convincing near matches with a decisive difference. Scholarship Institute VI was incorrectly linked to Scholarship Institute VAI at 0.998 despite the name difference and a singleton ground-truth label. Harli Norwood Markets at 444 Jennings Creek Road was linked to a record at 451 Jennings Creek Road at 0.975. Another error links Glaze Technologies Ltd to Glaze Technologies Private Limited. These illustrate why address numbers, distinguishing name tokens, and legal terms need careful treatment. Some pairs look plausible by eye, so the ground-truth labels should remain the standard for calling them errors.
Of 424 true singletons, 24 received at least one false match. Each such entity scores zero under the challenge metric, making these mistakes particularly costly.
Best next experiments

1. Improve retrieval for blank addresses and script or spelling variants: test more name candidates and character-based retrieval, measuring candidate recall and runtime.
2. Train on difficult nonmatches: emphasize pairs with almost identical names or addresses but conflicting numbers or distinguishing terms.
3. Evaluate sparse records separately: compare thresholds or features for missing-address pairs against the same held-out macro F₀.₅ metric.
   The current [predictor (line 15)](/home/kuttyappan/Documents/9thSem/amazon_ml/src/predict.py:15) uses the validation-selected threshold for all candidate pairs. The examples above explain observed held-out behavior; they do not establish how the unseen France portion performs.

12:33 AM

Okay here are 3 approaches to improve our scores. Spawn 3 GPT 6 Luna Extra High sub-agents to tackle each one individiually. Ensure results and code are seperated for ease of access
Let’s keep this to three practical experiments, each changing one part of the existing pipeline. No embeddings, graphs, or new infrastructure.
ExperimentChangeWhat stays fixedEffort

1. Retrieve more useful candidates	Search and candidate budgets	Matcher initially	Medium
2. Train a better matcher	Training examples and a few features	Existing candidates	Medium
3. Improve acceptance thresholds	Final decision rule	Retrieval and model	Small
4. Fix candidate truncation before adding new retrieval methods
   All 1,476 missed links share some normalized text. First test whether a less restrictive version of our existing search recovers them.

- Increase name-only candidates from 12 → 50 → 100.
- Increase combined candidates from 40 → 100 → 200.
- Test searching more query words and relaxing the frequent-word cutoff, separately.
- Reserve a small candidate quota specifically for targets with blank addresses.
- Only add character n-gram retrieval if these simpler changes leave substantial spelling-related misses.
  Measure recovered links, the oracle macro F₀.₅ ceiling, and runtime after each change. Then retrain the matcher on the winning candidate configuration because broader retrieval introduces different negatives.
  Why this could help: it directly attacks the 1,476 links the model currently never sees.

2. Give the current LightGBM model better training examples
   The initial matcher trained on just 35,000 Source 1 entities. We have much more labeled data available.

- Compare training on 35,000, 100,000, and 250,000 entities.
- Include more true matches with blank addresses, native-script names, and substantial name changes.
- Mine high-scoring false matches from a separate training pool: similar names, conflicting house numbers, and distinguishing token differences.
- Add a few explicit features: missing-address flag, unmatched rare name tokens, legal-form differences, and better separation of house numbers from unit/postal numbers.
- Keep calibration data representative; oversampling difficult cases can distort probabilities.
  Start with more data and harder negatives, then test the extra features separately. Do not turn house-number or legal-form differences into unconditional rejection rules.
  Why this could help: it targets both the 1,287 retrieved-but-rejected matches and the 277 false positives without changing the architecture.

3. Replace the single threshold with a small, validated threshold table
   Test whether one threshold is unnecessarily restrictive for some records and too permissive for others.
   Compare:

- One global threshold.
- Separate thresholds for targets with and without addresses.
- For blank-address targets, separate thresholds for distinctive versus common names, using name frequency in the supplied data.
  Choose thresholds jointly to maximize entity-level macro F₀.₅, including singleton penalties. Do not assume the missing-address threshold should be lower—the experiment must establish that.
  Why this could help: it can recover borderline positives such as the 0.691 example cheaply. It cannot recover missing candidates or fix confidently wrong predictions at 0.998.
  I’d run 3 first because it is cheap, then 1, then 2. Keep changes only when they improve a fresh evaluation set; leaderboard submissions should go to the strongest validated combinations.

Show less
