#!/usr/bin/env python3
"""
Expand all 04_* notebooks with required teaching sections.
Preserves all code cells and their outputs (including base64 images).
Version 2: More robust section handling.
"""
import json
import os
import re
import glob

NOTEBOOKS_DIR = 'notebooks/04_data_analysis'

# Notebooks config
NOTEBOOKS = {
    '04_01_eda.ipynb': {
        'title': 'Exploratory Data Analysis (EDA)',
        'code': '04-01',
        'difficulty': 1,
        'priority': 1,
        'mental_model': (
            "**Mental Model:** Think of EDA like being a detective at a crime scene. "
            "Before you write a single report, you walk the scene: you look around (inspect), "
            "take photos (visualize), collect evidence (compute statistics), and piece together "
            "what happened (analyze). You wouldn't start a trial without doing this — "
            "and you shouldn't start modeling without doing EDA."
        ),
        'decision_guidance': (
            "| Situation | What to Do | Why |\n"
            "|-----------|------------|-----|\n"
            "| Dataset has >30 features | Use correlation heatmap first | Find redundant features early |\n"
            "| Target is imbalanced | Check `value_counts()` on target | Plan resampling strategy |\n"
            "| Numeric features vary in scale | Look at min/max/median | Decide on scaling later |\n"
            "| Distributions are highly skewed | Note which features need transforms | Plan log/box-cox transforms |\n"
            "| Categories have many levels | Flag high-cardinality features | Plan encoding strategy |"
        ),
        'common_mistakes': (
            "- Skipping EDA and going straight to modeling\n"
            "- Only looking at mean (ignore median, std, min/max)\n"
            "- Not checking the target variable distribution\n"
            "- Ignoring correlation matrix (miss multicollinearity)\n"
            "- Forgetting to check data types and missing values"
        ),
        'hands_on': (
            "**Level 1 - Observation:** Run `df.head()`, `df.info()`, `df.describe()` and read the output.\n\n"
            "**Level 2 - Guided:** Create a histogram of the first numeric column using matplotlib.\n\n"
            "**Level 3 - Practice:** Build a correlation heatmap with seaborn, annotating values.\n\n"
            "**Level 4 - Challenge:** Identify the 3 most correlated feature pairs and explain what they mean.\n\n"
            "**Level 5 - Mastery:** Create a data quality report summarizing dtypes, missing values, outliers, and distributions in one view."
        ),
        'knowledge_check': (
            "1. What does `df.describe()` show, and what can you learn from it that `df.head()` cannot tell you?\n"
            "2. Why is a correlation matrix useful? What range of values indicates strong correlation?\n"
            "3. What does a boxplot reveal that a histogram does not?\n"
            "4. Why is EDA important before modeling? What happens if you skip it?\n"
            "5. How would you detect an imbalanced target variable?"
        ),
        'exit_criteria': (
            "- [ ] I can load a dataset and inspect its structure\n"
            "- [ ] I can compute and interpret summary statistics\n"
            "- [ ] I can create histograms, boxplots, and scatter plots\n"
            "- [ ] I can build and interpret a correlation matrix\n"
            "- [ ] I can identify data quality issues from EDA"
        ),
        'next_step': 'Proceed to `04_02_data_cleaning.ipynb` to learn how to fix the issues EDA reveals.'
    },
    '04_02_data_cleaning.ipynb': {
        'title': 'Data Cleaning & Fixing',
        'code': '04-02',
        'difficulty': 1,
        'priority': 1,
        'mental_model': (
            "**Mental Model:** Data cleaning is like washing vegetables before cooking. "
            "You wouldn't throw dirty vegetables into a pot — you'd rinse off the dirt, "
            "remove the wilted leaves, and trim the bad parts. Data cleaning does the same: "
            "remove junk, fix errors, and prepare clean ingredients for your model."
        ),
        'decision_guidance': (
            "| Situation | What to Do | Why |\n"
            "|-----------|------------|-----|\n"
            "| Mixed data types in one column | Parse or coerce types | Prevents downstream errors |\n"
            "| Trailing whitespace in strings | Use `.str.strip()` | Prevents mismatched categories |\n"
            "| Inconsistent date formats | Use `pd.to_datetime()` with format | Enables time-based analysis |\n"
            "| Duplicate rows | Check with `.duplicated()` | Prevents inflated counts |\n"
            "| Garbage characters in numeric columns | Coerce to NaN, then investigate | Clean numeric data |"
        ),
        'common_mistakes': (
            "- Cleaning only the training set (not applying same steps to test)\n"
            "- Over-cleaning: removing too much data\n"
            "- Not documenting cleaning decisions\n"
            "- Changing data without understanding why\n"
            "- Forgetting to re-run cleaning after new data arrives"
        ),
        'hands_on': (
            "**Level 1 - Observation:** Run `df.info()` to identify data type issues.\n\n"
            "**Level 2 - Guided:** Fix one string column with `.str.strip()` and `.str.lower()`.\n\n"
            "**Level 3 - Practice:** Convert a column to datetime using `pd.to_datetime()` with error handling.\n\n"
            "**Level 4 - Challenge:** Find and remove all duplicate rows, documenting how many were removed.\n\n"
            "**Level 5 - Mastery:** Write a cleaning pipeline that handles type conversion, duplicates, and whitespace in sequence."
        ),
        'knowledge_check': (
            "1. What is the difference between `.dropna()` and `.fillna()`? When would you use each?\n"
            "2. Why should you apply the same cleaning steps to both training and test sets?\n"
            "3. What is the danger of cleaning data without understanding the context?\n"
            "4. How would you handle a column that has mixed types (e.g., numbers and 'N/A' strings)?\n"
            "5. Why is it important to document your cleaning decisions?"
        ),
        'exit_criteria': (
            "- [ ] I can identify data type issues with `df.info()`\n"
            "- [ ] I can fix string columns (strip, lowercase, replace)\n"
            "- [ ] I can convert columns to appropriate types\n"
            "- [ ] I can find and remove duplicate rows\n"
            "- [ ] I can apply cleaning steps consistently to new data"
        ),
        'next_step': 'Proceed to `04_03_missing_values.ipynb` to learn how to handle missing data.'
    },
    '04_03_missing_values.ipynb': {
        'title': 'Missing Values & Imputation',
        'code': '04-03',
        'difficulty': 2,
        'priority': 1,
        'mental_model': (
            "**Mental Model:** Missing data is like a puzzle with missing pieces. "
            "You have three options: (1) try to guess what the missing piece looks like (imputation), "
            "(2) leave the gap and work around it (drop rows/columns), or (3) accept that "
            "the puzzle is incomplete and adjust your expectations. The right choice depends on "
            "how many pieces are missing and how important they are."
        ),
        'decision_guidance': (
            "| Situation | What to Do | Why |\n"
            "|-----------|------------|-----|\n"
            "| <5% missing in a column | Drop rows or simple imputation | Small impact |\n"
            "| 5-30% missing | Use mean/median/mode imputation | Preserves data distribution |\n"
            "| >30% missing | Consider dropping the column | Too unreliable to impute |\n"
            "| Missing values have a pattern | Investigate data collection process | May be informative |\n"
            "| Missing in target variable | Do NOT impute — drop those rows | Imputing target leaks info |"
        ),
        'common_mistakes': (
            "- Imputing missing values before train/test split (data leakage)\n"
            "- Using mean for skewed data (use median instead)\n"
            "- Not checking if missing values are informative\n"
            "- Imputing the target variable\n"
            "- Forgetting to apply the same imputation to test data"
        ),
        'hands_on': (
            "**Level 1 - Observation:** Run `df.isnull().sum()` to see missing counts per column.\n\n"
            "**Level 2 - Guided:** Drop rows where the target is missing using `df.dropna(subset=['target'])`.\n\n"
            "**Level 3 - Practice:** Fill missing numeric values with the median using `df.fillna(df.median())`.\n\n"
            "**Level 4 - Challenge:** Use `SimpleImputer` from sklearn to impute with different strategies.\n\n"
            "**Level 5 - Mastery:** Compare model performance with and without imputation to quantify the impact."
        ),
        'knowledge_check': (
            "1. What is the difference between MCAR, MAR, and MNAR missingness? Why does it matter?\n"
            "2. Why is it dangerous to impute missing values before splitting the data?\n"
            "3. When should you use mean vs. median for imputation?\n"
            "4. What is the risk of dropping rows with missing values?\n"
            "5. How would you handle a column where 50% of values are missing?"
        ),
        'exit_criteria': (
            "- [ ] I can identify and quantify missing values\n"
            "- [ ] I can choose the right imputation strategy based on the situation\n"
            "- [ ] I can use sklearn's SimpleImputer\n"
            "- [ ] I can avoid data leakage in imputation\n"
            "- [ ] I can evaluate the impact of imputation on model performance"
        ),
        'next_step': 'Proceed to `04_04_outliers.ipynb` to learn how to detect and handle outliers.'
    },
    '04_04_outliers.ipynb': {
        'title': 'Outlier Detection & Treatment',
        'code': '04-04',
        'difficulty': 2,
        'priority': 2,
        'mental_model': (
            "**Mental Model:** Outliers are like that one person who always shows up to the party "
            "an hour early or three hours late. They're real people, but they skew the vibe. "
            "You need to decide: is this person interesting and worth keeping (a genuine extreme value), "
            "or are they causing chaos (an error)? The IQR method is your bouncer — it flags anyone "
            "too far from the crowd."
        ),
        'decision_guidance': (
            "| Situation | What to Do | Why |\n"
            "|-----------|------------|-----|\n"
            "| Outlier is a data entry error | Remove or correct | It's garbage data |\n"
            "| Outlier is genuine but extreme | Use robust scaling or transforms | Preserves information |\n"
            "| Many outliers in one feature | Consider log transform | Reduces skewness |\n"
            "| Outliers only in training set | Cap at 99th percentile | Prevents overfitting to extremes |\n"
            "| Outliers are the target signal | Keep them — they're important | Domain-dependent |"
        ),
        'common_mistakes': (
            "- Removing all outliers without investigating first\n"
            "- Using mean/std for outlier detection (use IQR or median)\n"
            "- Applying outlier treatment before train/test split\n"
            "- Removing outliers that represent real phenomena\n"
            "- Not visualizing outliers before deciding on treatment"
        ),
        'hands_on': (
            "**Level 1 - Observation:** Create boxplots for all numeric features to spot outliers visually.\n\n"
            "**Level 2 - Guided:** Calculate IQR bounds for one column and count outliers.\n\n"
            "**Level 3 - Practice:** Apply the IQR method to flag outliers in multiple columns.\n\n"
            "**Level 4 - Challenge:** Cap outliers at the 1st and 99th percentiles and compare distributions.\n\n"
            "**Level 5 - Mastery:** Compare model performance with outlier removal vs. robust scaling vs. no treatment."
        ),
        'knowledge_check': (
            "1. What is the IQR method and how does it define an outlier?\n"
            "2. When should you keep outliers vs. remove them?\n"
            "3. Why is the median more robust to outliers than the mean?\n"
            "4. What is the danger of removing outliers from the training set but not the test set?\n"
            "5. How do you distinguish between a genuine extreme value and a data error?"
        ),
        'exit_criteria': (
            "- [ ] I can visualize outliers with boxplots\n"
            "- [ ] I can detect outliers using IQR and Z-score methods\n"
            "- [ ] I can decide when to remove, cap, or keep outliers\n"
            "- [ ] I can apply outlier treatment consistently to new data\n"
            "- [ ] I can evaluate the impact of outlier treatment on model performance"
        ),
        'next_step': 'Proceed to `04_05_feature_scaling_encoding.ipynb` to learn feature transformations.'
    },
    '04_05_feature_scaling_encoding.ipynb': {
        'title': 'Feature Scaling & Encoding',
        'code': '04-05',
        'difficulty': 2,
        'priority': 2,
        'mental_model': (
            "**Mental Model:** Feature scaling is like converting currencies before comparing prices. "
            "If one product costs 1000 Japanese Yen and another costs 10 US Dollars, you can't compare "
            "them directly — you need to convert to the same unit. Scaling puts all features on the "
            "same playing field so your model can compare them fairly. Encoding is like translating "
            "text into numbers so a calculator can understand it."
        ),
        'decision_guidance': (
            "| Situation | What to Do | Why |\n"
            "|-----------|------------|-----|\n"
            "| Features have very different scales | Use StandardScaler or MinMaxScaler | Prevents scale dominance |\n"
            "| Ordinal categories (low/med/high) | Use OrdinalEncoder | Preserves order |\n"
            "| Nominal categories (colors, cities) | Use OneHotEncoder | Prevents false ordering |\n"
            "| High-cardinality categorical | Use TargetEncoder or frequency encoding | Prevents dimension explosion |\n"
            "| Skewed numeric features | Apply log transform before scaling | Reduces skewness |"
        ),
        'common_mistakes': (
            "- Scaling the target variable (usually unnecessary)\n"
            "- Fitting scaler on entire dataset before train/test split (data leakage)\n"
            "- One-hot encoding ordinal features (loses order information)\n"
            "- Not scaling tree-based models (they don't need it, but linear models do)\n"
            "- Forgetting to inverse-transform predictions after scaling"
        ),
        'hands_on': (
            "**Level 1 - Observation:** Compare feature ranges with `df.describe()` to see scale differences.\n\n"
            "**Level 2 - Guided:** Apply StandardScaler to one numeric column and compare before/after.\n\n"
            "**Level 3 - Practice:** One-hot encode a categorical column using `pd.get_dummies()`.\n\n"
            "**Level 4 - Challenge:** Use ColumnTransformer to apply different transforms to different column types.\n\n"
            "**Level 5 - Mastery:** Build a pipeline that scales numeric and encodes categorical features, then evaluates model performance."
        ),
        'knowledge_check': (
            "1. What is the difference between StandardScaler and MinMaxScaler? When would you use each?\n"
            "2. Why is it important to fit the scaler only on training data?\n"
            "3. What is the curse of one-hot encoding for high-cardinality features?\n"
            "4. Do tree-based models need feature scaling? Why or why not?\n"
            "5. What is ordinal encoding and when is it appropriate?"
        ),
        'exit_criteria': (
            "- [ ] I can identify when features need scaling\n"
            "- [ ] I can apply StandardScaler and MinMaxScaler correctly\n"
            "- [ ] I can one-hot encode nominal features\n"
            "- [ ] I can ordinal encode ordinal features\n"
            "- [ ] I can use ColumnTransformer for mixed-type datasets"
        ),
        'next_step': 'Proceed to `04_06_train_val_test_splits.ipynb` to learn proper data splitting.'
    },
    '04_06_train_val_test_splits.ipynb': {
        'title': 'Train/Validation/Test Splits',
        'code': '04-06',
        'difficulty': 1,
        'priority': 1,
        'mental_model': (
            "**Mental Model:** Splitting data is like a teacher creating practice exams and final exams. "
            "The training set is the textbook (students learn from it). The validation set is the practice exam "
            "(students test their knowledge). The test set is the final exam (the real measure of understanding). "
            "If you study the final exam beforehand, your grade is meaningless — that's data leakage."
        ),
        'decision_guidance': (
            "| Situation | What to Do | Why |\n"
            "|-----------|------------|-----|\n"
            "| Small dataset (<1000 rows) | Use 70/15/15 or even 80/10/10 | Need enough test data |\n"
            "| Large dataset (>100k rows) | Use 80/10/10 or 90/5/5 | Training set is large enough |\n"
            "| Time series data | Use temporal split (no shuffle) | Future data must not leak to training |\n"
            "| Classification with rare class | Use StratifiedShuffleSplit | Preserves class proportions |\n"
            "| Multiple models to compare | Use same test set for all | Fair comparison |"
        ),
        'common_mistakes': (
            "- Splitting before any preprocessing (scaler fits on train only)\n"
            "- Using random split for time series (causes future data leakage)\n"
            "- Not stratifying when classes are imbalanced\n"
            "- Creating too many splits (diminishing returns)\n"
            "- Not fixing random state (results not reproducible)"
        ),
        'hands_on': (
            "**Level 1 - Observation:** Run `train_test_split` with default parameters and check split sizes.\n\n"
            "**Level 2 - Guided:** Split with `test_size=0.2` and `random_state=42`, verify reproducibility.\n\n"
            "**Level 3 - Practice:** Use `StratifiedShuffleSplit` for an imbalanced classification dataset.\n\n"
            "**Level 4 - Challenge:** Create a temporal split for a time series dataset.\n\n"
            "**Level 5 - Mastery:** Build a function that handles splitting for different data types (tabular, time series, image) with validation."
        ),
        'knowledge_check': (
            "1. Why do we need three separate datasets (train, validation, test)?\n"
            "2. What is data leakage and how does improper splitting cause it?\n"
            "3. Why is `random_state` important for reproducibility?\n"
            "4. When should you use stratified splitting vs. random splitting?\n"
            "5. Why can't you use random splitting for time series data?"
        ),
        'exit_criteria': (
            "- [ ] I can split data into train/validation/test sets\n"
            "- [ ] I can choose appropriate split ratios for different dataset sizes\n"
            "- [ ] I can use stratified splitting for imbalanced classes\n"
            "- [ ] I can create temporal splits for time series\n"
            "- [ ] I understand why splitting order matters for preprocessing"
        ),
        'next_step': 'Proceed to `04_07_data_leakage.ipynb` to learn about avoiding data leakage.'
    },
    '04_07_data_leakage.ipynb': {
        'title': 'Data Leakage Prevention',
        'code': '04-07',
        'difficulty': 3,
        'priority': 1,
        'mental_model': (
            "**Mental Model:** Data leakage is like studying for an exam by looking at the answer key. "
            "You'll ace the practice test, but you haven't actually learned anything. When you deploy your model, "
            "it will fail spectacularly because it memorized the answers instead of learning the patterns. "
            "Data leakage happens when information from the future or target accidentally gets into your training process."
        ),
        'decision_guidance': (
            "| Situation | What to Do | Why |\n"
            "|-----------|------------|-----|\n"
            "| Scaling before split | Split first, then fit scaler on train only | Prevents test data leakage |\n"
            "| Encoding before split | Split first, then fit encoder on train only | Prevents test data leakage |\n"
            "| Feature engineering uses target | Use target encoding with cross-validation | Prevents target leakage |\n"
            "| Time series with future features | Only use lag features up to current time | Prevents temporal leakage |\n"
            "| Feature selection uses all data | Select features using only training data | Prevents information leakage |"
        ),
        'common_mistakes': (
            "- Fitting preprocessing on entire dataset before splitting\n"
            "- Using target encoding without cross-validation\n"
            "- Including time-derived features that use future information\n"
            "- Performing feature selection on all data before splitting\n"
            "- Not checking for proxy variables that leak the target"
        ),
        'hands_on': (
            "**Level 1 - Observation:** Examine the code in this notebook and identify where leakage could occur.\n\n"
            "**Level 2 - Guided:** Fix a leaking pipeline by moving the scaler inside the cross-validation loop.\n\n"
            "**Level 3 - Practice:** Implement proper target encoding with cross-validation.\n\n"
            "**Level 4 - Challenge:** Find and fix data leakage in a provided buggy pipeline.\n\n"
            "**Level 5 - Mastery:** Audit a complete ML pipeline for all forms of data leakage and document findings."
        ),
        'knowledge_check': (
            "1. What is data leakage and why is it dangerous?\n"
            "2. What is the difference between data leakage and overfitting?\n"
            "3. How does preprocessing before train/test split cause leakage?\n"
            "4. What is temporal leakage in time series data?\n"
            "5. How do you detect data leakage in an existing pipeline?"
        ),
        'exit_criteria': (
            "- [ ] I can identify all forms of data leakage\n"
            "- [ ] I can prevent leakage in preprocessing pipelines\n"
            "- [ ] I can implement target encoding with cross-validation\n"
            "- [ ] I can audit a pipeline for temporal leakage\n"
            "- [ ] I can explain why leakage causes inflated model performance"
        ),
        'next_step': 'Proceed to `04_08_synthesis.ipynb` to apply everything you learned in a comprehensive exercise.'
    },
    '04_08_synthesis.ipynb': {
        'title': 'Data Preparation Synthesis',
        'code': '04-08',
        'difficulty': 3,
        'priority': 2,
        'mental_model': (
            "**Mental Model:** This is your final exam for data preparation. Like a chef who has learned "
            "to chop, sauté, season, and plate individually, you now need to combine all these skills "
            "into one coherent dish. The pipeline is your recipe — it ensures every step happens in the "
            "right order, every time, for every piece of data."
        ),
        'decision_guidance': (
            "| Situation | What to Do | Why |\n"
            "|-----------|------------|-----|\n"
            "| Multiple data quality issues | Build a Pipeline that handles all in order | Consistency and reproducibility |\n"
            "| Mixed numeric and categorical | Use ColumnTransformer | Different transforms for different types |\n"
            "| Need to evaluate different strategies | Use cross-validation with Pipeline | Fair comparison without leakage |\n"
            "| Deploying to production | Save the entire Pipeline | Reproducibility and no leakage |\n"
            "| Team collaboration | Document all preprocessing decisions | Reproducibility and debugging |"
        ),
        'common_mistakes': (
            "- Not using Pipeline (manual steps get out of order)\n"
            "- Fitting preprocessing outside cross-validation\n"
            "- Not saving the fitted pipeline for production\n"
            "- Not documenting preprocessing decisions\n"
            "- Skipping the validation set and testing directly"
        ),
        'hands_on': (
            "**Level 1 - Observation:** Review the completed pipeline in this notebook and trace each step.\n\n"
            "**Level 2 - Guided:** Modify the pipeline to use a different imputation strategy.\n\n"
            "**Level 3 - Practice:** Add a new feature to the pipeline and evaluate its impact.\n\n"
            "**Level 4 - Challenge:** Compare three different preprocessing pipelines on the same dataset.\n\n"
            "**Level 5 - Mastery:** Build an end-to-end pipeline from raw data to model evaluation, including feature engineering."
        ),
        'knowledge_check': (
            "1. Why is using a Pipeline important for reproducibility?\n"
            "2. What is the correct order of operations in a data preparation pipeline?\n"
            "3. How do you handle new categorical levels that weren't in the training data?\n"
            "4. What is the difference between fitting a pipeline and transforming with it?\n"
            "5. How would you debug a pipeline that produces different results each time?"
        ),
        'exit_criteria': (
            "- [ ] I can build a complete preprocessing Pipeline with ColumnTransformer\n"
            "- [ ] I can evaluate different preprocessing strategies using cross-validation\n"
            "- [ ] I can save and load fitted pipelines for production\n"
            "- [ ] I can document all preprocessing decisions clearly\n"
            "- [ ] I can explain the full data preparation workflow to someone else"
        ),
        'next_step': 'You have completed the Data Analysis & Preparation phase. Proceed to the next phase of your ML journey!'
    }
}


def make_md_cell(source_lines):
    """Create a new markdown cell."""
    return {
        'cell_type': 'markdown',
        'metadata': {},
        'source': source_lines
    }


def process_title_cell(cell, config):
    """Modify the first cell (title cell)."""
    source = ''.join(cell['source'])

    # Update title
    source = re.sub(r'# 04\.\d+ - .*', f"# {config['code']} - {config['title']}", source)

    # Add Difficulty and Priority
    source = source.replace(
        '**Phase:** 04 - Data Analysis & Preparation',
        f'**Phase:** 04 - Data Analysis & Preparation\n\n'
        f'**Difficulty:** {config["difficulty"]}/3 | **Priority:** {config["priority"]}/2'
    )

    # Replace Learning Objectives with checkbox format
    lo_match = re.search(r'## 4\. Learning Objectives.*?(?=## 5\.)', source, re.DOTALL)
    if lo_match:
        lo_text = lo_match.group(0)
        objectives = re.findall(r'- (.+)', lo_text)
        checkbox_items = '\n'.join(f'- [ ] {obj.strip()}' for obj in objectives)
        lo_new = f"## 4. Learning Objectives\n\nBy the end of this notebook, you should be able to:\n{checkbox_items}\n"
        source = source.replace(lo_text, lo_new)

    # Replace Mental Model
    mm_match = re.search(r'## 5\. Mental Model.*', source, re.DOTALL)
    if mm_match:
        mm_text = mm_match.group(0)
        mm_new = f"## 5. Mental Model\n\n{config['mental_model']}\n\nKey: understand the data before you model it.\n"
        source = source.replace(mm_text, mm_new)

    cell['source'] = [source]
    return cell


def find_and_replace_section(cells, pattern, replacement, is_last_cell=False):
    """Find a section in the last cell and replace it."""
    cell = cells[-1]
    source = ''.join(cell['source'])

    match = re.search(pattern, source, re.DOTALL)
    if match:
        source = source[:match.start()] + replacement + source[match.end():]
        cell['source'] = [source]
    return cells


def process_notebook(filename, config):
    """Process a single notebook."""
    filepath = os.path.join(NOTEBOOKS_DIR, filename)

    with open(filepath, 'r', encoding='utf-8') as f:
        nb = json.load(f)

    cells = nb['cells']

    # Step 1: Modify title cell
    cells[0] = process_title_cell(cells[0], config)

    # Step 2: Insert Decision Guidance and Common Mistakes after "Why Does This Matter"
    for i, cell in enumerate(cells):
        if cell['cell_type'] == 'markdown':
            src = ''.join(cell['source'])
            if 'Why Does This Matter' in src:
                dg_cell = make_md_cell([
                    "## 2a. Decision Guidance\n\n",
                    f"{config['decision_guidance']}\n"
                ])
                cm_cell = make_md_cell([
                    "## 2b. Common Mistakes to Avoid\n\n",
                    f"{config['common_mistakes']}\n"
                ])
                cells.insert(i + 1, dg_cell)
                cells.insert(i + 2, cm_cell)
                break

    # Step 3: Modify last cell
    last_cell = cells[-1]
    last_src = ''.join(last_cell['source'])

    # Replace Closed-Book Recall with Knowledge Check
    kb_match = re.search(r'## \d+\. Closed-Book Recall.*?(?=## \d+\. Teach-Back)', last_src, re.DOTALL)
    if kb_match:
        kb_new = f"## 19. Knowledge Check\n\n{config['knowledge_check']}\n\n"
        last_src = last_src[:kb_match.start()] + kb_new + last_src[kb_match.end():]

    # Find Further Experiment section and replace with Hands-On Practice
    he_match = re.search(r'## \d+\. Further Experiment.*?(?=## \d+\. Verification)', last_src, re.DOTALL)
    if he_match:
        he_new = f"## 22. Hands-On Practice\n\n{config['hands_on']}\n\n"
        last_src = last_src[:he_match.start()] + he_new + last_src[he_match.end():]

    # Find Summary and add Exit Criteria + Next Step after it
    # The Summary is followed by Further Experiment (or Hands-On now)
    summary_match = re.search(r'(## \d+\. Summary\n.*?\n)(## \d+\. )', last_src, re.DOTALL)
    if summary_match:
        summary_end = summary_match.end(1)
        summary_new = (
            f"{summary_match.group(1)}\n"
            f"## 21a. Exit Criteria\n\n{config['exit_criteria']}\n\n"
            f"## 21b. Next Step\n\n{config['next_step']}\n\n"
        )
        last_src = last_src[:summary_match.start(1)] + summary_new + last_src[summary_match.start(2):]

    last_cell['source'] = [last_src]

    nb['cells'] = cells

    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)

    return len(cells)


if __name__ == '__main__':
    for filename, config in NOTEBOOKS.items():
        try:
            cell_count = process_notebook(filename, config)
            print(f"OK: {filename} -> {cell_count} cells")
        except Exception as e:
            import traceback
            print(f"ERR: {filename} -> {e}")
            traceback.print_exc()
