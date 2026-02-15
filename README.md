#  Sports vs. Politics Classifier

**Author:** Akshat Jain (Roll No: M25CSA003)  
**Assignment:** NLU Assignment 1 - Problem 4  

##  Project Overview
This project implements a text classification system to distinguish between **Sports** and **Politics** news articles. Using a dataset of BBC News articles, we compared three different machine learning algorithms to determine the most effective approach for this task.


##  Methodology (Problem 4)
1.  **Preprocessing:** Text cleaning, lowercasing, and removal of stop words.
2.  **Feature Engineering:** TF-IDF Vectorization (Top 5000 features).
3.  **Models Trained:**
    - Multinomial Naive Bayes (Accuracy: 100%)
    - Logistic Regression (Accuracy: 100%)
    - Random Forest Classifier (Accuracy: 98.9%)

##  How to Run
1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/m25csa003-glitch/NLU-Assignment-1.git](https://github.com/m25csa003-glitch/NLU-Assignment-1.git)
    ```
2.  **Install dependencies:**
    ```bash
    pip install pandas scikit-learn
    ```
3.  **Run the classifier:**
    ```bash
    python M25CSA003_prob4.py
    ```

##  Conclusion
The distinct vocabulary between sports and politics makes this task highly suitable for TF-IDF approaches. Naive Bayes and Logistic Regression proved to be the most efficient models.