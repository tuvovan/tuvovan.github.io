---
title: Deduplication
tags: [Deep Learning, Artificial Intelligence, Batch]
style: fill
color: secondary
description: My Understanding about this Awesome Layer.
---

<!--
Source: [GitHub](https://github.com/amorehead/jazz-nn)

![](https://amorehead.github.io/assets/img/jazz_nn.jpg) -->

## Deduplication

### Motivation:

Recently we are working with a company that owns some different shops across the countries. Each shop allows customer to register account say to have some discounts or that kind of benefits. The problem is, each customer can open multiple accounts at each shop or even at multiple shops, causing the duplication in the database. And one day, when the infrastructure need upgrades or moving to the new location, "deduplication" is needed to deal with the duplicated data.

When a customer signs up, one needs those kinds of information to fill in

- first name

- last name

- email

- date of birth

- phone number

Example:
| Syntax | Description | Email | Date of birth | Phone number |
| ----------- | ----------- | ----- | ------------- | ------------ |
| Ana | Laurel | ana_laurel@yahoo.com | 02/01/1990 | 3102105770|

### Algorithm:

So we can imagine that we have a few approaches as follow:

- Hard coding: compare each line with other, and check:

  - if they exactly match → a match

  - otherwise → a distinct

- Dedupe: active learning from user labelling.

  - loss function: affine gap

  - model: regularized logistic regression

- Use tokenizer and train a classifier (our method) as shown below:

![Our approach](https://tuvovan.github.io/assets/img/dedupe.drawio.png)

- data: labelled data pairs (match pairs and distinct pairs, 40 and 40 respectively)

- Main idea: As we can see, our data has 5 terms and 2 of them (name (including first and last) and email) are critical terms that can be used as feature by vectorize it (convert word to vector) to train a ML model. Date of birth and phone number are numeric values, and can be used as well .

- Main steps:

  - Clean data

    - replace NaN value in date of birth with “01-01-1800”

    - remove “-” and convert to integer (“01-01-1800” → 01011800)

    - replace NaN value in phone number with “99999999999”

  - Vectorize:

    - use TfidfVectorizer library of sklearn

    - the final feature has shape of 40 _ 198 for match and 40 _ 198 for distinct.

  - Train and test model: as the test data is not available right now so models are trained and tested using the same data (quite dummy)
    - Logistic Regression: `Recall: 0.8 Precision: 0.9831932773109244`
    - Linear SVM: Recall: `0.8 Precision: 0.9831932773109244`
    - XGBoost: `Recall: 1.0 Precision: 1.0`

- Some comments on the results:
  - XGBoost seems to have the best performance then we decided to go with it. To apply the model on the real data and perform dedupe, we first group the user's data into different group using email and phone number. The model will run on each group to avoid the complexity.

### Reference <a href="https://colab.research.google.com/drive/1NOFt6MPTGalJHCjBmhxOoCOn_D3Q1M0R?usp=sharing">Google Colab</a>
