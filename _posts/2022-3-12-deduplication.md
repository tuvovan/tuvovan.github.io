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

| Firstname | Lastname | Email | Date of birth | Phone number |
| ----------- | ----------- | ----- | ------------- | ------------ |
| Ana | Laurel | ana_laurel@yahoo.com | 02/01/1990 | 3102105770|

### Algorithm:

So we can imagine that we have a few approaches as follow:

- Hard coding: compare each line with other, and check:

  - if they exactly match → a match

  - otherwise → a distinct

- [Dedupe](https://github.com/dedupeio/dedupe): active learning from user labelling.

  - loss function: affine gap

  - model: regularized logistic regression

- Use tokenizer and train a classifier (our method) as shown below:

![Our approach](https://tuvovan.github.io/assets/img/dedupe.drawio.png)

- data: labelled data pairs (match pairs and distinct pairs, 40 and 40 respectively)

- Main idea: As we can see, our data has 5 terms and 2 of them (name (including first and last) and email) are critical terms that can be used as feature by vectorize it (convert word to vector) to train a ML model. Date of birth and phone number are numeric values, and can be used as well .

- Main steps:
  - Collect data
  We used [dedupe](https://github.com/dedupeio/dedupe) active labelling tool to label the data.
  The data will be stored in a *json* format as follow:
```
  {
  "distinct": [
    {
      "__class__": "tuple",
      "__value__": [
        {
          "firstname": "natalie",
          "lastname": "ellerbrock",
          "cellphone": null,
          "emailname": "chadellerbrock@hotmail.com",
          "birthdate": "1987-09-10"
        },
        {
          "firstname": "dwyn",
          "lastname": "cannon",
          "cellphone": "2409180709.0",
          "emailname": "cannon20906@gmail.com",
          "birthdate": "1996-10-24"
        }
      ]
    }
  ],
  "match": [
    {
      "__class__": "tuple",
      "__value__": [
        {
          "firstname": "melissa d aka",
          "lastname": "that betch",
          "cellphone": "6023779749",
          "emailname": "dibblern@gmail.com",
          "birthdate": "1980-01-29"
        },
        {
          "firstname": "melissa d aka",
          "lastname": "that betch",
          "cellphone": "6023779749",
          "emailname": "dibblern@gmail.com",
          "birthdate": "1980-01-29"
        }
      ]
    }
  ]
}
```

  - Clean data

    - replace NaN value in date of birth with “01-01-1800”

    - remove “-” and convert to integer (“01-01-1800” → 01011800)

    - replace NaN value in phone number with “99999999999”
    
```
# helper function to encode date and cell number
def date_encode(entry):
  entry = entry.replace('-', '')
  entry = list(entry)
  entry_list = [int(i) for i in entry]
  return entry_list

def cell_encode(entry):
  entry = entry.split('.')[0]
  int_str = str(int(entry))
  if len(int_str) < 11:
    int_str += '0' * (11 - len(int_str))
  in_str_list = [int(i) for i in int_str]
  return in_str_list
```
  - Vectorize:

    - use TfidfVectorizer library of sklearn
    - the final feature has shape of 40 _ 198 for match and 40 _ 198 for distinct.
```
  # encoding
  tfidf_vectorizer1 = TfidfVectorizer(lowercase=False,max_features= 46)
  tfidf_vectorizer2 = TfidfVectorizer(lowercase=False,max_features= 34)
  tfidf_vectorizer3 = TfidfVectorizer(lowercase=False,max_features= 10)
  tfidf_vectorizer4 = TfidfVectorizer(lowercase=False,max_features= 10)

  named1_tfidf = tfidf_vectorizer1.fit_transform(df_name_d['name1'])
  named2_tfidf = tfidf_vectorizer1.transform(df_name_d['name2'])

  maild1_tfidf = tfidf_vectorizer2.fit_transform(df_mail_d['mail1'])
  maild2_tfidf = tfidf_vectorizer2.transform(df_mail_d['mail2'])

  celld1_tfidf = np.array(df_cell_d['cell1'].tolist())
  celld2_tfidf = np.array(df_cell_d['cell2'].tolist())

  birthd1_tfidf = np.array(df_birth_d['birth1'].tolist())
  birthd2_tfidf = np.array(df_birth_d['birth2'].tolist())
```

  - Train and test model: as the test data is not available right now so models are trained and tested using the same data (quite dummy)
    - Logistic Regression: `Recall: 0.8 Precision: 0.9831932773109244`
    - Linear SVM: Recall: `0.8 Precision: 0.9831932773109244`
    - XGBoost: `Recall: 1.0 Precision: 1.0`

- Some comments on the results:
  - XGBoost seems to have the best performance then we decided to go with it. To apply the model on the real data and perform dedupe, we first group the user's data into different group using email and phone number. The model will run on each group to avoid the complexity.

### Reference 
<a href="https://colab.research.google.com/drive/1NOFt6MPTGalJHCjBmhxOoCOn_D3Q1M0R?usp=sharing">Google Colab</a>
