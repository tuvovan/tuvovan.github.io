---
title: Deduplication
tags:
  [
    Deep Learning,
    Artificial Intelligence,
    Machine Learning,
    XGBoost,
    Logistic Regression,
  ]
style: fill
color: secondary
description: My approach to dedupe users's data.
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

| Firstname | Lastname | Email                | Date of birth | Phone number |
| --------- | -------- | -------------------- | ------------- | ------------ |
| Ana       | Laurel   | ana_laurel@yahoo.com | 02/01/1990    | 3102105770   |

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
    The data will be stored in a _json_ format as follow:

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
  - Linear SVM: `Recall: 0.8 Precision: 0.9831932773109244`
  - XGBoost: `Recall: 1.0 Precision: 1.0`

- Some comments on the results:
  - XGBoost seems to have the best performance then we decided to go with it. To apply the model on the real data and perform dedupe, we first group the user's data into different group using email and phone number. The model will run on each group to avoid the complexity.

```
  inference algorithm
  from tqdm import tqdm
  def find_dup(df, tf1, tf2):
      df_og = df.copy()
      df = df.reset_index()  # make sure indexes pair with number of rows
      df['cellphone'] = df['cellphone'].replace(' ', '99999999999', regex = True)
      df['birthdate'] = df['birthdate'].replace(' ', '1800-01-01', regex = True)
      df['firstname'] = df['firstname'].replace(' ', 'firstname', regex = True)
      df['lastname'] = df['lastname'].replace(' ', 'lastname', regex = True)
      df['birthdate'] = df['birthdate'] = df['birthdate'].apply(date_encode_pd)
      df['cellphone'] = df['cellphone'].apply(cell_encode_pd)
      # print(df['firstname'][:5])
      name = [str(first) + ' ' + str(last) for first, last in zip(list(df['firstname']), list(df['lastname']))]
      df['name'] = name
      name_tfidf = tf1.transform(df['name'])
      mail_tfidf = tf2.transform(df['emailname'])
      cell_tfidf = np.array(df['cellphone'].tolist())
      birth_tfidf = np.array(df['birthdate'].tolist())

      # new_df = pd.DataFrame(columns=df.columns)
      dups = []
      for i in tqdm(range(name_tfidf.shape[0]-1)):
          new_df = pd.DataFrame(columns=df.columns)
          namei, emaili,	celli,	birthi = name_tfidf[i], mail_tfidf[i], cell_tfidf[i], birth_tfidf[i]
          # new_df = new_df.append( [df.iloc[i]] )
          for j in range(i+1, name_tfidf.shape[0]):
              namej, emailj,	cellj,	birthj = name_tfidf[j], mail_tfidf[j], cell_tfidf[j], birth_tfidf[j]

              # x1 = hstack((namei, namej)).toarray()
              # x2 = hstack((emaili, emailj)).toarray()
              # x3 = np.hstack((celli, cellj))
              # x4 = np.hstack((birthi, birthj))
              # x3, x4 = np.expand_dims(x3, axis=0), np.expand_dims(x4, axis=0)
              x1 = namei.toarray() - namej.toarray()
              x2 = emaili.toarray() - emailj.toarray()
              x3 = celli - cellj
              x4 = birthi - birthj
              x3, x4 = np.expand_dims(x3, axis=0), np.expand_dims(x4, axis=0)
              data = np.hstack((x1, x2, x3, x4))
              pad_need = 99 - data.shape[1]
              data = np.expand_dims(np.pad(data[0], (0, pad_need), 'constant'), axis=0)
              predict_y = cal_clf.predict_proba(data[:,:99])[0]
              # print(predict_y, np.argmax(predict_y))
              if np.argmax(predict_y) == 1:
                  print(predict_y)
                  dups.append(j)
                  # new_df = new_df.append( [df.iloc[j]] )

      df_og = df_og.reset_index(drop=True)
      dups = set(dups)
      not_dups = set(range(len(df))) - dups
      new_df = df_og.loc[list(not_dups)]
      print(new_df.head())
      new_df.to_csv('/content/drive/MyDrive/Colab Notebooks/duplicated.csv', mode='a', header=False)
```

```
  rs_df = pd.DataFrame(columns=data.columns)
  rs_df.to_csv('/content/drive/MyDrive/Colab Notebooks/duplicated.csv')
  for cell in cellphone_list:
      sub_df = data.loc[data.cellphone==cell]
      if len(sub_df) < 2:
          continue
      email_list = sub_df['emailname'].unique().tolist()
      for email in email_list:
          subsub_df = sub_df.loc[sub_df.emailname==email]
          if len(subsub_df) < 2:
              continue
          find_dup(subsub_df, tfidf_vectorizer1, tfidf_vectorizer2, tfidf_vectorizer3, tfidf_vectorizer4)
```

### Reference

<a href="https://colab.research.google.com/drive/1NOFt6MPTGalJHCjBmhxOoCOn_D3Q1M0R?usp=sharing">Google Colab</a>
