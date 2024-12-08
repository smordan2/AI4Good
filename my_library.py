def test_load():
  return 'loaded'
def compute_probs(neg,pos):
  p0 = neg/(neg+pos)
  p1 = pos/(neg+pos)
  return [p0,p1]
def cond_prob(table, evidence, evidence_value, target, target_value):
  t_subset = up_table_subset(table, target, 'equals', target_value)
  e_list = up_get_column(t_subset, evidence)
  p_b_a = sum([1 if v==evidence_value else 0 for v in e_list])/len(e_list)
  return p_b_a + .01  #Laplace smoothing factor
def cond_probs_product(full_table, evidence_row, target_column, target_column_value):
  assert target_column in full_table
  assert target_column_value in up_get_column(full_table, target_column)
  assert isinstance(evidence_row, list)
  assert len(evidence_row) == len(up_list_column_names(full_table)) - 1 
  evidence_columns = up_list_column_names(full_table)[:-1]
  evidence_complete = up_zip_lists(evidence_columns, evidence_row)
  cond_prob_list = [cond_prob(full_table, evidence_column, evidence_value, target_column, target_column_value) for evidence_column, evidence_value in evidence_complete]
  partial_numerator = up_product(cond_prob_list)
  return partial_numerator
def prior_prob(full_table, the_column, the_column_value):
  assert the_column in full_table
  assert the_column_value in up_get_column(full_table, the_column)
  t_list = up_get_column(full_table, the_column)
  p_a = sum([1 if v==the_column_value else 0 for v in t_list])/len(t_list)
  return p_a
def naive_bayes(full_table, evidence_row, target_column):
  assert target_column in full_table
  assert isinstance(evidence_row, list)
  assert len(evidence_row) == len(up_list_column_names(full_table)) - 1 
  neg = cond_probs_product(full_table, evidence_row, target_column, 0) * prior_prob(full_table, target_column, 0)
  pos = cond_probs_product(full_table, evidence_row, target_column, 1) * prior_prob(full_table, target_column, 1)
  neg, pos = compute_probs(neg, pos)
  return [neg, pos] 
def metrics(zipped_list):
  assert isinstance(zipped_list, list)
  assert all([isinstance(v, list) for v in zipped_list])
  assert all([len(v)==2 for v in zipped_list])
  assert all([isinstance(a,(int,float)) and isinstance(b,(int,float)) for a,b in zipped_list]), f'zipped_list contains a non-int or non-float'
  assert all([float(a) in [0.0,1.0] and float(b) in [0.0,1.0] for a,b in zipped_list]), f'zipped_list contains a non-binary value'

  #first compute the sum of all 4 cases. See code above
  tn = sum([1 if pair==[0,0] else 0 for pair in pred_act_list]) #act list is 400 long
  tp = sum([1 if pair==[1,1] else 0 for pair in pred_act_list])
  fp = sum([1 if pair==[1,0] else 0 for pair in pred_act_list])
  fn = sum([1 if pair==[0,1] else 0 for pair in pred_act_list])

  #now can compute precicision, recall, f1, accuracy. Watch for divide by 0.
  if (tp + fp) == 0 :
    precision = 0
  else:
    precision = tp / (tp + fp)
  if (tp + fn) == 0 :
    recall = 0
  else:
    recall = tp / (tp + fn)
  if (precision + recall) == 0 :
    f1 = 0
  else:
    f1 = 2 * (precision * recall) / (precision + recall)
  accuracy = (tp + tn) / (tp + tn + fp + fn)

  #now build dictionary with the 4 measures - round values to 2 places
  four_dictionary = {'Precision': round(precision,2),
                     'Recall': round(recall,2),
                     'F1': round(f1,2),
                     'Accuracy': round(accuracy,2)}
  #finally, return the dictionary
  return four_dictionary
def run_random_forest(train, test, target, n):
  #target is target column name
  #n is number of trees to use

  assert target in train   #have not dropped it yet
  assert target in test

  #your code below - copy, paste and align from above
  rf_clf = RandomForestClassifier(n_estimators=n, max_depth=2, random_state=0)
  feature = up_drop_column(train, target)
  actual = up_get_column(train, target)
  rf_clf.fit(feature, actual)
  probs = rf_clf.predict_proba(up_drop_column(test, target))
  pos_probs = [p for n,p in probs]

  all_mets = []
  for t in thresholds:
    predictions = [1 if pos>t else 0 for pos in pos_probs]
    pred_act_list = up_zip_lists(predictions, up_get_column(test, target))
    mets = metrics(pred_act_list)
    mets['Threshold'] = t
    all_mets = all_mets + [mets]

  metrics_table = up_metrics_table(all_mets)
  return metrics_table

  return metrics_table
def try_archs(train_table, test_table, target_column_name, architectures, thresholds):
  for arch in architectures:
    probs = up_neural_net(train_table, test_table, arch, target_column_name)

    pos_probs = [pos for neg,pos in probs]

    all_mets = []
    for t in thresholds:
      predictions = [1 if pos>=t else 0 for pos in pos_probs]
      pred_act_list = up_zip_lists(predictions, up_get_column(test_table, target_column_name))
      mets = metrics(pred_act_list)
      mets['Threshold'] = t
      all_mets = all_mets + [mets]

    print(f'Architecture: {arch}')
    display(up_metrics_table(all_mets))
