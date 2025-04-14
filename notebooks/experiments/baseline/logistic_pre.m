%% 1. Load Preprocessed Data
% Training data
train_aux = readtable('/data/users3/gnagaboina1/python/DATA/train/aux.csv', 'VariableNamingRule', 'preserve');
train_conn = readtable('/data/users3/gnagaboina1/python/DATA/train/connectome_matrices.csv', 'VariableNamingRule', 'preserve');
train_labels = readtable('/data/users3/gnagaboina1/python/DATA/train/labels.csv', 'VariableNamingRule', 'preserve');

% Test data
test_aux = readtable('/data/users3/gnagaboina1/python/DATA/aux.csv', 'VariableNamingRule', 'preserve');
test_conn = readtable('/data/users3/gnagaboina1/python/DATA/connectome_matrices.csv', 'VariableNamingRule', 'preserve');

%% 2. Match Participants Across Training Tables
% Ensure all training tables have common participant IDs
[common_train_ids, idx_aux, idx_conn] = intersect(train_aux.participant_id, train_conn.participant_id, 'stable');
[common_train_ids, idx_aux, idx_labels] = intersect(common_train_ids, train_labels.participant_id, 'stable');

% Filter training tables
train_aux = train_aux(idx_aux, :);
train_conn = train_conn(idx_conn, :);
train_labels = train_labels(idx_labels, :);

% Verify test_aux and test_conn have matching IDs
[~, idx_test_aux, idx_test_conn] = intersect(test_aux.participant_id, test_conn.participant_id, 'stable');
test_aux = test_aux(idx_test_aux, :);
test_conn = test_conn(idx_test_conn, :);

%% 3. Process Connectome Data
% Training connectome features
conn_data = train_conn{:, 2:end};
train_conn_feat = table(...
    mean(conn_data, 2), std(conn_data, 0, 2), max(conn_data, [], 2), min(conn_data, [], 2), ...
    'VariableNames', {'conn_mean', 'conn_std', 'conn_max', 'conn_min'});

% Test connectome features
conn_data_test = test_conn{:, 2:end};
test_conn_feat = table(...
    mean(conn_data_test, 2), std(conn_data_test, 0, 2), max(conn_data_test, [], 2), min(conn_data_test, [], 2), ...
    'VariableNames', {'conn_mean', 'conn_std', 'conn_max', 'conn_min'});

%% 4. Combine All Features
% Exclude participant_id from features
feature_vars = setdiff(train_aux.Properties.VariableNames, 'participant_id', 'stable');
X_train_full = [table2array(train_aux(:, feature_vars)), table2array(train_conn_feat)];
X_test = [table2array(test_aux(:, feature_vars)), table2array(test_conn_feat)];

%% 5. Load Target Variables
y_adhd_full = train_labels.ADHD_Outcome; % Actual ADHD target
y_sex_full = train_labels.Sex_F;         % Actual Sex target (1 = Female, 0 = Not Female)

%% 6. Split Training Data into Train and Validation Sets
rng(42);
cv = cvpartition(size(X_train_full, 1), 'HoldOut', 0.2); % 80% train, 20% validation
idx_train = training(cv);
idx_val = test(cv);

X_train = X_train_full(idx_train, :);
X_val = X_train_full(idx_val, :);
y_adhd_train = y_adhd_full(idx_train);
y_adhd_val = y_adhd_full(idx_val);
y_sex_train = y_sex_full(idx_train);
y_sex_val = y_sex_full(idx_val);

% Standardize using training statistics
mu = mean(X_train);
sigma = std(X_train);
sigma(sigma == 0) = 1;
Z_train = (X_train - mu) ./ sigma;
Z_val = (X_val - mu) ./ sigma;
Z_test = (X_test - mu) ./ sigma;

%% 7. Feature Selection for ADHD and Sex
corr_adhd = corr(Z_train, y_adhd_train);
corr_sex = corr(Z_train, y_sex_train);
top_features_adhd = find(abs(corr_adhd) > 0.05);
top_features_sex = find(abs(corr_sex) > 0.05);
selected_features = unique([top_features_adhd; top_features_sex]);
Z_train_selected = Z_train(:, selected_features);
Z_val_selected = Z_val(:, selected_features);
Z_test_selected = Z_test(:, selected_features);

% Add interaction terms
[~, idx_adhd] = sort(abs(corr_adhd), 'descend');
[~, idx_sex] = sort(abs(corr_sex), 'descend');
interact1 = Z_train(:, idx_adhd(1)) .* Z_train(:, idx_adhd(2));
interact2 = Z_train(:, idx_sex(1)) .* Z_train(:, idx_sex(2));
Z_train_enhanced = [Z_train_selected, interact1, interact2];
Z_val_enhanced = [Z_val_selected, Z_val(:, idx_adhd(1)) .* Z_val(:, idx_adhd(2)), Z_val(:, idx_sex(1)) .* Z_val(:, idx_sex(2))];
Z_test_enhanced = [Z_test_selected, Z_test(:, idx_adhd(1)) .* Z_test(:, idx_adhd(2)), Z_test(:, idx_sex(1)) .* Z_test(:, idx_sex(2))];

%% 8. Compute Class Weights for Imbalance
adhd_pos_ratio = sum(y_adhd_train == 1) / length(y_adhd_train);
sex_pos_ratio = sum(y_sex_train == 1) / length(y_sex_train);
adhd_weights = ones(size(y_adhd_train));
adhd_weights(y_adhd_train == 1) = 1 / adhd_pos_ratio;
adhd_weights(y_adhd_train == 0) = 1 / (1 - adhd_pos_ratio);
sex_weights = ones(size(y_sex_train));
sex_weights(y_sex_train == 1) = 1 / sex_pos_ratio;
sex_weights(y_sex_train == 0) = 1 / (1 - sex_pos_ratio);

% Overlay 2x weight for female ADHD
female_adhd = (y_adhd_train == 1) & (y_sex_train == 1);
adhd_weights(female_adhd) = adhd_weights(female_adhd) * 2;

%% 9. Train ADHD Model with Logistic Regression and Hyperparameter Tuning
lambdas = logspace(-4, 0, 10);
cv = cvpartition(size(Z_train_enhanced, 1), 'KFold', 5);
f1_scores_adhd = zeros(length(lambdas), 1);
for i = 1:length(lambdas)
    mdl = fitclinear(Z_train_enhanced, y_adhd_train, 'Learner', 'logistic', ...
        'Regularization', 'ridge', 'Lambda', lambdas(i), 'Weights', adhd_weights);
    pred = predict(mdl, Z_val_enhanced);
    f1_scores_adhd(i) = compute_weighted_f1(y_adhd_val, pred, y_sex_val, 2);
end
best_lambda_adhd = lambdas(find(f1_scores_adhd == max(f1_scores_adhd), 1));
adhd_model_logistic = fitclinear(Z_train_enhanced, y_adhd_train, 'Learner', 'logistic', ...
    'Regularization', 'ridge', 'Lambda', best_lambda_adhd, 'Weights', adhd_weights);

% Optimize threshold for ADHD using raw scores
adhd_val_scores = predict(adhd_model_logistic, Z_val_enhanced); % Raw linear scores
adhd_val_prob = 1 ./ (1 + exp(-adhd_val_scores)); % Convert to probabilities
thresholds = 0.3:0.05:0.7;
f1_scores_thresh = zeros(length(thresholds), 1);
for i = 1:length(thresholds)
    pred = double(adhd_val_prob >= thresholds(i));
    f1_scores_thresh(i) = compute_weighted_f1(y_adhd_val, pred, y_sex_val, 2);
end
best_threshold_adhd = thresholds(find(f1_scores_thresh == max(f1_scores_thresh), 1));

%% 10. Train Sex Model with Logistic Regression and Hyperparameter Tuning
f1_scores_sex = zeros(length(lambdas), 1);
for i = 1:length(lambdas)
    mdl = fitclinear(Z_train_enhanced, y_sex_train, 'Learner', 'logistic', ...
        'Regularization', 'ridge', 'Lambda', lambdas(i), 'Weights', sex_weights);
    pred = predict(mdl, Z_val_enhanced);
    f1_scores_sex(i) = compute_weighted_f1(y_sex_val, pred, y_sex_val, 1);
end
best_lambda_sex = lambdas(find(f1_scores_sex == max(f1_scores_sex), 1));
sex_model_logistic = fitclinear(Z_train_enhanced, y_sex_train, 'Learner', 'logistic', ...
    'Regularization', 'ridge', 'Lambda', best_lambda_sex, 'Weights', sex_weights);

% Optimize threshold for Sex
sex_val_scores = predict(sex_model_logistic, Z_val_enhanced);
sex_val_prob = 1 ./ (1 + exp(-sex_val_scores));
f1_scores_thresh_sex = zeros(length(thresholds), 1);
for i = 1:length(thresholds)
    pred = double(sex_val_prob >= thresholds(i));
    f1_scores_thresh_sex(i) = compute_weighted_f1(y_sex_val, pred, y_sex_val, 1);
end
best_threshold_sex = thresholds(find(f1_scores_thresh_sex == max(f1_scores_thresh_sex), 1));

%% 11. Validate Models and Compute Weighted F1 Scores
adhd_val_scores = predict(adhd_model_logistic, Z_val_enhanced);
adhd_val_prob = 1 ./ (1 + exp(-adhd_val_scores));
adhd_val_pred = double(adhd_val_prob >= best_threshold_adhd);
weighted_f1_adhd = compute_weighted_f1(y_adhd_val, adhd_val_pred, y_sex_val, 2);

sex_val_scores = predict(sex_model_logistic, Z_val_enhanced);
sex_val_prob = 1 ./ (1 + exp(-sex_val_scores));
sex_val_pred = double(sex_val_prob >= best_threshold_sex);
weighted_f1_sex = compute_weighted_f1(y_sex_val, sex_val_pred, y_sex_val, 1);

leaderboard_score = (weighted_f1_adhd + weighted_f1_sex) / 2;

fprintf('Logistic - Weighted F1 Score for ADHD_Outcome (2x Female ADHD): %.4f\n', weighted_f1_adhd);
fprintf('Logistic - Weighted F1 Score for Sex_F: %.4f\n', weighted_f1_sex);
fprintf('Logistic - Final Leaderboard Score (Average): %.4f\n', leaderboard_score);

%% 12. Make Predictions on Test Set
test_adhd_scores = predict(adhd_model_logistic, Z_test_enhanced);
test_adhd_prob = 1 ./ (1 + exp(-test_adhd_scores));
test_adhd_pred = double(test_adhd_prob >= best_threshold_adhd);

test_sex_scores = predict(sex_model_logistic, Z_test_enhanced);
test_sex_prob = 1 ./ (1 + exp(-test_sex_scores));
test_sex_pred = double(test_sex_prob >= best_threshold_sex);

%% 13. Create Submission File
submission_logistic = table(test_aux.participant_id, test_adhd_pred, test_sex_pred, ...
    'VariableNames', {'participant_id', 'ADHD_Outcome', 'Sex_F'});
writetable(submission_logistic, 'submission_logistic_enhanced.csv');
disp('Enhanced Logistic Regression prediction complete. Results saved to submission_logistic_enhanced.csv');

%% Weighted F1 Function (Unchanged)
function f1 = compute_weighted_f1(y_true, y_pred, y_sex, weight_factor)
    conf_mat = confusionmat(y_true, y_pred);
    tp = conf_mat(2, 2); fp = conf_mat(1, 2); fn = conf_mat(2, 1);
    female_true = (y_true == 1) & (y_sex == 1);
    female_pred_correct = female_true & (y_pred == 1);
    female_missed = female_true & (y_pred == 0);
    weighted_tp = tp + (weight_factor - 1) * sum(female_pred_correct);
    weighted_fn = fn + (weight_factor - 1) * sum(female_missed);
    weighted_fp = fp;
    precision = weighted_tp / (weighted_tp + weighted_fp);
    recall = weighted_tp / (weighted_tp + weighted_fn);
    f1 = 2 * (precision * recall) / (precision + recall);
    if isnan(f1), f1 = 0; end
end