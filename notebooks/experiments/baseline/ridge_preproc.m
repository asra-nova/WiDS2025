%% 1. Load Preprocessed Data
% Training data
train_aux = readtable('/data/users3/gnagaboina1/python/DATA/train/aux.csv', 'VariableNamingRule', 'preserve');
train_conn = readtable('/data/users3/gnagaboina1/python/DATA/train/connectome_matrices.csv', 'VariableNamingRule', 'preserve');
train_labels = readtable('/data/users3/gnagaboina1/python/DATA/train/labels.csv', 'VariableNamingRule', 'preserve');

% Test data
test_aux = readtable('/data/users3/gnagaboina1/python/DATA/aux.csv', 'VariableNamingRule', 'preserve');
test_conn = readtable('/data/users3/gnagaboina1/python/DATA/connectome_matrices.csv', 'VariableNamingRule', 'preserve');

%% 2. Match Participants Across Training Tables
[common_train_ids, idx_aux, idx_conn] = intersect(train_aux.participant_id, train_conn.participant_id, 'stable');
[common_train_ids, idx_aux, idx_labels] = intersect(common_train_ids, train_labels.participant_id, 'stable');

% Filter training tables
train_aux = train_aux(idx_aux, :);
train_conn = train_conn(idx_conn, :);
train_labels = train_labels(idx_labels, :);

% Verify test_aux and test_conn
[~, idx_test_aux, idx_test_conn] = intersect(test_aux.participant_id, test_conn.participant_id, 'stable');
test_aux = test_aux(idx_test_aux, :);
test_conn = test_conn(idx_test_conn, :);

%% 3. Process Connectome Data with Enhanced Features
% Training connectome features
conn_data = train_conn{:, 2:end};
train_conn_feat = table(...
    mean(conn_data, 2), std(conn_data, 0, 2), max(conn_data, [], 2), min(conn_data, [], 2), ...
    median(conn_data, 2), skewness(conn_data, 0, 2), ...
    'VariableNames', {'conn_mean', 'conn_std', 'conn_max', 'conn_min', 'conn_median', 'conn_skewness'});

% Test connectome features
conn_data_test = test_conn{:, 2:end};
test_conn_feat = table(...
    mean(conn_data_test, 2), std(conn_data_test, 0, 2), max(conn_data_test, [], 2), min(conn_data_test, [], 2), ...
    median(conn_data_test, 2), skewness(conn_data_test, 0, 2), ...
    'VariableNames', {'conn_mean', 'conn_std', 'conn_max', 'conn_min', 'conn_median', 'conn_skewness'});

%% 4. Combine All Features
feature_vars = setdiff(train_aux.Properties.VariableNames, 'participant_id', 'stable');
X_train_full = [table2array(train_aux(:, feature_vars)), table2array(train_conn_feat)];
X_test = [table2array(test_aux(:, feature_vars)), table2array(test_conn_feat)];

%% 5. Load Target Variables
y_adhd_full = train_labels.ADHD_Outcome;
y_sex_full = train_labels.Sex_F;

%% 6. Split Training Data into Train and Validation Sets
rng(42);
cv = cvpartition(size(X_train_full, 1), 'HoldOut', 0.2);
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

%% 7. Feature Selection with Correlation and RFE
corr_adhd = corr(Z_train, y_adhd_train);
corr_sex = corr(Z_train, y_sex_train);
top_features_adhd = find(abs(corr_adhd) > 0.1);
top_features_sex = find(abs(corr_sex) > 0.1);
selected_features = unique([top_features_adhd; top_features_sex]);

% Recursive Feature Elimination (RFE) for top 50 features
if length(selected_features) > 50
    mdl = fitrlinear(Z_train(:, selected_features), y_adhd_train, 'Regularization', 'ridge', 'Lambda', 1e-2);
    [~, idx_sort] = sort(abs(mdl.Beta), 'descend');
    selected_features = selected_features(idx_sort(1:50));
end
Z_train_selected = Z_train(:, selected_features);
Z_val_selected = Z_val(:, selected_features);
Z_test_selected = Z_test(:, selected_features);

% Add interaction terms and polynomial features
[~, idx_adhd] = sort(abs(corr_adhd), 'descend');
[~, idx_sex] = sort(abs(corr_sex), 'descend');
interact1 = Z_train(:, idx_adhd(1)) .* Z_train(:, idx_adhd(2));
interact2 = Z_train(:, idx_sex(1)) .* Z_train(:, idx_sex(2));
interact3 = Z_train(:, idx_adhd(1)) .* Z_train(:, idx_sex(1));
poly1 = Z_train(:, idx_adhd(1)).^2;
poly2 = Z_train(:, idx_sex(1)).^2;
Z_train_enhanced = [Z_train_selected, interact1, interact2, interact3, poly1, poly2];
Z_val_enhanced = [Z_val_selected, ...
    Z_val(:, idx_adhd(1)) .* Z_val(:, idx_adhd(2)), ...
    Z_val(:, idx_sex(1)) .* Z_val(:, idx_sex(2)), ...
    Z_val(:, idx_adhd(1)) .* Z_val(:, idx_sex(1)), ...
    Z_val(:, idx_adhd(1)).^2, ...
    Z_val(:, idx_sex(1)).^2];
Z_test_enhanced = [Z_test_selected, ...
    Z_test(:, idx_adhd(1)) .* Z_test(:, idx_adhd(2)), ...
    Z_test(:, idx_sex(1)) .* Z_test(:, idx_sex(2)), ...
    Z_test(:, idx_adhd(1)) .* Z_test(:, idx_sex(1)), ...
    Z_test(:, idx_adhd(1)).^2, ...
    Z_test(:, idx_sex(1)).^2];

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

%% 9. Train ADHD Model with Ridge Regression
lambdas = logspace(-5, 1, 15);
cv = cvpartition(size(Z_train_enhanced, 1), 'KFold', 10);
f1_scores_adhd = zeros(length(lambdas), 1);
for i = 1:length(lambdas)
    mdl = fitrlinear(Z_train_enhanced, y_adhd_train, ...
        'Learner', 'leastsquares', 'Regularization', 'ridge', ...
        'Lambda', lambdas(i), 'Weights', adhd_weights);
    pred_scores = predict(mdl, Z_val_enhanced);
    pred = double(pred_scores >= 0.5);
    f1_scores_adhd(i) = compute_weighted_f1(y_adhd_val, pred, y_sex_val, 2);
end
best_lambda_adhd = lambdas(find(f1_scores_adhd == max(f1_scores_adhd), 1));
adhd_model_ridge = fitrlinear(Z_train_enhanced, y_adhd_train, ...
    'Learner', 'leastsquares', 'Regularization', 'ridge', ...
    'Lambda', best_lambda_adhd, 'Weights', adhd_weights);

% Optimize threshold for ADHD
adhd_val_scores = predict(adhd_model_ridge, Z_val_enhanced);
thresholds = 0.2:0.02:0.8;
f1_scores_thresh = zeros(length(thresholds), 1);
for i = 1:length(thresholds)
    pred = double(adhd_val_scores >= thresholds(i));
    f1_scores_thresh(i) = compute_weighted_f1(y_adhd_val, pred, y_sex_val, 2);
end
best_threshold_adhd = thresholds(find(f1_scores_thresh == max(f1_scores_thresh), 1));

%% 10. Train Sex Model with Ridge Regression
f1_scores_sex = zeros(length(lambdas), 1);
for i = 1:length(lambdas)
    mdl = fitrlinear(Z_train_enhanced, y_sex_train, ...
        'Learner', 'leastsquares', 'Regularization', 'ridge', ...
        'Lambda', lambdas(i), 'Weights', sex_weights);
    pred_scores = predict(mdl, Z_val_enhanced);
    pred = double(pred_scores >= 0.5);
    f1_scores_sex(i) = compute_weighted_f1(y_sex_val, pred, y_sex_val, 1);
end
best_lambda_sex = lambdas(find(f1_scores_sex == max(f1_scores_sex), 1));
sex_model_ridge = fitrlinear(Z_train_enhanced, y_sex_train, ...
    'Learner', 'leastsquares', 'Regularization', 'ridge', ...
    'Lambda', best_lambda_sex, 'Weights', sex_weights);

% Optimize threshold for Sex
sex_val_scores = predict(sex_model_ridge, Z_val_enhanced);
f1_scores_thresh_sex = zeros(length(thresholds), 1);
for i = 1:length(thresholds)
    pred = double(sex_val_scores >= thresholds(i));
    f1_scores_thresh_sex(i) = compute_weighted_f1(y_sex_val, pred, y_sex_val, 1);
end
best_threshold_sex = thresholds(find(f1_scores_thresh_sex == max(f1_scores_thresh_sex), 1));

%% 11. Validate Models and Compute Weighted F1 Scores
adhd_val_scores = predict(adhd_model_ridge, Z_val_enhanced);
adhd_val_pred = double(adhd_val_scores >= best_threshold_adhd);
weighted_f1_adhd = compute_weighted_f1(y_adhd_val, adhd_val_pred, y_sex_val, 2);

sex_val_scores = predict(sex_model_ridge, Z_val_enhanced);
sex_val_pred = double(sex_val_scores >= best_threshold_sex);
weighted_f1_sex = compute_weighted_f1(y_sex_val, sex_val_pred, y_sex_val, 1);

% Ensemble rule: Adjust ADHD predictions based on strong Sex_F correlation
sex_val_pred_strong = sex_val_scores > 0.7;
adhd_val_pred(sex_val_pred_strong & adhd_val_scores > 0.3) = 1;
weighted_f1_adhd = compute_weighted_f1(y_adhd_val, adhd_val_pred, y_sex_val, 2);

leaderboard_score = (weighted_f1_adhd + weighted_f1_sex) / 2;

fprintf('Ridge - Weighted F1 Score for ADHD_Outcome (2x Female ADHD): %.4f\n', weighted_f1_adhd);
fprintf('Ridge - Weighted F1 Score for Sex_F: %.4f\n', weighted_f1_sex);
fprintf('Ridge - Final Leaderboard Score (Average): %.4f\n', leaderboard_score);

%% 12. Make Predictions on Test Set
test_adhd_scores = predict(adhd_model_ridge, Z_test_enhanced);
test_adhd_pred = double(test_adhd_scores >= best_threshold_adhd);

test_sex_scores = predict(sex_model_ridge, Z_test_enhanced);
test_sex_pred = double(test_sex_scores >= best_threshold_sex);

% Apply ensemble rule on test set
test_sex_pred_strong = test_sex_scores > 0.7;
test_adhd_pred(test_sex_pred_strong & test_adhd_scores > 0.3) = 1;

%% 13. Create Submission File
submission_ridge = table(test_aux.participant_id, test_adhd_pred, test_sex_pred, ...
    'VariableNames', {'participant_id', 'ADHD_Outcome', 'Sex_F'});
writetable(submission_ridge, 'submission_ridge_2x_weighted.csv');
disp('Enhanced Ridge Regression with 2x Female ADHD Weighting complete. Results saved to submission_ridge_2x_weighted.csv');

%% Weighted F1 Function
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