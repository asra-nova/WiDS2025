## full results schema
key: layer dims and dropout rate
value: f1 for 5 fold cv

## summary results schema
key: layer dims and dropout rate
value: mean f1 for 5 fold cv

## epoch_history schema
key: layer dims and dropout rate
value: best epoch for 5 fold cv

## ...2.json
is for a different layer_dims and 0.0 to 0.5 dropout threshold experiment
result was the same as the first one

## `256-128-64-32-0.0.pth`
is the optimal model trained for 100 epochs on the whole data.