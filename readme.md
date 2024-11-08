### This software is an implementation of the paper "Highly improve the accuracy of clustering algorithms based on shortest path distance"
#### How to invoke HIACSP
```
# use HIACSP.py -h to get the help information
  -h, --help            show this help message and exit
  --data_dir DATA_DIR   the director of the original dataset saved
  --save_dir SAVE_DIR   the director of results saved
  --data_name DATA_NAME
                        dataset's name
  --ratio RATIO         It is only used to verify the robustness of HIACSP's performance underdifferent overlap rate,
                        the value of ratio can be selected in [10, 20, 30, 40, 50], datasets are R15, Gauss17, Two,
                        and A1
  --load_data_local     Default value is True. If use local dataset, the value should be set as True, otherwise
                        setting as False
  --data_id DATA_ID     If you want to download dataset from uci, please set the data_id of the dataset you want
  --with_label          Default value is True. If the dataset contains labels in the last column, set this value as
                        True. Otherwise, this software will load labels from local file which name is
                        "args.data_dir/args.data_name/args.data_name-label.txt"
  --visualization       Whether to visualization the intermediate results, default value is False
  --K K                 Parameter K, range from 5 to 30
  --D D                 Iterations
  --T T                 Parameter T, range from 0.2 to 1.
  --verbose             Whether to print the process information, default value is False
  --p P                 hyper-parameter p
# you also can appoint the value of parameter K, T, and D, but we recommend that you use the default D-value since
# you can get all the results from D=1 to D=10

python HIACSP.py --T 1.0 --K 20 --data_name Aggregation.txt

# If you add --with_label and --load_data_local, the value of with_label and load_data_local will be set as Flase,
# if you add --verbose and --visualization, the value of verbose and visualization will be set as True.

```
