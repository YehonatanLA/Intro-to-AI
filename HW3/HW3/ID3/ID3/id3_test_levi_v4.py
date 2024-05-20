from ID3 import ID3
from utils import *
from DecisonTree import Leaf, Question, class_counts

tree = """root: split by 'Weakness' if the value >= 0.55
root->true: is a Leaf with predictions {'T': 3}
root->false: split by 'Weakness' if the value >= 0.30000000000000004
root->false->true: is a Leaf with predictions {'F': 2}
root->false->false: split by 'Weakness' if the value >= 0.15000000000000002
root->false->false->true: is a Leaf with predictions {'T': 1}
root->false->false->false: is a Leaf with predictions {'F': 1}"""

tree_pruning = """root: split by 'Weakness' if the value >= 0.55
root->true: is a Leaf with predictions {'T': 3}
root->false: split by 'Weakness' if the value >= 0.30000000000000004
root->false->true: is a Leaf with predictions {'F': 2}
root->false->false: is a Leaf with predictions {'F': 1, 'T': 1}"""


def normalize_output(output):
    # Remove all whitespace including spaces, tabs, and newlines
    return ''.join(output.split())


# Normalize both the expected output and the actual output before comparison
expected_tree = normalize_output(tree)
expected_tree_pruning = normalize_output(tree_pruning)


def helper_print_counts_and_entropy(name, rows, labels, id3):
    parent_counts = class_counts(rows, labels)
    parent_node_entropy = id3.entropy(rows, labels)
    return f"{name} counts = {parent_counts} therefore, {name} entropy: {parent_node_entropy}"


def helper_print_tree(curr_node, str_path):
    output = ""
    if isinstance(curr_node, Leaf):
        sorted_labels = {k: curr_node.predictions[k] for k in sorted(curr_node.predictions.keys())}
        output += f"{str_path}: is a Leaf with predictions {sorted_labels}\n"
        return output
    output += f"{str_path}: split by '{curr_node.question.column}' if the value >= {curr_node.question.value}\n"
    # Recursively generate the output for the true and false branches
    output += helper_print_tree(curr_node.true_branch, str_path + "->true")
    output += helper_print_tree(curr_node.false_branch, str_path + "->false")
    return output


def test_initiation(attributes_names):
    print("\n########## test_initiation ##########")
    id3 = ID3(attributes_names)
    assert id3.label_names == attributes_names, f"Attributes names are not as expected: {id3.label_names}"
    print(f"Attributes names: {id3.label_names}")


def test_entropy(attributes_names, x_train, y_train):
    print("\n########## test_entropy ##########")
    id3 = ID3(attributes_names)

    print(f"label_arr = {y_train}")

    counts = class_counts(x_train, y_train)
    assert counts == {'F': 3, 'T': 4}, f"Counts are not as expected: {counts}"
    print(f"counts = {counts}")

    entropy = id3.entropy(x_train, y_train)
    assert np.isclose(entropy, 0.9852281360342515, atol=1e-9), f"Entropy is not as expected: {entropy}"
    print(f"Entropy = {entropy}")


def test_information_gain(attributes_names):
    print("\n########## test_information_gain ##########")
    id3 = ID3(attributes_names)
    L_rows = [['F' 'T' 'F'], ['F' 'T' 'F'], ['F' 'T' 'T']]
    L_label = ['F', 'F', 'T']
    R_rows = [['T' 'F' 'F'], ['T' 'F' 'T'], ['T' 'T' 'T'], ['T' 'T' 'F']]
    R_label = ['F', 'T', 'T', 'T']
    parent_node_entropy = 0.9852281360342515
    info_gain = id3.info_gain(L_rows, L_label, R_rows, R_label, parent_node_entropy)
    print(f"Info Gain = {info_gain}")
    assert np.isclose(info_gain, 0.12808527889139443, atol=1e-9), f"Info Gain is not as expected: {info_gain}"


def test_partition(attributes_names, x_train, y_train):
    print("\n########## test_partition ##########")
    id3 = ID3(attributes_names)

    Fever_col_index = 0
    parent_entropy = id3.entropy(x_train, y_train)

    question = Question("Fever", Fever_col_index, 0.8)
    print(f"Question: {question}")

    gain, true_rows, true_labels, false_rows, false_labels = id3.partition(x_train, y_train, question,
                                                                           parent_entropy)

    assert class_counts(true_rows, true_labels) == {'T': 3, 'F': 1}, f"True counts are not as expected"
    assert class_counts(false_rows, false_labels) == {'T': 1, 'F': 2}, f"False counts are not as expected"
    assert np.isclose(gain, 0.12808527889139443, atol=1e-9), f"Gain is not as expected: {gain}"

    print(f"Gain: {gain}, amount true: {len(true_rows)}, amount false: {len(false_rows)}")
    print(f"True include {class_counts(true_rows, true_labels)}")
    print(f"False include {class_counts(false_rows, false_labels)}")


def test_find_best_split(attributes_names, x_train, y_train):
    print("\n########## find_best_split ##########")
    id3 = ID3(attributes_names)

    best_gain, best_question, best_true_rows, best_true_labels, best_false_rows, best_false_labels = id3.find_best_split(
        x_train, y_train)

    class_counts_true = class_counts(best_true_rows, best_true_labels)
    class_counts_false = class_counts(best_false_rows, best_false_labels)
    assert class_counts_true == {'T': 3} or class_counts_true == {'T': 1,
                                                                  'F': 3}, f"True counts are not as expected: {class_counts_true}"
    assert class_counts_false == {'T': 3} or class_counts_false == {'T': 1,
                                                                    'F': 3}, f"False counts are not as expected: {class_counts_false}"
    assert np.isclose(best_gain, 0.5216406363433185, atol=1e-9), f"Gain is not as expected: {best_gain}"
    assert best_question.column == 'Weakness', f"Best question is not as expected: {best_question.column}"
    assert best_question.value == 0.55, f"Best question is not as expected: {best_question.value}"

    print(f"Best gain {best_gain}")
    print(
        f"Best question: {best_question.column} if the value >= {best_question.value}")
    print(f"True include {class_counts(best_true_rows, best_true_labels)}")
    print(f"False include {class_counts(best_false_rows, best_false_labels)}")


def test_build_tree(attributes_names, x_train, y_train):
    print("\n########## test_build_tree ##########")
    id3 = ID3(attributes_names)
    root_node = id3.build_tree(x_train, y_train)

    tree = helper_print_tree(root_node, "root")
    n_tree = normalize_output(tree)
    assert expected_tree == n_tree, f"Tree is not as expected: {tree}"
    print(f"Tree:\n{tree}")


def test_fit(attributes_names, x_train, y_train):
    print("\n########## test_fit ##########")
    id3 = ID3(attributes_names)
    id3.fit(x_train, y_train)

    tree = helper_print_tree(id3.tree_root, "root")
    n_tree = normalize_output(tree)
    assert expected_tree == n_tree, f"Tree is not as expected: {tree}"
    print("pass")


def test_predict_sample(attributes_names, x_train, y_train, x_test):
    print("\n########## test_predict_sample ##########")
    id3 = ID3(attributes_names)
    id3.fit(x_train, y_train)

    sample = np.array(x_test[0])
    print(f"Person data: {sample}")

    prediction = id3.predict_sample(sample)
    assert prediction == y_test[0], f"Prediction is not as expected: {prediction}"
    print(f"Prediction: {prediction}")
    print(f"Actual: {y_test[0]}")


def test_predict(attributes_names, x_train, y_train, x_test, y_test):
    print("\n########## test_predict ##########")
    id3 = ID3(attributes_names)

    id3.fit(x_train, y_train)
    predictions = id3.predict(x_test)

    accuracy_val = accuracy(y_test, predictions)
    assert accuracy_val == 0.8, f"Accuracy is not as expected: {accuracy_val}"

    print(f"Sample predictions = {predictions}")
    print(f"Sample actual      = {y_test}")
    print(f"Accuracy: {accuracy_val}")


def test_min_for_pruning(attributes_names, x_train, y_train):
    print("\n########### test_min_for_pruning ##########")

    id3 = ID3(attributes_names, 2)
    id3.fit(x_train, y_train)

    tree_pruning = helper_print_tree(id3.tree_root, "root")
    n_tree_pruning = normalize_output(tree_pruning)

    assert expected_tree_pruning == n_tree_pruning, f"Tree is not as expected: {tree_pruning}"
    print(f"Tree pruning:\n{tree_pruning}")


def load_data_set_t(clf_type: str):
    """
    Uses pandas to load train and test dataset.
    :param clf_type: a string equals 'ID3' '
    :return: A tuple of attributes_names (the features row) with train and test datasets split.
    """
    assert clf_type == 'ID3', 'The parameter clf_type must be ID3'
    hw_path = str(pathlib.Path(__file__).parent.absolute())
    dataset_path = hw_path + f"\\{clf_type}-dataset\\"
    train_file_path = dataset_path + "\\Corona_Symptoms_Data_v4.csv"
    test_file_path = dataset_path + "\\Corona_Symptoms_Test_Data_v4.csv"
    # Import all columns omitting the fist which consists the names of the attributes

    train_dataset = pd.read_csv(train_file_path)
    train_dataset = train_dataset.reset_index(drop=True)

    test_dataset = pd.read_csv(test_file_path)
    test_dataset = test_dataset.reset_index(drop=True)

    attributes_names = list(train_dataset.columns)
    return attributes_names, train_dataset, test_dataset


if __name__ == '__main__':
    attributes_names, train_dataset, test_dataset = load_data_set_t('ID3')
    target_attribute = 'Corona'
    (x_train, y_train, x_test, y_test) = get_dataset_split(train_dataset, test_dataset, target_attribute)

    print("-------------------------------------------")
    print("------------------ TESTS ------------------")
    print("-------------------------------------------")

    test_initiation(attributes_names)
    test_entropy(attributes_names, x_train, y_train)
    test_information_gain(attributes_names)
    test_partition(attributes_names, x_train, y_train)
    test_find_best_split(attributes_names, x_train, y_train)
    test_build_tree(attributes_names, x_train, y_train)
    test_fit(attributes_names, x_train, y_train)
    test_predict_sample(attributes_names, x_train, y_train, x_test)
    test_predict(attributes_names, x_train, y_train, x_test, y_test)
    test_min_for_pruning(attributes_names, x_train, y_train)





